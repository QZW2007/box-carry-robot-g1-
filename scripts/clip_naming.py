#!/usr/bin/env python
"""Canonical clip-prefix resolution -- the single source of truth for names.

Every artifact in the pipeline hangs off ONE prefix, `<YYYYmmdd_HHMMSS>`:

    video/<p>.mp4, data/<p>.pt, data/<p>_smplx.npz, data/<p>_g1.pkl,
    data/<p>_6dof.npz, data/<p>_world_6dof.npz, data/<p>_boxpose_used.npz

That timestamp is the RECORDING folder name `record_rgbd.py` creates
(`video/bag/rgbd_<YYYYmmdd_HHMMSS>/`).  The BAG FILE NAME is deliberately NOT
part of the rule -- it is free-form (`realsense.db3`, `9.22.1.bag`, ...), so you
can rename the recording whatever you like without breaking the pipeline.

Resolution order for a bag/db3 path:
    1. `configs/clips.json` explicit override  (key = file name or stem)
    2. the file stem, when `<YYYYmmdd_HHMMSS>` is followed by `_` or by the
       end of the string -- i.e. `20260922_161727.bag` and
       `20260922_161727_unitree.bag`, but NOT `20260922_161727x.bag`
       (the timestamp is 15 chars; a 16th digit would make it a different,
       malformed stamp, so the rule refuses to guess where the stamp ends)
       NOTE: a suffix in the middle, e.g. `20260922_161727_realsense.bag`,
       is only accepted because `_` follows; the stem rule stops at the
       timestamp, so the PREFIX is still exactly `20260922_161727`
    3. the nearest ancestor directory `rgbd_<YYYYmmdd_HHMMSS>` (or bare `<ts>`)
    4. `metadata.json` -> `recorded_at`, walking up from the file
    5. None -> the caller reports the bag as unmatched

Pass `--bag-root` to the batch pipeline (or set `GMR_BAG_ROOTS`) to make the
whole pipeline read recordings that live OUTSIDE the repo, e.g. a drive full of
`rgbd_<YYYYmmdd_HHMMSS>/realsense.bag` folders.  Nothing is copied.

Run this file directly to print the map and any unmatched bags:
    python scripts/clip_naming.py
"""
import glob
import json
import os
import re
import sys
from datetime import datetime

TS_RE = re.compile(r"^(\d{8}_\d{6})(?:_|$)")
DIR_RE = re.compile(r"^rgbd_(\d{8}_\d{6})$|^(\d{8}_\d{6})$")
CLIPS_CFG = os.path.join("configs", "clips.json")
BAG_EXTS = ("bag", "db3")
DEFAULT_ROOTS = ("video",)
ROOTS_ENV = "GMR_BAG_ROOTS"
# cfg path -> (mtime, mapping); keyed by path so callers can pass a custom cfg
_OVERRIDES = {}


def bag_roots():
    """Directories searched for recordings: the repo's `video/` plus extras.

    Extras come from `GMR_BAG_ROOTS` (os.pathsep-separated, i.e. `;` on
    Windows), so `set GMR_BAG_ROOTS=D:\\bag` batches a whole drive without
    copying ~20 GB into the repo.  Later roots do not shadow earlier ones --
    a bag present in two roots is reported as a duplicate prefix, not merged.
    """
    roots = list(DEFAULT_ROOTS)
    for r in os.environ.get(ROOTS_ENV, "").split(os.pathsep):
        r = r.strip().strip('"')
        if r:
            roots.append(r)
    return roots


def bag_globs(roots=None):
    """Glob patterns for every bag/db3 under the configured roots."""
    out = []
    for r in (roots if roots is not None else bag_roots()):
        for ext in BAG_EXTS:
            out.append(os.path.join(r, "**", "*." + ext))
    return tuple(out)


def load_overrides(cfg=CLIPS_CFG, refresh=False):
    """configs/clips.json: {"9.22.2.bag": "20260922_180000", ...}.

    Keys may be a file name (`9.22.2.bag`) or a stem (`9.22.2`); the value may
    be the prefix string or a dict with a "prefix" key.  Missing or malformed
    file -> empty mapping (never raises: a bad config must not stop a batch
    scan).  Cached per cfg path and invalidated when its mtime changes.
    """
    try:
        mtime = os.path.getmtime(cfg)
    except OSError:
        mtime = None
    hit = _OVERRIDES.get(cfg)
    if hit and not refresh and hit[0] == mtime:
        return hit[1]
    out = {}
    if mtime is not None:
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in (raw or {}).items():
                if isinstance(v, str) and TS_RE.match(v):
                    out[k] = v
                elif isinstance(v, dict) and TS_RE.match(str(v.get("prefix", ""))):
                    out[k] = v["prefix"]
        except (OSError, ValueError):
            out = {}
    _OVERRIDES[cfg] = (mtime, out)
    return out


def _from_stem(stem):
    m = TS_RE.match(stem)
    return m.group(1) if m else None


def _from_dirs(path):
    """Nearest ancestor directory named rgbd_<ts> (or a bare <ts>)."""
    d = os.path.abspath(os.path.dirname(path))
    for _ in range(4):
        m = DIR_RE.match(os.path.basename(d))
        if m:
            return m.group(1) or m.group(2)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _from_metadata(path):
    """metadata.json recorded_at, walking up from the file's directory."""
    d = os.path.abspath(os.path.dirname(path))
    for _ in range(4):
        mj = os.path.join(d, "metadata.json")
        if os.path.exists(mj):
            try:
                with open(mj, "r", encoding="utf-8") as f:
                    info = json.load(f)
                raw = info.get("recorded_at")
                if raw:
                    return datetime.fromisoformat(raw).strftime("%Y%m%d_%H%M%S")
            except (OSError, ValueError, TypeError):
                return None
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def canonical_prefix(path, cfg=CLIPS_CFG):
    """Canonical <YYYYmmdd_HHMMSS> prefix for a bag/db3 (or any clip path).

    Returns None when nothing identifies the recording (report it, do not
    guess -- a wrong prefix silently orphans every downstream artifact).
    """
    if not path:
        return None
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    ov = load_overrides(cfg)
    if name in ov:
        return ov[name]
    if stem in ov:
        return ov[stem]
    return _from_stem(stem) or _from_dirs(path) or _from_metadata(path)


def all_bags(globs=None):
    """Every bag/db3 under the configured roots, de-duplicated and sorted."""
    seen, out = set(), []
    for pat in (globs if globs is not None else bag_globs()):
        for b in glob.glob(pat, recursive=True):
            k = os.path.normcase(os.path.abspath(b))
            if k not in seen:
                seen.add(k)
                out.append(b)
    return sorted(out)


def bag_map(globs=None, cfg=CLIPS_CFG):
    """Returns (mapping, unmatched): {bag_path: prefix} and [bag_path]."""
    mp, unmatched = {}, []
    for b in all_bags(globs):
        p = canonical_prefix(b, cfg)
        if p:
            mp[b] = p
        else:
            unmatched.append(b)
    return mp, unmatched


def find_bag(prefix, globs=None, cfg=CLIPS_CFG):
    """The bag whose canonical prefix is `prefix` (None if there is none).

    Also accepts the legacy "<prefix>_unitree" spelling used by the 9.11 clip.
    """
    want = {prefix}
    if prefix.endswith("_unitree"):
        want.add(prefix[:-len("_unitree")])
    mp, _ = bag_map(globs, cfg)
    hits = [b for b, p in sorted(mp.items()) if p in want]
    if not hits:
        return None
    # prefer the shallowest path (the canonical video/<prefix>.bag copy)
    return min(hits, key=lambda b: (b.count(os.sep), len(b)))


def unique_prefix_map(mp):
    """{prefix: [bags]} reversed, flagging prefixes claimed by >1 bag."""
    rev = {}
    for b, p in mp.items():
        rev.setdefault(p, []).append(b)
    return rev


def main():
    mp, unmatched = bag_map()
    rev = unique_prefix_map(mp)
    print("bag -> canonical prefix   (name every data/<prefix>*.pt after this)")
    print("search roots: %s%s"
          % (", ".join(bag_roots()),
             "" if len(bag_roots()) > 1 else
             "   (add more with --bag-root on the batch pipeline, "
             "or set %s)" % ROOTS_ENV))
    if not mp and not unmatched:
        print("  (no bag/db3 found -- check the roots above)")
    for p, bags in sorted(rev.items()):
        for b in bags:
            print("  %-58s -> %s" % (b, p))
        if len(bags) > 1:
            print("      [!] %d bags claim this prefix -- keep only one, or "
                  "map the others in %s" % (len(bags), CLIPS_CFG))
    if unmatched:
        print("\nunmatched (no timestamp in the name, folder or metadata.json;")
        print("add them to %s, e.g. {\"9.22.3.bag\": \"20260922_183000\"}):"
              % CLIPS_CFG)
        for b in unmatched:
            print("  " + b)
    ov = load_overrides()
    if ov:
        print("\noverrides in %s: %s" % (CLIPS_CFG, json.dumps(ov, ensure_ascii=False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

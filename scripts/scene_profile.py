#!/usr/bin/env python
"""scene_profile.py -- resolve "what scene am I looking at?" for one recording,
and report WHERE every parameter came from.

Motivation
----------
The pipeline grew one clip at a time, so its defaults were tuned on particular
takes: an orange box in the 9.17 room, a 0.30x0.30x0.45 box, 1280x720, a
0.45 m stool.  Those numbers are scattered across scripts as literals, so a new
recording with a different box colour, a different box size, a different
resolution or a different room silently inherits the wrong ones -- and the
failure shows up far downstream (the box "doesn't fit", the mask tracks a
t-shirt) with nothing pointing back at the assumption.

This module centralises the resolution of those parameters and, crucially,
records for each one:

    * the VALUE actually in effect
    * its SOURCE  (cli / config / bag-measured / resolution-scaled / default)
    * its CONFIDENCE (measured / derived / assumed)
    * a WARNING when a value is assumed and unmeasured

so a run can be audited before -- or long after -- the expensive steps.

It is deliberately dependency-light (numpy + optional pyrealsense2/cv2) and is
usable both as a library and as a CLI pre-flight tool:

    python scripts/scene_profile.py --bag video/<prefix>.bag
    python scripts/scene_profile.py --bag video/<prefix>.bag --hsv 5,90,80:28,255,255
    python scripts/scene_profile.py --bag video/<prefix>.bag --box-size auto
    python scripts/scene_profile.py --bag video/<prefix>.bag --json .workbuddy/<p>_scene.json

Exit code is 0 when the profile is usable, 1 when a hard problem was found
(missing depth stream, unlocked exposure, unreadable bag) -- so it can gate a
batch run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------
# Baseline resolution every literal in this repo was tuned against.  Anything
# derived per-pixel scales against this, so a 640x480 take does not inherit a
# 720p `--roi-pad` and a 4K take does not inherit a 720p `--min-area`.
# --------------------------------------------------------------------------
BASE_W, BASE_H = 1280, 720
BASE_ROI_PAD = 280        # box_traj_from_depth 720p value used by the batch
BASE_MIN_AREA = 400
BASE_DEPTH_STEP = 3

# Ranges outside these are almost certainly a typo, not a scene.
BOX_EDGE_RANGE_M = (0.05, 2.0)
DEFAULT_BOX_SIZE = (0.30, 0.30, 0.45)
DEFAULT_STOOL_H = 0.45
ORANGE_HSV = "5,90,80:28,255,255"
BLUE_HSV = "100,70,40:130,255,255"

BOXES_CFG = os.path.join("configs", "boxes.json")
CLIPS_CFG = os.path.join("configs", "clips.json")
SCENES_CFG = os.path.join("configs", "scenes.json")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _prov(value, source, confidence, note=""):
    """One resolved parameter with provenance."""
    d = {"value": value, "source": source, "confidence": confidence}
    if note:
        d["note"] = note
    return d


def parse_hsv(spec):
    """'H,S,V:H,S,V' -> (np.uint8[3], np.uint8[3])."""
    try:
        lo_s, hi_s = spec.split(":")
        lo = np.array([int(x) for x in lo_s.split(",")], dtype=np.uint8)
        hi = np.array([int(x) for x in hi_s.split(",")], dtype=np.uint8)
    except Exception:
        raise ValueError("bad --hsv %r; expected 'H,S,V:H,S,V'" % spec)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("bad --hsv %r; need 3 components each side" % spec)
    return lo, hi


def parse_size(spec):
    v = np.array([float(x) for x in str(spec).split(",")], dtype=float)
    if v.shape != (3,):
        raise ValueError("bad --box-size %r; expected 'W,D,H'" % spec)
    return v


def prefix_from_bag(bag):
    """Canonical prefix, mirroring scripts/clip_naming.py semantics.

    Explicit clips.json override -> a `<YYYYmmdd_HHMMSS>` file stem (with an
    optional suffix) -> the nearest `rgbd_<ts>` / `<ts>` ancestor directory ->
    metadata.json `recorded_at`.  Never guessed: returns None if none apply.
    """
    bag = os.path.abspath(bag)
    stem = os.path.splitext(os.path.basename(bag))[0]
    try:
        with open(CLIPS_CFG, "r", encoding="utf-8") as f:
            over = json.load(f)
        for key in (os.path.basename(bag), stem):
            if key in over:
                return str(over[key])
    except (OSError, ValueError):
        pass
    m = re.match(r"^(\d{8}_\d{6})(?:_|$)", stem)
    if m:
        return m.group(1)
    cur = os.path.dirname(bag)
    for _ in range(4):                       # walk up a few levels
        b = os.path.basename(cur)
        m = re.match(r"^rgbd_(\d{8}_\d{6})$", b) or re.match(r"^(\d{8}_\d{6})$", b)
        if m:
            return m.group(1)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    cur = os.path.dirname(bag)
    for _ in range(4):
        md = os.path.join(cur, "metadata.json")
        if os.path.exists(md):
            try:
                with open(md, "r", encoding="utf-8") as f:
                    rec = json.load(f).get("recorded_at") or ""
                m = re.match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", rec)
                if m:
                    return "%s%s%s_%s%s%s" % m.groups()
            except (OSError, ValueError):
                pass
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def read_metadata(bag):
    """metadata.json written by record_rgbd.py.

    Looks in the bag's own directory and its ancestors, and -- because a bag is
    often RENAMED/MOVED to `video/<prefix>.bag` while its metadata stays behind
    in the original `video/bag/rgbd_<ts>/` folder -- also in any recording
    folder whose timestamp matches the bag's prefix.
    """
    bag = os.path.abspath(bag)
    cur = os.path.dirname(bag)
    for _ in range(4):
        md = os.path.join(cur, "metadata.json")
        if os.path.exists(md):
            try:
                with open(md, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                return {}
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # fall back to any sibling recording folder whose metadata matches this
    # prefix.  Bags get renamed/moved to `video/<prefix>.bag` while their
    # metadata stays behind in the original folder (which may be named
    # `9.17`, `rgbd_<ts>`, or anything else), so match on recorded_at.
    ts = prefix_from_bag(bag)
    vid = os.path.dirname(bag)
    for md in sorted(glob.glob(os.path.join(vid, "*", "metadata.json")) +
                     glob.glob(os.path.join(vid, "metadata.json"))):
        try:
            with open(md, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        rec = str(d.get("recorded_at") or "")
        digits = re.sub(r"[^0-9]", "", rec)[:14]          # YYYYmmddHHMMSS
        stamp = ("%s_%s" % (digits[:8], digits[8:14])) if len(digits) >= 14 else ""
        if ts and (stamp == ts or rec.startswith(ts[:4] + "-" + ts[4:6] + "-" + ts[6:8])):
            return d
    return {}


# --------------------------------------------------------------------------
# config-driven parameter lookup (deliberately mirrors batch_pipeline's chain:
# name substring -> exact prefix -> prefixes list -> session startswith)
# --------------------------------------------------------------------------
def _cfg_load(path):
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, ValueError) as e:
        return {}, "%s unreadable (%s) -- ignoring it" % (path, e)


def _match_entry(cfg, prefix):
    """Return (name, spec, how) for the most specific matching entry."""
    if not prefix:
        return None, None, None
    for name, spec in cfg.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("prefix") == prefix:
            return name, spec, "prefix"
        if prefix in (spec.get("prefixes") or []):
            return name, spec, "prefixes"
        if spec.get("session") and prefix.startswith(str(spec["session"])):
            return name, spec, "session %s" % spec["session"]
    for name, spec in cfg.items():
        if isinstance(spec, dict) and name and name in prefix:
            return name, spec, "name substring"
    return None, None, None


def resolve_hsv(prefix, cli_hsv=None):
    """CLI -> configs/boxes.json (prefix/prefixes/session/name) -> orange default."""
    if cli_hsv:
        return _prov(cli_hsv, "modes: --hsv", "user-given")
    cfg, err = _cfg_load(BOXES_CFG)
    name, spec, how = _match_entry(cfg, prefix)
    if spec and spec.get("hsv"):
        note = err or ("entry %r" % name if name else "")
        return _prov(spec["hsv"], "configs/boxes.json (%s)" % how, "measured", note)
    return _prov(ORANGE_HSV, "built-in orange default", "assumed",
                 err or "no entry in configs/boxes.json for this prefix")


def resolve_box_size(prefix, cli_size=None):
    """CLI -> configs/scenes.json estimate -> default (loudly flagged)."""
    if cli_size:
        v = parse_size(cli_size)
        return _prov([float(x) for x in v], "modes: --box-size", "user-given")
    cfg, err = _cfg_load(SCENES_CFG)
    name, spec, how = _match_entry(cfg, prefix)
    if spec and spec.get("box_size"):
        return _prov([float(x) for x in spec["box_size"]],
                     "configs/scenes.json (%s)" % how, "measured", err or "")
    return _prov(list(DEFAULT_BOX_SIZE), "built-in default", "assumed",
                 "NEVER calipered for the 9.17/9.22 takes. The box size is the "
                 "fit's ONLY scale source: a wrong value moves the box and its "
                 "orientation systematically, not slightly. Caliper it and pass "
                 "--box-size, or let the fit estimate it with --box-size auto.")


def resolve_stool_h(prefix, cli_h=None):
    if cli_h:
        return _prov(float(cli_h), "modes: --stool_h", "user-given")
    cfg, err = _cfg_load(SCENES_CFG)
    name, spec, how = _match_entry(cfg, prefix)
    if spec and spec.get("stool_h"):
        return _prov(float(spec["stool_h"]), "configs/scenes.json (%s)" % how,
                     "measured", err or "")
    return _prov(DEFAULT_STOOL_H, "built-in default", "assumed",
                 "assumed 0.45 m; the rise gate is checked against it, so a "
                 "wrong stool height silently rescales that acceptance test")


def scale_resolution_params(width, height, roi_pad=None, min_area=None,
                           depth_step=None):
    """Scale the pixel-space knobs against the 1280x720 baseline.

    These were the least portable numbers in the pipeline: `--roi-pad 280
    --min-area 400` are 720p values, while the older 640x480 script defaults
    are 140/120.  Handing a 640x480 take the 720p pair makes the search window
    too wide (it latches onto the same-coloured jumper) and the area gate too
    strict (the box is ~4x fewer pixels).  Derive them unless overridden.
    """
    area_ratio = (float(width) * float(height)) / float(BASE_W * BASE_H)
    width_ratio = float(width) / float(BASE_W)
    out = {}
    if roi_pad is None:
        out["roi_pad"] = _prov(int(round(BASE_ROI_PAD * width_ratio)),
                               "scaled from 720p baseline (%d px)" % BASE_ROI_PAD,
                               "derived")
    else:
        out["roi_pad"] = _prov(int(roi_pad), "modes: --roi-pad", "user-given")
    if min_area is None:
        out["min_area"] = _prov(int(round(BASE_MIN_AREA * area_ratio)),
                                "scaled from 720p baseline (%d px^2)" % BASE_MIN_AREA,
                                "derived")
    else:
        out["min_area"] = _prov(int(min_area), "modes: --min-area", "user-given")
    if depth_step is None:
        out["depth_step"] = _prov(BASE_DEPTH_STEP, "720p baseline", "derived")
    else:
        out["depth_step"] = _prov(int(depth_step), "modes: --depth-step",
                                  "user-given")
    return out


# --------------------------------------------------------------------------
# bag introspection
# --------------------------------------------------------------------------
def probe_bag(bag):
    """Streams / intrinsics / depth-scale straight from the bag (the truth)."""
    info = {"ok": False, "problems": [], "streams": []}
    try:
        import pyrealsense2 as rs
    except ImportError:
        info["problems"].append("pyrealsense2 not importable -- cannot read the bag")
        return info
    pipe = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, bag, repeat_playback=False)
    try:
        prof = pipe.start(cfg)
    except RuntimeError as e:
        info["problems"].append("cannot open bag: %s" % e)
        return info
    try:
        dev = prof.get_device()
        has_depth = False
        for s in dev.sensors:
            for sp in s.get_stream_profiles():
                if sp.stream_type() == rs.stream.depth:
                    has_depth = True
        for s in prof.get_streams():
            if s.is_video_stream_profile():
                v = s.as_video_stream_profile()
                info["streams"].append({
                    "name": s.stream_name(), "w": v.width(), "h": v.height(),
                    "fps": v.fps(), "format": str(s.format())})
            else:
                info["streams"].append({"name": s.stream_name(),
                                        "format": str(s.format())})
        cps = prof.get_stream(rs.stream.color).as_video_stream_profile()
        ci = cps.get_intrinsics()
        info["color"] = {"w": ci.width, "h": ci.height,
                         "fx": ci.fx, "fy": ci.fy, "ppx": ci.ppx, "ppy": ci.ppy,
                         "model": str(ci.model),
                         "coeffs": [float(x) for x in ci.coeffs]}
        try:
            info["depth_scale"] = float(
                dev.first_depth_sensor().get_depth_scale())
        except Exception:
            info["depth_scale"] = None
        info["has_depth"] = has_depth
        if not has_depth:
            info["problems"].append(
                "NO DEPTH STREAM (colour-only recording) -- the depth route "
                "cannot run on this clip; re-record or use the ArUco route")
        try:
            playback = dev.as_playback()
            dur = playback.get_duration()
            info["duration_s"] = (dur.total_seconds() if hasattr(dur, "total_seconds")
                                  else dur / 1e9)
        except Exception:
            info["duration_s"] = None
        info["ok"] = has_depth
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
    return info


def grab_middle_frame(bag, max_scan=4000):
    """One middle colour frame, for colour/size advisories (~seconds)."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None
    pipe = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, bag, repeat_playback=False)
    try:
        prof = pipe.start(cfg)
    except RuntimeError:
        return None
    try:
        try:
            prof.get_device().as_playback().set_real_time(False)
        except Exception:
            pass
        align = rs.align(rs.stream.color)
        frames = []
        try:
            while len(frames) < max_scan:
                fs = align.process(pipe.wait_for_frames())
                c = fs.get_color_frame()
                if c:
                    frames.append(np.asanyarray(c.get_data()))
        except RuntimeError:
            pass
        return frames[len(frames) // 2] if frames else None
    finally:
        try:
            pipe.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------
# advisories measured from a single frame
# --------------------------------------------------------------------------
def advise_box_size(frame, K, roi_frac=0.5):
    """Rough box-size estimate from the largest saturated blob, for ADVISORY use.

    Deliberately conservative: this reads one frame, assumes the blob is a
    fronto-parallel face and that the blob's pixel width spans one full box
    edge.  That is good to maybe +/-20 %, which is enough to catch "this is not
    a 0.30 m box" and to seed a manual measurement -- NOT enough to replace a
    caliper.  Returns (est_w, est_h, confidence, detail) or None.
    """
    try:
        import cv2
    except ImportError:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    # ignore the very top/bottom bands: floor and ceiling are not the box
    h = frame.shape[0]
    band = np.zeros_like(sat)
    band[int(0.12 * h):int(0.92 * h)] = sat[int(0.12 * h):int(0.92 * h)]
    for thr in (140, 110, 90, 70):
        m = (band > thr).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < 400:
            continue
        x, y, w, hh = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                       stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
        fill = area / float(max(w * hh, 1))
        if fill < 0.55:
            continue
        # metres per pixel at an assumed 3 m standoff (the repo's parked range)
        return {"px_w": int(w), "px_h": int(hh), "fill": round(float(fill), 3),
                "sat_thr": thr,
                "at_3m_w": round(float(w) * 3.0 / K[0, 0], 3),
                "at_3m_h": round(float(hh) * 3.0 / K[1, 1], 3)}
    return None


def advise_hsv(frame):
    """Advisory colour clusters, using calib_hsv.py's estimator per blob.

    Does NOT replace `calib_hsv.py` (which lets a human frame the box face) --
    it reports the few largest saturated blobs so a first run can be started
    without a manual step, and says so.
    """
    try:
        import cv2
    except ImportError:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    h = frame.shape[0]
    band = np.zeros_like(sat)
    band[int(0.10 * h):int(0.94 * h)] = sat[int(0.10 * h):int(0.94 * h)]
    m = (band > 120).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    cands = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 500:
            continue
        sel = lab == k
        pix = hsv[sel].astype(np.float64)
        hh, s, v = pix[:, 0], pix[:, 1], pix[:, 2]
        ang = np.deg2rad(hh * 2.0)
        sm, cm = np.sin(ang).mean(), np.cos(ang).mean()
        R = float(np.hypot(sm, cm))
        hmean = float(np.rad2deg(np.arctan2(sm, cm))) % 360.0
        hstd = 0.0 if R > 0.999 else float(np.sqrt(-2.0 * np.log(max(R, 1e-6))))
        c = hmean / 2.0
        d = max(3.0 * hstd / 2.0, 10.0)
        lo = (max(0, int(round(c - d))), int(max(0, np.median(s) - 80)),
              int(max(0, np.median(v) - 80)))
        hi = (min(179, int(round(c + d))), 255, 255)
        cands.append({
            "area_px": int(stats[k, cv2.CC_STAT_AREA]),
            "hsv": "%d,%d,%d:%d,%d,%d" % (lo + hi),
            "h_mean_deg": round(hmean, 1),
            "h_std_deg": round(hstd, 1),
        })
    cands.sort(key=lambda d: -d["area_px"])
    return cands[:4]


# --------------------------------------------------------------------------
# the profile
# --------------------------------------------------------------------------
def build_profile(bag, prefix=None, hsv=None, box_size=None, stool_h=None,
                  roi_pad=None, min_area=None, depth_step=None,
                  sample_frame=True):
    prof = {"bag": os.path.abspath(bag), "warnings": [], "problems": []}
    if not os.path.exists(bag):
        prof["problems"].append("bag not found: %s" % bag)
        return prof
    prof["prefix"] = prefix or prefix_from_bag(bag)
    if not prof["prefix"]:
        prof["warnings"].append(
            "could not resolve a canonical prefix (no clips.json entry, no "
            "timestamp stem, no rgbd_<ts> folder, no metadata recorded_at) -- "
            "name the GVHMR output explicitly or pin it in configs/clips.json")

    meta = read_metadata(bag)
    prof["metadata"] = {k: meta.get(k) for k in
                        ("device_name", "width", "height", "fps", "recorded_at")}
    cs = meta.get("color_settings") or {}
    if "enable_auto_exposure" in cs:
        locked = float(cs["enable_auto_exposure"]) == 0.0
        prof["exposure"] = _prov(
            {"locked": locked, "exposure_us": cs.get("exposure"),
             "gain": cs.get("gain"), "white_balance": cs.get("white_balance")},
            "metadata.json", "measured")
        if not locked:
            prof["warnings"].append(
                "AUTO-EXPOSURE WAS ON -- depth and colour drift during the take; "
                "the 9.17 clip lost its HSV calibration this way. Lock exposure "
                "(record_rgbd.py --exposure N) and re-record for a quantitative run")
    else:
        prof["exposure"] = _prov(None, "metadata.json absent/incomplete", "unknown")
        prof["warnings"].append("no exposure record in metadata.json")

    info = probe_bag(bag)
    prof["bag_probe"] = info
    prof["problems"].extend(info["problems"])
    ci = info.get("color")
    if ci:
        prof["resolution"] = _prov({"w": ci["w"], "h": ci["h"]},
                                   "bag colour intrinsics", "measured")
        prof["intrinsics"] = _prov(
            {"fx": round(ci["fx"], 3), "fy": round(ci["fy"], 3),
             "cx": round(ci["ppx"], 3), "cy": round(ci["ppy"], 3),
             "model": ci["model"]}, "bag (live)", "measured")
        K = np.array([[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]],
                      [0, 0, 1.0]])
        prof.update(scale_resolution_params(ci["w"], ci["h"], roi_pad, min_area,
                                            depth_step))
    else:
        w = meta.get("width") or BASE_W
        h = meta.get("height") or BASE_H
        prof["resolution"] = _prov({"w": int(w), "h": int(h)},
                                   "metadata.json (bag unreadable)", "derived")
        prof["intrinsics"] = _prov(None, "unavailable", "unknown")
        prof["problems"].append("colour intrinsics unavailable")
        K = None
        prof.update(scale_resolution_params(int(w), int(h), roi_pad, min_area,
                                            depth_step))

    prof["hsv"] = resolve_hsv(prof.get("prefix"), hsv)
    prof["box_size"] = resolve_box_size(prof.get("prefix"), box_size)
    prof["stool_h"] = resolve_stool_h(prof.get("prefix"), stool_h)

    # advisories measured from ONE frame (cheap, explicitly low-confidence)
    if sample_frame and info.get("ok") and K is not None:
        fr = grab_middle_frame(bag)
        if fr is not None:
            prof["advisory"] = {}
            cl = advise_hsv(fr)
            if cl:
                prof["advisory"]["colour_candidates"] = cl
                if prof["hsv"]["source"].startswith("built-in"):
                    prof["warnings"].append(
                        "hsv is the built-in default; the largest saturated blob "
                        "suggests %s -- confirm with scripts/calib_hsv.py, or pass "
                        "--hsv" % cl[0]["hsv"])
            bsz = advise_box_size(fr, K)
            if bsz:
                prof["advisory"]["box_face_px"] = bsz
                prof["warnings"].append(
                    "box size is not measured; the largest box-like blob is "
                    "%dx%d px => ~%.2f m wide IF it sits at 3 m "
                    "(px*Z/fx). Caliper the box, or pass --box-size" %
                    (bsz["px_w"], bsz["px_h"], bsz["at_3m_w"]))

    # assumed values always surface as warnings, by design
    for key in ("hsv", "box_size", "stool_h"):
        p = prof.get(key)
        if isinstance(p, dict) and p.get("confidence") == "assumed":
            note = p.get("note") or ""
            msg = "%s is ASSUMED (%s)" % (key, p["source"])
            if note:
                msg += " -- " + note
            if msg not in prof["warnings"]:
                prof["warnings"].append(msg)
    return prof


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def render(prof, stream=sys.stdout):
    def w(s=""):
        stream.write(s + "\n")

    w("=" * 78)
    w("SCENE PROFILE  %s" % (prof.get("prefix") or "<prefix unresolved>"))
    w("  bag: %s" % prof.get("bag"))
    md = prof.get("metadata") or {}
    if md.get("device_name"):
        w("  device: %s  %sx%s @ %s fps  recorded %s"
          % (md.get("device_name"), md.get("width"), md.get("height"),
             md.get("fps"), md.get("recorded_at")))
    w("=" * 78)
    w("%-14s %-34s %-10s %s" % ("PARAM", "VALUE", "SOURCE", "CONFIDENCE"))
    w("-" * 78)

    def row(name, p, fmt=str):
        if not isinstance(p, dict):
            return
        v = p.get("value")
        if isinstance(v, dict):
            v = " ".join("%s=%s" % (k, x) for k, x in v.items())
        elif isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)
        w("%-14s %-34s %-10s %s" % (name, fmt(v)[:34],
                                    str(p.get("source"))[:10],
                                    p.get("confidence")))

    row("resolution", prof.get("resolution"))
    row("intrinsics", prof.get("intrinsics"))
    row("exposure", prof.get("exposure"))
    row("hsv", prof.get("hsv"))
    row("box_size", prof.get("box_size"))
    row("stool_h", prof.get("stool_h"))
    row("roi_pad", prof.get("roi_pad"))
    row("min_area", prof.get("min_area"))
    row("depth_step", prof.get("depth_step"))

    adv = prof.get("advisory") or {}
    if adv.get("colour_candidates"):
        w("-" * 78)
        w("colour candidates on the middle frame (advisory -- confirm with calib_hsv.py):")
        for c in adv["colour_candidates"]:
            w("   %-26s area %6d px   hue %.1f +/- %.1f deg"
              % (c["hsv"], c["area_px"], c["h_mean_deg"], c["h_std_deg"]))
    if adv.get("box_face_px"):
        b = adv["box_face_px"]
        w("box-like blob: %dx%d px (fill %.2f) -> ~%.2f x %.2f m if 3 m away"
          % (b["px_w"], b["px_h"], b["fill"], b["at_3m_w"], b["at_3m_h"]))

    if prof.get("warnings"):
        w("-" * 78)
        w("WARNINGS (%d) -- each one is an assumption you may want to replace:"
          % len(prof["warnings"]))
        for m in prof["warnings"]:
            w("  ! " + m)
    if prof.get("problems"):
        w("-" * 78)
        w("PROBLEMS (%d) -- these block the depth route:" % len(prof["problems"]))
        for m in prof["problems"]:
            w("  X " + m)
    w("=" * 78)
    w("RESULT: %s" % ("BLOCKED" if prof.get("problems") else
                      ("OK (with assumptions)" if prof.get("warnings") else "OK")))
    return prof


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag", required=True)
    ap.add_argument("--prefix", default=None,
                    help="override the canonical prefix")
    ap.add_argument("--hsv", default=None,
                    help="explicit 'H,S,V:H,S,V'; overrides configs/boxes.json")
    ap.add_argument("--box-size", default=None,
                    help="explicit 'W,D,H' in metres; overrides the default")
    ap.add_argument("--stool-h", dest="stool_h", type=float, default=None)
    ap.add_argument("--roi-pad", dest="roi_pad", type=int, default=None)
    ap.add_argument("--min-area", dest="min_area", type=int, default=None)
    ap.add_argument("--depth-step", dest="depth_step", type=int, default=None)
    ap.add_argument("--no-frame", action="store_true",
                    help="skip the single-frame colour/size advisory (faster)")
    ap.add_argument("--json", default=None, help="also write the profile as JSON")
    args = ap.parse_args()

    prof = build_profile(args.bag, args.prefix, args.hsv, args.box_size,
                         args.stool_h, args.roi_pad, args.min_area,
                         args.depth_step, sample_frame=not args.no_frame)
    render(prof)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(prof, f, ensure_ascii=False, indent=2, default=str)
        print("wrote %s" % args.json)
    return 1 if prof.get("problems") else 0


if __name__ == "__main__":
    sys.exit(main())

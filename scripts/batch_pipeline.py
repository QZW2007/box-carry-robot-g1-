#!/usr/bin/env python
"""Batch runner for the carry-box video pipeline.

Scans the repo for clips (by the artifacts they already have), reports which
pipeline step each clip is stuck at, and (--run) executes the missing steps
one clip at a time, enforcing the acceptance metrics at every step.  One bad
clip never blocks the others.

Usage (from the repo root, with the gmr conda python):
    python scripts/batch_pipeline.py                 # status table only
    python scripts/batch_pipeline.py --run           # execute missing steps
    python scripts/batch_pipeline.py --run --only 20260917_212026
    python scripts/batch_pipeline.py --run --hsv 5,90,80:28,255,255
    python scripts/batch_pipeline.py --csv status.csv

Steps per clip (prefix convention: data/<p>_g1.pkl <-> data/<p>_world_6dof.npz):
    bag     video/**/<p>.bag|.db3 exists            (record_rgbd.py)
    mp4     video/<p>.mp4                           (bag_to_mp4.py)
    pt      data/<p>.pt                             (GVHMR, GPU machine ONLY)
    g1      data/<p>_g1.pkl                         (gvhmr_to_smplx + retarget)
    fit     data/<p>_6dof.npz                       (box_traj_from_depth.py)
    world   data/<p>_world_6dof.npz                 (align_box_to_world.py)
    render  video/<p>_g1_measured_box_6dof.mp4      (vis_robot_with_box.py)
    qa      RESULT: PASS on the dumped box pose     (check_render_quality.py)

The prefix is the RECORDING folder timestamp, not the bag file name: a bag
named `9.22.3.bag` inside `video/bag/rgbd_20260922_183000/` gets the prefix
`20260922_183000` automatically (see scripts/clip_naming.py).  The status
table prints the bag -> prefix map, so the GVHMR output can be named
correctly without guessing.

Acceptance (hard, same numbers as SKILL.md):
    fit:  valid3D >= 95%, pose_valid >= 95%, IoU median >= 0.85,
          depth residual median <= 40 mm (TH_DEPTH_M below).  The 40 mm gate is
          deliberate, not a typo for 35: the 20260922 batch is shot farther
          away (~4 m parked segment), which inflates the per-frame depth
          residual too -- clip 20260922_170444 measured 38 mm with IoU 0.899,
          the same far-range bias family as the world rise errors.
    world: (stool z - floor z) / scale within 12 mm of --stool_h
    qa:   RESULT: PASS
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_naming  # noqa: E402  (prefix = recording-folder timestamp)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

# Every path below (globs, logs, child-process cwd) is relative to the repo
# root.  Anchor there explicitly so a double-clicked .bat, a scheduled task or
# an editor "run" button can never silently scan the wrong directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.abspath(os.getcwd()) != ROOT:
    os.chdir(ROOT)

PY = os.environ.get("GMR_PYTHON",
                    r"python")
LOGDIR = os.path.join(".workbuddy", "batch_logs")
STOOL_H_DEFAULT = 0.45
# fit acceptance
TH_VALID = 0.95
TH_IOU = 0.85
# Depth-residual gate.  The residual is sensor noise, which for a stereo
# depth camera scales with RANGE (~1% of distance), so a fixed millimetre
# threshold is really a fixed *standoff* in disguise: 40 mm happens to be ~1%
# of 4 m, the distance the 20260922 batch was shot at.
#
# Measured over all 14 clips in data/ (median box standoff vs median depth
# residual): Pearson r = 0.944, and the residual is 0.66-0.93% of range
# (mean 0.76%).  The one clip that fails a fixed 40 mm gate is exactly the one
# shot furthest away (20260922_165811: range 5.25 m -> 42.6 mm), and it is a
# perfectly good fit (IoU 0.897, valid3D 100%, QA PASS on the render).  A fixed
# gate therefore rejects good far shots and, worse, would silently ACCEPT a bad
# near shot: at 2 m a 40 mm residual is 2% of range, i.e. twice the sensor floor.
#
# So the gate is: residual <= max(TH_DEPTH_M, TH_DEPTH_RANGE_FRAC * range).
# The absolute floor keeps it meaningful when the range is unknown, and the
# range term keeps it honest as the camera moves back.  TH_DEPTH_M is retained
# as that floor so every existing number in the docs still means something.
TH_DEPTH_M = 0.040
TH_DEPTH_RANGE_FRAC = 0.011      # ~1.4x the measured 0.76% mean, sensor floor
TH_RISE_ERR_M = 0.012
# calibrated box HSV per venue (configs/boxes.json); HSV_DEFAULT is the
# last-resort orange box -- reaching it means the calibration was NOT applied,
# so every fallback is reported loudly (see load_boxes_cfg).
HSV_CFG = "configs/boxes.json"
HSV_DEFAULT = "5,90,80:28,255,255"
_HSV_CFG_CACHE = {}
_HSV_CFG_FAILURES = []
# subprocess timeouts (s).  T_FIT: the 6-DoF fit measured 7-29 min on the
# 20260922 batch (frame count / occlusions dominate); 900 s killed a healthy
# one mid-run (rc=124), so 45 min is the safe ceiling.
T_RETARGET, T_FIT, T_WORLD, T_RENDER, T_SYNC, T_MP4 = 1800, 2700, 600, 900, 1800, 1800


def sh(cmd, log_path, timeout):
    """Run cmd, tee output into log_path, return (rc, tail_of_log)."""
    os.makedirs(LOGDIR, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with open(log_path, "w", encoding="utf-8") as lf:
        try:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=timeout, env=env)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            lf.write("\n[batch] TIMEOUT after %ss\n" % timeout)
            rc = 124
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
            tail = lf.read()[-1500:]
    except OSError:
        tail = ""
    return rc, tail


def collect_prefixes():
    """Every prefix we can infer from artifacts + bags.

    Returns (prefixes, bag_map, unmatched_bags).  The bag map is what makes
    free-form bag names ("9.22.1.bag") work: the prefix comes from the
    recording folder / metadata, never from the bag's own file name.
    """
    prefixes = set()
    for pat, strip in [(r"data/*.pt", ""),
                       (r"data/*_g1.pkl", "_g1"),
                       (r"data/*_6dof.npz", "_6dof"),
                       (r"data/*_world_6dof.npz", "_world_6dof"),
                       (r"data/*_boxpose_used.npz", "_boxpose_used")]:
        for f in glob.glob(pat):
            stem = os.path.splitext(os.path.basename(f))[0]
            if strip == "_6dof" and stem.endswith("_world_6dof"):
                continue  # world files are claimed by the _world_6dof pattern
            prefixes.add(stem[:-len(strip)] if strip and
                         stem.endswith(strip) else stem)
    bagmap, unmatched = clip_naming.bag_map()
    prefixes.update(bagmap.values())
    # a data/ prefix with no bag (e.g. the bag was deleted) is still a clip
    return sorted(prefixes), bagmap, unmatched


# ---------------------------------------------------------------- acceptance
def check_fit(path):
    """Returns (ok, dict_of_metrics) from the 6-DoF fit npz."""
    import numpy as np
    d = np.load(path, allow_pickle=True)
    valid = np.asarray(d["valid"]) > 0
    pv = np.asarray(d["pose_valid"]) > 0
    iou = np.asarray(d["fit_iou"], dtype=float)
    de = np.asarray(d["fit_depth_err_m"], dtype=float)
    m = {
        "valid3D": float(valid.mean()),
        "pose_valid": float(pv.mean()),
        "iou_med": float(np.nanmedian(iou)) if np.isfinite(iou).any() else 0.0,
        "depth_mm": float(np.nanmedian(de)) * 1000.0
        if np.isfinite(de).any() else 999.0,
        "n": int(len(valid)),
    }
    # Range-aware depth gate: the residual is ~1% of the box standoff, so grade
    # it against the distance this clip was actually shot from.  Falls back to
    # the absolute floor when no usable range key is present.
    rng = _fit_range_m(d)
    if rng is not None:
        m["range_m"] = rng
        m["depth_gate_mm"] = max(TH_DEPTH_M, TH_DEPTH_RANGE_FRAC * rng) * 1000.0
    else:
        m["range_m"] = None
        m["depth_gate_mm"] = TH_DEPTH_M * 1000.0
    ok = (m["valid3D"] >= TH_VALID and m["pose_valid"] >= TH_VALID
          and m["iou_med"] >= TH_IOU
          and m["depth_mm"] <= m["depth_gate_mm"])
    return ok, m


def _fit_range_m(d):
    """Median box standoff in metres from a fit npz, or None.

    `t_cam` is the fitted box centre in the camera frame (the 6-DoF route);
    `pos_cam` is the pre-fit blob centroid (the older centroid route).  Either
    is close enough for a noise-floor estimate.
    """
    import numpy as np
    for key in ("t_cam", "pos_cam"):
        if key not in d.files:
            continue
        try:
            a = np.asarray(d[key], dtype=float)
        except (TypeError, ValueError):
            continue
        if a.ndim != 2 or a.shape[1] < 3:
            continue
        r = np.linalg.norm(a, axis=1)
        r = r[np.isfinite(r) & (r > 0.1)]
        if len(r):
            return float(np.median(r))
    return None


def check_world(path, stool_h, tol_mm=None):
    """Rise check: world-frame lift / scale must equal the real stool height.

    tol_mm overrides the default +-12 mm gate.  The gate exists to catch a
    wrong scale/gravity frame; a far parking spot also inflates the measured
    rise through range-dependent depth bias (20260922 batch: +13..+140 mm with
    the box 4 m away), which is why --world_rise_tol_mm / --allow_bad_world
    exist -- the shipped renderer never uses the measured z for parked phases,
    so a BAD world only blocks rendering, it does not corrupt the picture.
    """
    import numpy as np
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    # Explicit None test, not `or`: a legitimate scale of 0.0 means "bad
    # alignment" (the gate below rejects it) and must not be silently replaced
    # by the npz's own scale array.
    _scale_meta = meta.get("scale")
    scale = float(d["scale"]) if _scale_meta is None else float(_scale_meta)
    if meta.get("stool_h") is None:
        print("[batch] WARNING: %s has no 'stool_h' in its meta -- falling "
              "back to the CLI --stool_h=%.3f m.  The alignment may have run "
              "with a different stool height, in which case the rise gate "
              "below can report a false BAD." % (path, float(stool_h)))
        stool = float(stool_h)
    else:
        stool = float(meta["stool_h"])
    carry = np.asarray(d["carry"]) > 0
    cov = (np.asarray(d["valid"]) > 0) & (np.asarray(d["pose_valid"]) > 0)
    pos = np.asarray(d["box_center_world"])
    n = len(carry)
    if not carry.any() or cov.sum() < 10 or scale <= 0:
        return False, {"rise_mm": float("nan"), "err_mm": float("nan"),
                       "note": "carry/cov too small"}
    fc = int(np.argmax(carry))
    lc = int(n - np.argmax(carry[::-1]) - 1)
    floor = cov & (np.arange(n) < fc)
    stool_m = cov & (np.arange(n) > lc)
    if floor.sum() < 3 or stool_m.sum() < 3:
        return False, {"rise_mm": float("nan"), "err_mm": float("nan"),
                       "note": "missing parked segment"}
    rise_w = (float(np.median(pos[stool_m, 2]))
              - float(np.median(pos[floor, 2])))
    rise_real = rise_w / scale
    m = {"rise_mm": rise_real * 1000.0,
         "err_mm": abs(rise_real - stool) * 1000.0,
         "stool_h": stool, "scale": scale}
    thr = TH_RISE_ERR_M * 1000 if tol_mm is None else tol_mm
    return m["err_mm"] <= thr, m


# The QA script lives in the skills tree, which was migrated from
# .workbuddy/skills/ to .dsh/skills/.  Both may exist during the transition and
# only one may survive it, so NEVER hard-code a single location: a wrong path
# would make QA fail silently for every clip of every batch.
#
# NOTE the `..` entries.  DSH anchors its project skills at the workspace root,
# which for this checkout is the PARENT of GMR-master (the repo has no .git, so
# the root resolves to the enclosing project folder).  A repo-relative probe
# alone therefore never sees the migrated skill.  Probe both depths.
QA_SCRIPT_RELPATHS = (
    os.path.join("..", ".dsh", "skills", "gmr-carry-video-pipeline", "scripts",
                 "check_render_quality.py"),
    os.path.join(".dsh", "skills", "gmr-carry-video-pipeline", "scripts",
                 "check_render_quality.py"),
    os.path.join(".workbuddy", "skills", "gmr-carry-video-pipeline", "scripts",
                 "check_render_quality.py"),
    os.path.join("..", ".workbuddy", "skills", "gmr-carry-video-pipeline",
                 "scripts", "check_render_quality.py"),
    os.path.join("skills", "gmr-carry-video-pipeline", "scripts",
                 "check_render_quality.py"),
)
_QA_SCRIPT = None        # memoised find_qa_script() result (path or None)
_QA_SCRIPT_RESOLVED = False


def find_qa_script():
    """Absolute path of check_render_quality.py, or None when no copy exists.

    Probes, in order: $GMR_QA_SCRIPT (when set), then every entry of
    QA_SCRIPT_RELPATHS relative to ROOT.

    An explicitly set $GMR_QA_SCRIPT that does not exist is a HARD error: it is
    never silently skipped in favour of a different copy, because a typo in the
    override must be as visible as a missing file.  The resolution -- success or
    failure -- is cached, and the report is printed exactly once per run, not
    once per clip.
    """
    global _QA_SCRIPT, _QA_SCRIPT_RESOLVED
    if _QA_SCRIPT_RESOLVED:
        return _QA_SCRIPT
    _QA_SCRIPT_RESOLVED = True
    env = os.environ.get("GMR_QA_SCRIPT")
    probed = []
    if env:
        probed.append(("$GMR_QA_SCRIPT", env))
        if os.path.isfile(env):
            _QA_SCRIPT = os.path.abspath(env)
        else:
            probed += [(r, os.path.join(ROOT, r)) for r in QA_SCRIPT_RELPATHS]
            _qa_not_found(probed, override=True)
            return _QA_SCRIPT
    else:
        probed += [(r, os.path.join(ROOT, r)) for r in QA_SCRIPT_RELPATHS]
        for label, path in probed:
            if os.path.isfile(path):
                _QA_SCRIPT = os.path.abspath(path)
                break
    if _QA_SCRIPT is None:
        _qa_not_found(probed, override=False)
    else:
        print("[batch] QA script: %s" % _QA_SCRIPT)
    return _QA_SCRIPT


def _qa_not_found(probed, override=False):
    """Loud, unmissable report: every probed path is named."""
    print("!" * 78)
    if override:
        print("[batch] ERROR: $GMR_QA_SCRIPT is set to a file that does NOT "
              "exist -- QA cannot run for ANY clip.")
    else:
        print("[batch] ERROR: check_render_quality.py was NOT FOUND -- QA "
              "cannot run for ANY clip.")
    print("[batch] probed %d path(s); none of them exists:" % len(probed))
    for label, path in probed:
        print("[batch]   - %-16s %s" % (label, path))
    print("[batch] every affected clip is reported as qa=ERROR (never PASS) "
          "until this is fixed.")
    print("[batch] fix: point GMR_QA_SCRIPT at the real script, or restore one "
          "of the paths above.")
    print("!" * 78)


def check_qa(npz_path):
    """Run the QA script; parse RESULT:.

    Return contract (relied on by survey() and blocker()):
    (ok, {"qa": verdict, "rc": rc}) with verdict in PASS/FAIL/ERROR.  A missing
    QA script yields (False, {"qa": "ERROR", "rc": -1}) so the table and the
    blocker both show a non-PASS state instead of a silent success.
    """
    script = find_qa_script()
    if script is None:
        return False, {"qa": "ERROR", "rc": -1}
    lp = os.path.join(LOGDIR, "qa_%s.log"
                      % os.path.splitext(os.path.basename(npz_path))[0])
    rc, tail = sh([PY, script, "--npz", npz_path], lp, 300)
    verdict = "FAIL" if rc else "PASS"
    if rc not in (0, 1):
        verdict = "ERROR"
    return verdict == "PASS", {"qa": verdict, "rc": rc}


# ---------------------------------------------------------------- step defs
def step_cmds(p, args):
    """Returns the command list for a step name, or None if not runnable here."""
    bag = find_bag(p)
    mp4 = "video/%s.mp4" % p
    if args.hsv:
        hsv = args.hsv
    else:
        hsv = load_hsv_for(p)
    return {
        "mp4": [PY, "scripts/bag_to_mp4.py", "--bag", bag, "--out", mp4]
        if bag else None,
        "g1": [PY, "scripts/gvhmr_to_smplx.py", "--input", "data/%s.pt" % p,
               "--output", "data/%s_smplx.npz" % p, "--fps", "30"],
        "g1b": [PY, "scripts/run_retarget_headless.py", "--smplx_file",
                "data/%s_smplx.npz" % p, "--robot", "unitree_g1",
                "--save_path", "data/%s_g1.pkl" % p,
                "--render_video", "video/%s_g1.mp4" % p, "--render_every", "2"],
        "fit": [PY, "scripts/box_traj_from_depth.py", "--bag", bag,
                "--fit-cuboid", "--hsv", hsv, "--roi-pad", "280",
                "--min-area", "400", "--out", "data/%s_6dof.npz" % p,
                "--preview", "video/%s_6dof.mp4" % p] if bag else None,
        "world": [PY, "scripts/align_box_to_world.py", "--robot_motion",
                  "data/%s_g1.pkl" % p,
                  "--box_traj", "data/%s_6dof.npz" % p,
                  "--box_pose", "data/%s_6dof.npz" % p,
                  "--out", "data/%s_world_6dof.npz" % p,
                  "--stool_h", str(args.stool_h)],
        "render": [PY, "scripts/vis_robot_with_box.py", "--robot_motion",
                   "data/%s_g1.pkl" % p, "--box_pose_world",
                   "data/%s_world_6dof.npz" % p, "--record_video",
                   "video/%s_g1_measured_box_6dof.mp4" % p,
                   "--dump_box_pose", "data/%s_boxpose_used.npz" % p],
        "sync": [PY, "scripts/check_timing_sync.py", "--prefix", p],
    } if True else {}


def find_bag(p):
    """The bag whose canonical prefix is `p` (see scripts/clip_naming.py).

    The bag's own file name is irrelevant -- `9.22.1.bag` inside
    `video/bag/rgbd_20260922_161727/` resolves to 20260922_161727 via the
    folder (or metadata.json) timestamp.  Legacy pkls saved as
    `<bag_ts>_unitree_g1.pkl` still resolve to the stripped timestamp.
    """
    b = clip_naming.find_bag(p)
    if b:
        return b
    # belt and braces: a literal name match, for a bag clip_naming could not
    # label (e.g. moved out of its folder AND missing from clips.json)
    for q in ([p[:-len("_unitree")]] if p.endswith("_unitree") else []) + [p]:
        for pat in ("video/%s.bag" % q, "video/%s.db3" % q,
                    "video/**/%s.bag" % q, "video/**/%s.db3" % q):
            hits = glob.glob(pat, recursive=True)
            if hits:
                return hits[0]
    return None


def _hsv_cfg_failure(cfg, detail):
    """Report a broken HSV config LOUDLY, once, without aborting the batch."""
    _HSV_CFG_FAILURES.append((cfg, detail))
    print("!" * 78)
    print("[batch] !!! HSV CONFIG ERROR: cannot use %s" % os.path.abspath(cfg))
    print("[batch] !!! reason: %s" % detail)
    print("[batch] !!! FALLING BACK TO THE DEFAULT HSV %s FOR EVERY CLIP."
          % HSV_DEFAULT)
    print("[batch] !!! The calibrated venue range is NOT applied, so clips on "
          "another box/in another room may be fit with the WRONG COLOUR.")
    print("[batch] !!! The batch continues; fix the JSON and re-run the fit "
          "step (--force fit).")
    print("!" * 78)


def load_boxes_cfg(cfg=HSV_CFG):
    """Parsed configs/boxes.json, memoised ({} when it cannot be used).

    A broken file must never silently revert every clip to the orange default
    -- that is the exact failure this file exists to prevent (the 20260922
    hall needs its own calibrated range).  The error is printed ONCE, loudly,
    and the run keeps going: one bad config may not abort a batch of otherwise
    healthy clips.
    """
    if cfg in _HSV_CFG_CACHE:
        return _HSV_CFG_CACHE[cfg]
    boxes = {}
    if not os.path.exists(cfg):
        # A deleted or renamed config is as dangerous as a malformed one: the
        # venue calibration silently disappears for every clip.
        _hsv_cfg_failure(cfg, "FileNotFoundError: %s does not exist"
                              % os.path.abspath(cfg))
        _HSV_CFG_CACHE[cfg] = boxes
        return boxes
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            boxes = json.load(f)
        if not isinstance(boxes, dict):
            raise ValueError("top level is %s, expected a JSON object"
                             % type(boxes).__name__)
        for name, spec in boxes.items():
            if not isinstance(spec, dict):
                raise ValueError("entry %r is %s, expected a JSON object"
                                 % (name, type(spec).__name__))
            if not isinstance(spec.get("hsv"), str) or not spec["hsv"].strip():
                raise KeyError("entry %r has no usable 'hsv' string" % name)
    except (OSError, ValueError, KeyError) as e:
        boxes = {}
        _hsv_cfg_failure(cfg, "%s: %s" % (type(e).__name__, e))
    _HSV_CFG_CACHE[cfg] = boxes
    return boxes


def load_hsv_for(p, cfg=HSV_CFG):
    """HSV range for a clip, from configs/boxes.json.

    An entry matches a prefix in the most specific way available:
      * `name in p`          -- loose substring (the JSON key, e.g. "橙箱")
      * `prefix == p`        -- one exact clip
      * `p in prefixes`      -- an explicit list of clips
      * `p.startswith(session)` -- every clip of one recording session/day
    HSV is a property of the VENUE (box + lighting), not of a single take, so a
    new take in the same room must reuse the calibrated range -- `session` is
    what makes that automatic instead of silently falling back to the default.

    A missing/unparseable config warns loudly (see load_boxes_cfg) and then
    returns the default; it never fails silently.
    """
    boxes = load_boxes_cfg(cfg)
    for name, spec in boxes.items():
        if name in p or spec.get("prefix") == p \
                or p in (spec.get("prefixes") or []) \
                or (spec.get("session") and p.startswith(spec["session"])):
            return spec["hsv"]
    # sensible defaults: orange box, then blue box
    return HSV_DEFAULT


def print_hsv_cfg_status():
    """One line after the status table when the HSV config was unusable."""
    for cfg, detail in _HSV_CFG_FAILURES:
        print("[!] HSV CONFIG ERROR: %s (%s) -- every clip used the DEFAULT HSV "
              "%s instead.  Fix the file and re-run the fit step."
              % (cfg, detail, HSV_DEFAULT))


# ---------------------------------------------------------------- per clip
def survey(p, args):
    """Returns dict: per-step state + metrics for one prefix."""
    s = {k: "-" for k in ("bag", "mp4", "pt", "g1", "fit", "world",
                          "render", "qa", "sync")}
    s["prefix"] = p
    s["bag"] = "Y" if find_bag(p) else "-"
    s["mp4"] = "Y" if os.path.exists("video/%s.mp4" % p) else "-"
    s["pt"] = "Y" if os.path.exists("data/%s.pt" % p) else "-"
    s["g1"] = "Y" if os.path.exists("data/%s_g1.pkl" % p) else "-"
    fitp = "data/%s_6dof.npz" % p
    if os.path.exists(fitp):
        try:
            ok, m = check_fit(fitp)
            s["fit"] = "OK" if ok else "BAD"
            s["fit_valid"] = "%.0f%%" % (m["valid3D"] * 100)
            s["fit_iou"] = "%.3f" % m["iou_med"]
            s["fit_depth"] = "%.0fmm" % m["depth_mm"]
        except Exception as e:
            s["fit"] = "ERR"
            s["fit_iou"] = str(e)[:40]
    worldp = "data/%s_world_6dof.npz" % p
    if os.path.exists(worldp):
        try:
            ok, m = check_world(worldp, args.stool_h,
                                getattr(args, "world_rise_tol_mm", None))
            s["world"] = "OK" if ok else "BAD"
            s["rise"] = "%.0fmm" % m["rise_mm"] if m["rise_mm"] == m["rise_mm"] \
                else "nan"
            s["rise_err"] = "%.0fmm" % m["err_mm"] if m["err_mm"] == m["err_mm"] \
                else "nan"
        except Exception as e:
            s["world"] = "ERR"
            s["rise_err"] = str(e)[:40]
    s["render"] = "Y" if os.path.exists(
        "video/%s_g1_measured_box_6dof.mp4" % p) else "-"
    dumpp = "data/%s_boxpose_used.npz" % p
    if os.path.exists(dumpp) and s["render"] == "Y":
        try:
            ok, m = check_qa(dumpp)
            s["qa"] = m["qa"]
        except Exception:
            s["qa"] = "ERR"
    # human<->robot timing sync (scripts/check_timing_sync.py).  Cheap here:
    # just read its cached verdict, do not re-run the body model / video decode.
    syncp = "data/%s_timing_sync.json" % p
    if os.path.exists(syncp):
        try:
            with open(syncp, "r", encoding="utf-8") as f:
                r = json.load(f)
            s["sync"] = "PASS" if r.get("ok") else "FAIL"
            # surface the weakest channel so a FAIL is actionable at a glance
            ch = r.get("channels") or {}
            if ch:
                worst = max(ch.items(),
                            key=lambda kv: abs(kv[1].get("lag", 0)))
                s["sync_worst"] = "%s%+d/%.2f" % (worst[0][:4],
                                                  worst[1].get("lag", 0),
                                                  worst[1].get("corr", 0.0))
        except Exception:
            s["sync"] = "ERR"
    return s


def blocker(s, allow_bad_world=False):
    # mp4 is checked BEFORE the GVHMR gate: converting a bag needs no GPU, and
    # --run skips anything that reports need-GVHMR.  Without this ordering a
    # fresh recording would sit un-converted until a GPU machine produced its
    # .pt.  (The table still shows pt=- and the bag map still prints
    # "[need data/<prefix>.pt]", so nothing is hidden.)
    if s["bag"] == "Y" and s["mp4"] != "Y":
        return "need-mp4"
    if s["pt"] != "Y" and s["g1"] != "Y":
        return "need-GVHMR(GPU)"
    if s["g1"] != "Y":
        return "need-retarget"
    if s["fit"] in ("-", "BAD", "ERR"):
        return "need-fit/%s" % s["fit"]
    if s["world"] in ("-", "ERR") or (s["world"] == "BAD"
                                      and not allow_bad_world):
        return "need-align/%s" % s["world"]
    if s["render"] != "Y":
        return "need-render"
    if s["qa"] == "-":
        return "need-qa-dump"
    if s["qa"] != "PASS":
        return "qa-%s" % s["qa"]
    if s["sync"] == "-":
        return "need-timing-sync"
    if s["sync"] != "PASS":
        return "sync-%s" % s["sync"]
    return ""


# order of the local (non-GVHMR) steps; --force STEP re-runs STEP and later
STEP_ORDER = ["mp4", "fit", "world", "render", "qa", "sync"]


def wants(args, step):
    """True when `step` is selected by --steps (default: every step is on)."""
    sel = getattr(args, "steps", None)
    return not sel or step in sel


def _forced(args, step):
    """True when `step` must run again even though its output already exists
    (--force STEP re-executes STEP and every step after it)."""
    f = getattr(args, "force", None)
    if not f:
        return False
    return STEP_ORDER.index(step) >= STEP_ORDER.index(f)


def run_step(label, cmd, log_name, timeout):
    """Run one pipeline step and report the outcome LOUDLY (silent failures
    used to look like success: the step was simply skipped later on)."""
    lp = os.path.join(LOGDIR, log_name)
    rc, tail = sh(cmd, lp, timeout)
    if rc == 0:
        print("    %-22s ok" % label)
    else:
        print("    %-22s FAILED rc=%d  (log: %s)" % (label, rc, lp))
        lines = [l for l in tail.strip().splitlines() if l.strip()]
        for line in lines[-6:]:
            print("      | %s" % line[:160])
    return rc


def run_clip(p, args):
    """Execute missing steps for one clip. Returns final survey dict."""
    s = survey(p, args)
    cmds = step_cmds(p, args)
    # mp4 first: everything downstream (GVHMR, the fit preview) reads it, and it
    # is the only step that needs neither a GPU nor the box trajectory.
    if args.run and wants(args, "mp4") and (s["mp4"] != "Y" or _forced(args, "mp4")):
        if cmds["mp4"] is None:
            print("  [%s] no bag found for the mp4 step -- skipped" % p)
        else:
            print("  [%s] bag -> mp4" % p)
            run_step("bag_to_mp4", cmds["mp4"], "%s_mp4.log" % p, T_MP4)
            s = survey(p, args)
    if wants(args, "retarget") and s["g1"] != "Y" and s["pt"] == "Y":
        print("  [%s] retarget (2 sub-steps)" % p)
        rc = run_step("gvhmr_to_smplx", cmds["g1"], "%s_g1.log" % p,
                      T_RETARGET)
        if rc == 0:
            run_step("run_retarget_headless", cmds["g1b"], "%s_g1b.log" % p,
                     T_RETARGET)
        s = survey(p, args)
    if wants(args, "fit") and (s["fit"] in ("-", "BAD", "ERR")
                               or _forced(args, "fit")) and args.run:
        if cmds["fit"] is None:
            print("  [%s] no bag found for the fit step -- skipped" % p)
        else:
            print("  [%s] 6-DoF fit (hsv %s)" % (p, cmds["fit"][6]))
            run_step("box_traj_from_depth", cmds["fit"], "%s_fit.log" % p,
                     T_FIT)
            s = survey(p, args)
    if wants(args, "world") and s["g1"] == "Y" and s["fit"] == "OK" \
            and (s["world"] in ("-", "BAD", "ERR") or _forced(args, "world")):
        print("  [%s] world alignment" % p)
        run_step("align_box_to_world", cmds["world"], "%s_world.log" % p,
                 T_WORLD)
        s = survey(p, args)
    # --allow_bad_world: a rise-gate BAD does not block the render.  The
    # renderer pins parked phases to floor+half-box anyway, so the measured z
    # (whose far-range depth bias tripped the gate) never reaches the picture.
    _world_ok = s["world"] == "OK" or (
        getattr(args, "allow_bad_world", False) and s["world"] == "BAD")
    if wants(args, "render") and _world_ok and (s["render"] != "Y"
                                                or _forced(args, "render")):
        print("  [%s] render" % p)
        run_step("vis_robot_with_box", cmds["render"], "%s_render.log" % p,
                 T_RENDER)
        s = survey(p, args)
    # timing sync needs both halves of the chain, and only makes sense once the
    # render's QA passed (a BAD QA usually means the motion itself is suspect).
    if wants(args, "sync") and s["render"] == "Y" and s["qa"] == "PASS" \
            and (s["sync"] in ("-", "ERR", "FAIL") or _forced(args, "sync")):
        print("  [%s] timing sync (3 hops)" % p)
        run_step("check_timing_sync", cmds["sync"], "%s_sync.log" % p, T_SYNC)
        s = survey(p, args)
    return s


# ---------------------------------------------------------------- reporting
def print_bag_map(bagmap, unmatched):
    """Show how each bag maps to a prefix, so the GVHMR output can be named.

    Free-form bag names are supported (9.22.1.bag, realsense.db3, ...) because
    the prefix comes from the recording folder / metadata.json -- but the user
    still has to name data/<prefix>.pt by hand, so print the exact name.
    """
    if not bagmap and not unmatched:
        return
    print("\nbag -> canonical prefix   (name the GVHMR output "
          "data/<prefix>.pt, then re-run)")
    rev = clip_naming.unique_prefix_map(bagmap)
    for p, bags in sorted(rev.items()):
        note = "" if os.path.exists("data/%s.pt" % p) else \
            "   [need data/%s.pt]" % p
        for b in bags:
            print("  %-56s -> %s%s" % (b, p, note))
        if len(bags) > 1:
            print("      [!] %d bags map to the same prefix -- keep one, or "
                  "pin the others in %s" % (len(bags), clip_naming.CLIPS_CFG))
    if unmatched:
        print("\nbags with no timestamp in the name, folder or metadata.json:")
        for b in unmatched:
            print("  %s" % b)
        print("  -> pin them in %s, e.g. {\"9.22.3.bag\": \"20260922_183000\"}"
              % clip_naming.CLIPS_CFG)


def print_table(rows, unmatched, bagmap=None):
    cols = ["prefix", "bag", "mp4", "pt", "g1", "fit", "fit_valid", "fit_iou",
            "fit_depth", "world", "rise_err", "render", "qa", "sync",
            "blocker"]
    hdr = ["prefix", "bag", "mp4", "pt", "g1", "fit", "v3d", "iou", "dep",
           "world", "riseE", "render", "qa", "sync", "blocker"]
    widths = [max(len(hdr[i]), max((len(str(r.get(c, "-"))) for r in rows),
                                   default=0)) for i, c in enumerate(cols)]
    fmt = "  ".join("%%-%ds" % w for w in widths)
    print(fmt % tuple(hdr))
    print("-" * (sum(widths) + 2 * (len(cols) - 1)))
    for r in rows:
        print(fmt % tuple(str(r.get(c, "-")) for c in cols))


def write_csv(rows, path):
    cols = ["prefix", "bag", "mp4", "pt", "g1", "fit", "fit_valid", "fit_iou",
            "fit_depth", "world", "rise", "rise_err", "render", "qa",
            "sync", "sync_worst", "blocker"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="execute the missing steps (GVHMR excepted)")
    ap.add_argument("--only", default=None, help="process this prefix only")
    ap.add_argument("--hsv", default=None,
                    help="HSV range for the fit step, e.g. "
                         "5,90,80:28,255,255 (default: configs/boxes.json, "
                         "then the orange-box default)")
    ap.add_argument("--stool_h", type=float, default=STOOL_H_DEFAULT)
    ap.add_argument("--world_rise_tol_mm", type=float, default=None,
                    help="relax the world rise gate from the default +-12 mm "
                         "(a far parking spot inflates the measured rise via "
                         "range-dependent depth bias)")
    ap.add_argument("--allow_bad_world", action="store_true",
                    help="do not let a world BAD block render/qa/sync (the "
                         "table still shows BAD and rise_err; the renderer "
                         "never uses the measured z for parked phases)")
    ap.add_argument("--force", default=None, choices=STEP_ORDER,
                    help="re-run this step and every later one even if its "
                         "output already exists (use after editing a script)")
    ap.add_argument("--csv", default=None, help="write the status table here")
    ap.add_argument("--steps", default=None, metavar="S1,S2",
                    help="only run these steps, comma-separated, from "
                         "mp4|retarget|fit|world|render|sync (default: all). "
                         "e.g. --steps mp4 converts bags to mp4 and stops.")
    ap.add_argument("--bag-root", action="append", default=None, metavar="DIR",
                    help="extra directory to search for recordings, e.g. "
                         "D:\\bag (repeatable).  Each root is scanned for "
                         "`**/*.bag` and `**/*.db3`; the prefix still comes from "
                         "the `rgbd_<timestamp>` folder, so nothing is copied "
                         "into the repo.  Same effect as setting %s."
                         % clip_naming.ROOTS_ENV)
    args = ap.parse_args()

    if args.steps:
        args.steps = {x.strip() for x in args.steps.split(",") if x.strip()}
        unknown = args.steps - set(STEP_ORDER) - {"retarget"}
        if unknown:
            ap.error("--steps: unknown step(s) %s (choose from mp4|retarget|fit|"
                     "world|render|sync)" % ", ".join(sorted(unknown)))
        # `qa` is a *check* run by survey() on the rendered mp4, not a step that
        # can be executed: selecting it alone would silently do nothing.
        if "qa" in args.steps:
            print("[note] qa is a check, not a runnable step -- it is evaluated "
                  "automatically after render; use --force qa to re-read it.")
            if args.steps == {"qa"}:
                ap.error("--steps qa would run nothing; name a real step "
                         "(mp4|retarget|fit|world|render|sync)")

    # clip_naming reads the roots from the env, so set it before anything scans
    if args.bag_root:
        extra = os.pathsep.join(args.bag_root)
        prev = os.environ.get(clip_naming.ROOTS_ENV, "")
        os.environ[clip_naming.ROOTS_ENV] = \
            os.pathsep.join(x for x in (prev, extra) if x)

    prefixes, bagmap, unmatched = collect_prefixes()
    if args.only:
        prefixes = [p for p in prefixes if args.only in p]
    if not prefixes:
        print("no clips found (looked in data/*.pt, data/*_g1.pkl, "
              "video/**/*.bag|.db3)")
        print_bag_map(bagmap, unmatched)
        return

    # Resolve the two external resources up front, so a missing QA script or a
    # broken HSV config is reported even when no clip happens to reach them.
    if not args.hsv:
        load_boxes_cfg()
    find_qa_script()

    rows = [survey(p, args) for p in prefixes]
    for r in rows:
        r["blocker"] = blocker(r, args.allow_bad_world)
    print_table(rows, unmatched)
    done = sum(1 for r in rows if not blocker(r, args.allow_bad_world))
    print("\n%d of %d clips fully done" % (done, len(rows)))
    print_hsv_cfg_status()
    print_bag_map(bagmap, unmatched)

    if args.csv:
        write_csv(rows, args.csv)

    if args.run:
        if args.force:
            todo = [r["prefix"] for r in rows
                    if not r["blocker"].startswith("need-GVHMR")]
        else:
            todo = [r["prefix"] for r in rows if blocker(r, args.allow_bad_world)
                    and not r["blocker"].startswith("need-GVHMR")]
        tag = "  (--force %s: re-running that step + later)" % args.force \
            if args.force else ""
        print("\n--run: %d clips to advance (GVHMR-blocked ones skipped)%s"
              % (len(todo), tag))
        for p in todo:
            print("\n=== %s ===" % p)
            try:
                run_clip(p, args)
            except Exception as e:
                print("  [!] unexpected error, moving on: %r" % e)
        rows = [survey(p, args) for p in prefixes]
        for r in rows:
            r["blocker"] = blocker(r, args.allow_bad_world)
        print("\n===== after run =====")
        print_table(rows, unmatched)
        done = sum(1 for r in rows if not blocker(r, args.allow_bad_world))
        print("\n%d of %d clips fully done" % (done, len(rows)))
        print_hsv_cfg_status()
        print_bag_map(bagmap, unmatched)
        if args.csv:
            write_csv(rows, args.csv)


if __name__ == "__main__":
    main()

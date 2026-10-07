#!/usr/bin/env python
"""Per-clip HUMAN <-> ROBOT timing-sync QA.

Why this exists
---------------
The render can *look* like the robot crouches long before the real person, but
that impression is unreliable: the G1 is only 1.32 m tall with a big head and
short torso, so at the same torso lean it reads as "folded double" while a 1.7 m
human reads as "bending over".  Guessing from the video is how a wrong
"the robot is ~20 frames early" conclusion gets reached.

This script replaces the guess with a measurement, in three independent hops:

  1. SMPL-X reconstruction  ->  robot motion     (the retarget)
       runs the SMPL-X body model on `<p>_smplx.npz` and the G1 forward
       kinematics on `<p>_g1.pkl`, then cross-correlates four channels:
       pelvis height, chest height, knee flexion, torso lean.
  2. source video  ->  SMPL-X reconstruction     (the GVHMR step)
       the RealSense is static, so median-frame background subtraction recovers
       the real person's silhouette; its head-top row is compared with the
       SMPL-X head height onset.
  3. video frame axis  ->  bag/box axis
       the box is HSV-segmented (see `--hsv` / configs/boxes.json); its rise
       frame is compared with the depth-measured carry start + `frame_offset`.
       This validates the frame mapping everything else depends on.

A positive lag means "the robot leads the human" (it reaches a landmark first).

THE GATE IS HOP 1 ONLY.  Hops 2-3 are reported as *evidence*: a flaky source
video (unlocked auto-exposure, dropped frames) must not fail a clip whose
retarget is provably in sync.  Hop 3 additionally VALIDATES the frame axis, so
when it cannot be evaluated -- no `_world_6dof.npz`, a missing key, an
unparseable value -- the video hop is reported INCONCLUSIVE, never OK and never
OUT OF TOLERANCE: an un-validated axis must not produce a confident verdict.

Exit code is 0 when every HOP-1 channel is within tolerance, 1 otherwise, so the
batch pipeline can gate on it.

    python scripts/check_timing_sync.py --prefix 20260922_161727
    python scripts/check_timing_sync.py --all
    python scripts/check_timing_sync.py --prefix 20260922_165945 --hsv 4,75,79:24,255,255
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MAX_LAG = 30          # +/- frames searched around zero
LAG_TOL = 3           # frames; |lag| above this is reported
CORR_MIN = 0.90       # shape correlation floor

BOXES_JSON = os.path.join("configs", "boxes.json")
ORANGE_HSV = "5,90,80:28,255,255"     # last-resort default (orange box)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_hsv(s):
    """`"H,S,V:H,S,V"` -> (lo, hi) uint8 arrays (OpenCV H 0-179).

    Same accepted format as `box_traj_from_depth.py --hsv`.
    """
    lo, hi = s.split(":")
    lo = [int(v) for v in lo.split(",")]
    hi = [int(v) for v in hi.split(",")]
    if len(lo) != 3 or len(hi) != 3:
        raise ValueError("need three channels per bound, got %d:%d" % (len(lo), len(hi)))
    return (np.array(lo, np.uint8), np.array(hi, np.uint8))


def load_hsv_for(p, cfg=None):
    """HSV range for a clip, from configs/boxes.json -> (hsv, source).

    Local re-implementation of `batch_pipeline.load_hsv_for()` (same semantics,
    deliberately NOT imported: batch_pipeline has import side effects).  An entry
    matches a prefix in the most specific way available:
      * `name in p`             -- loose substring (the JSON key, e.g. "橙箱")
      * `spec["prefix"] == p`   -- one exact clip
      * `p in spec["prefixes"]` -- an explicit list of clips
      * `p.startswith(session)` -- every clip of one recording session/day
    HSV is a property of the VENUE (box + lighting), not of a single take, so a
    new take in the same room reuses the calibrated range instead of silently
    falling back to the orange default.
    """
    cfg = cfg or os.path.join(ROOT, BOXES_JSON)
    rel = os.path.relpath(cfg, ROOT)
    if not os.path.exists(cfg):
        return ORANGE_HSV, "orange default (%s not found)" % rel
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            boxes = json.load(f)
        for name, spec in boxes.items():
            if name in p:
                why = "name %s" % name
            elif spec.get("prefix") == p:
                why = "prefix %s" % p
            elif p in (spec.get("prefixes") or []):
                why = "prefixes list"
            elif spec.get("session") and p.startswith(spec["session"]):
                why = "session %s" % spec["session"]
            else:
                continue
            return spec["hsv"], "%s: %s" % (rel, why)
    except (OSError, ValueError, KeyError) as e:
        # reported, never hidden, and never fatal: a broken config must not stop
        # a QA run -- the caller sees the warning and the fallback on stdout.
        return ORANGE_HSV, "orange default (%s unreadable: %s)" % (rel, e)
    return ORANGE_HSV, "orange default (no match in %s)" % rel


def resolve_hsv(prefix, cli_hsv):
    """(hsv_string, source) by priority: --hsv > configs/boxes.json > orange."""
    if cli_hsv:
        return cli_hsv, "--hsv (CLI)"
    return load_hsv_for(prefix)


def smooth(c, k=5):
    c = np.asarray(c, dtype=np.float64)
    return np.convolve(np.pad(c, (k // 2, k // 2), mode="edge"),
                       np.ones(k) / k, mode="valid")


def best_lag(ref, mov, max_lag=MAX_LAG, min_len=20):
    """z-normalised cross-correlation.

    Returns (lag, corr) where lag > 0 means `mov` LEADS `ref`
    (mov[t+lag] lines up with ref[t]).
    """
    a = smooth(np.asarray(mov, dtype=np.float64))
    b = smooth(np.asarray(ref, dtype=np.float64))
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best, bl = -9.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(a) - lag]
        else:
            y, x = b[-lag:], a[:len(b) + lag]
        if len(x) < min_len:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best:
            best, bl = c, lag
    return bl, (best if best > -9 else float("nan"))


def onset(curve, frac=0.05, skip=3):
    """First frame the curve has moved `frac` of its full travel off baseline."""
    c = smooth(curve)
    base = float(np.median(c[:max(8, int(0.12 * len(c)))]))
    ext = float(c.max() if abs(c.max() - base) > abs(c.min() - base) else c.min())
    sgn = 1.0 if ext > base else -1.0
    rng = max(1e-9, abs(ext - base))
    idx = np.where(sgn * (c - base) > frac * rng)[0]
    idx = idx[idx >= skip]
    h = np.where(sgn * (c - base) > 0.5 * rng)[0]
    h = h[h >= skip]
    return (int(idx[0]) if len(idx) else -1), (int(h[0]) if len(h) else -1)


def knee_flexion(hip, knee, ankle):
    v1, v2 = knee - hip, ankle - knee
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1, 1)))


def lean_from_vertical(up_vec, up_axis):
    return np.degrees(np.arccos(np.clip(
        up_vec[:, up_axis] / (np.linalg.norm(up_vec, axis=1) + 1e-9), -1, 1)))


# --------------------------------------------------------------------------- #
# curves
# --------------------------------------------------------------------------- #
def human_curves(prefix):
    """SMPL-X human: pelvis height, chest (shoulder-mid) height, knee, lean."""
    from smplx.joint_names import JOINT_NAMES
    from general_motion_retargeting.utils.smpl import load_smplx_file

    f = os.path.join(ROOT, "data", "%s_smplx.npz" % prefix)
    if not os.path.exists(f):
        return None
    sd, bm, so, height = load_smplx_file(f, os.path.join(ROOT, "assets", "body_models"))
    J = so.joints.detach().numpy().reshape(len(sd["pose_body"]), -1, 3)
    nm = JOINT_NAMES[:J.shape[1]]

    def j(n):
        return J[:, nm.index(n), :]

    # up axis = the one the body model actually spans most in
    up = int(np.argmax([float(np.median(J[:, :, a].max(1) - J[:, :, a].min(1)))
                        for a in range(3)]))
    chest = 0.5 * (j("left_shoulder") + j("right_shoulder"))
    return {
        "n": len(J),
        "up_axis": up,
        "human_height": float(height),
        "pelvis_h": j("pelvis")[:, up],
        "chest_h": chest[:, up],
        "knee_flex": knee_flexion(j("left_hip"), j("left_knee"), j("left_ankle")),
        "torso_lean": lean_from_vertical(chest - j("pelvis"), up),
    }


def robot_curves(prefix):
    """G1 robot: pelvis height, chest (shoulder-mid) height, knee, lean."""
    import torch
    from general_motion_retargeting import KinematicsModel, ROBOT_XML_DICT
    from general_motion_retargeting.data_loader import load_robot_motion

    f = os.path.join(ROOT, "data", "%s_g1.pkl" % prefix)
    if not os.path.exists(f):
        return None
    km = KinematicsModel(str(ROBOT_XML_DICT["unitree_g1"]), device="cpu")
    md, _, _, _, _, _, _ = load_robot_motion(f)
    with torch.no_grad():
        bp, _ = km.forward_kinematics(
            torch.tensor(md["root_pos"], dtype=torch.float32),
            torch.tensor(md["root_rot"], dtype=torch.float32),
            torch.tensor(md["dof_pos"], dtype=torch.float32))
    bp = bp.numpy()
    bn = km.body_names

    def b(n):
        return bp[:, bn.index(n), :]

    # NOTE: head_link in this G1 XML is a dummy frame level with the pelvis;
    # head_mocap is the real head.  torso_link sits only 44 mm above the pelvis,
    # so the shoulder midpoint is the only chest proxy with a usable lever.
    chest = 0.5 * (b("left_shoulder_pitch_link") + b("right_shoulder_pitch_link"))
    return {
        "n": len(bp),
        "pelvis_h": b("pelvis")[:, 2],
        "chest_h": chest[:, 2],
        "knee_flex": knee_flexion(b("left_hip_pitch_link"), b("left_knee_link"),
                                  b("left_ankle_roll_link")),
        "torso_lean": lean_from_vertical(chest - b("pelvis"), 2),
    }


def video_head_row(prefix, motion, hsv_str=ORANGE_HSV):
    """Real person's head-top row from the source video (static camera).

    Also segments the box with `hsv_str` (`"H,S,V:H,S,V"`) for the mapping hop.
    """
    import cv2

    src = os.path.join(ROOT, "video", "%s.mp4" % prefix)
    if not os.path.exists(src):
        return None
    lo, hi = parse_hsv(hsv_str)
    cap = cv2.VideoCapture(src)
    grays, colors = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        colors.append(fr)
        grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not grays:
        return None
    gray = np.stack(grays)
    N, H, W = gray.shape
    bg = np.median(gray, axis=0).astype(np.uint8)
    roi = np.zeros((H, W), np.uint8)
    roi[int(0.22 * H):, int(0.12 * W):int(0.80 * W)] = 1

    head_row = np.full(N, np.nan)
    for t in range(N):
        m = ((cv2.absdiff(gray[t], bg) > 28) & (roi > 0)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] >= 400:
            head_row[t] = stats[k, cv2.CC_STAT_TOP]

    # box rise, to validate the video <-> bag frame mapping.  `lo`/`hi` come from
    # the caller's resolved --hsv / configs/boxes.json range, NOT a hard-coded
    # orange: the calibrated range is a property of the venue.
    hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in colors]
    box_row = np.full(N, np.nan)
    for t in range(N):
        bm = cv2.inRange(hsv[t], lo, hi)
        bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        if bm.sum():
            box_row[t] = np.nonzero(bm)[0].mean()
    return {"n": N, "head_row": head_row, "box_row": box_row, "motion": motion}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
CHANNELS = [("pelvis_h", 0.95), ("chest_h", 0.95),
            ("knee_flex", CORR_MIN), ("torso_lean", CORR_MIN)]


def check(prefix, use_video=True, quiet=False, hsv=None):
    def say(*a):
        if not quiet:
            print(*a)

    hum = human_curves(prefix)
    rob = robot_curves(prefix)
    if hum is None or rob is None:
        say("  %s: SKIP (need both _smplx.npz and _g1.pkl)" % prefix)
        return None

    hsv_str, hsv_src = resolve_hsv(prefix, hsv)
    try:
        parse_hsv(hsv_str)
    except (ValueError, IndexError, AttributeError) as e:
        say("  hsv %s (%s) is unparseable (%s) -> falling back to %s"
            % (hsv_str, hsv_src, e, ORANGE_HSV))
        hsv_str, hsv_src = ORANGE_HSV, "orange default (bad HSV string)"
    say("  hsv %s (from %s)" % (hsv_str, hsv_src))
    report_hsv = {"hsv": hsv_str, "source": hsv_src}

    N = min(hum["n"], rob["n"])
    # evaluate over the pickup, not the whole clip: walking dominates the rest
    pel = smooth(hum["pelvis_h"][:N])
    peak = int(np.argmin(pel))
    w0, w1 = max(0, peak - 60), min(N, peak + 60)

    say("  window pkl frames %d..%d   (pickup at ~%d, %d frames @30fps)"
        % (w0, w1, peak, w1 - w0))
    say("  %-12s %8s %8s   %s" % ("channel", "lag", "corr", "verdict"))
    report = {"prefix": prefix, "window": [w0, w1], "pickup_frame": peak,
              "channels": {}, "ok": True, "video_ok": None,
              "hsv": report_hsv,
              "gate": "smplx->robot retarget channels (hop 1 only)"}
    for name, cmin in CHANNELS:
        lag, corr = best_lag(hum[name][w0:w1], rob[name][w0:w1])
        ok = (abs(lag) <= LAG_TOL) and (corr >= cmin)
        report["channels"][name] = {"lag": int(lag), "corr": float(corr), "ok": bool(ok)}
        report["ok"] = report["ok"] and ok
        say("  %-12s %+8d %8.3f   %s" % (name, lag, corr, "OK" if ok else "OUT OF TOLERANCE"))

    if use_video:
        v = video_head_row(prefix, None, hsv_str)
        if v is None:
            say("  video: not found, skipping the GVHMR-vs-video hop")
            report["video"] = None
        else:
            # --- what the depth side measured, read EXPLICITLY --------------
            # `frame_offset` and the carry start differ per clip (offsets across
            # this repo run -11..+4, carry starts 114..205), so a fabricated
            # default would silently shift the whole video axis.  Missing input
            # is REPORTED and leaves these None -- never guessed.
            off, carry0 = None, None
            wpath = os.path.join(ROOT, "data", "%s_world_6dof.npz" % prefix)
            wrel = os.path.relpath(wpath, ROOT)
            try:
                with np.load(wpath, allow_pickle=True) as z:
                    try:
                        off = int(np.asarray(z["frame_offset"]).item())
                    except KeyError:
                        say("  frame_offset: MISSING from %s -> pkl->video offset "
                            "is UNKNOWN" % wrel)
                    except (TypeError, ValueError) as e:
                        say("  frame_offset: unparseable in %s (%s) -> pkl->video "
                            "offset is UNKNOWN" % (wrel, e))
                    try:
                        c = np.asarray(z["carry"]).astype(bool)
                    except KeyError:
                        say("  carry: MISSING from %s -> depth-measured carry "
                            "start is UNKNOWN" % wrel)
                        c = None
                    except (TypeError, ValueError) as e:
                        say("  carry: unparseable in %s (%s) -> depth-measured "
                            "carry start is UNKNOWN" % (wrel, e))
                        c = None
                    if c is not None:
                        if c.any():
                            carry0 = int(np.argmax(c))
                        else:
                            say("  carry: all-False in %s -> depth-measured "
                                "carry start is UNKNOWN" % wrel)
            except FileNotFoundError:
                say("  world npz MISSING (%s) -> frame_offset and depth-measured "
                    "carry start are UNKNOWN" % wrel)
            except Exception as e:
                say("  world npz UNREADABLE (%s: %s: %s) -> frame_offset and "
                    "depth-measured carry start are UNKNOWN"
                    % (wrel, type(e).__name__, e))
            say("  depth side: frame_offset=%s  carry_start=%s"
                % ("?" if off is None else "%+d" % off,
                   "?" if carry0 is None else "pkl f%d" % carry0))

            # --- mapping hop first: it VALIDATES the video's frame axis ---
            # Without a trustworthy video<->bag axis, a video-vs-reconstruction
            # timing number is meaningless: the hop is reported SKIPPED (not
            # OK, not OUT OF TOLERANCE) and the video hop stays INCONCLUSIVE.
            bre, _ = onset(v["box_row"])
            exp = None
            mapping_ok = None                      # None == "not evaluated"
            if carry0 is None or off is None:
                say("  mapping hop: SKIPPED -- %s, so the depth-predicted box "
                    "rise frame cannot be computed; the video frame axis is NOT "
                    "validated"
                    % ("no depth-measured carry start" if carry0 is None
                       else "no frame_offset"))
            else:
                exp = carry0 + off
                bdelta = bre - exp
                mapping_ok = abs(bdelta) <= 5
                say("  mapping hop: box rises in video at f%d (depth-measured "
                    "carry start pkl f%d + frame_offset %+d predicts f%d) -> "
                    "delta %+d  %s"
                    % (bre, carry0, off, exp, bdelta,
                       "OK" if mapping_ok else "OUT OF TOLERANCE"))

            report["video"] = {"frame_offset": off,
                               "carry_start_pkl_frame": carry0,
                               "box_rise_video_frame": int(bre)}
            if exp is not None:
                report["video"]["box_rise_predicted"] = int(exp)
            if mapping_ok is not None:
                report["video"]["mapping_ok"] = bool(mapping_ok)

            if off is None:
                # Without the offset the reconstruction cannot be placed on the
                # video axis at all, so hop 2 is not evaluable either.
                verdict = ("INCONCLUSIVE (video frame axis not validated: no "
                           "frame_offset, mapping hop SKIPPED)")
                report["video_ok"] = None
                report["video"]["verdict"] = verdict
                say("  video hop: SKIPPED -- %s; no timing number is reported "
                    "for the video hop" % verdict)
            else:
                vv = v["head_row"][:]
                fin = np.isfinite(vv)
                vv = np.interp(np.arange(len(vv)), np.nonzero(fin)[0], vv[fin])
                # Measure both sides over the SAME span of the SAME physical
                # motion: from 55 frames before the pickup to 25 after, in each
                # one's own axis.  A window anchored too late starts mid-crouch
                # and its baseline is junk.
                sw = slice(max(0, peak - 55), min(len(hum["pelvis_h"]), peak + 25))
                vw = slice(max(0, sw.start + off), min(len(vv), (peak + 25) + off))
                ve, _ = onset(vv[vw])
                he, _ = onset(hum["pelvis_h"][sw])
                lead = (sw.start + he + off) - (vw.start + ve)
                report["video"]["video_onset"] = int(vw.start + ve)
                report["video"]["smplx_onset_in_video_axis"] = int(sw.start + he + off)
                report["video"]["recon_leads_video_frames"] = int(lead)

                lead_ok = abs(lead) <= LAG_TOL
                if mapping_ok is None:
                    # Un-validated axis: NEVER a confident PASS or FAIL here.
                    verdict = ("INCONCLUSIVE (video frame axis not validated: "
                               "mapping hop SKIPPED)")
                    report["video_ok"] = None
                elif not mapping_ok:
                    verdict = "INCONCLUSIVE (video frame axis not validated)"
                    report["video_ok"] = None
                elif not lead_ok:
                    verdict = "OUT OF TOLERANCE"
                    report["video_ok"] = False
                else:
                    verdict = "OK"
                    report["video_ok"] = True
                # The GATE is the retarget channels only: they are what the
                # pipeline actually guarantees.  The video hop is evidence, kept
                # advisory so a flaky source video (auto-exposure, dropped
                # frames) cannot block a clip whose motion is provably in sync.
                say("  video hop: real body crouches at video f%d, reconstruction "
                    "at video f%d -> GVHMR leads by %+d frames  %s"
                    % (report["video"]["video_onset"],
                       report["video"]["smplx_onset_in_video_axis"], lead, verdict))
                report["video"]["verdict"] = verdict

    # Say plainly WHAT was gated: `report["ok"]` is hop 1 only.  The reader must
    # not read this PASS as "the video frame axis was validated" -- hop 3 reports
    # that separately (OK / OUT OF TOLERANCE / SKIPPED).
    say("  => %s" % ("PASS: robot and human are time-synchronised (hop 1: "
                     "retarget channels are the gate)"
                     if report["ok"] else
                     "FAIL: channels above are out of tolerance (hop 1: retarget "
                     "channels are the gate)"))
    vo = report.get("video_ok", None)
    if report.get("video") is not None and vo is None:
        say("     NOTE: the video hop is INCONCLUSIVE -- the video frame axis "
            "was not validated, so hop 1 alone decides this verdict.")
    elif report.get("video") is not None:
        say("     video hop: %s (advisory; not part of the gate)"
            % report["video"].get("verdict", "?"))
    out = os.path.join(ROOT, "data", "%s_timing_sync.json" % prefix)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    say("  wrote %s" % os.path.relpath(out, ROOT))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", help="clip prefix, e.g. 20260922_161727")
    ap.add_argument("--all", action="store_true", help="every clip with a _g1.pkl")
    ap.add_argument("--no_video", action="store_true", help="skip the video hop")
    ap.add_argument("--hsv", default=None,
                    help="box HSV range for the mapping hop, as "
                         "\"H,S,V:H,S,V\" (OpenCV hue 0-179), e.g. "
                         "\"4,75,79:24,255,255\".  Overrides configs/boxes.json. "
                         "Unset = configs/boxes.json (name/prefix/session match) "
                         "-> orange default %s.  A wrong range makes the mask "
                         "track clothing and the box rise frame garbage." % ORANGE_HSV)
    ap.add_argument("--json", help="write the combined report here")
    a = ap.parse_args()

    if a.all:
        prefixes = sorted(os.path.splitext(os.path.basename(p))[0][:-3]
                          for p in __import__("glob").glob(
                              os.path.join(ROOT, "data", "*_g1.pkl")))
    elif a.prefix:
        prefixes = [a.prefix]
    else:
        ap.error("pass --prefix or --all")

    results, all_ok = [], True
    for p in prefixes:
        print("=" * 74)
        print("### %s" % p)
        print("=" * 74)
        r = check(p, use_video=not a.no_video, hsv=a.hsv)
        if r:
            results.append(r)
            all_ok = all_ok and r["ok"]
        print()
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("combined report -> %s" % a.json)
    print("OVERALL: %s" % ("PASS" if all_ok and results else
                           ("NO CLIPS" if not results else "FAIL")))
    return 0 if (all_ok and results) else 1


if __name__ == "__main__":
    sys.exit(main())

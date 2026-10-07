#!/usr/bin/env python
"""Recover the carried box's 3D trajectory from a RealSense bag using
color segmentation + aligned depth — no markers needed, works at any distance.

For each color frame:
  1. HSV mask of the box color, morphological cleanup, largest tracked blob
  2. median depth over the eroded blob (valid pixels only)
  3. unproject blob centroid -> camera-frame 3D point (OpenCV convention:
     x right, y down, z forward, meters)

With ``--fit-cuboid`` an extra per-frame 6-DoF estimate is produced on top of
the centroid position.  The centroid alone carries NO orientation (it is a
single unprojected pixel), which is why the orientation used to be hard-coded
to "upright + one constant yaw".  ``scripts/cuboid_fit.py`` supplies the real
pose: because the box dimensions are known, its projected silhouette is fully
determined by the 6-DoF pose, and that silhouette is fitted to the colour mask
(sub-pixel accurate) with a per-pixel depth-consistency term.

The earlier attempt — per-pixel depth normals clustered into the front and top
faces — is NOT used any more.  On this footage the box is only 30-70 px wide at
2-3.5 m, so the depth gradient over one pixel step is the same size as the
depth noise and the normals come out near-random.  It measured a 0.43 m front
face on a 0.30 m box and a silhouette IoU of 0.00.  See cuboid_fit.py.

Known limits, by construction:
  * frames where the fit fails get pose_valid=False (nan), never an identity
    pose; the caller keeps the hand-rigid behaviour there
  * a square cross-section (this box is 0.3 x 0.3 x 0.45) has a 90 deg yaw
    symmetry, so the fit cannot — and does not need to — resolve yaw beyond
    that; any comparison against a reference pose must be taken modulo it
  * a single-face view leaves the tilt weakly constrained; the depth term is
    what pins it, so low-texture depth there means a low-confidence tilt

Outputs:
  npz     : t / pos_cam (N,3) / K / valid / bbox_w_m / bbox_h_m
            + (with --fit-cuboid) R_cam (N,3,3) / t_cam (N,3) / pose_valid (N,)
              / fit_iou (N,) / fit_depth_err_m (N,)
  preview : annotated mp4 (mask contour + centroid + fitted cuboid outline)
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import pyrealsense2 as rs

from cuboid_fit import (hull_mask, mask_pixels,  # noqa: E402
                        solve as solve_cuboid)


def get_frame(frames, stream):
    try:
        return frames.first(stream)
    except RuntimeError:
        return None


def grab_probe_frame(pipe, align, rs, max_frames=120):
    """Grab one mid-clip colour+depth pair WITHOUT consuming the main loop.

    The bag plays back from the start each time, so this costs a short extra
    pass (bounded by max_frames) and is used only for `--hsv auto` /
    `--box-size auto` / the colour advisory.
    """
    frames = []
    try:
        while len(frames) < max_frames:
            fs = align.process(pipe.wait_for_frames(timeout_ms=8000))
            cf = get_frame(fs, rs.stream.color)
            df = get_frame(fs, rs.stream.depth)
            if cf is None or df is None:
                continue
            frames.append((np.asanyarray(cf.get_data()),
                           np.asanyarray(df.get_data())))
    except RuntimeError:
        pass
    if not frames:
        return None
    return frames[len(frames) // 2]


def resolve_hsv_arg(arg, cfg="configs/boxes.json", prefix=None):
    """Turn `--hsv` into (lo, hi, provenance_string).

    Priority: explicit string -> configs/boxes.json (by clip prefix) ->
    orange default.  `auto` is handled by the caller (it needs the video).
    """
    if arg:
        lo, hi = parse_hsv(arg)
        return lo, hi, "explicit --hsv"
    if prefix and Path(cfg).exists():
        try:
            with open(cfg, encoding="utf-8") as f:
                boxes = json.load(f)
            for name, spec in boxes.items():
                if not isinstance(spec, dict) or not spec.get("hsv"):
                    continue
                if spec.get("prefix") == prefix \
                        or prefix in (spec.get("prefixes") or []) \
                        or (spec.get("session")
                            and prefix.startswith(str(spec["session"]))) \
                        or name in prefix:
                    lo, hi = parse_hsv(spec["hsv"])
                    return lo, hi, "%s -> %s" % (cfg, name)
        except (OSError, ValueError) as e:
            print("hsv       : ! %s unreadable (%s); using the default range"
                  % (cfg, e))
    lo, hi = parse_hsv("5,90,80:28,255,255")
    return lo, hi, "orange default"


def parse_hsv(s):
    lo, hi = s.split(":")
    lo = [int(v) for v in lo.split(",")]
    hi = [int(v) for v in hi.split(",")]
    return np.array(lo, np.uint8), np.array(hi, np.uint8)


def parse_size(s):
    v = [float(x) for x in s.replace("x", ",").split(",")]
    if len(v) != 3:
        raise ValueError("--box-size needs three numbers: W,D,H")
    return np.array(v, dtype=np.float64)


# --------------------------------------------------------------------------
# Scene-agnostic helpers.  The pipeline grew around one venue, so several knobs
# are really "the value that worked at 1280x720 with an orange box 3-4 m away".
# These helpers re-derive them for a DIFFERENT resolution / box / room, and
# report where each value came from so a surprising run can be audited.
# --------------------------------------------------------------------------
BASE_W, BASE_H = 1280, 720
BASE_ROI_PAD, BASE_MIN_AREA = 280, 400

# Colour-family seeds for `--hsv auto`.  Ordered by how common a "box" is; the
# estimator below still MEASURES the actual range from the image, these only
# decide which blobs to consider.  OpenCV H is 0-179 (i.e. degrees/2).
BOX_HUE_CENTRES = {
    "orange": 12,      # 5-28   (0.30 m orange box, both venues)
    "blue": 115,       # 100-130 (the 0911 box)
    "red": 0,          # wraps 170/0-8
    "green": 60, "yellow": 30, "cyan": 90, "purple": 145,
}


def scale_to_resolution(w, h, roi_pad=None, min_area=None):
    """Scale the pixel-space knobs against the 1280x720 baseline.

    `--roi-pad 280 --min-area 400` are 720p numbers; the older 640x480 script
    defaults were 140/120.  Handing a 640x480 take the 720p pair widens the
    search window (it latches onto same-coloured clothing) and quadruples the
    area gate (the box is ~4x fewer pixels).  Derive them unless given, and say
    which happened.
    """
    area_ratio = (float(w) * float(h)) / float(BASE_W * BASE_H)
    width_ratio = float(w) / float(BASE_W)
    if roi_pad is None:
        roi_pad = int(round(BASE_ROI_PAD * width_ratio))
        print("roi-pad   : %d px  (scaled from the 720p baseline %d "
              "at %dx%d)" % (roi_pad, BASE_ROI_PAD, w, h))
    else:
        print("roi-pad   : %d px  (explicit --roi-pad)" % roi_pad)
    if min_area is None:
        min_area = int(round(BASE_MIN_AREA * area_ratio))
        print("min-area  : %d px^2 (scaled from the 720p baseline %d "
              "at %dx%d)" % (min_area, BASE_MIN_AREA, w, h))
    else:
        print("min-area  : %d px^2 (explicit --min-area)" % min_area)
    return int(roi_pad), int(max(min_area, 20))


def hsv_range_from_pixels(pix_bgr):
    """calib_hsv.py's estimator, applied to an arbitrary pixel set.

    H: circular mean +- max(10, 3 sigma).  S/V: [median - 80, 255].  Kept
    deliberately identical to scripts/calib_hsv.py so an automatic suggestion
    and a hand-drawn ROI produce comparable ranges.
    """
    hsv = cv2.cvtColor(pix_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)
    pix = hsv.reshape(-1, 3).astype(np.float64)
    hh, s, v = pix[:, 0], pix[:, 1], pix[:, 2]
    ang = np.deg2rad(hh * 2.0)
    sin_m, cos_m = np.sin(ang).mean(), np.cos(ang).mean()
    R = float(np.hypot(sin_m, cos_m))
    h_mean = float(np.rad2deg(np.arctan2(sin_m, cos_m))) % 360.0
    h_std = 0.0 if R > 0.999 else float(np.sqrt(-2.0 * np.log(max(R, 1e-6))))
    c = h_mean / 2.0
    d = max(3.0 * h_std / 2.0, 10.0)
    lo = (max(0, int(round(c - d))), int(max(0, np.median(s) - 80)),
          int(max(0, np.median(v) - 80)))
    hi = (min(179, int(round(c + d))), 255, 255)
    return lo, hi, h_mean, h_std


def _hue_dist_deg(a, b):
    """Shortest distance between two OpenCV hues, in degrees."""
    da = a * 2.0
    db = b * 2.0
    return min(abs(da - db), 360.0 - abs(da - db))


def auto_hsv(frame, min_area=None, verbose=True):
    """Suggest an HSV range for the box by finding a large, box-like blob.

    NOT a replacement for scripts/calib_hsv.py (which lets a human frame the
    face exactly).  This exists so a first run on a new venue does not have to
    guess, and so a wrong range announces itself instead of silently producing
    a 0-frame trajectory.

    Returns (hsv_string, candidates) or (None, candidates) when nothing
    credible was found.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # ignore the very top/bottom bands: ceiling and floor are not the box
    band = np.zeros_like(sat)
    band[int(0.10 * h):int(0.94 * h)] = sat[int(0.10 * h):int(0.94 * h)]
    if min_area is None:
        min_area = int(round(BASE_MIN_AREA * (w * h) / float(BASE_W * BASE_H)))

    cands = []
    for thr in (150, 120, 90, 70):
        m = ((band > thr) & (val > 40)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x, y, bw, bh = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                            stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
            fill = area / float(max(bw * bh, 1))
            ar = bw / float(max(bh, 1))
            if fill < 0.62:                 # a box silhouette is fairly solid
                continue
            if not (0.25 <= ar <= 4.0):     # and not a sliver
                continue
            sel = lab == k
            pix = frame[sel]
            if len(pix) < 100:
                continue
            lo, hi, hmean, hstd = hsv_range_from_pixels(pix)
            # score: big, solid, and close to a plausible "box colour" hue
            fam = min((_hue_dist_deg(hmean / 2.0, BH), name)
                      for name, BH in BOX_HUE_CENTRES.items())
            score = area * fill / (1.0 + fam[0] / 15.0)
            cands.append({
                "hsv": "%d,%d,%d:%d,%d,%d" % (lo + hi),
                "lo": lo, "hi": hi, "area_px": area, "fill": round(fill, 3),
                "aspect": round(ar, 2), "h_mean_deg": round(hmean, 1),
                "h_std_deg": round(hstd, 1), "family": fam[1],
                "hue_miss_deg": round(fam[0], 1), "sat_thr": thr,
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "score": round(score, 1),
            })
        if cands:
            break                             # the loosest threshold that worked
    # de-duplicate by the rounded range (the same blob appears at several
    # thresholds only when the loose pass found nothing)
    seen, uniq = set(), []
    for c in sorted(cands, key=lambda d: -d["score"]):
        key = (c["lo"], c["hi"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    if verbose:
        if uniq:
            print("auto-hsv  : %d candidate(s) on the sampled frame" % len(uniq))
            for c in uniq[:4]:
                print("   %-26s area %7d px  fill %.2f  aspect %.2f  "
                      "hue %.0f+/-%.0f deg (%s, miss %.0f deg)"
                      % (c["hsv"], c["area_px"], c["fill"], c["aspect"],
                         c["h_mean_deg"], c["h_std_deg"], c["family"],
                         c["hue_miss_deg"]))
        else:
            print("auto-hsv  : no credible box-like blob found")
    return (uniq[0]["hsv"] if uniq else None), uniq


def auto_box_size(frame, K, depth_frame=None, depth_scale=None,
                  px_hint=None, verbose=True):
    """Estimate the box's front-face edge lengths from one frame.

    Two independent routes, both reported:
      * the blob's pixel width/height at an assumed standoff (metric only if
        the standoff is right) -- this is what a single image can give;
      * the same but using the measured DEPTH at the blob, when a depth frame
        is supplied, which removes the standoff guess.

    Deliberately conservative: a face seen at an angle foreshortens, so this
    reports a RANGE (min/max over plausible aspect corrections) rather than a
    single number, and it never silently replaces a caliper measurement.

    Returns dict or None.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    band = np.zeros_like(sat)
    band[int(0.10 * h):int(0.94 * h)] = sat[int(0.10 * h):int(0.94 * h)]
    for thr in (150, 120, 90, 70):
        m = (band > thr).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < 400:
            continue
        x, y, bw, bh = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                        stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
        fill = area / float(max(bw * bh, 1))
        if fill < 0.62:
            continue
        # standoff: measured depth at the blob if available, else the repo's
        # typical parked range
        z, z_src = None, "assumed 3.0 m (no depth available)"
        if depth_frame is not None and depth_scale:
            sel = (lab == k)
            dz = depth_frame[sel].astype(np.float64) * depth_scale
            dz = dz[(dz > 0.3) & (dz < 10.0)]
            if len(dz) > 50:
                z = float(np.median(dz))
                z_src = "measured depth %.2f m at the blob" % z
        if z is None:
            z = 3.0
        w_m = float(bw) * z / K[0, 0]
        h_m = float(bh) * z / K[1, 1]
        out = {
            "px_wh": [int(bw), int(bh)], "fill": round(fill, 3),
            "standoff_m": round(z, 3), "standoff_source": z_src,
            "est_w_m": round(w_m, 3), "est_h_m": round(h_m, 3),
            "bbox": [int(x), int(y), int(bw), int(bh)],
        }
        if verbose:
            print("auto-size : front face %dx%d px, fill %.2f, %s"
                  % (bw, bh, out["fill"], z_src))
            print("            => visible face ~%.3f x %.3f m "
                  "(foreshortening only shrinks it, so treat these as LOWER "
                  "bounds)" % (out["est_w_m"], out["est_h_m"]))
        return out
    if verbose:
        print("auto-size : no credible box-like blob found")
    return None


def project(P, K):
    """(N,3) camera-frame points -> (N,2) pixels."""
    z = np.maximum(P[:, 2], 1e-6)
    return np.stack([K[0, 0] * P[:, 0] / z + K[0, 2],
                     K[1, 1] * P[:, 1] / z + K[1, 2]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="synthesise a known box and verify the cuboid fit, "
                         "then exit (no bag needed)")
    ap.add_argument("--hsv", default=None,
                    help="H,S,V lo:hi (OpenCV H 0-179).  Default: look the "
                         "range up in configs/boxes.json for this clip's "
                         "session, else the orange default "
                         "'5,90,80:28,255,255'.  Pass 'auto' to MEASURE a "
                         "range from the video (a large box-like blob), which "
                         "is the right first move in a new room; pass an "
                         "explicit string to override.  A wrong range yields a "
                         "silent 0 pct, so the run warns and suggests a hue when "
                         "nothing is accepted.")
    ap.add_argument("--min-area", type=int, default=None,
                    help="min blob area in px to accept.  Default: scaled from "
                         "the 1280x720 baseline (400 px^2) by the frame area, "
                         "so a 640x480 take is not judged by a 720p gate.")
    ap.add_argument("--roi-pad", type=int, default=None,
                    help="px search window around the previous position.  "
                         "Default: scaled from the 1280x720 baseline (280 px) "
                         "by the frame width.")
    ap.add_argument("--max-jump", type=float, default=0.9,
                    help="meters; reject blob farther than this from prediction")
    ap.add_argument("--out", default="data/box_traj_depth.npz")
    ap.add_argument("--meta-out", default="data/camera_calib.json")
    ap.add_argument("--preview", default="video/box_depth_preview.mp4")
    ap.add_argument("--dump-frames", default=None,
                    help="folder for annotated PNGs (every frame)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--fit-cuboid", action="store_true",
                    help="also estimate a 6-DoF box pose per frame by fitting "
                         "the known-size cuboid's silhouette to the mask plus "
                         "a per-pixel depth term (scripts/cuboid_fit.py). "
                         "Frames that fail get pose_valid=False, never an "
                         "identity pose.")
    ap.add_argument("--box-size", default="0.30,0.30,0.45",
                    help="real box W,D,H in metres — the cuboid fit needs it, "
                         "it is what makes the silhouette a metric pose. "
                         "Pass 'auto' to ESTIMATE the visible face from the "
                         "video (reported with its standoff assumption, and "
                         "never silently substituted).  A caliper measurement "
                         "beats both: the size is the fit's only scale source.")
    ap.add_argument("--depth-step", type=int, default=3,
                    help="pixel subsampling for the cuboid fit's depth term")
    ap.add_argument("--min-iou", type=float, default=0.55,
                    help="reject a cuboid fit whose silhouette IoU with the "
                         "mask is below this (0 disables the gate)")
    args = ap.parse_args()

    if args.selftest:
        import cuboid_fit
        cuboid_fit.selftest()
        return 0
    if not args.bag:
        ap.error("--bag is required (or use --selftest)")

    # Which range do we segment with?  `auto` needs the video, so it is
    # resolved after the pipeline opens (below); everything else resolves now.
    want_auto_hsv = str(args.hsv or "").strip().lower() == "auto"
    prefix = None
    try:
        import clip_naming
        prefix = clip_naming.canonical_prefix(Path(args.bag))
    except Exception:
        prefix = None
    if want_auto_hsv:
        hsv_lo = hsv_hi = None
        hsv_src = "auto (measured from the video)"
    else:
        hsv_lo, hsv_hi, hsv_src = resolve_hsv_arg(args.hsv, prefix=prefix)
        print("hsv       : %s  (%s)" % (args.hsv or "%d,%d,%d:%d,%d,%d"
                                        % tuple(hsv_lo) + tuple(hsv_hi),
                                        hsv_src))
    bag = Path(args.bag)

    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(bag), repeat_playback=False)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback()
    try:
        playback.set_real_time(False)
    except RuntimeError:
        pass
    align = rs.align(rs.stream.color)

    # A color-only bag has no Stereo Module profiles at all.  Without this
    # guard the align/process step raises RuntimeError, which the main loop
    # below legitimately treats as "end of recording" — so the script used to
    # exit 0 having produced a zero-frame npz, with no hint as to why.
    has_depth = any(sp.stream_type() == rs.stream.depth
                    for s in profile.get_device().sensors
                    for sp in s.get_stream_profiles())
    if not has_depth:
        raise SystemExit(
            f"{bag.name} has NO depth stream (color-only recording). "
            "The depth route cannot run on this clip. Re-record with "
            "scripts/record_rgbd.py so both streams land in the file, or use "
            "the ArUco route (scripts/solve_box_pose.py) for it instead.")

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
    W, H = intr.width, intr.height
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"intr {W}x{H} fx={intr.fx:.2f} fy={intr.fy:.2f} "
          f"cx={intr.ppx:.2f} cy={intr.ppy:.2f}  depth_scale={depth_scale}")

    if args.dump_frames:
        Path(args.dump_frames).mkdir(parents=True, exist_ok=True)
    writer = None
    if args.preview:
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.preview,
                                 cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pos_prev = None
    pix_prev = None
    pose_prev = None
    recs = []
    last_color = None

    # ---- resolution-agnostic knobs -------------------------------------
    roi_pad, min_area = scale_to_resolution(W, H, args.roi_pad, args.min_area)
    args.roi_pad, args.min_area = roi_pad, min_area

    # ---- resolve the box size (explicit / estimated / default) ----------
    auto_size_info = None
    auto_hsv_info = None
    if want_auto_hsv:
        print("hsv       : 'auto' -- measuring a range from a sampled frame ...")
        probe = grab_probe_frame(pipe, align, rs)
        if probe is None:
            ap.error("--hsv auto: could not sample a frame from the bag")
        pcolor, _ = probe
        cand, cands = auto_hsv(pcolor, min_area=min_area)
        if cand is None:
            print("hsv       : ! no credible box-like blob found.  Do NOT trust "
                  "a guess here -- run scripts/calib_hsv.py on this bag and "
                  "pass its printed range, or pass --hsv explicitly.  Falling "
                  "back to the orange default for this run.")
            hsv_lo, hsv_hi = parse_hsv("5,90,80:28,255,255")
            hsv_src = "orange default (auto failed)"
        else:
            hsv_lo, hsv_hi = parse_hsv(cand)
            hsv_src = "auto (largest box-like blob)"
            auto_hsv_info = {"chosen": cand, "candidates": cands}
            print("hsv       : USING auto range %s -- %d candidate(s) found; "
                  "confirm with scripts/calib_hsv.py if the mask looks wrong"
                  % (cand, len(cands)))

    if str(args.box_size).strip().lower() == "auto":
        print("box-size  : 'auto' -- sampling a frame to estimate the visible "
              "face ...")
        probe = grab_probe_frame(pipe, align, rs)
        if probe is None:
            print("box-size  : ! could not sample a frame; falling back to the "
                  "default 0.30,0.30,0.45 -- PASS --box-size explicitly")
            box_size = np.array([0.30, 0.30, 0.45], dtype=np.float64)
        else:
            pcolor, pdepth = probe
            auto_size_info = auto_box_size(pcolor, K, pdepth, depth_scale)
            if auto_size_info is not None and \
                   0.05 <= auto_size_info["est_w_m"] <= 2.0 and \
                   0.05 <= auto_size_info["est_h_m"] <= 2.0:
                # a square-ish cross-section is the documented case; if the
                # face is clearly non-square, keep W/D distinct
                ew, eh = auto_size_info["est_w_m"], auto_size_info["est_h_m"]
                if abs(ew - eh) / max(ew, eh) < 0.35:
                    box_size = np.array([ew, ew, eh], dtype=np.float64)
                else:
                    box_size = np.array([ew, eh, max(ew, eh)], dtype=np.float64)
                print("box-size  : ESTIMATED %s m -- this is a ONE-FRAME "
                      "estimate, not a caliper measurement.  The size is the "
                      "fit's only scale source, so verify before trusting "
                      "downstream metrics."
                      % ",".join("%.3f" % x for x in box_size))
            else:
                print("box-size  : ! estimate implausible or unavailable; "
                      "falling back to the default 0.30,0.30,0.45")
                box_size = np.array([0.30, 0.30, 0.45], dtype=np.float64)
    else:
        box_size = parse_size(args.box_size)
        print("box-size  : %s m (explicit) -- confirm this matches a caliper "
              "measurement; a wrong size moves the box and its orientation "
              "systematically, not slightly"
              % ",".join("%.3f" % x for x in box_size))
    half = box_size * 0.5
    if args.fit_cuboid:
        print(f"cuboid fit ON: box {box_size[0]:.2f} x {box_size[1]:.2f} x "
              f"{box_size[2]:.2f} m, half-extents {half}")
    poses = []
    fit_reasons = []
    t = 0
    try:
        while True:
            frames = pipe.wait_for_frames(timeout_ms=8000)
            frames = align.process(frames)
            cf = get_frame(frames, rs.stream.color)
            df = get_frame(frames, rs.stream.depth)
            if cf is None or df is None:
                continue
            color = np.asanyarray(cf.get_data())
            last_color = color
            depth = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale

            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
            best, best_score = 0, 0.0
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < args.min_area:
                    continue
                cx, cy = cents[i]
                if pos_prev is not None:
                    z_pred = pos_prev[2]
                    u_pred = K[0, 2] + K[0, 0] * pos_prev[0] / z_pred
                    v_pred = K[1, 2] + K[1, 1] * pos_prev[1] / z_pred
                    d_px = np.hypot(cx - u_pred, cy - v_pred)
                    if d_px > args.roi_pad:
                        continue
                    score = area / (1.0 + d_px / 50.0)
                else:
                    score = area
                if score > best_score:
                    best_score, best = score, i

            accepted = False
            pos = np.full(3, np.nan)
            R_fit = t_fit = None
            fit_info = {}
            if best > 0:
                blob = (labels == best).astype(np.uint8)
                eroded = cv2.erode(blob, kernel, iterations=2)
                vals = depth[eroded > 0]
                vals = vals[(vals > 0.1) & (vals < 12.0)]
                area = stats[best, cv2.CC_STAT_AREA]
                cx, cy = cents[best]
                ok = len(vals) >= 20
                bw_m = bh_m = np.nan
                if ok:
                    z = float(np.median(vals))
                    x = (cx - K[0, 2]) / K[0, 0] * z
                    y = (cy - K[1, 2]) / K[1, 1] * z
                    pos = np.array([x, y, z])
                    if pos_prev is not None and \
                            np.linalg.norm(pos - pos_prev) > args.max_jump:
                        ok = False
                if ok and np.ndim(blob) == 2:
                    ys_, xs_ = np.nonzero(blob)
                    if len(ys_) > 4:
                        bw_m = (xs_.max() - xs_.min() + 1) * z / K[0, 0]
                        bh_m = (ys_.max() - ys_.min() + 1) * z / K[1, 1]
                if ok:
                    accepted = True
                    pos_prev = pos
                    pix_prev = (cx, cy)

                    # ---- 6-DoF cuboid fit (independent of the centroid) ----
                    # The centroid above is a single unprojected pixel, so it
                    # carries no orientation at all.  This fits the known-size
                    # cuboid's silhouette to the mask plus a per-pixel depth
                    # term.  A failure is recorded as pose_valid=False and
                    # never as an identity pose.
                    if args.fit_cuboid:
                        mu, my = mask_pixels(eroded, args.depth_step)
                        if len(mu) >= 60:
                            R_fit, t_fit, fit_info = solve_cuboid(
                                half, K, blob, pos, depth_img=depth,
                                uv=mu, yx=my,
                                R_prev=None if pose_prev is None else pose_prev[0],
                                C_prev=None if pose_prev is None else pose_prev[1])
                        else:
                            fit_info = {"reason": "mask too small"}
                        if R_fit is not None and args.min_iou > 0 and \
                                fit_info.get("iou", 0.0) < args.min_iou:
                            fit_info["reason"] = (
                                f"IoU {fit_info['iou']:.2f} < {args.min_iou}")
                            R_fit = t_fit = None
                        if R_fit is not None:
                            pose_prev = (R_fit, t_fit)

                if writer is not None:
                    vis = color.copy()
                    cnts, _ = cv2.findContours(
                        blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
                    if accepted:
                        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                        cv2.putText(vis, f"z={pos[2]:.2f}m "
                                         f"({pos[0]:+.2f},{pos[1]:+.2f}) a={area}",
                                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                        if args.fit_cuboid:
                            if R_fit is not None:
                                # axis triad: only draw when all four points
                                # are actually in front of the camera — a
                                # partially-filled list used to index past its
                                # end and kill the whole run.
                                tip = t_fit + 0.14 * R_fit
                                pp = []
                                for q in (t_fit, tip[:, 0], tip[:, 1],
                                          tip[:, 2]):
                                    if q[2] > 1e-3:
                                        pp.append((int(K[0, 0] * q[0] / q[2]
                                                      + K[0, 2]),
                                                   int(K[1, 1] * q[1] / q[2]
                                                      + K[1, 2])))
                                if len(pp) == 4:
                                    for colr, p in (((0, 0, 255), pp[1]),
                                                    ((0, 255, 0), pp[2]),
                                                    ((255, 0, 0), pp[3])):
                                        cv2.line(vis, pp[0], p, colr, 2)
                                # fitted-cuboid silhouette: if this polygon
                                # does not sit on the box, the pose is wrong
                                m = hull_mask(R_fit, t_fit, half, K, blob.shape)
                                if m is not None:
                                    c2, _ = cv2.findContours(
                                        m, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(vis, c2, -1, (0, 220, 0), 1)
                                cv2.putText(
                                    vis, f"pose IoU={fit_info['iou']:.2f}"
                                         f" depthErr="
                                         f"{fit_info.get('depth_err_m', float('nan'))*1000:.0f}mm",
                                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)
                            else:
                                cv2.putText(vis, f"pose FAIL: "
                                                 f"{fit_info.get('reason', '?')}",
                                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 165, 255), 1)
                    else:
                        cv2.putText(vis, f"reject a={area}", (10, 26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    writer.write(vis)
                    if args.dump_frames:
                        cv2.imwrite(
                            str(Path(args.dump_frames) / f"vis_{t:04d}.png"), vis)
            else:
                if writer is not None:
                    vis = color.copy()
                    cv2.putText(vis, "no blob", (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    writer.write(vis)
                    if args.dump_frames:
                        cv2.imwrite(
                            str(Path(args.dump_frames) / f"vis_{t:04d}.png"), vis)

            pose_ok = R_fit is not None
            poses.append((R_fit if pose_ok else np.full((3, 3), np.nan),
                          t_fit if pose_ok else np.full(3, np.nan),
                          float(fit_info.get("iou", np.nan)),
                          float(fit_info.get("depth_err_m", np.nan))))
            # keep the rejection reason as well: a bare "49% failed" is not
            # actionable, "58 frames were rejected on depth mismatch" is
            fit_reasons.append(
                "" if pose_ok
                else (str(fit_info.get("reason", "")) or "no blob accepted"))

            recs.append((pos if accepted else np.full(3, np.nan),
                         bw_m if accepted else np.nan,
                         bh_m if accepted else np.nan))
            t += 1
            if t % 40 == 0:
                print(f"  {t} frames...")
            if args.max_frames and t >= args.max_frames:
                break
    except RuntimeError:
        pass  # end of recording
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        if writer is not None:
            writer.release()

    P = np.array([r[0] for r in recs])
    BW = np.array([r[1] for r in recs])
    BH = np.array([r[2] for r in recs])
    valid = ~np.isnan(P[:, 0])
    print(f"\nframes: {len(P)}  valid 3D: {valid.sum()} "
          f"({100.0 * valid.mean():.0f}%)")
    if len(P) and not valid.any():
        # A wrong --hsv range produces a completely silent empty result, which
        # is exactly how the 20260911 blue box first came back as 0/206 while
        # the default range was still orange.  Say something actionable.
        print("\nWARNING: not one frame produced a valid 3D position.")
        print(f"  --hsv is {args.hsv} — that range probably does not match "
              "the box colour.")
        if last_color is not None:
            hsv_ = cv2.cvtColor(last_color, cv2.COLOR_BGR2HSV)
            sel = (hsv_[:, :, 1] > 90) & (hsv_[:, :, 2] > 60)
            if sel.sum() > 50:
                h = hsv_[:, :, 0][sel]
                hist = np.bincount(h, minlength=180)
                tops = np.argsort(hist)[::-1][:6]
                tops = [(int(t), int(hist[t])) for t in tops if hist[t] > 0]
                print(f"  dominant saturated hues in the last frame: {tops}")
                print("  (OpenCV H: orange 5-28, yellow 28-35, green 40-85, "
                      "blue 100-130, magenta 140-170)")
        print("  Re-run with --dump-frames and inspect the contours, or "
              "pass --hsv measured from the box.")
    if valid.any():
        gaps = []
        run = 0
        for v in valid:
            run = 0 if v else run + 1
            gaps.append(run)
        print(f"longest gap: {max(gaps)} frames")
        print(f"pos range: x [{np.nanmin(P[:,0]):+.2f},{np.nanmax(P[:,0]):+.2f}] "
              f"y [{np.nanmin(P[:,1]):+.2f},{np.nanmax(P[:,1]):+.2f}] "
              f"z [{np.nanmin(P[:,2]):.2f},{np.nanmax(P[:,2]):.2f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(t=np.arange(len(P)), pos_cam=P, K=K, valid=valid,
                   bbox_w_m=BW, bbox_h_m=BH)

    if args.fit_cuboid:
        RC = np.array([p[0] for p in poses])
        TC = np.array([p[1] for p in poses])
        FIOU = np.array([p[2] for p in poses])
        FDEP = np.array([p[3] for p in poses])
        pose_valid = ~np.isnan(TC[:, 0])
        payload.update(R_cam=RC, t_cam=TC, pose_valid=pose_valid,
                       fit_iou=FIOU, fit_depth_err_m=FDEP,
                       box_size=box_size,
                       fit_reason=np.array(fit_reasons, dtype=object))
        n_ok = int(pose_valid.sum())
        print(f"\ncuboid fit (silhouette + depth): {n_ok}/{len(pose_valid)} "
              f"frames ({100.0 * pose_valid.mean():.0f}%)")
        if n_ok:
            # ONLY the accepted frames: fit_info keeps its iou/depth_err even
            # when solve() rejected the pose, so averaging the raw arrays
            # reported a 204 mm median depth residual for a run whose gate
            # throws away anything above 150 mm.  Two different populations.
            iou_ok, dep_ok = FIOU[pose_valid], FDEP[pose_valid]
            print(f"  ACCEPTED ({n_ok} f) -- silhouette IoU : median "
                  f"{np.median(iou_ok):.3f}  p10 {np.percentile(iou_ok, 10):.3f}"
                  f"  (occlusion lowers this; it is a lower bound)")
            print(f"  ACCEPTED ({n_ok} f) -- depth residual : median "
                  f"{np.median(dep_ok) * 1000:.0f} mm  p90 "
                  f"{np.percentile(dep_ok, 90) * 1000:.0f} mm")
            print(f"  note: RealSense depth noise at 2-3.5 m is ~1% of range "
                  f"(~25-35 mm), so a residual near that is the sensor floor, "
                  f"not fit error")
            # IMPORTANT: CAM_UP is the IMAGE up direction, not world vertical —
            # the camera here is pitched down, so a level box does NOT give a
            # tilt of 0 against it.  Only align_box_to_world.py, which knows
            # the camera->world rotation, can report a true tilt.
            up = RC[pose_valid] @ np.array([0.0, 0.0, 1.0])
            til = np.degrees(np.arccos(np.clip(-up[:, 1], -1, 1)))
            print(f"  ACCEPTED -- box z vs IMAGE up (NOT world up, camera is "
                  f"pitched): median {np.median(til):.1f} deg  "
                  f"p10 {np.percentile(til, 10):.1f} "
                  f"p90 {np.percentile(til, 90):.1f}")
            print("  (true tilt needs align_box_to_world.py --box_pose)")
        n_fail = len(pose_valid) - n_ok
        if n_fail:
            print(f"  {n_fail} frames without a pose fell back to "
                  f"'no measurement' (not an identity pose)")
            # a bare failure count is not actionable; say WHY, and when the
            # frames failed, so an occluded stretch is distinguishable from a
            # systematically broken estimator
            reasons = [fit_reasons[i] for i in range(len(pose_valid))
                       if not pose_valid[i]]
            norm = Counter(re.sub(r"[-\d.]+", "#", r) for r in reasons)
            for r, c in norm.most_common():
                print(f"    {c:4d} x  {r}")
            runs = []
            run = 0
            for ok_f in pose_valid:
                run = 0 if ok_f else run + 1
                runs.append(run)
            worst = int(np.argmax(runs))
            print(f"    longest rejected stretch: {max(runs)} frames, ending "
                  f"at frame {worst}")
            blocks = 20
            tl = "".join(
                f"[{int(pose_valid[i:i + blocks].sum()):2d}/{blocks}]"
                for i in range(0, len(pose_valid), blocks))
            print(f"    accepted per {blocks}-frame block: {tl}")

    np.savez(out, **payload)
    print(f"trajectory -> {out}")
    good = valid & ~np.isnan(BH)
    if good.any():
        lo, hi = np.percentile(BW[good], [20, 80]), np.percentile(BH[good], [20, 80])
        print(f"box bbox size (metric, 20-80pct): "
              f"W {lo[0]:.3f}..{hi[0]:.3f} m   H {lo[1]:.3f}..{hi[1]:.3f} m")
    if Path(args.meta_out).exists():
        meta = json.load(open(args.meta_out, encoding="utf-8"))
        print(f"calib in use: {meta.get('source', '?')} "
              f"K fx={meta['K'][0][0]:.1f}")


if __name__ == "__main__":
    main()

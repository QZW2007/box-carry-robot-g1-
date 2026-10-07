#!/usr/bin/env python
"""Calibrate the HSV colour range for box segmentation (once per box).

Why: box_traj_from_depth.py silently returns an empty trajectory when --hsv
does not match the box colour.  This tool measures the range from ONE frame
so nobody has to guess.

Usage (gmr conda python, from the repo root):
    python scripts/calib_hsv.py --bag video/9.17/realsense.bag
    python scripts/calib_hsv.py --bag video/xxx/realsense.bag --frame 300
    python scripts/calib_hsv.py --image some_frame.png
    python scripts/calib_hsv.py --bag ... --roi 400,150,220,260 --no-show
    python scripts/calib_hsv.py --bag ... --name orange --save

Workflow
  1. a window shows one frame of the bag (middle frame by default,
     --frame N picks another, or --image uses a still picture)
  2. drag a rectangle covering ONLY the box surface - no background, no
     hands, no floor - then press ENTER (ESC cancels)
  3. the tool prints a robust "h,s,v:h,s,v" string (circular-hue mean +
     percentile margins) and a segmentation preview of the whole frame
  4. with --save the range is merged into configs/boxes.json, which
     batch_pipeline.py reads automatically.  Use --name <colour word> and
     optionally --prefix <timestamp prefix> so the batch runner can match.

Headless mode: pass --roi x,y,w,h and --no-show to skip all windows
(the preview is written to --out-preview instead).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import cv2

CFG = os.path.join("configs", "boxes.json")


# ------------------------------------------------------------- cn-safe IO
def imread_cn(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_cn(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)
    return ok


# ------------------------------------------------------------- bag frame
def grab_frame(bag, frame_idx):
    """Return (img, frames_seen).  frame_idx >= 0 stops there, -1 = last."""
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(bag, repeat_playback=False)
    pipe.start(cfg)
    img, count = None, 0
    try:
        while True:
            fs = pipe.wait_for_frames()
            c = fs.get_color_frame()
            if not c:
                continue
            img = np.asanyarray(c.get_data())
            if 0 <= frame_idx <= count:
                break
            count += 1
    except RuntimeError:
        pass  # end of playback
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
    return img, count


def get_frame(args):
    if args.image:
        img = imread_cn(args.image)
        if img is None:
            sys.exit("cannot read image: %s" % args.image)
        return img
    if not os.path.exists(args.bag):
        sys.exit("bag not found: %s" % args.bag)
    print("[1/4] reading %s ..." % args.bag)
    if args.frame >= 0:
        img, _ = grab_frame(args.bag, args.frame)
    else:
        _, total = grab_frame(args.bag, -1)          # count pass
        if total == 0:
            sys.exit("no color frames found in the bag")
        img, _ = grab_frame(args.bag, total // 2)    # middle frame
    if img is None:
        sys.exit("could not extract a frame (bad bag?)")
    return img


# ------------------------------------------------------------- hsv stats
def hsv_range_from_roi(roi_bgr):
    """Median-anchored margins -> lo/hi triples (cv2 units).

    H: circular mean +- max(10, 3 sigma).  S/V: [median - 80, 255] - the
    medians are robust against dark labels / shadows inside the ROI, and
    the wide fixed margin admits the shaded faces of the box while the
    hue still keeps white walls / navy clothes out.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv.reshape(-1, 3).astype(np.float64)
    h, s, v = pix[:, 0], pix[:, 1], pix[:, 2]
    ang = np.deg2rad(h * 2.0)                       # cv2 H*2 = degrees
    sin_m, cos_m = np.sin(ang).mean(), np.cos(ang).mean()
    R = float(np.hypot(sin_m, cos_m))
    h_mean_deg = float(np.rad2deg(np.arctan2(sin_m, cos_m))) % 360.0
    h_std_deg = 0.0 if R > 0.999 else float(np.sqrt(-2.0 * np.log(max(R, 1e-6))))
    c = h_mean_deg / 2.0                            # back to cv2 units
    d = max(3.0 * h_std_deg / 2.0, 10.0)
    lo_h, hi_h = c - d, c + d
    lo_s = int(max(0, np.median(s) - 80))
    lo_v = int(max(0, np.median(v) - 80))
    hi_s, hi_v = 255, 255
    return (int(round(lo_h)), lo_s, lo_v), (int(round(hi_h)), hi_s, hi_v), \
        c, h_std_deg


def fmt_range(lo, hi):
    return "%d,%d,%d:%d,%d,%d" % (int(round(lo[0])), lo[1], lo[2],
                                  int(round(hi[0])), hi[1], hi[2])


def range_strings(lo, hi):
    """One or two strings if the hue arc wraps around 0/179."""
    h1, h2 = int(round(lo[0])), int(round(hi[0]))
    if 0 <= h1 <= h2 <= 179:
        return [fmt_range(lo, hi)]
    if h1 < 0:      # wraps the bottom: [180+h1, 179] + [0, h2]
        return ["%d,%d,%d:179,%d,%d" % (max(0, 180 + h1), lo[1], lo[2],
                                        hi[1], hi[2]),
                "0,%d,%d:%d,%d,%d" % (lo[1], lo[2], min(179, h2),
                                      hi[1], hi[2])]
    # wraps the top: [h1, 179] + [0, h2 - 180]
    return ["%d,%d,%d:179,%d,%d" % (max(0, h1), lo[1], lo[2], hi[1], hi[2]),
            "0,%d,%d:%d,%d,%d" % (lo[1], lo[2], max(0, h2 - 180),
                                  hi[1], hi[2])]


# ------------------------------------------------------------- preview
def make_preview(frame, lo, hi):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    cover = mask.mean() / 255.0 * 100.0
    dim = (frame.astype(np.float32) * 0.35).astype(np.uint8)
    overlay = dim.copy()
    overlay[mask > 0] = frame[mask > 0]
    h = min(frame.shape[0], 480)
    sc = h / frame.shape[0]
    w = int(frame.shape[1] * sc)
    small = lambda im: cv2.resize(im, (w, h))
    cv2.putText(small(frame), "original", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(small(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), "mask", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(small(overlay), "selected by range", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    panel = np.hstack([small(frame),
                       small(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
                       small(overlay)])
    return panel, cover


# ------------------------------------------------------------- config
def save_cfg(name, prefix, hsv_str, bag):
    cfg = {}
    if os.path.exists(CFG):
        try:
            with open(CFG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            cfg = {}
    spec = {"hsv": hsv_str}
    if prefix:
        spec["prefix"] = prefix
    spec["source"] = os.path.basename(bag or "image")
    cfg[name] = spec
    os.makedirs(os.path.dirname(CFG), exist_ok=True)
    with open(CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("saved '%s' -> %s" % (name, CFG))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--frame", type=int, default=-1,
                    help="frame index (default: middle of the bag)")
    ap.add_argument("--roi", default=None,
                    help="headless ROI x,y,w,h (skips the drag window)")
    ap.add_argument("--no-show", action="store_true",
                    help="no windows; write the preview to --out-preview")
    ap.add_argument("--out-preview", default="configs/calib_preview.png")
    ap.add_argument("--save", action="store_true",
                    help="merge the measured range into configs/boxes.json")
    ap.add_argument("--name", default=None, help="entry name for --save")
    ap.add_argument("--prefix", default=None,
                    help="clip prefix for the batch runner to match exactly")
    args = ap.parse_args()

    if not args.bag and not args.image:
        sys.exit("need --bag or --image")
    if args.save and not args.name:
        sys.exit("--save needs --name (a colour word or the clip prefix)")

    frame = get_frame(args)
    if args.no_show:
        frame_png = os.path.join(os.path.dirname(args.out_preview) or ".",
                                 "_calib_frame.png")
        os.makedirs(os.path.dirname(frame_png), exist_ok=True)
        imwrite_cn(frame_png, frame)

    # ---- ROI
    if args.roi:
        x, y, w, h = (int(t) for t in args.roi.split(","))
    elif args.no_show:
        sys.exit("--no-show needs --roi x,y,w,h")
    else:
        print("[2/4] drag a rectangle over ONLY the box, then press ENTER")
        x, y, w, h = cv2.selectROI("select the BOX only, ENTER to confirm",
                                   frame, showCrosshair=True)
        cv2.destroyAllWindows()
        if w == 0 or h == 0:
            sys.exit("cancelled")
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        sys.exit("empty ROI")

    # ---- stats
    lo, hi, h_c, h_sd = hsv_range_from_roi(roi)
    print("[3/4] measured HSV (cv2 units: H 0-179, S/V 0-255)")
    print("      hue centre %.1f  circular std %.1f deg" % (h_c, h_sd))
    print("      initial --hsv %s" % fmt_range(lo, hi))
    strs = range_strings(lo, hi)
    for i, s in enumerate(strs):
        print("      use --hsv %s%s" % (s, "   (arc part %d)" % (i + 1)
                                        if len(strs) > 1 else ""))
    if lo[1] < 60:
        print("      WARNING: S floor is low (%d) - white walls may match; "
              "if the preview shows background pixels raise it to >=60"
              % lo[1])

    # ---- preview
    panel, cover = make_preview(frame, lo, hi)
    print("      coverage of the full frame: %.1f%%" % cover)
    if cover > 40:
        print("      WARNING: >40%% of the frame matched - the range is too "
              "wide, redraw a tighter ROI or raise the S/V floors")
    if cover < 0.2:
        print("      WARNING: almost nothing matched - the range is too tight")
    if args.no_show:
        os.makedirs(os.path.dirname(args.out_preview) or ".", exist_ok=True)
        imwrite_cn(args.out_preview, panel)
        print("      preview written to %s" % args.out_preview)
    else:
        print("[4/4] preview: left=original  middle=mask  right=kept pixels")
        cv2.imshow("hsv calibration (any key to close)", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if len(strs) > 1:
        print("      NOTE: the hue arc wraps 0/179; --save stores part 1 "
              "only - pick the bigger part by hand if needed")

    if args.save:
        save_cfg(args.name, args.prefix, strs[0], args.bag)
        print("batch_pipeline.py / load_hsv_for will pick this up "
              "automatically (name substring or exact prefix match)")
    print("\nready-to-paste fit command:")
    print("  python scripts/box_traj_from_depth.py --bag %s --fit-cuboid "
          "--hsv %s --roi-pad 280 --min-area 400 "
          "--out data/<prefix>_6dof.npz --preview video/<prefix>_6dof.mp4"
          % (args.bag or args.image, strs[0]))


if __name__ == "__main__":
    main()

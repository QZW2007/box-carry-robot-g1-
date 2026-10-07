#!/usr/bin/env python3
"""Capture chessboard stills for camera calibration, straight off the D455.

Why not the RealSense Viewer's snapshot button:

  * Its save location is not configurable - PNGs land in the user's home folder,
    and every shot also drops a .raw and a _metadata.csv next to it;
  * whatever post-processing you left enabled in the Viewer (align, colorizer,
    any filter) is baked into the snapshot.  An rs-align'd colour frame is
    RECTIFIED, which is a DIFFERENT camera model.  Calibrating on those frames
    yields intrinsics that silently do not match your recording.

This script opens the colour stream ONLY, applies nothing, writes raw frames
into a folder you choose, and tells you the exact resolution at the end so you
can confirm it equals the resolution you will record the carry video at.

Usage:
    python scripts/capture_calib_shots.py --out calib_shots --width 1280 --height 720
    python scripts/capture_calib_shots.py --out calib_shots --width 640 --height 480 --no-preview

Keys (preview window):
    space   save the current frame
    q / esc quit

Without a preview (headless / over SSH) press Enter in the terminal to save,
and type q then Enter to quit.  You will be framing blind - use it only if you
have no display, and double-check each shot afterwards.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(f"Missing Python dependency: {exc}") from exc


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save raw colour frames for chessboard calibration.")
    parser.add_argument("--out", type=Path, default=Path("calib_shots"),
                        help="folder for the PNGs (created if missing)")
    parser.add_argument("--serial", help="camera serial, if several are attached")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30, choices=(6, 15, 30, 60, 90))
    parser.add_argument("--no-preview", action="store_true",
                        help="no OpenCV window; drive capture from the terminal")
    return parser.parse_args()


def stdin_listener(q: "queue.Queue[str]") -> None:
    """Feed terminal lines into a queue so the capture loop never blocks."""
    for line in sys.stdin:
        q.put(line.strip().lower())


def main() -> int:
    args = arguments()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("shot_*.png"))
    if existing:
        print(f"NOTE: {out} already holds {len(existing)} shot_*.png; "
              f"new shots continue from shot_{len(existing) + 1:03d}.png")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    # Colour only.  No depth, therefore no align, therefore no rectification.
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)

    q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=stdin_listener, args=(q,), daemon=True).start()

    saved = 0
    gui = not args.no_preview
    window = "D455 colour - SPACE saves, q quits"
    started = False
    w = h = None
    try:
        profile = pipeline.start(config)
        started = True
        vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        w, h = vsp.width(), vsp.height()
        print(f"colour stream: {w}x{h} @ {vsp.fps()} fps")
        if (w, h) != (args.width, args.height):
            print(f"WARNING: asked for {args.width}x{args.height} but the device "
                  f"gave {w}x{h}. The calibration must match your VIDEO, so "
                  f"check this before shooting.")
        print(f"saving to: {out}")
        print("SPACE saves a shot, q quits.\n"
              if gui else "Press Enter to save, q then Enter to quit.\n")

        while True:
            frames = pipeline.wait_for_frames()
            frame = frames.get_color_frame()
            if not frame:
                continue
            fmt = frame.profile.format()
            img = np.asanyarray(frame.get_data())
            if fmt == rs.format.rgb8:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif fmt is not rs.format.bgr8:
                raise SystemExit(f"unhandled colour format {fmt}")

            do_save = do_quit = False
            if gui:
                try:
                    cv2.imshow(window, img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        do_save = True
                    elif key in (ord("q"), 27):
                        do_quit = True
                except cv2.error:
                    print("OpenCV has no GUI backend - switching to terminal "
                          "control. Press Enter to save, q then Enter to quit.")
                    gui = False
            else:
                while not q.empty():
                    cmd = q.get()
                    if cmd in ("", "s"):
                        do_save = True
                    elif cmd == "q":
                        do_quit = True

            if do_save:
                saved += 1
                path = out / f"shot_{saved:03d}.png"
                if not cv2.imwrite(str(path), img):
                    raise SystemExit(f"failed to write {path}")
                print(f"[{saved:3d}] {path.name}  ({w}x{h})")
            if do_quit:
                break
            if not gui:
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        if started:
            pipeline.stop()
        if gui:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print(f"\n{saved} shot(s) saved to {out}")
    if saved:
        print(f"resolution written: {w}x{h}")
        print("calibrate_camera.py refuses mixed resolutions, so every shot here "
              "must be the same size - and the SAME size as your carry video.")
        print("\nnext, on the machine that has the project:")
        print(f"  python scripts/calibrate_camera.py --img_dir {out.name} "
              f"--rows 9 --cols 6 --square_mm 25 --out data/camera_calib.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

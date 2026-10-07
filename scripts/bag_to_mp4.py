"""Convert a RealSense .bag recording to a plain video file for solve_box_pose.py.

Why this is needed: the RealSense Viewer's Recorder writes .bag files, and
solve_box_pose.py reads its video with cv2.VideoCapture, which cannot open a
.bag.  This pulls the colour frames straight out of the recording and writes
them out unchanged.

Critically, it does NOT align, rectify or post-process anything.  Aligning the
colour stream (rs-align) RECTIFIES it, which changes the camera model and would
silently invalidate data/camera_calib.json.  Raw frames in, raw frames out.

Usage:
    pip install pyrealsense2
    python scripts/bag_to_mp4.py --bag video/take.bag --out video/take.mp4
    python scripts/bag_to_mp4.py --bag video/take.bag --out video/take.avi --lossless
"""

import argparse
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lossless", action="store_true",
                        help="use FFV1 in .avi - no compression artifacts, big file")
    parser.add_argument("--fps", type=float, default=None,
                        help="override the frame rate (default: as recorded)")
    args = parser.parse_args()

    import pyrealsense2 as rs

    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, args.bag, repeat_playback=False)
    cfg.enable_stream(rs.stream.color)          # colour only, nothing else
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    playback = prof.get_device().as_playback()
    playback.set_real_time(False)               # decode as fast as possible

    vsp = prof.get_stream(rs.stream.color).as_video_stream_profile()
    fps = args.fps or float(vsp.fps())
    print(f"source: {args.bag}")
    print(f"colour stream: {vsp.width()}x{vsp.height()} @ {vsp.fps()} fps "
          f"({vsp.format()})")

    writer = None
    n = 0
    while True:
        try:
            frames = pipe.wait_for_frames()
        except RuntimeError:
            break                              # end of file
        f = frames.get_color_frame()
        if not f:
            continue
        img = np.asanyarray(f.get_data())
        fmt = f.profile.format()
        if fmt == rs.format.rgb8:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif fmt == rs.format.yuyv:
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
        elif fmt != rs.format.bgr8:
            raise SystemExit(f"unhandled colour format {fmt}")

        if writer is None:
            h, w = img.shape[:2]
            if args.lossless:
                fourcc = cv2.VideoWriter_fourcc(*"FFV1")
                out = os.path.splitext(args.out)[0] + ".avi"
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = args.out
            writer = cv2.VideoWriter(out, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise SystemExit(f"could not open writer for {out}")
            print(f"writing {w}x{h} @ {fps} fps -> {out}"
                  f"{' (lossless FFV1)' if args.lossless else ''}")
        writer.write(img)
        n += 1

    pipe.stop()
    if writer is not None:
        writer.release()
    print(f"\n{n} frames written ({n / fps:.1f} s)" if n else "\nno frames!")
    if n:
        print(f"solve with: python scripts/solve_box_pose.py --video {out} ...")
        print("CHECK: the written resolution must equal width_height in "
              "data/camera_calib.json, or the intrinsics are not valid.")


if __name__ == "__main__":
    main()

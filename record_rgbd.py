#!/usr/bin/env python3
"""Record an Intel RealSense RGB-D stream.

The recording is always saved as a lossless RealSense ``.db3`` file.  With
``--save-frames`` it also writes RGB and colorized-depth MP4s plus aligned
16-bit depth PNGs, which are convenient for CV datasets.  The PNG values are raw depth units;
multiply them by ``depth_scale`` in metadata.json to obtain metres.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        f"Missing Python dependency: {exc}. Create the project environment "
        "and install requirements.txt first."
    ) from exc


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record RGB, depth, and RealSense metadata.")
    default_output = Path(__file__).resolve().parent / "recordings"
    parser.add_argument("--output", type=Path, default=default_output, help=f"Directory for this run (default: {default_output}).")
    parser.add_argument("--serial", help="Optional camera serial number when multiple cameras are attached.")
    parser.add_argument("--width", type=int, default=640, help="RGB/depth width (default: 640).")
    parser.add_argument("--height", type=int, default=480, help="RGB/depth height (default: 480).")
    parser.add_argument("--fps", type=int, default=30, choices=(6, 15, 30, 60, 90), help="Frame rate.")
    parser.add_argument("--duration", type=float, default=0, help="Seconds to record; 0 means run until q/Ctrl-C.")
    parser.add_argument("--save-frames", action="store_true", help="Also save RGB/depth MP4s and aligned depth PNG frames.")
    parser.add_argument("--preview", action="store_true", help="Show RGB and colorized-depth preview; press q to stop.")
    parser.add_argument("--exposure", type=float, help="Locked exposure, in the device's own units. Required unless --keep-auto-exposure.")
    parser.add_argument("--gain", type=float, help="Locked gain.")
    parser.add_argument("--white-balance", type=float, dest="white_balance", help="Locked white balance in Kelvin.")
    parser.add_argument("--power-line", type=int, dest="power_line", choices=(0, 1, 2, 3), help="Anti-flicker: 1 = 50 Hz, 2 = 60 Hz, 3 = auto, 0 = off.")
    parser.add_argument("--keep-auto-exposure", action="store_true", help="Do NOT lock exposure/gain. Recordings will flicker mid-take.")
    return parser.parse_args()


def apply_color_settings(profile: rs.pipeline_profile, args: argparse.Namespace) -> dict:
    """Lock the colour sensor to fixed exposure/gain/white-balance.

    RealSense image controls are runtime state, not device state: a fresh
    pipeline start comes up with auto-exposure and auto-white-balance ENABLED,
    regardless of what was set in realsense-viewer beforehand.  Anything tuned
    there does not reach this script, so it has to be set explicitly here.
    """
    sensor = profile.get_device().first_color_sensor()
    if not args.keep_auto_exposure:
        sensor.set_option(rs.option.enable_auto_exposure, 0)   # must precede exposure
        if args.exposure is None:
            print("WARNING: auto exposure disabled but --exposure not given; "
                  "the image may be far too dark or too bright.")
        else:
            sensor.set_option(rs.option.exposure, args.exposure)
        if args.gain is not None:
            sensor.set_option(rs.option.gain, args.gain)
    if args.white_balance is not None:
        sensor.set_option(rs.option.enable_auto_white_balance, 0)
        sensor.set_option(rs.option.white_balance, args.white_balance)
    if args.power_line is not None:
        sensor.set_option(rs.option.power_line_frequency, args.power_line)

    applied = {}
    for name, opt in (("enable_auto_exposure", rs.option.enable_auto_exposure),
                      ("exposure", rs.option.exposure),
                      ("gain", rs.option.gain),
                      ("enable_auto_white_balance", rs.option.enable_auto_white_balance),
                      ("white_balance", rs.option.white_balance),
                      ("power_line_frequency", rs.option.power_line_frequency)):
        if sensor.supports(opt):
            applied[name] = sensor.get_option(opt)
    return applied


def camera_info(device: rs.device, profile: rs.pipeline_profile, depth_scale: float, args: argparse.Namespace, settings: dict) -> dict:
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    requested_info = (
        ("device_name", rs.camera_info.name),
        ("serial_number", rs.camera_info.serial_number),
        ("firmware_version", rs.camera_info.firmware_version),
    )
    info = {name: device.get_info(key) for name, key in requested_info if device.supports(key)}
    info.update(
        recorded_at=datetime.now().astimezone().isoformat(),
        width=args.width,
        height=args.height,
        fps=args.fps,
        depth_scale_metres_per_unit=depth_scale,
        color_intrinsics=str(color_profile.get_intrinsics()),
        depth_intrinsics=str(depth_profile.get_intrinsics()),
        color_settings=settings,
        depth_is_aligned_to_color=True,
        depth_storage="16-bit PNG; metres = pixel_value * depth_scale_metres_per_unit",
    )
    return info


def main() -> int:
    args = arguments()
    run_dir = args.output.expanduser().resolve() / datetime.now().strftime("rgbd_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    recording_path = run_dir / "realsense.db3"

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    # librealsense 2.58+ uses the MCAP/rosbag2-style .db3 recording format.
    config.enable_record_to_file(str(recording_path))

    color_writer = None
    depth_writer = None
    frame_index = 0
    start = time.monotonic()
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        device = profile.get_device()
        depth_scale = device.first_depth_sensor().get_depth_scale()
        settings = apply_color_settings(profile, args)
        if not args.keep_auto_exposure:
            print("colour settings locked:", settings)
        (run_dir / "metadata.json").write_text(json.dumps(camera_info(device, profile, depth_scale, args, settings), indent=2) + "\n")
        align = rs.align(rs.stream.color)
        if args.save_frames:
            (run_dir / "depth_aligned_png").mkdir()
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            color_writer = cv2.VideoWriter(str(run_dir / "color.mp4"), codec, args.fps, (args.width, args.height))
            depth_writer = cv2.VideoWriter(str(run_dir / "depth_colormap.mp4"), codec, args.fps, (args.width, args.height))
            if not color_writer.isOpened() or not depth_writer.isOpened():
                raise RuntimeError("Could not open an MP4 writer; the .db3 recording is still available.")

        print(f"Recording to: {run_dir}")
        print("Press q in the preview window or Ctrl-C to stop.")
        while not args.duration or time.monotonic() - start < args.duration:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            if color_writer is not None:
                color_writer.write(color)
                depth_view = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                depth_writer.write(depth_view)
                if not cv2.imwrite(str(run_dir / "depth_aligned_png" / f"{frame_index:06d}.png"), depth):
                    raise RuntimeError("Failed to write a depth PNG.")
            if args.preview:
                depth_view = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                try:
                    cv2.imshow("D455 RGB | aligned depth", np.hstack((color, depth_view)))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("OpenCV has no GUI backend: continuing without preview. Use realsense-viewer for live preview.")
                    args.preview = False
            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        if color_writer is not None:
            color_writer.release()
        if depth_writer is not None:
            depth_writer.release()
        if started:
            pipeline.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    elapsed = time.monotonic() - start
    print(f"Saved {frame_index} aligned RGB-D frames in {elapsed:.1f}s: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

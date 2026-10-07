"""Read the Intel RealSense D455 factory RGB intrinsics -> camera_calib.json.

Why this instead of calibrate_camera.py: D400-series devices are calibrated
per unit at the factory, and the RGB intrinsics + distortion coefficients are
readable straight from the device.  No chessboard needed.

The catch it handles for you: intrinsics are defined PER (stream, resolution).
RealSense modes are CROPS of each other, not scales - 640x480 (4:3) and
1280x720 (16:9) do not even share an aspect ratio - so you must use the entry
for the exact resolution you record at, never a rescaled one.

Usage:
    pip install pyrealsense2
    python scripts/d455_intrinsics.py --width 640 --height 480 --out data/camera_calib.json
    python scripts/d455_intrinsics.py --list          # show every colour mode

Writes the same JSON schema as scripts/calibrate_camera.py, so
solve_box_pose.py consumes it unchanged.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=None,
                        help="prefer this frame rate if the mode offers several")
    parser.add_argument("--out", default="data/camera_calib.json")
    parser.add_argument("--list", action="store_true",
                        help="list all colour modes and exit")
    args = parser.parse_args()

    import pyrealsense2 as rs

    devices = list(rs.context().query_devices())
    if not devices:
        raise SystemExit("no RealSense device found - is it plugged in?")
    dev = devices[0]
    print(f"device: {dev.get_info(rs.camera_info.name)}  "
          f"s/n {dev.get_info(rs.camera_info.serial_number)}")

    # ---- collect every colour mode the device reports ----
    modes = []
    for sensor in dev.sensors:
        for prof in sensor.get_stream_profiles():
            if prof.stream_type() != rs.stream.color:
                continue
            v = prof.as_video_stream_profile()
            modes.append((v.width(), v.height(), prof.fps(), v))

    if not modes:
        raise SystemExit("device exposes no colour stream")

    if args.list:
        print("\ncolour modes available:")
        for w, h, fps, _ in sorted(set((m[0], m[1], m[2]) for m in modes)):
            print(f"  {w}x{h} @ {fps} fps")
        return

    # ---- pick the requested resolution, preferring the requested fps ----
    cands = [m for m in modes if m[0] == args.width and m[1] == args.height]
    if not cands:
        avail = sorted(set((m[0], m[1]) for m in modes))
        raise SystemExit(f"{args.width}x{args.height} is not a colour mode on "
                         f"this device. available: {avail}\n"
                         f"(run with --list to see frame rates too)")
    if args.fps is not None:
        exact = [m for m in cands if m[2] == args.fps]
        if exact:
            cands = exact
        else:
            print(f"NOTE: {args.width}x{args.height} @ {args.fps} fps not "
                  f"offered; using {max(m[2] for m in cands)} fps profile "
                  f"(intrinsics are identical across frame rates)")

    w, h, fps, prof = cands[0]
    intr = prof.get_intrinsics()

    K = [[intr.fx, 0.0, intr.ppx],
         [0.0, intr.fy, intr.ppy],
         [0.0, 0.0, 1.0]]
    # RealSense coeffs come in the same Brown-Conrady order OpenCV expects:
    # [k1, k2, p1, p2, k3]
    dist = list(intr.coeffs)
    model = str(intr.model).split(".")[-1]

    print(f"\ncolour stream {w}x{h} @ {fps} fps")
    print(f"distortion model: {model}")
    print(f"fx={intr.fx:.4f}  fy={intr.fy:.4f}  "
          f"cx={intr.ppx:.4f}  cy={intr.ppy:.4f}")
    print(f"coeffs = {[round(float(c), 6) for c in dist]}")

    if "brown_conrady" not in model and "none" not in model:
        print(f"\nWARNING: model is '{model}', not Brown-Conrady. OpenCV's "
              f"solvePnP in solve_box_pose.py assumes Brown-Conrady; verify "
              f"before trusting the distortion term.")
    if max(abs(float(c)) for c in dist) > 0.5:
        print("\nWARNING: a distortion coefficient looks unusually large - "
              "read the model name before using these numbers.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "K": K,
            "dist": [float(c) for c in dist],
            "width_height": [w, h],
            "rms_px": None,
            "source": {
                "device": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "stream": "color",
                "fps": fps,
                "distortion_model": model,
                "note": "factory calibration read from the device - no "
                        "chessboard. NOT valid if the recording resolution "
                        "differs from width_height.",
            },
        }, fh, indent=2)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()

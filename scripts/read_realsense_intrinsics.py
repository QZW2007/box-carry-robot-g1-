#!/usr/bin/env python
"""Read factory intrinsics from a RealSense recording (.bag/.db3) or a live device
and write them in the same JSON format the box-pose pipeline expects.

Priority of sources (first available wins):
  1. --bag <file>   : playback a recording, read intrinsics at record time (best)
  2. --live         : read from the connected device
  3. no args        : every .bag/.db3 under video/, first that parses

Output: data/camera_calib.json with keys K / dist / width_height (identical to
scripts/calibrate_camera.py output) plus depth-side extras.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import pyrealsense2 as rs

DIST_NAMES = {
    rs.distortion.none: "none",
    rs.distortion.modified_brown_conrady: "brown",
    rs.distortion.brown_conrady: "brown",
    rs.distortion.inverse_brown_conrady: "fisheye",
}


def intr_to_dict(intr):
    coeffs = [float(c) for c in intr.coeffs]
    while len(coeffs) < 5:
        coeffs.append(0.0)
    return {
        "K": [[float(intr.fx), 0.0, float(intr.ppx)],
              [0.0, float(intr.fy), float(intr.ppy)],
              [0.0, 0.0, 1.0]],
        "dist": [coeffs],
        "dist_model": DIST_NAMES.get(intr.model, str(intr.model)),
        "width_height": [int(intr.width), int(intr.height)],
    }


def rot_quat(R):
    """3x3 rotation -> xyzw quaternion (mujoco convention)."""
    q_w = 0.5 * np.sqrt(max(1.0 + float(np.trace(R)), 0.0))
    q = np.zeros(4)
    if q_w < 1e-8:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        q[i] = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 0.0)) / 2
        q[j] = (R[j, i] + R[i, j]) / (4 * q[i] + 1e-12)
        q[k] = (R[k, i] + R[i, k]) / (4 * q[i] + 1e-12)
        return q
    q[3] = q_w
    q[0] = (R[2, 1] - R[1, 2]) / (4 * q_w)
    q[1] = (R[0, 2] - R[2, 0]) / (4 * q_w)
    q[2] = (R[1, 0] - R[0, 1]) / (4 * q_w)
    return q


def get_frame(frames, stream):
    try:
        return frames.first(stream)
    except RuntimeError:
        return None


def read_frames(bag_path, count_frames, dump_n, dump_dir):
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(bag_path), repeat_playback=False)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback()
    try:
        playback.set_real_time(False)
    except RuntimeError:
        pass
    dur = playback.get_duration().total_seconds()
    print(f"playback: {Path(bag_path).name}  duration {dur:.1f}s")

    color_intr = depth_intr = None
    color_fps = depth_fps = None
    extr = None
    n_color = n_depth = 0
    dumped = 0
    import cv2
    if dump_dir is not None and dump_n != 0:
        dump_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            frames = pipe.wait_for_frames(timeout_ms=8000)
            cf = get_frame(frames, rs.stream.color)
            df = get_frame(frames, rs.stream.depth)
            if color_intr is None and cf is not None:
                vp = cf.get_profile().as_video_stream_profile()
                color_intr, color_fps = vp.get_intrinsics(), vp.fps()
            if depth_intr is None and df is not None:
                vp = df.get_profile().as_video_stream_profile()
                depth_intr, depth_fps = vp.get_intrinsics(), vp.fps()
                if color_intr is not None:
                    extr = vp.get_extrinsics_to(
                        cf.get_profile().as_video_stream_profile())
            n_color += cf is not None
            n_depth += df is not None
            if cf is not None and dump_dir is not None and \
                    (dump_n < 0 or dumped < dump_n):
                img = np.asanyarray(cf.get_data())
                cv2.imwrite(str(dump_dir / f"frame_{n_color - 1:04d}.png"), img)
                dumped += 1
            if n_color >= 5000:
                break
    except RuntimeError:
        pass  # end of recording
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
    return color_intr, depth_intr, color_fps, depth_fps, extr, n_color, n_depth


def read_from_live():
    pipe = rs.pipeline()
    cfg = rs.config()
    profile = pipe.start(cfg)
    try:
        cf = profile.get_stream(rs.stream.color).as_video_stream_profile()
        df = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        return (cf.get_intrinsics(), df.get_intrinsics(), cf.fps(), df.fps(),
                df.get_extrinsics_to(cf), 0, 0)
    finally:
        pipe.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=None, help="path to .bag / .db3 recording")
    ap.add_argument("--live", action="store_true", help="read from connected device")
    ap.add_argument("--out", default="data/camera_calib.json")
    ap.add_argument("--probe-frames", type=int, default=0,
                    help="dump first N color frames as PNG for inspection")
    ap.add_argument("--frames-out", default="video/realsense_frames")
    args = ap.parse_args()

    color_intr = depth_intr = None
    result = None
    if args.live:
        try:
            print("reading live device...")
            result = read_from_live() + ("live",)
        except RuntimeError as e:
            print(f"live read failed: {e}")
    elif args.bag:
        result = read_frames(args.bag, True, args.probe_frames,
                             Path(args.frames_out) if args.probe_frames else None
                             ) + (args.bag,)
    else:
        candidates = sorted(Path("video").glob("*.bag")) + \
                     sorted(Path("video").glob("*.db3"))
        for c in candidates:
            print(f"reading {c} ...")
            try:
                result = read_frames(c, True, args.probe_frames,
                                     Path(args.frames_out) if args.probe_frames
                                     else None) + (str(c),)
                break
            except RuntimeError as e:
                print(f"  failed: {e}")

    if result is None:
        print("no source available (no device, no readable bag)")
        sys.exit(1)

    color_intr, depth_intr, color_fps, depth_fps, extr, n_color, n_depth, src = result
    if color_intr is None:
        print("no color stream in source")
        sys.exit(1)

    out = intr_to_dict(color_intr)
    out["source"] = src
    out["rms_px"] = 0.0
    out["n_images"] = 0
    out["note"] = "factory intrinsics from RealSense (record-time)"
    out["fps"] = color_fps
    out["n_frames_color"] = n_color
    if depth_intr is not None:
        out["depth"] = intr_to_dict(depth_intr)
        out["depth"]["fps"] = depth_fps
        out["n_frames_depth"] = n_depth
        if extr is not None:
            R = np.array(extr.rotation, dtype=float).reshape(3, 3)
            out["depth_to_color"] = {
                "rotation_xyzw": rot_quat(R).tolist(),
                "translation_m": [float(t) for t in extr.translation],
                "baseline_m": float(np.linalg.norm(extr.translation)),
            }
    print(f"\ncolor {color_intr.width}x{color_intr.height}@{color_fps}  "
          f"fx={color_intr.fx:.2f} fy={color_intr.fy:.2f} "
          f"cx={color_intr.ppx:.2f} cy={color_intr.ppy:.2f}  "
          f"model={out['dist_model']} coeffs={out['dist'][0]}")
    print(f"color frames: {n_color}")
    if depth_intr is not None:
        print(f"depth  {depth_intr.width}x{depth_intr.height}@{depth_fps}  "
              f"fx={depth_intr.fx:.2f}  frames: {n_depth}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()

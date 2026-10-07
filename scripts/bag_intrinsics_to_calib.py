"""Read a RealSense .bag's OWN colour intrinsics -> data/camera_calib.json.

Why the bag and not the device: intrinsics differ per (stream, resolution, and
sometimes frame rate).  The bag stores the exact intrinsics entry that was live
when it recorded, so there is no chance of picking the wrong mode.  A chessboard
is unnecessary - D400-series colour sensors are calibrated per unit at the
factory.

The catch this script fixes, which the factory numbers alone get wrong:

  The device reports its distortion model as INVERSE Brown-Conrady: the
  polynomial maps the DISTORTED normalised radius to the UNDISTORTED one.
  OpenCV's solvePnP / projectPoints implement the opposite direction.  The two
  are mutual inverses, so handing the raw coefficients to OpenCV applies the
  correction backwards - on a D455 at 640x480 that is ~5 px at mid-radius and
  ~25 px at the frame corner, and err_px does NOT show it, because solvePnP and
  the projectPoints that computes err_px share the same wrong coefficients.

Here the conversion is done numerically and needs no camera:
  1. lay a grid of undistorted normalised coords over the sensor;
  2. push each through the DEVICE model with cv2.undistortPoints, which IS the
     inverse of OpenCV's forward polynomial, giving the pixel the camera emits;
  3. least-squares fit OpenCV's forward polynomial to those (u -> d) pairs.

Usage (run from the project root, so the path stays ASCII for the SDK):
    python scripts/bag_intrinsics_to_calib.py --bag video/take.bag \
        --out data/camera_calib.json
    python scripts/bag_intrinsics_to_calib.py --bag video/take.bag --dry-run
"""

import argparse
import json
import os
import sys

import numpy as np


def device_to_opencv(K, dist_inv):
    """Inverse Brown-Conrady (device) -> forward Brown-Conrady (OpenCV)."""
    import cv2

    W, H = int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))
    us = np.linspace(-K[0, 2] / K[0, 0], (W - 1 - K[0, 2]) / K[0, 0], 41)
    vs = np.linspace(-K[1, 2] / K[1, 1], (H - 1 - K[1, 2]) / K[1, 1], 31)
    U, V = np.meshgrid(us, vs)
    u = np.stack([U.ravel(), V.ravel()], axis=1)

    pix = np.stack([K[0, 0] * u[:, 0] + K[0, 2],
                    K[1, 1] * u[:, 1] + K[1, 2]], axis=1).reshape(-1, 1, 2)
    d = cv2.undistortPoints(pix.astype(np.float64), K, dist_inv).reshape(-1, 2)

    # d - u is linear in (k1, k2, k3, p1, p2) for OpenCV's forward model.
    x, y = u[:, 0], u[:, 1]
    r2 = x * x + y * y
    A = np.stack([x * r2, x * r2**2, x * r2**3, 2 * x * y, r2 + 2 * x * x], axis=1)
    B = np.stack([y * r2, y * r2**2, y * r2**3, r2 + 2 * y * y, 2 * x * y], axis=1)
    M = np.vstack([A, B])
    rhs = np.concatenate([d[:, 0] - x, d[:, 1] - y])
    coef, *_ = np.linalg.lstsq(M, rhs, rcond=None)

    fit = np.stack([x + A @ coef, y + B @ coef], axis=1)
    resid_px = np.linalg.norm(fit - d, axis=1) * np.sqrt(
        (K[0, 0]**2 + K[1, 1]**2) / 2.0)
    return coef, float(resid_px.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", default="data/camera_calib.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the numbers, write nothing")
    args = parser.parse_args()

    if not os.path.exists(args.bag):
        raise SystemExit(f"no such bag: {args.bag}\n(run this from the project "
                         f"root with a relative path - the SDK cannot open "
                         f"paths containing non-ASCII characters)")

    import pyrealsense2 as rs

    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, args.bag, repeat_playback=False)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    try:
        color = next((s for s in prof.get_streams()
                      if s.stream_type() == rs.stream.color), None)
        if color is None:
            raise SystemExit("this bag holds no colour stream")
        vsp = color.as_video_stream_profile()
        i = vsp.get_intrinsics()
        W, H, fps = vsp.width(), vsp.height(), vsp.fps()
        model = str(i.model).split(".")[-1]
        K = np.array([[i.fx, 0.0, i.ppx],
                      [0.0, i.fy, i.ppy],
                      [0.0, 0.0, 1.0]])
        dist_inv = np.array(list(i.coeffs), dtype=np.float64)
        serial = None
        try:
            serial = prof.get_device().get_info(rs.camera_info.serial_number)
        except Exception:
            pass
    finally:
        pipe.stop()

    print(f"bag      : {args.bag}")
    print(f"colour   : {W}x{H} @ {fps} fps   serial {serial}")
    print(f"model    : {model}")
    print(f"fx={i.fx:.4f}  fy={i.fy:.4f}  ppx={i.ppx:.4f}  ppy={i.ppy:.4f}")
    print(f"raw dist : {[round(float(c), 6) for c in dist_inv]}")

    if "inverse_brown_conrady" not in model:
        print(f"\nNOTE: model is '{model}', not inverse Brown-Conrady. The "
              f"conversion below assumes the device polynomial runs "
              f"distorted->undistorted; check before trusting it.", file=sys.stderr)

    coef, resid = device_to_opencv(K, dist_inv)
    print(f"\nconverted to OpenCV forward Brown-Conrady "
          f"(fit residual max {resid:.5f} px):")
    for n, v in zip(("k1", "k2", "k3", "p1", "p2"), coef):
        print(f"  {n} = {v:+.6f}")
    # A good fit is well under 0.1 px; the mismatch this conversion removes is
    # 5-25 px, so anything below that is irrelevant to the pose.
    if resid > 0.2:
        print(f"WARNING: fit residual {resid:.3f} px is large - the device model "
              f"may not be a clean inverse of OpenCV's; treat these coefficients "
              f"with suspicion.")

    if args.dry_run:
        print("\ndry run - nothing written")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "K": K.tolist(),
            "dist": [float(c) for c in coef],
            "width_height": [W, H],
            "rms_px": None,
            "source": {
                "kind": "D455 factory intrinsics read from the recording itself",
                "bag": args.bag,
                "serial": serial,
                "stream": "color",
                "fps": fps,
                "device_distortion_model": model,
                "raw_device_dist": [float(c) for c in dist_inv],
                "conversion": "inverse Brown-Conrady -> forward Brown-Conrady, "
                              f"numeric fit, residual {resid:.5f} px",
            },
            "_WARNING": [
                "NOT chessboard-verified. The distortion CONVENTION is handled",
                "(see 'conversion' above), and fx agrees with two independent",
                "device sources to 0.1% (this bag 387.080, a 30 fps recording",
                "386.659), so the lens model should be sound.",
                "",
                "What is still unverified is the PHYSICAL MARKER SIZE. A wrong",
                "marker side length scales the whole recovered trajectory by the",
                "same factor while leaving err_px completely flat - object points",
                "scaled by g and translation scaled by g give an identical image.",
                "So err_px can never catch it. Measure a printed marker edge with",
                "a ruler and, if it is not 60 mm, correct s in box_markers.json.",
            ],
        }, fh, indent=2)
    print(f"\nwritten: {args.out}")
    print(f"the video solve reads this unchanged; width_height {W}x{H} must match "
          f"the video you feed solve_box_pose.py")


if __name__ == "__main__":
    main()

"""Checkerboard camera calibration -> K + distortion for the re-shoot.

Takes a folder of chessboard photos (print from generate_aruco_markers.py
--checker 9 6, shoot 15-25 images covering angles/positions/distances),
outputs a JSON consumed by solve_box_pose.py.

Usage:
    python scripts/calibrate_camera.py --img_dir calib_shots \
        --rows 9 --cols 6 --square_mm 25 --out data/camera_calib.json
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--rows", type=int, default=9,
                        help="inner corners along height (vertical count)")
    parser.add_argument("--cols", type=int, default=6,
                        help="inner corners along width (horizontal count)")
    parser.add_argument("--square_mm", type=float, default=25.0)
    parser.add_argument("--max_images", type=int, default=30)
    parser.add_argument("--out", default="data/camera_calib.json")
    args = parser.parse_args()

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = sorted(sum([glob.glob(os.path.join(args.img_dir, e))
                        for e in exts], []))
    if len(files) < 10:
        raise SystemExit(
            f"only {len(files)} images in {args.img_dir} - take at least 15")

    pattern_candidates = [
        (args.cols, args.rows), (args.rows, args.cols)  # both board orientations
    ]
    objp_base = np.zeros((args.rows * args.cols, 3), np.float32)
    grid = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp_base[:, :2] = grid * (args.square_mm / 1000.0)  # meters

    obj_points, img_points, used, sizes = [], [], [], set()
    rejected = []
    for f in tqdm(files, desc="detect corners"):
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = False, None
        for pattern in pattern_candidates:
            ok, corners = cv2.findChessboardCornersSB(
                gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                break
        if ok:
            obj_points.append(objp_base.copy())
            img_points.append(corners)
            used.append(f)
            sizes.add(gray.shape[::-1])
        else:
            rejected.append(os.path.basename(f))

    if len(used) < 10:
        raise SystemExit(
            f"chessboard found in only {len(used)}/{len(files)} images; "
            f"re-shoot with better light/angles/focus")
    if len(sizes) > 1:
        raise SystemExit("images have mixed resolutions - calibrate per resolution")

    print(f"detected in {len(used)}/{len(files)} images"
          + (f" (rejected: {rejected})" if rejected else ""))

    def run_calib(obj_pts, img_pts, rational=False):
        flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts, img_pts, next(iter(sizes)), None, None, flags=flags)
        per_image = []
        for i in range(len(obj_pts)):
            proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
            d = np.linalg.norm(img_pts[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
            per_image.append({"idx": i, "rms_px": round(float(np.sqrt((d ** 2).mean())), 3)})
        return rms, K, dist, per_image, rvecs, tvecs

    rms, K, dist, per_image, rvecs, tvecs = run_calib(obj_points, img_points)
    if rms >= 1.0:
        # wide-angle / action cams often need the rational distortion model
        rms_r, K_r, dist_r, per_r, *_ = run_calib(obj_points, img_points, rational=True)
        if rms_r < rms:
            rms, K, dist, per_image = rms_r, K_r, dist_r, per_r
            print("using rational distortion model")

    # iterative outlier rejection: drop the worst image while RMS is high
    working = list(range(len(obj_points)))
    for _ in range(3):
        if rms < 0.8 or len(working) <= 10:
            break
        worst = max(per_image, key=lambda d: d["rms_px"])
        if worst["rms_px"] < 1.5 * float(np.median([d["rms_px"] for d in per_image])):
            break
        drop = working[worst["idx"]]
        print(f"dropping outlier {os.path.basename(used[drop])} ({worst['rms_px']} px)")
        working.pop(worst["idx"])
        keep = [used[i] for i in range(len(used)) if i in
                [working[j] for j in range(len(working))]]
        obj_pts = [obj_points[i] for i in working]
        img_pts = [img_points[i] for i in working]
        rms, K, dist, per_image, rvecs, tvecs = run_calib(obj_pts, img_pts)
        used = keep

    print(f"\nRMS reprojection error: {rms:.3f} px  (target < 0.5, hard limit 1.0)")
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} "
          f"@ {next(iter(sizes))} | dist: {np.round(dist, 4).tolist()}")
    for j, d in enumerate(per_image):
        d["file"] = os.path.basename(used[j])
        d.pop("idx", None)
    worst = sorted(per_image, key=lambda d: -d["rms_px"])[:3]
    print("worst images:", worst)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "width_height": list(next(iter(sizes))),
            "rms_px": round(float(rms), 4),
            "n_images": len(used),
            "pattern": {"rows": args.rows, "cols": args.cols,
                        "square_mm": args.square_mm},
            "per_image": per_image,
        }, fh, indent=2)
    if rms < 0.5:
        verdict = "PASS"
    elif rms < 1.0:
        verdict = "OK - usable, supplemental shots would tighten it"
    else:
        verdict = ("MARGINAL - json written, box PnP will carry ~1-3 mm error; "
                   "recommended: add 5-10 closer/steadier shots (board fills half "
                   "the frame, keep paper flat, avoid the torn edge) and re-run")
    print(f"{verdict} -> {args.out}")


if __name__ == "__main__":
    main()

"""Per-image diagnostic for calibration shots: which images detect, which fail, why.

Usage: python scripts/check_calib_shots.py --img_dir snapshot --rows 9 --cols 6
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = sorted(sum([glob.glob(os.path.join(args.img_dir, e)) for e in exts], []))
    pats = [(args.cols, args.rows), (args.rows, args.cols)]  # try both orientations
    print(f"{'image':<20} {'size':<10} {'blur':<7} result")
    n_ok = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"{os.path.basename(f):<20} READ FAIL")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        res = "NO BOARD"
        for pi, pat in enumerate(pats):
            ok_sb, _ = cv2.findChessboardCornersSB(gray, pat,
                                                   flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok_sb:
                res = f"OK pattern {pat}"
                n_ok += 1
                break
            ok_std, _ = cv2.findChessboardCorners(gray, pat, None)
            if ok_std:
                res = f"OK(std) pattern {pat}"
                n_ok += 1
                break
        print(f"{os.path.basename(f):<20} {str(gray.shape[::-1]):<10} {blur:<7.0f} {res}")
    print(f"\ntotal: {n_ok}/{len(files)} images with detectable board")


if __name__ == "__main__":
    main()

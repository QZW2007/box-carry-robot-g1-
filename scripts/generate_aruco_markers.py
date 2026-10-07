"""Generate print-ready calibration materials at exact physical size.

Outputs (into --out dir, default ./print_materials):
  * ArUco marker PNGs, one per id, with white border, embedded DPI so that
    printing at 100% scale reproduces --size_mm exactly.
  * A chessboard PNG (--rows x --cols inner corners, --square_mm squares).

Print at 100% scale ("actual size", NOT "fit to page"), on matte paper if
possible (glossy reflects), then verify one square/marker with a ruler.

Usage:
    python scripts/generate_aruco_markers.py --ids 0 1 2 3 --size_mm 60
    python scripts/generate_aruco_markers.py --checker 9 6 --square_mm 25
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", default="DICT_4X4_50",
                        help="ArUco dictionary name")
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="marker ids to generate")
    parser.add_argument("--size_mm", type=float, default=60.0,
                        help="printed marker side length in mm")
    parser.add_argument("--border_mm", type=float, default=12.0,
                        help="white border around the marker (aids detection)")
    parser.add_argument("--checker", type=int, nargs=2, default=None,
                        metavar=("ROWS", "COLS"),
                        help="also generate a chessboard with this many inner "
                             "corners (e.g. 9 6) using --square_mm")
    parser.add_argument("--square_mm", type=float, default=25.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", default="print_materials")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    marker_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dict))

    s_px = mm_to_px(args.size_mm, args.dpi)
    b_px = mm_to_px(args.border_mm, args.dpi)
    canvas = np.full((s_px + 2 * b_px, s_px + 2 * b_px), 255, np.uint8)

    for mid in args.ids:
        m = cv2.aruco.generateImageMarker(marker_dict, mid, s_px)
        canvas[b_px:b_px + s_px, b_px:b_px + s_px] = m
        path = os.path.join(args.out, f"aruco_{args.dict}_id{mid}_{int(args.size_mm)}mm.png")
        Image.fromarray(canvas).save(path, dpi=(args.dpi, args.dpi))
        print(f"marker id {mid}: {path} "
              f"({args.size_mm:.0f} mm + {args.border_mm:.0f} mm border)")

    if args.checker is not None:
        rows, cols = args.checker
        # rows*cols inner corners -> (rows+1)x(cols+1) squares, plus 1-square
        # white border so corner detection is not starved at the edges.
        q = mm_to_px(args.square_mm, args.dpi)
        w, h = (cols + 3) * q, (rows + 3) * q
        board = np.full((h, w), 255, np.uint8)
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:  # origin offset makes (0,0) start white-bordered
                    board[q + r * q:q + (r + 1) * q, q + c * q:q + (c + 1) * q] = 0
        path = os.path.join(
            args.out, f"chessboard_{rows}x{cols}_{int(args.square_mm)}mm.png")
        Image.fromarray(board).save(path, dpi=(args.dpi, args.dpi))
        print(f"chessboard ({rows}x{cols} inner corners): {path} "
              f"-> print size {((cols + 3) * args.square_mm / 10):.0f} x "
              f"{((rows + 3) * args.square_mm / 10):.0f} cm")

    print("\nPrint at 100% scale / 'actual size'. Verify with a ruler:")
    print(f"  marker side should measure {args.size_mm:.0f} mm exactly")


if __name__ == "__main__":
    main()

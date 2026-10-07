# -*- coding: utf-8 -*-
"""Characterise the REAL box rotation rate in a --box_pose_world clip, so you
can size the renderer's slew limiter (`--quat_max_step_deg`) from the data.

The renderer clips the per-frame angular step of the measured box pose to
`--quat_max_step_deg` (default 3.5).  The limiter compares each frame to the
already-CLIPPED previous frame, so once it clips it keeps clipping until it
catches up: a cap below the real rate stretches the turn into a long ramp and
the box visibly LAGS the body ("箱子转的相对于机器人来说有一些延迟").  A cap
tuned on a quiet clip (where the box truly does not rotate) is therefore WRONG
on a clip where the box is genuinely turned in the hands.

This tool prints the symmetric-aware per-frame angular step of the fitted
`box_R_world` over the carry, plus median / p95 / max, and suggests a cap.

The step is measured as the nearest of the box's 8 symmetries (4 z-rotations
of 90 deg x 2 x-flips), so a 90-deg branch hop does NOT read as motion.

Usage:
    python raw_steps.py <world_6dof.npz> [--start F] [--end F] [--thresh 3]

Rule of thumb: set `--quat_max_step_deg` at or just above the printed p95
(a cap >= the real peak clips NOTHING -- confirm in the render log with
"0 frames clipped", and that no isolated spike came back, see spike_chk).
"""
import argparse
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def syms8():
    out = []
    for k in range(4):
        Rz = R.from_euler("z", 90.0 * k, degrees=True)
        out.append(Rz)
        out.append(Rz * R.from_euler("x", 180.0, degrees=True))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="the *_world_6dof.npz used as --box_pose_world")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--thresh", type=float, default=3.0,
                    help="only list frames whose step exceeds this (deg/frame)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    if "box_R_world" not in d.files:
        sys.exit("no box_R_world in " + args.npz)
    B = d["box_R_world"]
    n = B.shape[0]
    cov = d["carry"].astype(bool) if "carry" in d.files else np.ones(n, bool)

    M = np.full((n, 3, 3), np.nan)
    for i in range(n):
        if np.all(np.isfinite(B[i])):
            M[i] = B[i]

    S = syms8()
    steps = np.full(n, np.nan)
    for i in range(1, n):
        if not (np.isfinite(M[i]).all() and np.isfinite(M[i - 1]).all()):
            continue
        Mi = R.from_matrix(M[i])
        Mp = R.from_matrix(M[i - 1]).inv()
        dd = np.degrees((Mp * Mi).magnitude())
        best = min(np.degrees((Mp * Mi * s).magnitude()) for s in S)
        steps[i] = min(dd, 90.0 - dd, best)

    cidx = np.nonzero(cov)[0]
    lo = args.start if args.start is not None else int(cidx.min())
    hi = args.end if args.end is not None else int(cidx.max())
    print(f"# {args.npz}: {n} frames, carry f{lo}..f{hi}")
    print(f"# frames with symmetric-aware angular step > {args.thresh} deg/frame:")
    for i in range(lo, hi + 1):
        if np.isfinite(steps[i]) and steps[i] > args.thresh:
            print(f"  f{i:<4} {steps[i]:7.2f} deg/frame")

    seg = steps[lo:hi + 1]
    seg = seg[np.isfinite(seg)]
    if len(seg):
        print()
        print(f"  carry steps: median {np.median(seg):.2f}  "
              f"p95 {np.percentile(seg, 95):.2f}  max {seg.max():.2f} deg/frame")
        p95 = float(np.percentile(seg, 95))
        print(f"  => suggested --quat_max_step_deg >= {max(3.5, round(p95)):.0f} "
              f"(at/above p95 clips almost nothing yet still catches flips)")
        print(f"     (a default of 3.5/4.5 clips "
              f"{int((seg > 4.5).sum())} of {len(seg)} carry frames here)")


if __name__ == "__main__":
    main()

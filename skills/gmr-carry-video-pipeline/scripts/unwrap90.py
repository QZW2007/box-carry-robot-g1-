# -*- coding: utf-8 -*-
"""Diagnose whether the box REALLY turns in a --box_pose_world clip.

A square-section box is identical modulo 90 deg, so the cuboid fit's absolute
yaw is blind to a real 90-deg turn AND the renderer's fixed-anchor symmetry
snap folds such a turn back to ~0.  This tool unwraps the fitted yaw MOD 90
(allow +-90 deg branch hops, chained from the previous frame) to recover the
true continuous trajectory, and prints the turn amplitude.

Usage:
    python unwrap90.py <world_6dof.npz> [--start F] [--end F]

A SMOOTH, multi-frame ramp (4-9 deg/frame, one direction for many frames that
then reverses) in the printed series == a REAL box turn.  Render it with
`--box_sym_snap continuity` or it will be folded away.
Isolated single-frame jumps of 20-80 deg that immediately return == fit noise.

Also prints the raw (un-unwrapped) yaw so you can see the mod-90 aliasing.
"""
import argparse
import sys

import numpy as np

BLOCK = 90.0


def face_yaw(M):
    """Box local +x axis azimuth in the world XY plane (deg)."""
    x = M[:, 0]
    return np.degrees(np.arctan2(x[1], x[0]))


def unwrap_mod(a, period=BLOCK, half=None):
    """Unwrap a series allowing only +-period/2 hops (chained), so a turn that
    lands on an equivalent symmetry branch is followed instead of folded."""
    if half is None:
        half = period / 2.0
    o = a.copy()
    for i in range(1, len(o)):
        if not (np.isfinite(o[i]) and np.isfinite(o[i - 1])):
            continue
        # bring o[i] within +-half of o[i-1] using whole-period hops
        while o[i] - o[i - 1] > half:
            o[i] -= period
        while o[i] - o[i - 1] < -half:
            o[i] += period
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="the *_world_6dof.npz used as --box_pose_world")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    if "box_R_world" not in d.files:
        sys.exit("no box_R_world in " + args.npz)
    B = d["box_R_world"]
    n = B.shape[0]
    cov = d["carry"].astype(bool) if "carry" in d.files else np.ones(n, bool)

    raw = np.array([face_yaw(B[i]) if np.all(np.isfinite(B[i])) else np.nan
                    for i in range(n)])
    u = unwrap_mod(raw)

    lo = args.start if args.start is not None else int(np.nonzero(cov)[0].min())
    hi = args.end if args.end is not None else int(nonzero_last(cov))
    print(f"# {args.npz}: {n} frames, carry f{lo}..f{hi}")
    print("#  f    raw_yaw   unwrap_mod90   carry")
    for f in range(lo, hi + 1, args.step):
        tag = "C" if cov[f] else "."
        print(f"  {f:<5} {raw[f]:9.1f} {u[f]:14.1f}   {tag}")

    seg = u[lo:hi + 1]
    seg = seg[np.isfinite(seg)]
    if len(seg) >= 3:
        # robust plateau = 20th/80th percentile of the two ends vs the extreme
        k = max(3, len(seg) // 12)
        pre = np.median(seg[:k])
        post = np.median(seg[-k:])
        mid = np.median(seg)
        peak_hi = np.nanmax(seg)
        peak_lo = np.nanmin(seg)
        amp = max(peak_hi - min(pre, post), max(pre, post) - peak_lo)
        print()
        print(f"  pre-turn median  (first {k}): {pre:8.1f}")
        print(f"  post-turn median (last  {k}): {post:8.1f}")
        print(f"  peak high / low  : {peak_hi:8.1f} / {peak_lo:8.1f}")
        print(f"  => turn amplitude ~= {amp:.0f} deg")
        if amp > 20:
            print("  => REAL turn likely: render with --box_sym_snap continuity")
        else:
            print("  => no significant turn in this clip")


def nonzero_last(cov):
    idx = np.nonzero(cov)[0]
    return idx[-1] if len(idx) else len(cov) - 1


if __name__ == "__main__":
    main()

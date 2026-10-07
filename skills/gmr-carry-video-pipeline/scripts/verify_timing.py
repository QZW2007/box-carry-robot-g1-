# -*- coding: utf-8 -*-
"""Verify the RENDERED box rotation TIMING against the measurement and the
robot body -- the halfway metric.

For each of three yaw series -- BODY (g1 `root_rot`), RAW box (fitted
`box_R_world`), REN box (the dumped `boxpose_used.npz`) -- find the frames
where the curve crosses 25 / 50 / 75 % of the way from the pre-turn plateau to
the far plateau, separately for the OUT half (away) and the BACK half (return).
A faithful render must reproduce the RAW crossings; the BODY column tells you
how far the (real) box trails the body.

Why: a slew limiter / temporal filter that clips real motion bakes in a PHASE
LAG that shows up as the box finishing its turn late relative to the robot
("箱子转的相对于机器人来说有一些延迟").  The lag is usually asymmetric -- the
OUT half is short and passes, the BACK half is where the clip bites -- so both
halves must be measured.

Usage:
    python verify_timing.py <g1.pkl> <world_6dof.npz> <boxpose_used.npz>
                            [--base F0 F1] [--pivot F0 F1]

--base F0 F1  frames of the pre-turn plateau used to zero each curve
              (default 146 152).
--pivot F0 F1 frames of the far/turn plateau used for the OUT half's target
              (default 176 185).

Reads the frame offset from the world npz `meta` only for reporting; all series
share one index (the render index) so no shifting is done.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def unwrap(a, period):
    o = a.copy()
    for i in range(1, len(o)):
        if not (np.isfinite(o[i]) and np.isfinite(o[i - 1])):
            continue
        while o[i] - o[i - 1] > period / 2:
            o[i] -= period
        while o[i] - o[i - 1] < -period / 2:
            o[i] += period
    return o


def yaw_from_mats(mats):
    out = np.full(len(mats), np.nan)
    for i in range(len(mats)):
        if np.all(np.isfinite(mats[i])):
            out[i] = np.degrees(np.arctan2(mats[i][1, 0], mats[i][0, 0]))
    return out


def yaw_from_quats(quats, cov):
    out = np.full(len(quats), np.nan)
    for i in range(len(quats)):
        if cov[i]:
            M = R.from_quat(quats[i]).as_matrix()
            out[i] = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    return out


def halves(y, a_span, b_span, rng, period):
    """Crossing frames of 25/50/75 % from plateau(a_span) to plateau(b_span)."""
    a = np.nanmedian(y[a_span[0]:a_span[1]])
    b = np.nanmedian(y[b_span[0]:b_span[1]])
    res = []
    for wq in (0.25, 0.5, 0.75):
        tgt = a + (b - a) * wq
        found = None
        for f in rng:
            if not np.isfinite(y[f]):
                continue
            if (b < a and y[f] <= tgt) or (b > a and y[f] >= tgt):
                found = f
                break
        res.append(found)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("g1_pkl")
    ap.add_argument("world_npz")
    ap.add_argument("boxpose_npz")
    ap.add_argument("--base", type=int, nargs=2, default=[146, 152])
    ap.add_argument("--pivot", type=int, nargs=2, default=[176, 185])
    args = ap.parse_args()

    for p in (args.g1_pkl, args.world_npz, args.boxpose_npz):
        if not os.path.isfile(p):
            sys.exit("missing " + p)

    with open(args.g1_pkl, "rb") as f:
        p = pickle.load(f)
    rr = p["root_rot"]
    body = unwrap(np.degrees(np.arctan2(
        2 * (rr[:, 3] * rr[:, 2] + rr[:, 0] * rr[:, 1]),
        1 - 2 * (rr[:, 1] ** 2 + rr[:, 2] ** 2))), 360)
    n = len(body)

    w = np.load(args.world_npz, allow_pickle=True)
    rawu = unwrap(yaw_from_mats(w["box_R_world"]), 90)
    if "meta" in w.files:
        try:
            off = json.loads(str(w["meta"])).get("frame_offset")
            print("# frame_offset from meta:", off)
        except Exception:
            pass

    d = np.load(args.boxpose_npz, allow_pickle=True)
    renu = unwrap(yaw_from_quats(d["box_quat_xyzw"], d["cov"].astype(bool)), 90)

    b0, b1 = args.base
    bodN = body - np.nanmedian(body[b0:b1])
    rawN = rawu - np.nanmedian(rawu[b0:b1])
    renN = renu - np.nanmedian(renu[b0:b1])

    piv = args.pivot
    end = (min(n - 2, 236), min(n, 248))
    out_rng = range(b1, min(n, piv[1] + 6))
    back_rng = range(piv[1], end[1])

    bOut = halves(bodN, (b0, b1), piv, out_rng, 360)
    rOut = halves(rawN, (b0, b1), piv, out_rng, 90)
    eOut = halves(renN, (b0, b1), piv, out_rng, 90)
    bBack = halves(bodN, piv, end, back_rng, 360)
    rBack = halves(rawN, piv, end, back_rng, 90)
    eBack = halves(renN, piv, end, back_rng, 90)

    print("OUT  25/50/75:  BODY", bOut, " RAW", rOut, " REN", eOut)
    print("BACK 25/50/75:  BODY", bBack, " RAW", rBack, " REN", eBack)
    print("BACK lag vs body: RAW", [rBack[i] - bBack[i] for i in range(3)],
          " REN", [eBack[i] - bBack[i] for i in range(3)])
    print("  (a faithful render should match RAW; do not 'improve' on it)")
    lo = np.nanmin(renN[b0:max(n, piv[1] + 100)])
    hi = np.nanmax(renN[b0:max(n, piv[1] + 100)])
    print(f"REN amplitude in carry: {hi - lo:.1f} deg "
          f"(raw {np.nanmax(rawN[b0:]) - np.nanmin(rawN[b0:]):.1f})")
    ss = [abs(renN[f] - renN[f - 1]) for f in range(b0, min(n, 281))
          if np.isfinite(renN[f]) and np.isfinite(renN[f - 1])]
    if ss:
        print(f"REN max per-frame step: {max(ss):.1f} deg/frame")


if __name__ == "__main__":
    main()

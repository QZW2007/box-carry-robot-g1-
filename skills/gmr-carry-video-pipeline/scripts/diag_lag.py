# -*- coding: utf-8 -*-
"""How many frames is the retargeted robot behind the measured box?

The box is rigidly held while carried, so |hand_mid - box_centre| must be
roughly CONSTANT over the carry.  If the retargeted robot's whole body runs
late, the distance curve is a U whose minimum sits at the true lag: shifting
the box timeline by that many frames lines the two up again.

Reports, for every candidate lag tau, the median / p90 hand-to-box distance
over the carry phase, plus the distance curve's own minimum frame.
"""
import sys

import numpy as np
import torch

from general_motion_retargeting import KinematicsModel, ROBOT_XML_DICT
from general_motion_retargeting.data_loader import load_robot_motion

PKL = sys.argv[1] if len(sys.argv) > 1 else "data/20260922_161727_g1.pkl"
WORLD = sys.argv[2] if len(sys.argv) > 2 else \
    "data/20260922_161727_world_6dof.npz"

md, fps, _, _, _, _, _ = load_robot_motion(PKL)
km = KinematicsModel(str(ROBOT_XML_DICT["unitree_g1"]), device="cpu")
with torch.no_grad():
    bp, _ = km.forward_kinematics(
        torch.tensor(md["root_pos"], dtype=torch.float32),
        torch.tensor(md["root_rot"], dtype=torch.float32),
        torch.tensor(md["dof_pos"], dtype=torch.float32))
bp = bp.numpy()
bn = km.body_names
hm = (bp[:, bn.index("left_wrist_yaw_link"), :]
      + bp[:, bn.index("right_wrist_yaw_link"), :]) / 2.0

w = np.load(WORLD, allow_pickle=True)
pos = np.asarray(w["box_center_world"], dtype=float)
carry = np.asarray(w["carry"]) > 0
cov = ((np.asarray(w["valid"]) > 0) & (np.asarray(w["pose_valid"]) > 0)
       & np.isfinite(pos).all(axis=1))
n = min(len(hm), len(pos))
fc = int(np.argmax(carry))
lc = int(n - np.argmax(carry[::-1]) - 1)
print("n=%d  carry %d..%d  (fps %s)" % (n, fc, lc, fps))

print()
print("tau   median_xy   p90_xy   median_3d   n_used")
best = None
for tau in range(0, 41, 2):
    # compare hand(t) with box(t - tau) over the carry
    idx = [t for t in range(fc + tau, lc + 1) if cov[t - tau] and t < n]
    if len(idx) < 20:
        continue
    dxy = np.array([np.linalg.norm(hm[t, :2] - pos[t - tau, :2]) for t in idx])
    d3 = np.array([np.linalg.norm(hm[t] - pos[t - tau]) for t in idx])
    if best is None or np.median(dxy) < best[1]:
        best = (tau, float(np.median(dxy)))
    print("%3d   %.3f      %.3f    %.3f       %d"
          % (tau, np.median(dxy), np.percentile(dxy, 90), np.median(d3),
             len(idx)))
print()
if best:
    print("best tau (by median hand-box horizontal distance) = %d frames "
          "= %.2f s" % (best[0], best[0] / float(fps or 30)))

# how well does a late robot explain the approach?  find when the hands
# first get close to the box's PARKED spot
parked = pos[fc, :2]
d0 = np.array([np.linalg.norm(hm[t, :2] - parked) for t in range(n)])
print()
print("horizontal distance from the hands to the box's parked spot:")
for t in range(max(fc - 30, 0), min(fc + 45, n), 2):
    print("  f%3d  %.3f m%s" % (t, d0[t], "   <- carry starts" if t == fc
                                else ""))
hit = np.nonzero(d0 <= 0.25)[0]
hit = hit[hit >= fc - 30]
print("hands come within 0.25 m of the parked spot at f%d"
      % (hit[0] if len(hit) else -1))
hit2 = np.nonzero(d0 <= 0.15)[0]
hit2 = hit2[hit2 >= fc - 30]
print("hands come within 0.15 m of the parked spot at f%d"
      % (hit2[0] if len(hit2) else -1))

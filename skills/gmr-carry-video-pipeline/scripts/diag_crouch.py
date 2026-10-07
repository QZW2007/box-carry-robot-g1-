# -*- coding: utf-8 -*-
"""Why does the robot squat BEFORE it reaches the box?

Prints, frame by frame around the pickup, the robot's own body state (root and
hand heights from FK) next to the box pose the renderer actually drew.  If the
pelvis/hands descend while the hands are still well short of the box, the
approach is mistimed; if the hands are already at the box, the mismatch is
somewhere else.

The FK here is on the RAW pkl (no arm despike / no left-arm mirror), which is
exact for the root and the legs -- and those are what "walks up and then
squats" is about.  The arm despike/mirror only touches the left arm over the
last 12 frames before the carry phase.
"""
import json
import sys

import numpy as np
import torch

from general_motion_retargeting import KinematicsModel, ROBOT_XML_DICT
from general_motion_retargeting.data_loader import load_robot_motion

PKL = sys.argv[1] if len(sys.argv) > 1 else "example/20260922_162859/data/20260922_162859_g1.pkl"
USED = sys.argv[2] if len(sys.argv) > 2 else \
    "example/20260922_162859/data/20260922_162859_boxpose_used.npz"
F0 = int(sys.argv[3]) if len(sys.argv) > 3 else 120
F1 = int(sys.argv[4]) if len(sys.argv) > 4 else 178

md, fps, _, _, _, _, _ = load_robot_motion(PKL)
km = KinematicsModel(str(ROBOT_XML_DICT["unitree_g1"]), device="cpu")
with torch.no_grad():
    bp, _ = km.forward_kinematics(
        torch.tensor(md["root_pos"], dtype=torch.float32),
        torch.tensor(md["root_rot"], dtype=torch.float32),
        torch.tensor(md["dof_pos"], dtype=torch.float32))
bp = bp.numpy()
bn = km.body_names


def bidx(n):
    return bn.index(n)


lw = bp[:, bidx("left_wrist_yaw_link"), :]
rw = bp[:, bidx("right_wrist_yaw_link"), :]
hm = (lw + rw) / 2.0
lk = bp[:, bidx("left_knee_link"), :]
rk = bp[:, bidx("right_knee_link"), :]
la = bp[:, bidx("left_ankle_roll_link"), :]
ra = bp[:, bidx("right_ankle_roll_link"), :]
root = np.asarray(md["root_pos"], dtype=float)

u = np.load(USED, allow_pickle=True)
box = u["box_pos"]
carry = u["carry"].astype(bool)
first_c, last_c = int(u["first_c"]), int(u["last_c"])
print("pkl=%s  n=%d  fps=%s" % (PKL, len(root), fps))
print("carry %d..%d   box half height 0.225 (0.45 box)" % (first_c, last_c))
print("floor z (min ankle over clip) = %+.4f" % float(np.minimum(la[:, 2],
                                                                ra[:, 2]).min()))
print()
print("  f    root_x root_y root_z | pelvis_z knee_z  hand_y hand_z "
      "| box_x  box_y  box_z  boxTop | dXY   dZ        d3D  c")
for f in range(F0, min(F1, len(root))):
    kz = 0.5 * (lk[f, 2] + rk[f, 2])
    bt = box[f, 2] + 0.225
    dxy = float(np.hypot(hm[f, 0] - box[f, 0], hm[f, 1] - box[f, 1]))
    dz = float(hm[f, 2] - box[f, 2])
    d3 = float(np.linalg.norm(hm[f] - box[f]))
    print("%4d  %+6.3f %+6.3f %+6.3f | %+6.3f %+6.3f  %+6.3f %+6.3f "
          "| %+6.3f %+6.3f %+6.3f %+6.3f | %.3f %+6.3f  %.3f  %s"
          % (f, root[f, 0], root[f, 1], root[f, 2], root[f, 2], kz,
             hm[f, 1], hm[f, 2], box[f, 0], box[f, 1], box[f, 2], bt,
             dxy, dz, d3, "C" if carry[f] else "."))

# ---- when does each thing actually happen? -------------------------------
print()
root_z = root[:, 2]
base = float(np.median(root_z[max(first_c - 60, 0):max(first_c - 20, 1)]))
print("standing pelvis z (median f%d..f%d) = %+.4f"
      % (max(first_c - 60, 0), max(first_c - 20, 1), base))
thr = base - 0.03
dip = np.nonzero(root_z < thr)[0]
dip = dip[dip < first_c + 5]
if len(dip):
    print("pelvis first drops 30 mm below standing at f%d" % dip[0])
hm_z = hm[:, 2]
med_hi = float(np.median(hm_z[:max(first_c - 40, 1)]))
desc = np.nonzero(hm_z < med_hi - 0.10)[0]
if len(desc):
    print("hands first 100 mm below their early height at f%d "
          "(early hand z median %+.3f)" % (desc[0], med_hi))
print("carry (box leaves the floor) starts at f%d" % first_c)
print("hand-to-box 3D distance at f%d-4..f%d:"
      % (first_c, first_c + 6))
for f in range(first_c - 4, first_c + 7):
    print("   f%d  %.3f m%s" % (f, np.linalg.norm(hm[f] - box[f]),
                                "  <- carry" if f == first_c else ""))

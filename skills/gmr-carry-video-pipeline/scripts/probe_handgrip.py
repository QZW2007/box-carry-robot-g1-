# -*- coding: utf-8 -*-
"""Probe: hand geometry vs MEASURED box trajectory during carry.

Replicates the renderer's FK (KinematicsModel on the g1 pkl) and compares,
per frame: wrist span, measured-vs-handmid offset, and left/right rubber-hand
centres expressed in the MEASURED box frame (and in the RENDERED box frame
for cross-check).  Tells us whether moving the box back to the measurement
would clear the hands (grip on the faces) or not.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from general_motion_retargeting import KinematicsModel, ROBOT_XML_DICT  # noqa
from general_motion_retargeting.data_loader import load_robot_motion  # noqa

P = "20260922_165945"
HALF = np.array([0.15, 0.15, 0.225])

motion_data, fps, _, _, _, _, _ = load_robot_motion("data/%s_g1.pkl" % P)
km = KinematicsModel(str(ROBOT_XML_DICT["unitree_g1"]), device="cpu")
root_pos = torch.tensor(motion_data["root_pos"], dtype=torch.float32)
root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float32)  # xyzw
dof_pos = torch.tensor(motion_data["dof_pos"], dtype=torch.float32)
with torch.no_grad():
    body_pos, _ = km.forward_kinematics(root_pos, root_rot, dof_pos)
body_pos = body_pos.numpy()
BN = km.body_names

print("body list has rubber hands:",
      "left_rubber_hand" in BN, "right_rubber_hand" in BN)


def bp(name):
    return body_pos[:, BN.index(name), :]


lw, rw = bp("left_wrist_yaw_link"), bp("right_wrist_yaw_link")
try:
    lh, rh = bp("left_rubber_hand"), bp("right_rubber_hand")
except ValueError:
    lh, rh = lw, rw
mid = (lw + rw) / 2.0
n = len(mid)

w = np.load("data/%s_world_6dof.npz" % P, allow_pickle=True)
print("world npz keys:", w.files)
cmeas = np.asarray(w["box_pos_world"], dtype=float)
Rmeas = np.asarray(w["box_R_world"], dtype=float)  # (N,3,3), NaN when uncov
carry = np.asarray(w["carry"]).astype(bool)

d = np.load("data/%s_boxpose_used.npz" % P, allow_pickle=True)
rpos = np.asarray(d["box_pos"], dtype=float)
rq = np.asarray(d["box_quat_xyzw"], dtype=float)
x, y, z, ww = rq[:, 0], rq[:, 1], rq[:, 2], rq[:, 3]
Rr = np.empty((n, 3, 3))
Rr[:, 0, 0] = 1 - 2 * (y * y + z * z)
Rr[:, 0, 1] = 2 * (x * y - z * ww)
Rr[:, 0, 2] = 2 * (x * z + y * ww)
Rr[:, 1, 0] = 2 * (x * y + z * ww)
Rr[:, 1, 1] = 1 - 2 * (x * x + z * z)
Rr[:, 1, 2] = 2 * (y * z - x * ww)
Rr[:, 2, 0] = 2 * (x * z - y * ww)
Rr[:, 2, 1] = 2 * (y * z + x * ww)
Rr[:, 2, 2] = 1 - 2 * (x * x + y * y)

print()
print("f     span  |cM-mid|  em_z    L in MEAS box (x,y,z mm)      "
      "R in MEAS box (x,y,z mm)     L/R in REND box x   carry")
ci = int(np.nonzero(carry)[0][0])
cj = int(np.nonzero(carry)[0][-1])
n_in_m = 0
n_in_r = 0
ncov = 0
spans = []
for t in range(138, min(cj + 12, n)):
    covm = np.isfinite(cmeas[t]).all() and np.isfinite(Rmeas[t]).all()
    if t % 2 == 0 or t == ci or t == cj:
        span = float(np.linalg.norm(rw[t] - lw[t]))
        em = cmeas[t] - mid[t] if covm else np.full(3, np.nan)
        lo_m = Rmeas[t].T @ (lh[t] - cmeas[t]) if covm else np.full(3, np.nan)
        ro_m = Rmeas[t].T @ (rh[t] - cmeas[t]) if covm else np.full(3, np.nan)
        lo_r = Rr[t].T @ (lh[t] - rpos[t])
        ro_r = Rr[t].T @ (rh[t] - rpos[t])
        inmL = np.all(np.abs(lo_m) < HALF) if covm else False
        inmR = np.all(np.abs(ro_m) < HALF) if covm else False
        inrL = np.all(np.abs(lo_r) < HALF)
        inrR = np.all(np.abs(ro_r) < HALF)
        tag = ("M" if (inmL or inmR) else " ") + ("R" if (inrL or inrR) else " ")
        print("f%03d %5.0f %7.0f %+7.0f  [%+6.0f %+6.0f %+6.0f] "
              "[%+6.0f %+6.0f %+6.0f]  [%+6.0f %+6.0f]  %d %s"
              % (t, span * 1000,
                 np.linalg.norm(em) * 1000 if covm else np.nan,
                 em[2] * 1000 if covm else np.nan,
                 *(lo_m * 1000 if covm else np.full(3, np.nan)),
                 *(ro_m * 1000 if covm else np.full(3, np.nan)),
                 lo_r[0] * 1000, ro_r[0] * 1000, carry[t], tag))
    if carry[t] and covm:
        ncov += 1
        spans.append(float(np.linalg.norm(rw[t] - lw[t])))
        if np.all(np.abs(Rmeas[t].T @ (lh[t] - cmeas[t])) < HALF) or \
           np.all(np.abs(Rmeas[t].T @ (rh[t] - cmeas[t])) < HALF):
            n_in_m += 1
        if np.all(np.abs(Rr[t].T @ (lh[t] - rpos[t])) < HALF) or \
           np.all(np.abs(Rr[t].T @ (rh[t] - rpos[t])) < HALF):
            n_in_r += 1

print()
print("carry frames covered: %d" % ncov)
print("wrist span over carry: min %.0f  med %.0f  max %.0f mm"
      % (min(spans) * 1000, float(np.median(spans)) * 1000,
         max(spans) * 1000))
print("frames with a rubber-hand CENTRE inside the MEASURED box: %d"
      % n_in_m)
print("frames with a rubber-hand CENTRE inside the RENDERED box: %d"
      % n_in_r)

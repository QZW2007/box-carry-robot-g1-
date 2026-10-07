# -*- coding: utf-8 -*-
"""Box-vs-hand gap through the pickup, from the dumped used box pose."""
import sys

import numpy as np
import torch

from general_motion_retargeting import KinematicsModel, ROBOT_XML_DICT
from general_motion_retargeting.data_loader import load_robot_motion

PKL = "example/20260922_162859/data/20260922_162859_g1.pkl"
USED = sys.argv[1] if len(sys.argv) > 1 else \
    "example/20260922_162859/data/20260922_162859_boxpose_used.npz"
F0, F1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (140, 200)

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

u = np.load(USED, allow_pickle=True)
box, carry = u["box_pos"], u["carry"].astype(bool)
fc, lc = int(u["first_c"]), int(u["last_c"])
print("dump first_c=%d last_c=%d" % (fc, lc))
print("  f   box_x  box_y  box_z | hand_x hand_y hand_z | gap_xy gap_z "
      "boxStep")
step = np.linalg.norm(np.diff(box, axis=0), axis=1) * 1000.0
for f in range(F0, min(F1, len(box))):
    gxy = float(np.hypot(hm[f, 0] - box[f, 0], hm[f, 1] - box[f, 1]))
    gz = float(hm[f, 2] - box[f, 2])
    print("%4d  %+6.3f %+6.3f %+6.3f | %+6.3f %+6.3f %+6.3f | %.3f %+6.3f "
          "%6.1f %s" % (f, box[f, 0], box[f, 1], box[f, 2], hm[f, 0], hm[f, 1],
                        hm[f, 2], gxy, gz, step[f], "C" if carry[f] else "."))

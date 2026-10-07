# -*- coding: utf-8 -*-
"""Hand centres in box frame during turn/walk f185..265 (config via argv)."""
import sys
import numpy as np

npz = sys.argv[1] if len(sys.argv) > 1 else ".workbuddy/hd_v16.npz"
out = sys.argv[2] if len(sys.argv) > 2 else ".workbuddy/hd_walk.txt"
d = np.load(npz, allow_pickle=True)
bodies = [str(b) for b in d["bodies"]]
idx = {b: i for i, b in enumerate(bodies)}
LR, RR = idx["left_rubber_hand"], idx["right_rubber_hand"]
dep, cen = d["depth_mm"], d["center_box"] * 1000.0

lines = ["frame | L depth cen(box x,y,z) | R depth cen(box x,y,z)  "
         "(box half 150,150,225)"]
for t in range(185, 266, 4):
    lc, rc = cen[t, LR], cen[t, RR]
    lines.append("f%d | L %3.0f (%+4.0f,%+4.0f,%+4.0f) | R %3.0f "
                 "(%+4.0f,%+4.0f,%+4.0f)"
                 % (t, dep[t, LR], lc[0], lc[1], lc[2],
                    dep[t, RR], rc[0], rc[1], rc[2]))

# also the box yaw vs body yaw proxy: box_quat yaw over these frames
from scipy.spatial.transform import Rotation as R
q = d["box_quat"]
ys = []
for t in range(185, 266, 4):
    yaw = R.from_quat(q[t]).as_euler("xyz")[2]
    ys.append("f%d:%+.0f" % (t, np.degrees(yaw)))
lines.append("box yaw: " + " ".join(ys))

with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("DONE")

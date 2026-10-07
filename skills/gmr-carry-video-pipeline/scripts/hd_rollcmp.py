# -*- coding: utf-8 -*-
"""Compare rubber-hand penetration between two hd_*.npz dumps (v19 roll85 vs
v20 roll0), per phase."""
import numpy as np

def summ(path):
    d = np.load(path, allow_pickle=True)
    bodies = [str(b) for b in d["bodies"]]
    dep = d["depth_mm"]
    out = {}
    for ph, (a, b) in (("approach f139-147", (139, 148)),
                       ("lift f148-171", (148, 172)),
                       ("turn f172-212", (172, 213)),
                       ("walk f213-269", (213, 270)),
                       ("place f270-353", (270, 354))):
        seg = dep[a:b]
        row = []
        for nm in ("left_rubber_hand", "right_rubber_hand"):
            i = bodies.index(nm)
            s = seg[:, i]
            row.append("%s max %5.1f  p90 %5.1f  >0: %3d fr"
                       % (nm.split("_")[0][0].upper(), s.max(),
                          np.percentile(s, 90), int((s > 0.5).sum())))
        out[ph] = " | ".join(row)
    return out

a = summ(".workbuddy/hd_v19.npz")
b = summ(".workbuddy/hd_v20.npz")
lines = []
for ph in a:
    lines.append(ph)
    lines.append("  v19(roll85): " + a[ph])
    lines.append("  v20(roll 0): " + b[ph])
with open(".workbuddy/hd_roll_cmp.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("DONE")

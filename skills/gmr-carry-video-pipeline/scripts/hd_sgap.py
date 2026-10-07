# -*- coding: utf-8 -*-
"""True SURFACE gap between each rubber-hand mesh and the box slab, per frame.

For every frame: transform the hand mesh vertices into the box frame (same
convention as the renderer's --hand_diag), compute the signed distance of the
closest vertex to the box SURFACE:
    d < 0  -> the nearest vertex is inside the box (touching / penetrating)
    d >= 0 -> gap in mm between the hand and the box (floating)
Also report the *press-axis* gap: min over the two faces of |box x|, so a hand
pressing a SIDE face is judged against that face.

Output: one line every 2 frames for f150..f353, plus a summary of the
"floating" runs (gap > 20 mm) so we can see exactly WHICH phase detaches.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

import mujoco as mj  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from general_motion_retargeting.mj_compat import load_model  # noqa: E402

P = "20260922_165945"
HALF = np.array([0.15, 0.15, 0.225])
XML = os.path.join("assets", "unitree_g1", "g1_mocap_29dof_box.xml")
BOXNPZ = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "数据", P, "%s_boxpose_used.npz" % P)
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else ".workbuddy/hd_sgap.txt"


def body_pose_from_pkl():
    """Replicate the delivered v20 ARMS: load pkl, apply despike + mirror +
    abduct + hand roll/shrink is not needed for GEOMETRY of the hand centre,
    but the arm dofs decide the hand position, so apply the same chains."""
    import pickle
    with open("data/%s_g1.pkl" % P, "rb") as f:
        md = pickle.load(f)
    return md


def main():
    model = load_model(XML)
    # hand mesh vertices (geom-local, deduped) exactly like --hand_diag
    hand_geoms = []
    seen = set()
    for g in range(model.ngeom):
        if model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        bnm = model.body(model.geom_bodyid[g]).name or ""
        if "rubber_hand" not in bnm:
            continue
        mid = model.geom_dataid[g]
        key = (model.geom_bodyid[g], mid)
        if key in seen:
            continue
        seen.add(key)
        adr = model.mesh_vertadr[mid]
        num = model.mesh_vertnum[mid]
        hand_geoms.append((g, bnm, model.mesh_vert[adr:adr + num].astype(float)))
    print("hand geoms:", [(g, b, len(V)) for g, b, V in hand_geoms])

    md = body_pose_from_pkl()
    root_pos = np.asarray(md["root_pos"], dtype=float)
    root_rot = np.asarray(md["root_rot"], dtype=float)
    dof = np.asarray(md["dof_pos"], dtype=float)

    d = np.load(BOXNPZ, allow_pickle=True)
    box_pos = np.asarray(d["box_pos"], dtype=float)
    box_quat = np.asarray(d["box_quat_xyzw"], dtype=float)

    data = mj.MjData(model)
    mocap_id = model.body("carried_box").mocapid
    n = len(root_pos)
    lines = []
    float_runs = {"L": [], "R": []}
    cur = {"L": None, "R": None}
    for t in range(n):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot[t][[3, 0, 1, 2]]
        data.qpos[7:] = dof[t]
        data.mocap_pos[mocap_id] = box_pos[t]
        data.mocap_quat[mocap_id] = R.from_quat(box_quat[t]).as_quat(
            scalar_first=True)
        mj.mj_forward(model, data)
        cbx = R.from_quat(box_quat[t])          # box -> world
        out = {}
        for g, bnm, Vl in hand_geoms:
            Vw = data.geom_xpos[g] + Vl @ data.geom_xmat[g].reshape(3, 3).T
            loc = cbx.inv().apply(Vw - box_pos[t])
            a = np.abs(loc)
            # signed distance to the surface of the slab (negative inside)
            sd = (a - HALF[None, :]).max(axis=1)
            dmin = float(sd.min())              # closest vertex
            out[bnm] = dmin
        L, Rr = out["left_rubber_hand"], out["right_rubber_hand"]
        side = "L" if L < Rr else "R"
        # a hand counts as FLOATING when its closest vertex is > 20 mm away
        for tag, val in (("L", L), ("R", Rr)):
            f = val > 0.020
            if f and cur[tag] is None:
                cur[tag] = t
            elif not f and cur[tag] is not None:
                float_runs[tag].append((cur[tag], t - 1))
                cur[tag] = None
        if t >= 145:
            lines.append("f%03d  L gap %+6.1f mm   R gap %+6.1f mm"
                         % (t, L * 1000, Rr * 1000))
    for tag in ("L", "R"):
        if cur[tag] is not None:
            float_runs[tag].append((cur[tag], n - 1))

    hdr = ["# TRUE surface gap of each rubber hand to the box (mm)",
           "# gap < 0 = touching/inside, > 0 = floating",
           "# floating runs (gap > 20 mm):"]
    for tag in ("L", "R"):
        tot = sum(b - a + 1 for a, b in float_runs[tag])
        hdr.append("#   %s: %d frames in %d runs: %s"
                   % (tag, tot, len(float_runs[tag]),
                      ", ".join("f%d-%d" % (a, b) for a, b in float_runs[tag])))
    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write("\n".join(hdr + lines) + "\n")
    print("DONE", OUTFILE)


main()

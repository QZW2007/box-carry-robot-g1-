# -*- coding: utf-8 -*-
"""Wrist-roll A/B probe: render the delivered carry pose (raw dofs at a walk
frame, v19 uses raw arms there) with different --hand_roll_deg values and
show a zoomed box+hands view from the delivered camera azimuth (-97.2 deg).

Grid: rows = candidate roll values, cols = frames.  Decide which roll puts
the PALM on the box (inward) instead of the back of the hand (outward).
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

import cv2  # noqa: E402
import mujoco as mj  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from general_motion_retargeting.mj_compat import load_model  # noqa: E402
from general_motion_retargeting.data_loader import load_robot_motion  # noqa

P = "20260922_165945"
FRAMES = [160, 230]          # grasp-ish + mid-walk (v19: raw arms at 230)
ROLLS = [85, 0, -85, 180]    # current + candidates
CAM_AZ = -97.2
CAM_EL = -16.0
CAM_DIST = 1.15

motion_data, fps, _, _, _, _, _ = load_robot_motion("data/%s_g1.pkl" % P)
root_pos = np.asarray(motion_data["root_pos"], dtype=float)
root_rot = np.asarray(motion_data["root_rot"], dtype=float)      # xyzw
dof_pos = np.asarray(motion_data["dof_pos"], dtype=float)

d = np.load(os.path.join("数据", P, "%s_boxpose_used.npz" % P),
            allow_pickle=True)
box_pos = np.asarray(d["box_pos"], dtype=float)
box_quat = np.asarray(d["box_quat_xyzw"], dtype=float)

XML = os.path.join("assets", "unitree_g1", "g1_mocap_29dof_box.xml")


def build(roll_deg):
    model = load_model(XML)
    if roll_deg:
        for nm, sgn in (("left_rubber_hand", +1.0),
                        ("right_rubber_hand", -1.0)):
            roll = R.from_rotvec([np.deg2rad(sgn * roll_deg), 0, 0])
            bid = model.body(nm).id
            model.body_quat[bid] = roll.as_quat(scalar_first=True)
    # hand_shrink 0.62 (match the delivered render)
    s = 0.62
    for nm in ("left_rubber_hand", "right_rubber_hand"):
        mesh_id = model.mesh(nm).id
        rq = R.from_quat(model.mesh_quat[mesh_id][[1, 2, 3, 0]]).as_matrix()
        adr = model.mesh_vertadr[mesh_id]
        num = model.mesh_vertnum[mesh_id]
        V = model.mesh_vert[adr:adr + num].reshape(-1, 3).copy()
        Vg = V @ rq
        x0 = Vg[:, 0].min()
        Vg[:, 0] = x0 + (Vg[:, 0] - x0) * s
        model.mesh_vert[adr:adr + num] = Vg @ rq.T
    return model


tiles = []
for roll in ROLLS:
    model = build(roll)
    data = mj.MjData(model)
    mocap_id = model.body("carried_box").mocapid
    rend = mj.Renderer(model, height=480, width=640)
    row = []
    for t in FRAMES:
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot[t][[3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[t]
        data.mocap_pos[mocap_id] = box_pos[t]
        data.mocap_quat[mocap_id] = R.from_quat(box_quat[t]).as_quat(
            scalar_first=True)
        mj.mj_forward(model, data)
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = box_pos[t] + np.array([0.0, 0.0, 0.05])
        cam.azimuth = CAM_AZ
        cam.elevation = CAM_EL
        cam.distance = CAM_DIST
        rend.update_scene(data, camera=cam)
        img = rend.render().copy()
        cv2.putText(img, "roll %+d  f%d" % (roll, t), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        row.append(img)
    rend.close()
    tiles.append(np.hstack(row))

sheet = np.vstack(tiles)
out = os.path.join(".workbuddy", "out", "wristroll_ab.jpg")
ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
buf.tofile(out)
print("wrote", out, sheet.shape)

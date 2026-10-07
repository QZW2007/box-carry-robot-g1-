"""Visualize the Unitree G1 carrying a box, with the box trajectory drawn in 3D.

Pipeline:
    robot motion pkl -> FK wrist/elbow/ankle poses -> per-frame hand frame
    -> 3-phase box pose (on ground -> smooth blend -> rigidly attached to hands)
    -> MuJoCo mocap box + trajectory polyline, interactive viewer or offscreen mp4.

The default box-to-hand offset/orientation are eyeballed placeholders; once the
ArUco re-shoot data (真值) is available they get replaced by calibrated values.
Phase boundaries default to the 20260911 video (grasp ~85, carried ~100) and
should be re-detected for new footage: hands come together low = grasp moment.

Usage examples:
    # interactive window
    python scripts/vis_robot_with_box.py --robot_motion data/20260911211441_unitree_g1.pkl

    # headless render with trajectory + save box trajectory
    python scripts/vis_robot_with_box.py --robot_motion data/xxx_g1.pkl \
        --record_video video/xxx_box.mp4 --save_box_traj data/xxx_box_traj.npz
"""

import argparse
import json
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm

import mujoco as mj

from general_motion_retargeting import KinematicsModel, ROBOT_XML_DICT
from general_motion_retargeting.data_loader import load_robot_motion
from general_motion_retargeting.mj_compat import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")

BOX_XML = os.path.join(ASSETS, "unitree_g1", "g1_mocap_29dof_box.xml")


# --------------------------------------------------------------------------- #
# hand frame + 3-phase box pose
# --------------------------------------------------------------------------- #

def compute_stool_pose(motion, box_pos, box_quat_xyzw, hand_mid, hand_rot,
                       box_size=(0.30, 0.24, 0.30),
                       stool_size=(0.38, 0.30, 0.45),
                       stool_offset=(0.476, 0.902),
                       box_ground_xy=None,
                       root_pos=None):
    """Static four-leg stool, placed RELATIVE TO THE BOX in the robot world.

    Why not an absolute camera solve?  Because GVHMR's world is scaled by an
    unknown factor: it normalises depth with an assumed focal of 800 while
    the RealSense bag's real focal is 386.76 (2.07x).  Solving the camera
    pose in that world gives an inconsistent answer (the box comes out
    3.4 m deep when the robot world says 1.8 m).

    What IS trustworthy:
      * the pkl world's DIRECTIONS (the robot really walks along +Y and its
        up-axis really is +Z), and
      * the source image's RATIOS.

    So we measure the stool-vs-box separation in pixels, convert it to real
    metres with the BOX itself as the ruler (its width is a known 0.30 m),
    and then place the stool along the pkl world's direction vectors.

    Source image, frame 0 (640x480), real intrinsics fx=386.76 cx=321.93:
        box   ground contact  u=214, v=391 ;  front face 44 px wide = 0.30 m
        stool leg base        u=320, v=467
    ->  box depth 2.64 m, stool depth 1.74 m, camera height 1.00 m
    ->  stool is +0.48 m to the box's image-right, +0.90 m closer to camera

    Parameters
    ----------
    stool_offset : (d_lateral, d_closer)  METRES, measured in the camera's
        frame.  d_lateral is along image-right; d_closer is toward the
        camera.  Both are already in real metres (box-as-ruler), so the
        pkl world's scale error cannot leak in.

    Returns stool_pos (3,), stool_quat_xyzw (4,) -- constant over time.
    """
    if box_ground_xy is None:
        box_ground_xy = box_pos[0, :2]
    box_ground_xy = np.asarray(box_ground_xy, dtype=float)

    # ---- pkl world directions ----
    # "toward the camera" comes from the robot's own net walk: the person
    # walks toward the camera in the video, and the robot reproduces it.
    if root_pos is None:
        root_pos = motion["root_pos"]
    walk = np.asarray(root_pos[-1][:2], dtype=float) \
        - np.asarray(root_pos[0][:2], dtype=float)
    n = np.linalg.norm(walk)
    if n < 1e-6:                     # robot stood still; fall back to +Y
        toward_cam = np.array([0.0, 1.0])
    else:
        toward_cam = walk / n
    # image-right = view_dir x up, with view_dir = away from the camera
    view_dir = np.array([-toward_cam[0], -toward_cam[1], 0.0])
    img_right = np.cross(view_dir, np.array([0.0, 0.0, 1.0]))[:2]
    img_right /= np.linalg.norm(img_right) + 1e-9

    d_lat, d_closer = stool_offset
    offset = d_lat * img_right + d_closer * toward_cam
    stool_xy = box_ground_xy + offset

    # the stool's yaw: keep the seat parallel to the box's resting yaw
    q0 = box_quat_xyzw[0]
    yaw = np.arctan2(2.0 * (q0[3] * q0[2] + q0[0] * q0[1]),
                     1.0 - 2.0 * (q0[1] ** 2 + q0[2] ** 2))

    stool_pos = np.array([stool_xy[0], stool_xy[1], 0.0])
    stool_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    return stool_pos, stool_quat


def compute_hand_frames(body_pos, body_names):
    """Per-frame grasping frame from wrist/elbow positions.

    Origin: midpoint of the two wrists. Axes (columns of R):
      y: left wrist -> right wrist (roughly world-right while carrying)
      x: forearm direction (elbow -> wrist), orthogonalized against y
      z: x cross y (roughly up)
    """
    def idx(name):
        return body_names.index(name)

    lw = body_pos[:, idx("left_wrist_yaw_link"), :]
    rw = body_pos[:, idx("right_wrist_yaw_link"), :]
    le = body_pos[:, idx("left_elbow_link"), :]
    re = body_pos[:, idx("right_elbow_link"), :]

    mid = (lw + rw) / 2.0
    y = rw - lw
    y /= np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8

    f = (lw - le) + (rw - re)
    x = f - (np.sum(f * y, axis=-1, keepdims=True) * y)  # Gram-Schmidt
    x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8

    z = np.cross(x, y)

    rot = np.stack([x, y, z], axis=-1)  # (N, 3, 3), columns are axes
    return mid, rot


def compute_box_trajectory(motion, phase_lift, phase_carry,
                           box_offset, box_yaw_deg,
                           ground_xy=None, ground_z=None):
    """Returns box_pos (N,3), box_quat_xyzw (N,3?), phase labels (N,)."""
    import torch

    km = KinematicsModel(str(ROBOT_XML_DICT["unitree_g1"]), device="cpu")
    root_pos = torch.tensor(motion["root_pos"], dtype=torch.float32)
    root_rot = torch.tensor(motion["root_rot"], dtype=torch.float32)  # xyzw
    dof_pos = torch.tensor(motion["dof_pos"], dtype=torch.float32)
    with torch.no_grad():
        body_pos, _ = km.forward_kinematics(root_pos, root_rot, dof_pos)
    body_pos = body_pos.numpy()

    hand_mid, hand_rot = compute_hand_frames(body_pos, km.body_names)
    n = len(hand_mid)

    # ---- carried phase: box rigidly attached to the hand frame ----
    yaw = np.deg2rad(box_yaw_deg)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1],
    ])
    carry_rot = hand_rot @ R_yaw  # (N,3,3)
    carry_pos = hand_mid + np.einsum("nij,j->ni", hand_rot, box_offset)
    carry_quat = R.from_matrix(carry_rot).as_quat()  # xyzw

    # ---- ground phase: static box, placed at the grasp spot ----
    if ground_xy is None:
        grasp_slice = slice(max(phase_lift - 5, 0), phase_carry + 1)
        ground_xy = np.median(hand_mid[grasp_slice, :2], axis=0)
    ankles = np.concatenate([
        body_pos[:, km.body_names.index("left_ankle_roll_link"), 2],
        body_pos[:, km.body_names.index("right_ankle_roll_link"), 2],
    ])
    floor_z = float(ankles.min())
    if ground_z is None:
        ground_z = floor_z + motion["box_half_height"]
    # face the same direction the box has right after pickup -> tiny blend
    yaw0 = np.arctan2(carry_rot[phase_carry, 1, 0], carry_rot[phase_carry, 0, 0])
    ground_rot = np.array([
        [np.cos(yaw0), -np.sin(yaw0), 0],
        [np.sin(yaw0),  np.cos(yaw0), 0],
        [0, 0, 1],
    ])
    ground_pos = np.array([ground_xy[0], ground_xy[1], ground_z])
    ground_quat = R.from_matrix(ground_rot).as_quat()

    # ---- assemble with smooth blend over [phase_lift, phase_carry] ----
    box_pos = np.zeros((n, 3))
    box_quat = np.zeros((n, 4))
    phase = np.zeros(n, dtype=int)  # 0 ground, 1 blend, 2 carry
    for t in range(n):
        if t < phase_lift:
            box_pos[t], box_quat[t], phase[t] = ground_pos, ground_quat, 0
        elif t < phase_carry:
            s = (t - phase_lift) / max(phase_carry - phase_lift, 1)
            s = s * s * (3 - 2 * s)  # smoothstep
            box_pos[t] = (1 - s) * ground_pos + s * carry_pos[t]
            key = R.from_quat(np.stack([ground_quat, carry_quat[t]]))
            box_quat[t] = Slerp([0, 1], key)(s).as_quat()
            phase[t] = 1
        else:
            box_pos[t], box_quat[t], phase[t] = carry_pos[t], carry_quat[t], 2

    return box_pos, box_quat, phase, hand_mid, hand_rot, body_pos, floor_z


# --------------------------------------------------------------------------- #
# trajectory drawing into a user scene
# --------------------------------------------------------------------------- #

def _cmap(t):
    """simple blue -> red time gradient"""
    c0, c1 = np.array([0.25, 0.55, 1.0]), np.array([1.0, 0.30, 0.25])
    return list((1 - t) * c0 + t * c1) + [0.9]


def draw_trajectory(scene, pts, every=3, radius=0.012,
                    ground_proj=True, box_height=0.3,
                    ground_z=0.0, upto=None):
    """Time-gradient polyline through box centers + dotted ground projection.

    `upto` = highest frame index to draw.  Pass the CURRENT frame (minus a few)
    so the path is drawn as a growing trail: the future part of the path runs
    toward the camera and would otherwise be painted across the carried box's
    front face (it looks exactly like the box is clipping through something).

    Appends user geoms at scene.geoms[scene.ngeom] (the same write-through
    pattern as viewer.user_scn); call AFTER mjv_updateScene filled the scene.
    """
    n_total = len(pts)
    last = (n_total - 1) if upto is None else min(max(upto, 0), n_total - 1)
    idxs = list(range(0, last + 1, every))
    if idxs[-1] != last:
        idxs.append(last)

    def sphere(p, size, rgba):
        g = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(g, type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=[size, size, size], pos=p,
                        mat=np.eye(3).flatten(), rgba=rgba)
        scene.ngeom += 1

    def capsule(pa, pb, width, rgba):
        g = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(g, type=mj.mjtGeom.mjGEOM_CAPSULE,
                        size=[width, 0, 0], pos=(pa + pb) / 2,
                        mat=np.eye(3).flatten(), rgba=rgba)
        mj.mjv_connector(g, type=mj.mjtGeom.mjGEOM_CAPSULE,
                         width=width, from_=pa, to=pb)
        scene.ngeom += 1

    for i in idxs:
        sphere(pts[i], radius, _cmap(i / max(n_total - 1, 1)))
    for a, b in zip(idxs[:-1], idxs[1:]):
        capsule(pts[a], pts[b], radius * 0.5,
                _cmap(((a + b) / 2) / max(n_total - 1, 1)))

    if ground_proj:
        for i in idxs:
            p = pts[i].copy()
            p[2] = ground_z + 0.012
            sphere(p, radius * 0.55, [0.9, 0.9, 0.9, 0.6])


def chordal_mean(Rs):
    """Mean of a stack of rotation matrices: project their arithmetic mean
    back onto SO(3) with an SVD.  This is the standard way to average
    rotations (element-wise averaging of matrices is not a rotation), and it
    is what a STATIC phase should use -- the box does not rotate while it
    sits untouched."""
    M = np.asarray(Rs).sum(axis=0) / len(Rs)
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def trail_cut(box_pos, t, skip_m):
    """Largest index <= t whose box position lies farther than `skip_m` from
    the current one; -1 if the whole trail is still inside the box volume."""
    d = np.linalg.norm(box_pos[:t + 1] - box_pos[t], axis=1)
    idx = np.nonzero(d > skip_m)[0]
    return int(idx[-1]) if len(idx) else -1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

# per-video box parameters. resolution order: built-in defaults < sidecar
# json < explicit CLI flags. The sidecar (<pkl_name>_box.json) is what makes
# new footage double-click-friendly: calibrate once, then reuse forever.
BUILTIN_DEFAULTS = {
    # real box: 300 x 300 mm footprint, 450 mm main body (the two 40 mm top
    # bosses are drawn separately from this size, see the box_boss_* geoms)
    "box_size": [0.30, 0.30, 0.45],
    "phase_lift": 85,
    "phase_carry": 100,
    # box centre in the hand frame (x forward = along the forearms, y right,
    # z up), world metres.  The G1's visible hand is the rubber_hand shell on
    # the wrist chain (wrist_roll -> pitch -> yaw -> rubber_hand).  Its
    # longest fingertip, measured on the shipped --hand_shrink 0.62 mesh
    # along the forearm axis, reaches:
    #     0.205-0.220 m past the wrist_roll_link origin  (the "wrist")
    #     0.121-0.136 m past the wrist_yaw_link origin   (the frame hand_mid
    #                                                     is built from, see
    #                                                     compute_hand_frames)
    #     0.079-0.095 m past the rubber_hand body origin
    # (ranges are left/right hand).  The box is 0.30 m deep, so with too
    # small an x the fingertips punch straight through its front face.  0.12
    # (plus the 0.08 box_clear_x folded into the carry offset) puts the front
    # face just beyond the fingertips and the back face at the palms.
    "box_offset": [0.12, 0.0, 0.05],
    "box_yaw_deg": 0.0,
    "ground_xy": None,
    # ---- grip cosmetics --------------------------------------------------
    # minimum the box CENTRE must sit ahead of the hands along the forearm
    # direction.  The box's own front face already reaches half a box depth
    # (0.15 m) beyond its centre and the shortened hand reaches ~0.12 m, so a
    # small number is plenty; 0.08 leaves a clear gap in front of the finger
    # tips without letting the box drift away from the palms.  Folded into the
    # rigid hand-frame offset (see the clearance pass).
    "box_clear_x": 0.08,
    # Roll both rubber-hand shells about the forearm axis.  This is the
    # PALM-FACING parameter: it spins the palm normal in the plane normal to
    # the forearm, so it decides whether the palm or the BACK of the hand ends
    # up on the box.
    #
    # Default is 0 (stock orientation = palm on the box).  It used to be 85,
    # a value tuned on ONE clip (0911) to stop the shell's thumb poking into a
    # 0.30 m box side.  That roll also turned the PALM OUTWARD, and because
    # batch_pipeline.py passed no --hand_roll_deg, EVERY batch-rendered clip
    # inherited 85 and shipped with the hand back-to-front.  Only the one clip
    # with a hand-tuned run script (165945, run_v29.ps1 -> --hand_roll_deg 0)
    # was correct -- which is exactly the "almost every video has the hands
    # facing the wrong way" report.  Keep 0 as the default; if a specific take
    # needs the thumb lifted, set it per clip (see --hand_roll_left_deg /
    # --hand_roll_right_deg, which can differ because the two shells are
    # mirror images and their optimal rolls are measured ~26-37 deg apart).
    "hand_roll_deg": 0.0,
    # per-hand overrides; None => fall back to hand_roll_deg above
    "hand_roll_left_deg": None,
    "hand_roll_right_deg": None,
    # shorten the rigid hand shell along its finger axis; the stock G1 hand is
    # oversized next to a 0.30 m box and cannot curl, so its finger tips would
    # otherwise cross the box faces.
    "hand_shrink": 0.62,
    # ---- static scene prop: the white four-leg stool ----
    # null => derive its pose from the box's ground pose using the
    # image-measured offset below. Set to [x, y] to pin it absolutely.
    "chair_xy": None,
    # real stool measured off the clip with the 0.30 m box as the ruler:
    # square seat ~0.30, seat top at 0.45 (rendered a touch larger than the
    # box so the box does not sit flush with the seat edges)
    "chair_size": [0.32, 0.32, 0.45],
    # (lateral = image-right, closer = toward camera), REAL metres measured
    # in the source video with the box (0.30 m wide) as the scale ruler.
    "chair_offset": [0.476, 0.902],
    "chair_yaw_deg": None,              # null => parallel to the box
    "chair_visible": True,
}


def resolve_params(args):
    """Merge built-in defaults < sidecar json < explicit CLI flags."""
    sidecar_path = os.path.splitext(args.robot_motion)[0] + "_box.json"
    sidecar = {}
    if os.path.exists(sidecar_path):
        with open(sidecar_path, encoding="utf-8") as fh:
            sidecar = json.load(fh)
        print(f"box params loaded from sidecar: {sidecar_path}")

    eff = {}
    for key, dflt in BUILTIN_DEFAULTS.items():
        # not every key has a CLI flag (chair_visible is sidecar/json-only)
        cli_val = getattr(args, key, None)
        eff[key] = cli_val if cli_val is not None else sidecar.get(key, dflt)

    if args.save_params:
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(eff, fh, indent=2, ensure_ascii=False)
        print(f"box params saved -> {sidecar_path}")
    return eff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_motion", required=True, help="robot motion pkl")
    parser.add_argument("--no_arm_despike", action="store_true",
                        help="disable the rolling-median clamp on the 14 arm "
                             "dofs (GVHMR occlusion artifacts, e.g. the left "
                             "elbow twitch during the carry).")
    parser.add_argument("--no_mirror_left", action="store_true",
                        help="disable mirroring the right arm onto the left "
                             "during the carry phase (--box_pose_world only).")
    parser.add_argument("--mirror_left_end", type=int, default=None,
                        help="end the left-arm mirror at this frame instead "
                             "of the carry end (12-frame ramp-out follows).  "
                             "The mirror spreads the two hands wider than "
                             "the box while the body pivots against the "
                             "yaw-anchored box (~60 deg on 165945), so both "
                             "hands float off the box edges during the "
                             "turn; ending the mirror hands the left arm "
                             "back to the raw retarget, which re-grips "
                             "around the box like the source video.  The "
                             "right-arm place mirror still uses the carry "
                             "end.")
    parser.add_argument("--no_mirror_right_place", action="store_true",
                        help="disable mirroring the left arm onto the RIGHT "
                             "arm after the carry phase: the retargeted "
                             "right hand dives into the placed box (G1's "
                             "shorter reach) while the raw left hand lands "
                             "on the box edge, so post-carry the right arm "
                             "follows the mirrored left arm.")
    parser.add_argument("--carry_abduct_deg", type=float, default=30.0,
                        help="add this much shoulder-ROLL abduction (left "
                             "dof16 += deg, right dof23 -= deg, so both hands "
                             "swing OUTWARD) over --carry_abduct_win.  Why it "
                             "is needed and how the default was chosen: the "
                             "retargeted wrist span (~280 mm) is NARROWER than "
                             "the 300 mm box while the rigid shell reaches "
                             "120-220 mm past the wrist, so the hands sit "
                             "INSIDE the box volume.  Measured on 161727 the "
                             "worst hand-shell vertex penetrates the box by "
                             "82 mm at 0 deg; the shell clears the box at "
                             "27 deg (26 deg still clips 4.1 mm, 27 deg is "
                             "exactly 0) and 30 deg leaves margin.  Cosmetic "
                             "only: the box/clamp math keeps the RAW dofs, so "
                             "the box trajectory is unchanged (verified "
                             "max|d| 0.33 mm, QA PASS).  0 disables.")
    parser.add_argument("--carry_abduct_win", type=int, nargs=2, default=None,
                        metavar=("F0", "F1"),
                        help="frame window for --carry_abduct_deg (12-frame "
                             "smoothstep ramps on both ends).  Default AUTO: "
                             "the measured carry span (first_c..last_c) grown "
                             "by 8 frames on each side, so the ramp-in "
                             "completes by the pickup.  Pass F0 F1 to confine "
                             "it to the collapse window only.")
    parser.add_argument("--carry_grip_slack", type=float, default=0.05,
                        help="--box_pose_world only: how far (m) the carried "
                             "box centre may deviate from the hand midpoint "
                             "(low-passed measured drift). 0 = exactly "
                             "between the hands.")
    parser.add_argument("--carry_grip_slack_walk", type=float, default=None,
                        help="optional second slack (m) for the turn/walk "
                             "part of the carry: from --carry_grip_slack"
                             "_walk_win F0 the drift clamp ramps down to "
                             "this so the box comes back BETWEEN the hands "
                             "while walking (a wide slack leaves the "
                             "wandering retargeted hands drifting around "
                             "the measured box: detached-looking arm, hand "
                             "straddling the camera-facing face).")
    parser.add_argument("--carry_grip_slack_walk_win", type=int, nargs=2,
                        default=None, metavar=("F0", "F1"),
                        help="frame window over which the slack ramps from "
                             "--carry_grip_slack down to "
                             "--carry_grip_slack_walk (smoothstep)")
    parser.add_argument("--carry_body_clear", type=float, default=0.23,
                        help="--box_pose_world only: keep the carried box "
                             "centre at least this far (m, horizontal) from "
                             "the spine capsule so it cannot clip the torso. "
                             "0 disables the keep-out.")
    parser.add_argument("--place_hand_z_follow", type=int, nargs=2,
                        default=None, metavar=("F0", "F1"),
                        help="place-descent window over which the box TOP is "
                             "tied to the actual WRIST-MIDPOINT height, so "
                             "the (shorter) G1 arms keep their hands ON the "
                             "box while the body lowers it -- the measured "
                             "z has the box sinking away below the hands "
                             "during the descent.  Weighted in/out with "
                             "smoothstep ramps and always released onto the "
                             "stool rest height, so the drop-off is "
                             "unchanged.")
    parser.add_argument("--place_hand_z_drop", type=float, default=0.06,
                        help="with --place_hand_z_follow: box top sits this "
                             "far (m) BELOW the wrist midpoint (default "
                             "0.06, the G1 hand length that reaches down to "
                             "the box top).")
    parser.add_argument("--box_size", type=float, nargs=3, default=None,
                        metavar=("W", "D", "H"), help="full box dims in meters")
    parser.add_argument("--phase_lift", type=int, default=None,
                        help="frame when hands reach the box (blend starts)")
    parser.add_argument("--phase_carry", type=int, default=None,
                        help="frame when the box is fully carried")
    parser.add_argument("--box_offset", type=float, nargs=3, default=None,
                        help="box center in hand frame (x forward, y right, z up), meters")
    parser.add_argument("--box_yaw_deg", type=float, default=None,
                        help="box yaw in hand frame, degrees")
    parser.add_argument("--ground_xy", type=float, nargs=2, default=None,
                        help="explicit ground box XY; default auto from grasp pose")
    parser.add_argument("--chair_xy", type=float, nargs=2, default=None,
                        help="absolute world XY of the stool; default: derive "
                             "from the box ground pose + --chair_offset")
    parser.add_argument("--chair_size", type=float, nargs=3, default=None,
                        metavar=("W", "D", "H"),
                        help="stool seat width/depth and total height, metres")
    parser.add_argument("--chair_offset", type=float, nargs=2, default=None,
                        metavar=("DLAT", "DCLOSE"),
                        help="stool position relative to the box, in REAL "
                             "metres in the camera frame: dlat = image-right, "
                             "dclose = toward the camera. Measured with the "
                             "0.30 m box as the scale ruler.")
    parser.add_argument("--chair_yaw_deg", type=float, default=None,
                        help="stool yaw; default follows the box")
    parser.add_argument("--chair_push_out", type=float, default=0.0,
                        metavar="M",
                        help="measured mode only: push the stool this many "
                             "metres AWAY from the robot along the "
                             "robot->box drop direction, so the robot does not "
                             "crouch INTO the stool.  The box's landing spot "
                             "is left alone (it is a measurement); only the "
                             "stool moves.  0 = sit the stool exactly under "
                             "the box (old behaviour, overlaps the legs). "
                             "KEEP M <= 0.010: the box is 0.30 m and the stool "
                             "seat is 0.32 m, so the box seats on the stool only "
                             "while the two are near-concentric -- that leaves "
                             "(0.32 - 0.30) / 2 = 10 mm overhang margin per "
                             "side.  A push-out of M slides the stool M away "
                             "from the box, so any M > 0.010 makes the box "
                             "overhang the seat edge and drop off it.  Values "
                             "above 0.010 are accepted but warned about.")
    parser.add_argument("--chair_min_clear", type=float, default=0.0,
                        metavar="M",
                        help="measured mode only: report the minimum clear "
                             "distance between the stool footprint and the "
                             "robot's leg joints during the drop, and warn if "
                             "it is below M.  0 = off (default).  The 20260917 "
                             "clip measures 0 mm -- the robot faces the stool "
                             "straight on, so its knees sit inside the seat "
                             "footprint; see --chair_push_out.  NOTE the "
                             "trade-off this flag exists to expose: clearance "
                             "and box support are the SAME dial.  The box is "
                             "0.30 m and the seat 0.32 m, so there is only "
                             "(0.32 - 0.30) / 2 = 10 mm of overhang margin per "
                             "side; buying more clearance needs a bigger "
                             "--chair_push_out, and past 0.010 m the box no "
                             "longer sits fully on the seat.  Behaviour when "
                             "the clearance target cannot be met within that "
                             "limit: accept a smaller clearance (or reposition "
                             "the stool by hand with --chair_xy) rather than "
                             "pushing the stool further out.")
    parser.add_argument("--no_chair", action="store_true",
                        help="hide the stool (restores the pre-chair render)")
    parser.add_argument("--box_clear_x", type=float, default=None,
                        help="world m; minimum the box CENTRE must stay ahead "
                             "of the hands along the horizontal forearm "
                             "direction. The G1 hand is one rigid shell whose "
                             "finger tips reach 0.121-0.136 m past the "
                             "wrist_yaw_link origin (0.205-0.220 m past the "
                             "wrist_roll_link origin) along the forearm -- the "
                             "shipped --hand_shrink 0.62 mesh, left/right -- "
                             "while "
                             "the box's own front face already reaches 0.15 m "
                             "past its centre, so a small value (0.05) keeps a "
                             "clear gap. Applied as a single rigid offset over "
                             "the whole carry phase, not per frame.")
    parser.add_argument("--box_traj_world", default=None,
                        help="npz from align_box_to_world.py: drive the box "
                             "with the depth-measured world trajectory "
                             "instead of the synthesized 3-phase one")
    parser.add_argument("--box_pose_world", default=None,
                        help="npz from align_box_to_world.py --box_pose (or "
                             "any file with box_R_world/box_center_world): "
                             "drive the box with the fitted 6-DoF pose - real "
                             "POSITION and real ORIENTATION - on every frame "
                             "that has a measurement.  Frames without one keep "
                             "the hand-rigid fallback.  This supersedes the "
                             "centroid route's two approximations: the "
                             "front-face-centroid offset correction and the "
                             "hard-coded 'upright + one constant yaw' pose "
                             "(which is what --box_clear_x / --hand_roll_deg / "
                             "--hand_shrink exist to paper over).")
    parser.add_argument("--pose_blend", type=int, default=8,
                        help="frames of position/rotation blending at every "
                             "measurement-coverage boundary in "
                             "--box_pose_world mode, so a fallback stretch does "
                             "not snap the box")
    parser.add_argument("--pose_z", choices=("rest", "measured"), default="rest",
                        help="--box_pose_world only: where the box's z comes "
                             "from. 'rest' (default) pins the parked phases to "
                             "floor + half box / stool + half box -- the "
                             "shipped behaviour, kept because the alignment's "
                             "absolute z does not satisfy the stool rest height "
                             "on this clip -- and uses the measured z only "
                             "while carried. 'measured' uses the measured z "
                             "everywhere, which is the honest raw result but "
                             "leaves the box ~0.24 m under its own stool.")
    parser.add_argument("--measured_blend", type=int, default=10,
                        help="frames of smooth blending at parked/carry "
                             "boundaries in measured mode")
    parser.add_argument("--grasp_reach", type=float, default=0.24,
                        help="--box_pose_world only: hold the box on its "
                             "parked spot until the robot's hand midpoint "
                             "comes within this many metres of it "
                             "horizontally, instead of picking the box up on "
                             "the video's schedule. The retargeted G1 reaches "
                             "the box later than the person did, so lifting on "
                             "the measured frame made the box rise and drift "
                             "to a robot whose hands were still ~0.3 m away. "
                             "0 = disable the wait and restore the old "
                             "frame-count behaviour.")
    parser.add_argument("--grasp_dz", type=float, default=0.12,
                        help="--grasp_reach only: extra metres above the box "
                             "top that still count as 'the hands are at the "
                             "box' (a hand passing above it is not a grip).")
    parser.add_argument("--grasp_z_maxstep", type=float, default=0.040,
                        help="--grasp_reach only: cap on the box's per-frame z"
                             " step (metres) through the delayed pickup, so "
                             "the lift starts smoothly from the floor instead "
                             "of popping. Must stay below the QA 50 mm jump "
                             "threshold.")
    parser.add_argument("--meas_smooth", type=int, default=9,
                        help="--box_pose_world only: centered moving-average "
                             "window (frames) applied to the measured box "
                             "POSITION to kill depth/fit jitter. 1 = off.")
    parser.add_argument("--meas_smooth_quat", type=int, default=15,
                        help="--box_pose_world only: same, for the ORIENTATION "
                             "(pitch/yaw wobble); wider than the position "
                             "window because tilt noise is pure fit noise. "
                             "1 = off.")
    parser.add_argument("--meas_median", type=int, default=9,
                        help="--box_pose_world only: odd window of the robust "
                             "median filter applied to the measured "
                             "orientation before smoothing.  The fit lands on "
                             "a wrong local minimum on a large fraction of "
                             "frames; a median (not a mean) removes them.")
    parser.add_argument("--meas_median_spike_only", action="store_true",
                        help="--box_pose_world only: apply the median filter "
                             "ONLY to isolated spikes (a frame whose two "
                             "neighbours agree with each other but not with "
                             "it), preserving genuine monotone ramps such as a "
                             "real box turn.  Without this the filter also "
                             "flattens real rotations.")
    parser.add_argument("--meas_median_spike_thr", type=float, default=25.0,
                        help="with --meas_median_spike_only: the two "
                             "neighbours must be within this many degrees of "
                             "each other for the middle frame to count as an "
                             "isolated spike (default 25 deg).")
    parser.add_argument("--box_sym_snap", choices=["anchor", "continuity"],
                        default="anchor",
                        help="--box_pose_world only: how the cuboid fit's "
                             "symmetry branch is resolved.  'anchor' (default) "
                             "snaps every frame to the symmetry-equivalent "
                             "representative closest to a FIXED anchor (the "
                             "floor-parked median): it removes a spurious "
                             "wrong-branch drift but ALSO folds away a genuine "
                             "90-deg box turn (a square box is identical "
                             "modulo 90 deg, so a real 90-deg turn lands on the "
                             "'same' branch).  'continuity' instead chains the "
                             "branch choice from the previous frame, seeding "
                             "only the first frame from the anchor: the "
                             "trajectory follows the real turn (165945 turns "
                             "~90 deg right at f165-f195 and back by f206) "
                             "while still suppressing per-frame branch flips.")
    parser.add_argument("--carry_lpf", type=int, default=15,
                        help="--box_pose_world only: during the carry phase "
                             "the box is driven by the hand-rigid position "
                             "plus a low-passed measured-vs-hands residual "
                             "(this many frames window), so contact stays "
                             "stable while gross measured drift is kept. "
                             "1 = raw measured position.")
    parser.add_argument("--carry_meas_gain", type=float, default=1.0,
                        help="gain on the low-passed residual above; 0 = pure "
                             "hand-rigid, 1 = full measured gross drift.")
    parser.add_argument("--carry_quat_max_deg", type=float, default=60.0,
                        help="--box_pose_world only: while the box is carried, "
                             "clip the measured orientation to this many "
                             "degrees away from the hand-rigid orientation "
                             "(symmetry-aware). The fit lands on wrong "
                             "branches when the arms occlude the box, which "
                             "renders as the box spinning in the hands. "
                             "0 disables the clip.")
    parser.add_argument("--quat_max_step_deg", type=float, default=3.5,
                        help="--box_pose_world only: maximum box rotation per "
                             "frame (deg). Real motion here is ~0.4 deg/frame; "
                             "the fast steps are fit/retarget noise that shows "
                             "up as the box twisting and swaying. 0 disables.")
    parser.add_argument("--carry_tilt_anchor_deg", type=float, default=0.0,
                        help="--box_pose_world only: during the carry phase, "
                             "keep the box's tilt (its z-axis vs world up) "
                             "within this many degrees of the interpolation "
                             "between the two parked phases' median tilts -- "
                             "a box picked up flat and put down flat is "
                             "carried flat.  Kills the ~15-20 deg sway the "
                             "fit shows under self-occlusion. 0 disables.")
    parser.add_argument("--carry_yaw_anchor_deg", type=float, default=0.0,
                        help="--box_pose_world only: during the carry phase, "
                             "keep the box's yaw within this many degrees of "
                             "the interpolation between the two parked "
                             "phases' median yaws (a square box's yaw is "
                             "wrapped to the nearest 90-deg symmetry branch "
                             "first).  Kills the grasp-moment yaw swing the "
                             "fit shows when the hands occlude the box.  "
                             "0 disables.")
    parser.add_argument("--body_yaw_monotone", action="store_true",
                        help="--box_pose_world only: clamp the unwrapped "
                             "root yaw to the [grasp, post-turn-plateau] "
                             "band, removing a spurious pre-turn excursion "
                             "in the opposite direction of the real turn "
                             "(165945: a -26 deg dip during the stand-up "
                             "that swept the hands across the anchored box) "
                             "while keeping the turn itself untouched.  "
                             "Direction auto-detected.  Default off.")
    parser.add_argument("--carry_orient", choices=["measured", "hand"],
                        default="measured",
                        help="--box_pose_world only: orientation source for "
                             "the CARRIED box.  'measured' follows the depth "
                             "fit (its yaw turns early and wanders under "
                             "self-occlusion); 'hand' drives the carried "
                             "box rigidly from the wrists -- a held box "
                             "turns WITH the person, which is what the "
                             "source video shows (the person pivots to face "
                             "the walking direction and the box rides the "
                             "turn).  Parked phases always use the "
                             "measurement.")
    parser.add_argument("--drop_guard", type=int, default=12,
                        help="--box_pose_world only: ramp the carry z onto "
                             "the stool rest height over this many frames at "
                             "the end of the carry phase, so the box cannot "
                             "clip through the seat at touchdown. 0 = off.")
    parser.add_argument("--parked_freeze_band", type=int, default=8,
                        help="--box_pose_world only: a parked (static) box is "
                             "frozen to its own median pose; this many frames "
                             "next to the carry phase ramp back to the "
                             "measurement so pickup/dropoff stay smooth.")
    parser.add_argument("--dump_box_pose", default=None,
                        help="save the box pose the renderer actually used "
                             "to this npz (QA: static phases must have ~0 "
                             "std)")
    parser.add_argument("--traj_every", type=int, default=3,
                        help="one trajectory dot every N frames")
    parser.add_argument("--traj_skip_m", type=float, default=0.34,
                        help="do not draw trajectory points closer than this "
                             "to the CURRENT box position (world m). The path "
                             "runs through the box's own volume; the pieces "
                             "that cross its faces show up as slivers stuck on "
                             "the box (looks like clipping). 0.34 clears a "
                             "0.30 x 0.30 x 0.45 box's half-diagonal.")
    parser.add_argument("--record_video", default=None,
                        help="offscreen mp4 path; omit for interactive window")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cam_distance", type=float, default=3.2)
    parser.add_argument("--cam_azimuth", type=float, default=None,
                        help="camera position azimuth in deg. Default: auto "
                             "-- derive from R_world_from_cam in "
                             "--box_pose_world so the render views the scene "
                             "from the SAME direction as the source video "
                             "(315 was the 0911 clip's manual calibration; "
                             "each clip's alignment differs, e.g. 165945 "
                             "maps to ~83 deg). Falls back to 315 when the "
                             "npz has no camera rotation.")
    parser.add_argument("--cam_elevation", type=float, default=-16.0)
    parser.add_argument("--save_box_traj", default=None,
                        help="save box_pos/box_quat/phase to this npz")
    parser.add_argument("--save_params", action="store_true",
                        help="save resolved box params next to the pkl "
                             "(<pkl_name>_box.json); later runs auto-load them")
    parser.add_argument("--hand_roll_deg", type=float, default=None,
                        help="rotate both rubber-hand meshes about the forearm "
                             "axis. This is the PALM-FACING parameter: it spins "
                             "the palm normal about the forearm, so it decides "
                             "whether the palm or the BACK of the hand ends up "
                             "on the box. Default 0 = stock orientation = palm "
                             "on the box. The old default of 85 was tuned on one "
                             "clip to stop the shell's thumb poking into the "
                             "box, and it turned the palm OUTWARD; do not "
                             "reinstate it globally.")
    parser.add_argument("--hand_roll_left_deg", type=float, default=None,
                        help="override --hand_roll_deg for the LEFT hand only. "
                             "The two shells are mirror images and their measured "
                             "optimal rolls differ by ~26-37 deg, so a single "
                             "value may not suit both.")
    parser.add_argument("--hand_roll_right_deg", type=float, default=None,
                        help="override --hand_roll_deg for the RIGHT hand only.")
    parser.add_argument("--hand_shrink", type=float, default=None,
                        help="shorten the visible rubber-hand mesh along its "
                             "own finger axis (1.0 = untouched). The G1 hand "
                             "is ONE rigid shell - its fingers cannot curl - "
                             "and it is oversized next to a 0.30 m box, so the "
                             "finger tips cross the box faces. Scaling the mesh "
                             "toward the wrist end removes that overlap without "
                             "pushing the box away from the palms. Applied to "
                             "both hands; 0.7 is a good starting point.")
    parser.add_argument("--ground_z", type=float, default=0.0,
                        help="world z of the floor plane (trajectory shadow)")
    parser.add_argument("--diag_contact", action="store_true",
                        help="temporarily make the box/stool collidable, run "
                             "mj_forward on every frame and print the deepest "
                             "hand<->box and box<->stool interpenetration (mm). "
                             "Use it to tune --box_clear_x / --box_offset "
                             "numerically instead of eyeballing renders.")
    parser.add_argument("--place_root_clear_m", type=float, default=0.0,
                        help="enforce a minimum root-to-box resting-centre "
                             "horizontal distance from a few frames before "
                             "the carry ends to the end of the clip.  The "
                             "retargeted 165945 root ends 0.029 m from the "
                             "box centre (pelvis INSIDE the box footprint -> "
                             "the box swallows the torso up to the chin for "
                             "the last ~1.5 s), and the bend-over descent "
                             "clips the pinned box too.  The deficit is "
                             "smoothed by a (2*12+1) = 25-frame box filter "
                             "(pad = 12), so the correction lands as a slow "
                             "~7 mm/frame foot slide rather than a jump.  "
                             "0.27 = square-box support (~0.16-0.21) + torso "
                             "half depth (~0.09) + margin.")
    parser.add_argument("--place_waist_lift_deg", type=float, default=0.0,
                        help="reduce waist_pitch (dof 14) by up to this many "
                             "degrees from the carry end on, smoothstep over "
                             "RAMP2 = 14 frames.  The retargeted G1 stays folded "
                             "~30 deg after the box lands and the helmet "
                             "grazes into the box top-back corner; lifting "
                             "the torso arcs the head clear (the real person "
                             "is taller, so the same reach needs less fold). "
                             "12-16 is a good starting range.")
    parser.add_argument("--place_bow_deg", type=float, default=0.0,
                        help="keep the waist folded FORWARD by at least this "
                             "many degrees (dof 14, POSITIVE = fold) over "
                             "--place_bow_win, i.e. a FLOOR applied AFTER "
                             "--place_waist_lift_deg.  The source person bends "
                             "right over the box while setting it down "
                             "(dof 14 measured +30 deg on 165945) and the "
                             "retargeted G1 has the same fold -- but "
                             "--place_waist_lift_deg then subtracts ~20 deg "
                             "and stands the torso upright, which both opens "
                             "the helmet clearance AND pulls the hands up and "
                             "back off the box (right hand ends up buried in "
                             "the front face).  This restores the bow so the "
                             "hands ride the box sides/top like the source.")
    parser.add_argument("--place_bow_win", type=int, nargs=2, default=None,
                        help="frame window for --place_bow_deg (12-frame "
                             "smoothstep ramps in and out).  Required with "
                             "--place_bow_deg.")
    parser.add_argument("--place_diag", action="store_true",
                        help="per-frame depth of pelvis/torso/head/wrist body "
                             "centres inside the final box OBB over the "
                             "placement window to the end (uses data.xpos, "
                             "works even though the model registers no "
                             "contacts).  Tune --place_root_clear_m and "
                             "--place_waist_lift_deg with it.")
    parser.add_argument("--hand_diag", action="store_true",
                        help="per-frame TRUE mesh-vertex intersection of both "
                             "rubber hands (and wrist links) vs the box OBB "
                             "over the whole sequence, plus the minimal "
                             "push-out vector at the worst frames.  Read-only "
                             "diagnostic for --hand_push_out_mm.")
    parser.add_argument("--hand_push_out_mm", type=float, default=0.0,
                        help="visually push the rubber-hand geoms out of the "
                             "box along the minimal-penetration direction "
                             "(clamped to this many mm, EMA-smoothed).  "
                             "Visual-only: the skeleton/box trajectories are "
                             "untouched.  0 disables.  Tune with --hand_diag.")
    parser.add_argument("--hand_diag_npz", type=str, default=None,
                        help="optional .npz path: per-frame penetration depth "
                             "(mm) of every hand/wrist mesh, the mesh centre "
                             "in the box frame, and the MTV push vector -- "
                             "for offline frame-level analysis")
    args = parser.parse_args()

    if args.box_pose_world and not args.box_traj_world:
        # the align output is a superset: same parked/carry info, plus the
        # fitted orientation, so it can serve as the measured trajectory too
        args.box_traj_world = args.box_pose_world
        print("--box_pose_world also supplies the measured world trajectory "
              "(superset of --box_traj_world)")

    eff = resolve_params(args)
    print(f"box params: lift={eff['phase_lift']} carry={eff['phase_carry']} "
          f"offset={eff['box_offset']} yaw={eff['box_yaw_deg']} "
          f"size={eff['box_size']}")
    if not args.no_chair and eff.get("chair_visible", True):
        print(f"chair: size={eff['chair_size']} offset={eff['chair_offset']} "
              f"xy={eff['chair_xy']}")
    else:
        print("chair: hidden")

    motion_data, fps, root_pos, root_rot_wxyz, dof_pos, _, _ = load_robot_motion(
        args.robot_motion)

    # ---- remove the spurious pre-turn yaw excursion (opt-in) --------------
    # The retargeted root yaw sometimes executes a spurious excursion in the
    # OPPOSITE direction of the eventual turn.  On this clip: the source
    # video shows the person standing in profile while lifting (f148-190),
    # then pivoting ~60 deg to face the walking direction (f190-212) while
    # RE-GRIPPING around the box -- the box's own world yaw stays ~fixed
    # (its label keeps facing the camera, and the fit's floor->stool medians
    # differ by only 13 deg).  The retargeted root yaw instead DIPS -26 deg
    # during the stand-up (f148-180) before turning +69 deg: the dip sweeps
    # the hands across the anchored box ("hands on different sides at 5 s
    # vs 6 s").  Clamp the unwrapped root yaw to the [grasp, post-turn
    # plateau] band: the excursion is removed while the turn itself is kept
    # untouched; the direction is auto-detected from plateau-vs-grasp.
    # Enable with --body_yaw_monotone.
    if getattr(args, "body_yaw_monotone", False) and \
            getattr(args, "box_pose_world", None):
        wc = np.load(args.box_pose_world, allow_pickle=True)
        carr = np.asarray(wc["carry"]).astype(bool)
        if len(carr) == len(motion_data["root_pos"]) and carr.any():
            i0 = int(np.nonzero(carr)[0][0])
            i1 = int(np.nonzero(carr)[0][-1])
            rr = np.asarray(motion_data["root_rot"], dtype=float)
            Rm = R.from_quat(rr).as_matrix()
            yaw_u = np.unwrap(np.arctan2(Rm[:, 1, 0], Rm[:, 0, 0]))
            y0 = float(yaw_u[i0])
            t0 = min(i1 + 5, len(yaw_u) - 1)
            t1 = min(i1 + 35, len(yaw_u))
            y1 = float(np.median(yaw_u[t0:t1]))
            lo, hi = (y0, y1) if y1 >= y0 else (y1, y0)
            target = np.clip(yaw_u, lo, hi)
            target[:i0] = yaw_u[:i0]      # pre-grasp motion untouched
            delta = target - yaw_u
            if np.abs(delta).max() > np.radians(1.0):
                dq = R.from_euler("z", np.degrees(delta), degrees=True)
                rr_new = (dq * R.from_quat(rr)).as_quat()
                rr_new /= np.linalg.norm(rr_new, axis=1, keepdims=True)
                motion_data["root_rot"] = rr_new
                root_rot_wxyz = np.ascontiguousarray(
                    rr_new[:, [3, 0, 1, 2]])
            print(f"body yaw monotone clamp: grasp {np.degrees(y0):.1f} deg,"
                  f" post-turn plateau {np.degrees(y1):.1f} deg -> removed "
                  f"{np.degrees(np.abs(delta)).max():.1f} deg of spurious "
                  f"pre-turn excursion on "
                  f"{int((np.abs(delta) > 1e-3).sum())} frames (the turn "
                  f"itself is kept)")

    # ---- keep the root OUT of the pinned box (placement + tail) -----------
    # Measured on 165945: the retargeted root ends the clip 0.029 m from the
    # box resting centre -- the pelvis stands INSIDE the box footprint, the
    # box swallows the lower torso up to the chin for the last ~1.5 s, and
    # during the bend-over descent the chest/head clip the pinned box too.
    # Enforce a minimum root-to-resting-centre distance along a fixed ray
    # (frozen at the window start) from a few frames before the carry ends
    # to the end of the clip.  The deficit is smoothed by a (2*12+1) =
    # 25-frame box filter (pad = 12, edge-padded) so the correction becomes
    # a slow backward foot slide (~7 mm/frame) instead of a jump.  No ramp
    # out: holding a ~0.27 m standing distance
    # matches the real person's tail pose better than the retargeted 0.03 m.
    if args.place_root_clear_m > 0 and args.box_pose_world:
        wc = np.load(args.box_pose_world, allow_pickle=True)
        carr = np.asarray(wc["carry"]).astype(bool)
        bc = np.asarray(wc["box_center_world"], dtype=float)
        if len(carr) == len(root_pos) and carr.any():
            last_c = int(np.nonzero(carr)[0][-1])
            fin = np.isfinite(bc).all(axis=1)
            stc = fin & (np.arange(len(fin)) > last_c + 5)
            tgt = np.median(bc[stc, :2], axis=0)   # stool resting xy
            w0 = max(last_c - 8, 0)
            dir0 = root_pos[w0, :2] - tgt
            dir0 = dir0 / (np.linalg.norm(dir0) + 1e-9)
            proj = (root_pos[:, :2] - tgt[None, :]) @ dir0
            need = np.clip(args.place_root_clear_m - proj, 0.0, None)
            need[:w0] = 0.0
            # edge-padded smoothing: constant need stays constant at the tail;
            # 25-frame window spreads the onset slide to ~6 mm/frame
            pad = 12
            kp = np.pad(need, (pad, pad), mode="edge")
            sm = np.convolve(kp, np.ones(2 * pad + 1) / (2 * pad + 1.0),
                             mode="valid")[:len(need)]
            root_pos[:, :2] += sm[:, None] * dir0[None, :]
            print(f"place root keep-out: root held >= "
                  f"{args.place_root_clear_m * 1000:.0f} mm from the box "
                  f"rest xy ({tgt[0]:.3f}, {tgt[1]:.3f}) along "
                  f"({dir0[0]:+.2f}, {dir0[1]:+.2f}) from f{w0} on "
                  f"(max correction {sm.max() * 1000:.0f} mm, smoothed "
                  f"over 25 frames -> <= {np.abs(np.diff(sm)).max() * 1000:.1f} "
                  f"mm/frame foot slide)")

    # ---- straighten the waist once the box lands (head clears the box) ----
    # With the box pinned on the stool the retargeted G1 stays folded
    # ~23-30 deg at the waist (dof 14 measured); the real person folds with
    # the same intent but at 1.615 m tall needs less fold for the same
    # reach, so the G1 helmet ends up grazing ~5 cm into the box top-back
    # corner (f284-302 on 165945).  Reduce waist_pitch (dof 14, POSITIVE =
    # fold forward -- verified by FK probe: -0.2 rad raises head_mocap
    # +0.065 m) from the carry end on, smoothstep-ramped over RAMP2 = 14
    # frames so the release keeps looking natural.  The head arcs up-and-back
    # clear of the box top and the sliding-out hands retract off the box edge.
    # No ramp out: the real person also stands tall through the tail.
    if args.place_waist_lift_deg > 0 and args.box_pose_world and \
            np.asarray(dof_pos).ndim == 2 and dof_pos.shape[1] == 29:
        wc = np.load(args.box_pose_world, allow_pickle=True)
        carr = np.asarray(wc["carry"]).astype(bool)
        if len(carr) == len(dof_pos) and carr.any():
            last_c = int(np.nonzero(carr)[0][-1])
            w0 = max(last_c - 6, 0)
            RAMP2 = 14
            wgt = np.zeros(len(dof_pos))
            wgt[w0:] = 1.0
            for k in range(RAMP2):
                u = (k + 1) / (RAMP2 + 1.0)
                wgt[w0 + k] = u * u * (3.0 - 2.0 * u)
            dof_pos[:, 14] -= np.radians(args.place_waist_lift_deg) * wgt
            print(f"place waist lift: waist_pitch reduced up to "
                  f"{args.place_waist_lift_deg:.0f} deg from f{w0} on "
                  f"(ramp {RAMP2} frames) -> head arcs clear of the box top")

    # ---- restore the setting-down BOW (hands stay on the box) -------------
    # The lift above trades the real bow away for helmet clearance, but that
    # same bow is what carries the hands down the box: with the torso upright
    # the retargeted wrists pull up/back and the right rubber hand ends up
    # ~84 mm inside the FRONT face while the left hovers off the left side.
    # Source video f294-311: the person folds right over the box, both hands
    # on the upper sides.  This applies a FLOOR to dof 14 over the place
    # window (after the lift), so the fold is never straightened below
    # `deg`; the lift still wins wherever the source fold is smaller.
    BOW = float(getattr(args, "place_bow_deg", 0.0) or 0.0)
    if BOW > 0.0 and np.asarray(dof_pos).ndim == 2 and \
            dof_pos.shape[1] == 29:
        win = getattr(args, "place_bow_win", None)
        assert win, "--place_bow_win F0 F1 is required with --place_bow_deg"
        b0, b1 = int(win[0]), int(win[1])
        RAMP3 = 12
        wbo = np.zeros(len(dof_pos))
        wbo[b0:b1 + 1] = 1.0
        for k in range(1, RAMP3 + 1):
            u = 1.0 - k / (RAMP3 + 1.0)
            u = u * u * (3.0 - 2.0 * u)
            if b0 - k >= 0:
                wbo[b0 - k] = u
            if b1 + k < len(wbo):
                wbo[b1 + k] = u
        # floor: never let the weighted fold drop below `BOW` degrees
        fold = np.asarray(dof_pos[:, 14], dtype=np.float64)
        tgt = np.radians(BOW) * wbo
        raised = np.where(fold < tgt, tgt - fold, 0.0)  # deg added
        dof_pos[:, 14] = fold + raised
        print(f"place bow: waist_pitch (dof14) floored at {BOW:.0f} deg over "
              f"f{b0}..{b1} (ramp {RAMP3}); restored up to "
              f"{np.degrees(raised.max()):.1f} deg of the source fold the "
              f"waist-lift had removed ({int((raised > 1e-6).sum())} frames "
              f"touched) -> hands ride the box sides/top")


    # ---- despike the ARM dofs ----------------------------------------------
    # GVHMR occasionally loses an occluded arm and invents a fast joint
    # excursion that is not in the source video.  Measured on this clip: the
    # left elbow bent 4 -> 60 -> -3 deg within 0.7 s during the carry while
    # the person held the box steadily (source frames checked); genuine
    # elbow motion anywhere in the clip stays below 5.9 deg/frame.  The
    # bogus excursion also jolts hand_mid -> the carried box bounces
    # vertically (up to +58 mm on the elbow snap frame).  Fix at the source:
    # clamp every arm dof to a ROLLING-MEDIAN baseline (window 61 -- the
    # bogus bump is shorter than half the window, so the baseline itself is
    # unpolluted) and cap the deviation.  Legs/waist are left alone: the
    # gait and foot contact are trustworthy.
    if not getattr(args, "no_arm_despike", False) and \
            np.asarray(motion_data["dof_pos"]).ndim == 2 and \
            np.asarray(motion_data["dof_pos"]).shape[1] == 29:
        from scipy.ndimage import median_filter
        ARM_DOFS = range(15, 29)      # G1 29-dof: 7 joints x L/R arm
        MW, TH = 61, 12.0             # baseline window / deviation cap (deg)
        MW2 = 51                      # median filter for sub-threshold wobble
        dof_np = np.asarray(motion_data["dof_pos"], dtype=np.float64).copy()
        n_clamped = {}
        for j in ARM_DOFS:
            x = np.degrees(dof_np[:, j])
            base = median_filter(x, size=MW, mode="nearest")
            dev = x - base
            over = np.abs(dev) > TH
            if over.any():
                x[over] = base[over] + np.sign(dev[over]) * TH
                n_clamped[j] = int(over.sum())
            # Second pass: the clamp only removes deviations > TH, but the
            # same occlusion artifact also produces BELOW-threshold
            # dips-and-recoveries (measured: wrist roll snapped 13 deg,
            # wrist yaw dipped 10 deg and returned, the clamped elbow kept
            # a +-10 deg bump).  A median filter is exactly the right tool:
            # it PRESERVES monotone ramps and step edges (all the genuine
            # arm motion shapes) and deletes dips shorter than half the
            # window.  Window 51: the bogus bumps (20-30 frames) are shorter
            # than 25.5, genuine ramps survive untouched.
            x = median_filter(x, size=MW2, mode="nearest")
            dof_np[:, j] = np.radians(x)
        motion_data["dof_pos"] = dof_np
        dof_pos = dof_np
        if n_clamped:
            print("arm-dof despike (rolling-median clamp 61/12 deg + "
                  "median 51): clamped frames per dof: " + ", ".join(
                      f"dof{j}={n}" for j, n in sorted(n_clamped.items())))
        else:
            print("arm-dof despike: clamp found nothing; median 51 applied")

        # ---- mirror the RIGHT arm onto the LEFT during the carry ---------
        # Even despiked, the LEFT arm's retargeted carry pose is visibly
        # wrong (the wrist dorsiflexes so the hand reads as "pried upward"
        # against the box edge; L-vs-mirror(R) deviates up to 24 deg in
        # wrist roll / elbow).  The RIGHT arm renders correctly, and a
        # person carrying a box grips it symmetrically -- so during the
        # carry phase drive the left arm with the mirrored right arm
        # (pitch/elbow keep sign, roll/yaw negate; verified against the
        # medians: mirror(R) matches L within a few deg where L is good).
        # Smooth 12-frame ramps at both ends of the carry phase.
        if getattr(args, "box_pose_world", None) and \
                not getattr(args, "no_mirror_left", False):
            mc = np.load(args.box_pose_world)["carry"].astype(bool)
            if len(mc) == len(dof_np) and mc.any():
                i0, i1 = np.nonzero(mc)[0][[0, -1]]
                i1l = i1
                mle = getattr(args, "mirror_left_end", None)
                if mle is not None:
                    i1l = max(min(int(mle), i1), i0 + 1)
                BW = 12
                w = np.zeros(len(mc))
                w[i0:i1l + 1] = 1.0
                for k in range(1, BW + 1):
                    u = 1.0 - k / (BW + 1.0)
                    u = u * u * (3.0 - 2.0 * u)
                    if i0 - k >= 0:
                        w[i0 - k] = u
                    if i1l + k < len(w):
                        w[i1l + k] = u
                # (left, right, mirror sign): pitch/elbow keep, roll/yaw flip
                for JL, JR, sgn in [(15, 22, 1), (16, 23, -1), (17, 24, -1),
                                    (18, 25, 1), (19, 26, -1), (20, 27, 1),
                                    (21, 28, -1)]:
                    dof_np[:, JL] = ((1.0 - w) * dof_np[:, JL]
                                     + w * sgn * dof_np[:, JR])
                motion_data["dof_pos"] = dof_np
                dof_pos = dof_np
                mle_note = (f" (early end f{i1l} via --mirror_left_end; "
                            f"raw left arm afterwards)"
                            if i1l != i1 else "")
                print(f"left-arm mirror: left arm driven by the mirrored "
                      f"right arm during carry f{i0}..{i1l} "
                      f"(+/-{BW}-frame ramps){mle_note}")

                # ---- mirror the LEFT arm onto the RIGHT after the carry ---
                # At the place phase the retargeted RIGHT hand dives INTO
                # the box (G1's shorter reach stops the hand near the box
                # centre, deepest ~123 mm in 165945) while the raw LEFT
                # hand lands on the box edge outside -- so post-carry
                # drive the right arm with the mirrored left arm instead.
                # The 12-frame ramp overlaps the left-mirror ramp-out:
                # while the left arm is still ~mirror(right), mirror(left)
                # ~ the right arm itself, so the hand only starts moving
                # once the left arm returns to its (good) raw place pose.
                # Disable with --no_mirror_right_place.
                if not getattr(args, "no_mirror_right_place", False):
                    PW = 12
                    w2 = np.zeros(len(mc))
                    if i1 + 1 < len(w2):
                        w2[i1 + 1:] = 1.0
                    for k in range(1, PW + 1):
                        u = 1.0 - k / (PW + 1.0)
                        u = u * u * (3.0 - 2.0 * u)
                        if i1 + 1 - k >= 0:
                            w2[i1 + 1 - k] = u
                    for JL, JR, sgn in [(15, 22, 1), (16, 23, -1),
                                        (17, 24, -1), (18, 25, 1),
                                        (19, 26, -1), (20, 27, 1),
                                        (21, 28, -1)]:
                        dof_np[:, JR] = ((1.0 - w2) * dof_np[:, JR]
                                         + w2 * sgn * dof_np[:, JL])
                    motion_data["dof_pos"] = dof_np
                    dof_pos = dof_np
                    print(f"right-arm mirror at place: right arm driven by "
                          f"the mirrored left arm from f{i1 + 1} on "
                          f"(ramp {PW}) -- right hand clears the placed box")

    # ---- widen the collapsed retargeted grip (shoulder abduction) --------
    # During the lift the retargeted hands pull IN toward the chest (165945:
    # wrist span 213 -> 140 mm while the box is 300 mm wide), so the hands
    # end up physically inside the box volume no matter where the box is
    # anchored -- the box either swallows the arms (hand-midpoint anchor) or
    # the hands dip into its mouth (measured anchor).  This is an AMPLITUDE
    # error of the retargeting (the G1 is 1.32 m vs a 1.615 m human), the
    # same class the left-arm mirror already fixes.  Swing both arms outward
    # about the shoulder roll axes over the collapse window: left dof16 += d,
    # right dof23 -= d keeps the mirrored symmetric grip.  Applied AFTER the
    # mirror so both arms stay symmetric; 12-frame smoothstep ramps keep the
    # velocity continuous (a 25 deg offset over 12 frames = ~2.5 deg/frame,
    # below the QA slew limit).
    ABD = float(getattr(args, "carry_abduct_deg", 0.0) or 0.0)
    # The abduction must stay COSMETIC: it swings the arms out so the rigid
    # hand shells stop stabbing into the box, but it must not move the box
    # itself.  `compute_box_trajectory` anchors the carried box on the hand
    # midpoint (`hand_mid`), which the abduction shifts, so with a single dof
    # array a 35 deg swing moved the delivered box trajectory by up to 80 mm.
    # Hence two arrays: the RAW one feeds the box/clamp/contact math, the
    # abducted one only what is rendered (dof_pos -> set_frame).
    ABD_RAW = None
    if ABD > 0.0:
        win = getattr(args, "carry_abduct_win", None)
        # AUTO window: the measured carry span grown by PAD frames on each
        # side.  The grip is collapsed through the WHOLE carry, not just the
        # lift, so a hand-picked window leaks clipping back in after it ends
        # (measured on 161727: window 148..195 -> 24.9 % vert-in-box, window
        # 140..last_c -> 0.0 %).  The carry mask lives in the measured world
        # npz (`carry`), which --box_pose_world provides.
        PAD = 8
        _cw = None
        if getattr(args, "box_pose_world", None):
            _cw = np.asarray(
                np.load(args.box_pose_world, allow_pickle=True)["carry"],
                dtype=bool)
        if win is None:
            assert _cw is not None and _cw.any(), \
                "--carry_abduct_win F0 F1 is required with --carry_abduct_deg " \
                "when --box_pose_world does not provide a carry mask"
            _ci = np.nonzero(_cw)[0]
            w0 = max(int(_ci[0]) - PAD, 0)
            w1 = min(int(_ci[-1]) + PAD, len(_cw) - 1)
            print(f"abduction window: AUTO -> measured carry "
                  f"f{int(_ci[0])}..{int(_ci[-1])} grown by {PAD} "
                  f"-> f{w0}..{w1}")
        else:
            w0, w1 = int(win[0]), int(win[1])
        RAMP = 12
        dof_np = np.asarray(motion_data["dof_pos"], dtype=np.float64).copy()
        wab = np.zeros(len(dof_np))
        wab[w0:w1 + 1] = 1.0
        for k in range(1, RAMP + 1):
            u = 1.0 - k / (RAMP + 1.0)
            u = u * u * (3.0 - 2.0 * u)
            if w0 - k >= 0:
                wab[w0 - k] = u
            if w1 + k < len(wab):
                wab[w1 + k] = u
        dof_np[:, 16] += wab * np.radians(ABD)
        dof_np[:, 23] -= wab * np.radians(ABD)
        # keep the pre-abduction dofs for everything that must NOT move
        ABD_RAW = np.asarray(motion_data["dof_pos"], dtype=np.float64).copy()
        motion_data["dof_pos"] = dof_np
        dof_pos = dof_np
        print(f"shoulder abduction: left dof16 +{ABD:.0f} deg / right dof23 "
              f"-{ABD:.0f} deg over f{w0}..{w1} (ramp {RAMP}) -- widens the "
              f"collapsed retargeted grip so the hands reach the box sides; "
              f"box/clamp math keeps the RAW dofs so the box trajectory is "
              f"unchanged")

    motion = {
        "root_pos": motion_data["root_pos"],
        "root_rot": motion_data["root_rot"],  # xyzw, for KinematicsModel
        # raw (pre-abduction) dofs: the box anchor follows the hands, so
        # feeding it the abducted arms would move the delivered trajectory
        "dof_pos": (ABD_RAW if ABD_RAW is not None
                    else motion_data["dof_pos"]),
        "box_half_height": eff["box_size"][2] / 2.0,
    }
    box_pos, box_quat_xyzw, phase, hand_mid, hand_rot, body_pos, floor_z = \
        compute_box_trajectory(
            motion, eff["phase_lift"], eff["phase_carry"],
            np.array(eff["box_offset"]), eff["box_yaw_deg"],
            ground_xy=eff["ground_xy"])

    # leg body indices, for the stool-placement clearance check below
    _LEG_NAMES = ("left_knee_link", "left_ankle_roll_link", "left_hip_pitch_link",
                  "right_knee_link", "right_ankle_roll_link",
                  "right_hip_pitch_link")
    try:
        _bn = KinematicsModel(str(ROBOT_XML_DICT["unitree_g1"]), device="cpu").body_names
        _LEG_IDS = [_bn.index(n) for n in _LEG_NAMES if n in _bn]
    except Exception:
        _LEG_IDS = []
    print(f"leg bodies used for the stool clearance check: {len(_LEG_IDS)}")

    # ---- measured-trajectory mode: drive the box with the depth-measured
    # world trajectory (output of scripts/align_box_to_world.py) instead of
    # the synthesized 3-phase one.
    #   * the clip ends with the box placed ON THE STOOL, so the trailing
    #     static segment is rested on the stool top, not on the floor;
    #   * a depth centroid carries no orientation, and the hand frame spins
    #     the box with the wrists -> the box is kept UPRIGHT with one
    #     constant yaw taken from its own net travel direction;
    #   * the stool is then placed under the box's final spot so the box
    #     really lands on it.
    measured_chair_xy = None
    measured_yaw = None
    if getattr(args, "box_traj_world", None):
        md = np.load(args.box_traj_world)
        mpos = md["box_pos_world"]
        mcarry = md["carry"].astype(bool)
        mvalid = md["valid"].astype(bool)
        mscale = float(md["scale"]) if "scale" in md.files else 1.0
        n_f = len(box_pos)
        if len(mpos) != n_f:
            raise SystemExit(
                f"box_traj_world has {len(mpos)} frames but the motion has "
                f"{n_f}; rerun align_box_to_world.py with the same "
                f"--robot_motion pkl")
        idx = np.arange(n_f)
        carry_i_all = np.nonzero(mcarry & mvalid)[0]
        first_c, last_c = int(carry_i_all[0]), int(carry_i_all[-1])
        # `floor_meas` = the frames the MEASUREMENT calls parked on the mat.
        # The render-side mask `on_floor` is set further down, once the
        # grasp-wait gate has decided when the box is really picked up.
        floor_meas = mvalid & (idx < first_c)
        on_stool = mvalid & (idx > last_c)       # parked ON THE STOOL
        # ---- split the two roles of the depth data -----------------------
        # The depth point is the VISIBLE-SURFACE centroid, not the box
        # centre, and frame-to-frame it scatters 10-24 cm against the hands
        # (measured on 20260911).  Driving contact with it is what made the
        # hands sink into the box.  So:
        #   * carry phase  -> box is RIGIDLY ATTACHED to the retargeted
        #                     hands (that is what a carried box does);
        #   * parked phases-> keep the measured spots, after correcting the
        #                     front-face-for-centre bias below.
        p_start = np.median(mpos[floor_meas, :2], axis=0)
        p_end = np.median(mpos[on_stool, :2], axis=0)
        travel = p_end - p_start
        nt = np.linalg.norm(travel)
        travel = travel / nt if nt > 1e-6 else np.array([0.0, 1.0])
        # the parked blob is the FRONT face -> its centre sits half a box
        # depth toward the camera; subtract that to get the box centre
        corr = mscale * eff["box_size"][1] / 2.0 * travel
        start_xy = p_start - corr
        end_xy = p_end - corr
        # box front (local -y) faces the camera = along the travel direction
        yaw_lock = float(np.arctan2(travel[0], -travel[1]))
        box_quat_xyzw = np.repeat(
            R.from_euler("z", yaw_lock).as_quat()[None, :], n_f, axis=0)

        # a few mm of clearance kills the coplanar-face z-fighting between
        # the box bottom and the floor / the stool seat
        clear = 0.005
        rest_floor = (floor_z + motion["box_half_height"] + clear)
        rest_stool = (floor_z + eff["chair_size"][2]
                      + motion["box_half_height"] + clear)

        # ---- the GRASP WAIT -------------------------------------------
        # The retargeted G1 reaches the box LATER than the person in the
        # video did.  Measured on 20260922: the box (depth-measured, so
        # truthful) starts leaving the floor at f148, while the robot's hand
        # midpoint is still 0.32 m short of its parked spot horizontally --
        # the hands only come level with it around f166.  Rendering the
        # measured lift on the measured schedule therefore shows the box
        # rising and drifting toward a robot whose hands are visibly NOT on
        # it: the robot "squats down before it has reached the box".
        # So the box WAITS.  It is held on its parked spot until the hands
        # have actually arrived -- horizontally within --grasp_reach of the
        # parked centre, and low enough to be gripping rather than passing
        # above the box -- and only then is it handed over to the rigid grip
        # below.  The measured pickup frame `first_c` still drives every
        # measurement-side statistic (parked medians, anchors, the stool).
        # --grasp_reach 0 disables the wait (old frame-count behaviour).
        carry_start = first_c
        if args.grasp_reach > 0 and floor_meas.sum() > 3:
            half_h = eff["box_size"][2] / 2.0
            # Search from the measured pickup onward only: the wait may DELAY
            # a pickup that the robot cannot make in time, never advance it
            # (frames before first_c have the hands swinging past the parked
            # spot on the way in, which would grab early).
            for t in range(first_c, min(first_c + 90, last_c) + 1):
                dxy = float(np.hypot(*(hand_mid[t, :2] - start_xy)))
                dz = float(hand_mid[t, 2] - (rest_floor + half_h))
                if dxy <= args.grasp_reach and dz <= half_h + args.grasp_dz:
                    carry_start = t
                    break
            if carry_start != first_c:
                print(f"grasp wait: the box stays parked until f{carry_start} "
                      f"-- {carry_start - first_c} frames after the measured "
                      f"pickup at f{first_c}; the retargeted hands only come "
                      f"within {args.grasp_reach:.2f} m of the parked spot "
                      f"there (hand-to-parked-spot "
                      f"{np.hypot(*(hand_mid[carry_start, :2] - start_xy)):.3f}"
                      f" m horizontally, "
                      f"{hand_mid[carry_start, 2] - rest_floor:+.3f} m "
                      f"vertically)")
            else:
                print("grasp wait: the hands are already at the box on the "
                      "measured pickup frame -- no wait needed")
        on_floor = mvalid & (idx < carry_start)   # parked on the mat
        carry_i = np.nonzero(mcarry & mvalid & (idx >= carry_start))[0]
        off_carry = np.array(eff["box_offset"])
        b = max(args.measured_blend, 1)
        tgt = box_pos.copy()
        tgt[carry_i] = hand_mid[carry_i] + np.einsum(
            "nij,j->ni", hand_rot[carry_i], off_carry)
        if on_floor.any():
            tgt[on_floor] = np.r_[start_xy, rest_floor]
        if on_stool.any():
            tgt[on_stool] = np.r_[end_xy, rest_stool]

        # ---- clearance ------------------------------------------------
        # The G1's visible hand mesh is a single RIGID shell whose longest
        # finger tip reaches 0.205-0.220 m past the wrist_roll_link origin
        # along the forearm -- equivalently 0.121-0.136 m past the
        # wrist_yaw_link origin that hand_mid is built from -- measured on
        # the shipped --hand_shrink 0.62 mesh (left/right hand).  It cannot
        # curl around the box.
        # So the box has to sit far enough ahead of the hands that the
        # fingertips never emerge through its front face.  Crucially the
        # push is folded into the *rigid* hand-frame offset (constant over
        # the whole carry phase) instead of being applied per frame - a
        # per-frame push would let the box slide in the hands and read as
        # jitter.  One push value = max shortfall over the carry phase.
        if eff["box_clear_x"] and len(carry_i):
            # only the *pure carry* frames may drive this: the blend frames
            # interpolate between the parked spot and the grip, so they
            # legitimately hold the box far from the hands and would otherwise
            # inflate the offset hugely.  And take the MEDIAN shortfall, not
            # the max: a couple of frames where the wrist is cocked would
            # otherwise shove the whole box far ahead of the palms.
            short = []
            for t in carry_i:
                xh = hand_rot[t][:, 0].copy()
                xh[2] = 0.0
                nx = np.linalg.norm(xh)
                if nx < 1e-6:
                    continue
                xh /= nx
                along = float((tgt[t] - hand_mid[t]) @ xh)
                short.append(eff["box_clear_x"] - along)
            need = max(0.0, float(np.median(short))) if short else 0.0
            if need > 1e-4:
                off_carry[0] += need
                tgt[carry_i] = hand_mid[carry_i] + np.einsum(
                    "nij,j->ni", hand_rot[carry_i], off_carry)
                print(f"clearance: rigid offset x {eff['box_offset'][0]:.3f} "
                      f"-> {off_carry[0]:.3f} m (box stays rigid in the hands)")

        # smooth blend across both parked<->carry boundaries
        for t0 in (carry_start, last_c):
            lo, hi = max(t0 - b, 0), min(t0 + b + 1, n_f - 1)
            a, c = tgt[lo].copy(), tgt[hi].copy()
            for t in range(lo, hi + 1):
                w = (t - lo) / max(hi - lo, 1)
                w = w * w * (3 - 2 * w)
                tgt[t] = (1 - w) * a + w * c

        box_pos = tgt
        measured_yaw = yaw_lock
        if on_stool.any():
            measured_chair_xy = end_xy

        print(f"measured mode: {args.box_traj_world}  "
              f"carry {carry_start}..{last_c} (rigid to hands, offset "
              f"{np.round(off_carry, 3)})  floor {on_floor.sum()} f "
              f"@ z={rest_floor:+.3f}  stool {on_stool.sum()} f "
              f"@ z={rest_stool:+.3f}  yaw {np.degrees(yaw_lock):+.1f} deg  "
              f"start XY {np.round(start_xy, 3)}  end XY "
              f"{np.round(end_xy, 3)}  blend {args.measured_blend}")

        # ---- optional: real 6-DoF pose (position AND orientation) --------
        # --box_pose_world replaces the two approximations above wherever the
        # cuboid fit produced a measurement:
        #   * position: the fitted box CENTRE, so the front-face-centroid
        #     correction (`corr`) is not needed and is not applied;
        #   * orientation: the fitted R, so the box is no longer forced
        #     upright with one constant yaw.
        # Everything above still runs, because it provides the fallback for
        # the frames without a measurement (plus the parked medians and the
        # stool placement).
        if getattr(args, "box_pose_world", None):
            pR = md["box_R_world"] if "box_R_world" in md.files else None
            pc = (md["box_center_world"] if "box_center_world" in md.files
                  else None)
            if pR is None or pc is None:
                raise SystemExit(
                    f"{args.box_pose_world} has no box_R_world/"
                    f"box_center_world. Re-run align_box_to_world.py with "
                    f"--box_pose <npz from box_traj_from_depth.py "
                    f"--fit-cuboid>.")
            if len(pc) != n_f:
                raise SystemExit(
                    f"{args.box_pose_world} has {len(pc)} frames but the "
                    f"motion has {n_f}")
            cov = (np.isfinite(pc).all(axis=1)
                   & np.isfinite(pR).all(axis=(1, 2)))
            if "pose_valid" in md.files:
                cov &= md["pose_valid"].astype(bool)

            # ---- snap the symmetry branch CONTINUOUS before anything else --
            # The cuboid fit resolves the pose only modulo the box's symmetry
            # group (square section: 4 yaws x 2 flips, order 8).  When the
            # arms occlude the box the solver can re-lock onto an equivalent
            # branch; on this clip the yaw drifted a full 180 deg during the
            # pickup (f148-f204 sit ~170 deg from the floor median while the
            # tilt stays < 10 deg) and stayed there for the whole carry.  A
            # 180-deg yaw is INVISIBLE on a square box -- but an absolute
            # orientation timeline shows it as a giant rotation, and no filter
            # removes it because the measurement genuinely rests on the
            # flipped branch for tens of frames.  Fix: snap every frame to
            # the symmetry-equivalent representative closest to a FIXED
            # anchor (the floor-parked median).
            # Real motion is slow (median 0.4 deg/frame), so the nearest
            # branch is always the right one and a branch flip contributes
            # ~0 deg instead of ~180.  The reference must be a FIXED anchor
            # (the floor-parked median), NOT the previous frame: the fit's
            # wrong-branch yaw drifts CONTINUOUSLY (deltas < 45 deg), so a
            # chained previous-frame reference would just follow the drift
            # branch by branch.
            syms8 = []
            for k in range(4):
                Rz = R.from_euler("z", 90.0 * k, degrees=True)
                syms8.append(Rz)
                syms8.append(Rz * R.from_euler("x", 180.0, degrees=True))
            anchor = floor_meas & cov
            if anchor.sum() > 3:
                Mi = sum(R.from_matrix(pR[j]).as_matrix()
                         for j in np.nonzero(anchor)[0])
                U, _, Vt = np.linalg.svd(Mi)
                R_anchor = R.from_matrix(U @ Vt)
            else:
                R_anchor = R.from_matrix(pR[np.nonzero(cov)[0][0]])
            snap_mode = str(getattr(args, "box_sym_snap", "anchor"))
            if snap_mode == "continuity":
                # Chain the branch choice from the previous (already snapped)
                # frame, seeded by the anchor.  This tracks a GENUINE box turn
                # (a real 90-deg yaw lands on an equivalent branch and would be
                # folded away by a fixed anchor) while still removing the
                # isolated per-frame branch flips the fit makes under
                # self-occlusion -- those show up as a single frame far from
                # its neighbour, which the previous-frame reference rejects.
                n_snap = 0
                ref = R_anchor
                prev_j = None
                for j in np.nonzero(cov)[0]:
                    Rj = R.from_matrix(pR[j])
                    angs = [np.degrees((ref.inv() * Rj * S_).magnitude())
                            for S_ in syms8]
                    k = int(np.argmin(angs))
                    if k:
                        Rj = Rj * syms8[k]
                        n_snap += 1
                    pR[j] = Rj.as_matrix()
                    # only advance the reference on a CONTIGUOUS frame; a gap
                    # keeps the last known branch (the fit just has no data).
                    if prev_j is not None and j == prev_j + 1:
                        ref = Rj
                    elif prev_j is None:
                        ref = Rj
                    prev_j = j
                if n_snap:
                    print(f"  symmetry branch snap (continuity): {n_snap} "
                          f"of {int(cov.sum())} frames re-expressed on a "
                          f"symmetry-equivalent branch, chained from the "
                          f"previous frame so a genuine box turn is preserved")
                else:
                    print(f"  symmetry branch snap (continuity): 0 of "
                          f"{int(cov.sum())} frames needed re-expressing -- "
                          f"the measured branch is already continuous, so a "
                          f"genuine box turn (a real 90-deg yaw lands on an "
                          f"equivalent branch) is followed rather than folded")
            else:
                n_snap = 0
                for j in np.nonzero(cov)[0]:
                    Rj = R.from_matrix(pR[j])
                    angs = [np.degrees((R_anchor.inv() * Rj * S_).magnitude())
                            for S_ in syms8]
                    k = int(np.argmin(angs))
                    if k:
                        Rj = Rj * syms8[k]
                        n_snap += 1
                    pR[j] = Rj.as_matrix()
                if n_snap:
                    print(f"  symmetry branch snap (anchor): {n_snap} of "
                          f"{int(cov.sum())} frames re-expressed on a "
                          f"symmetry-equivalent branch to keep the orientation "
                          f"sequence continuous")

            # ---- temporal smoothing: kill depth/fit jitter -----------------
            # The measured pos/quat carry per-frame depth noise that reads as
            # box wobble in the render.  Centered moving average over covered
            # frames (NaN-aware so gaps fall out); rotations averaged as
            # rotvecs, which is valid for the small frame-to-frame deltas.
            def _nan_ma(a, w):
                """Centered moving average along axis 0, ignoring NaNs."""
                out = np.full_like(a, np.nan)
                hh = w // 2
                for i in range(len(a)):
                    seg = a[max(0, i - hh):i + hh + 1]
                    fin = np.isfinite(seg).all(axis=-1)
                    if fin.any():
                        out[i] = seg[fin].mean(axis=0)
                return out

            S = max(int(getattr(args, "meas_smooth", 9)), 1)
            SQ = max(int(getattr(args, "meas_smooth_quat", 15)), 1)
            if S > 1:
                pc_s = _nan_ma(pc.copy(), S)
                cov_s = np.isfinite(pc_s).all(axis=1) & cov
                pc = np.where(cov_s[:, None], pc_s, pc)
                print(f"  measured position smoothed (window {S} frames)")
            if SQ > 1:
                # Orientation is where the fit noise shows up as pitch/yaw
                # wobble (the box just off the floor is the worst: only the
                # front face is well constrained), and unlike position the
                # user wants no wobble here at all -> its own, wider window.
                rv_full = np.full((n_f, 3), np.nan)
                rv_full[cov] = R.from_matrix(pR[cov]).as_rotvec()
                # Unwrap the rotvec REPRESENTATION along the timeline first.
                # scipy returns angles in [0, pi], so a rotation hovering near
                # 180 deg flips between +pi and -pi: two neighbours that are
                # physically 6 deg apart become 354 deg apart in components.
                # Averaging/median-filtering across that flip produces a
                # completely different rotation -- this is what made the box
                # twist in place while it sat untouched.  Replace any frame
                # whose "long way round" representative (angle - 2*pi) is
                # closer to its predecessor with that representative.
                sel0 = np.nonzero(cov)[0]
                rv = rv_full[sel0]
                th = np.linalg.norm(rv, axis=1, keepdims=True)
                alt = rv * np.where(th > 1e-9, (th - 2.0 * np.pi)
                                    / np.maximum(th, 1e-9), 1.0)
                n_flip = 0
                for i in range(1, len(rv)):
                    if np.linalg.norm(alt[i] - rv[i - 1]) < \
                            np.linalg.norm(rv[i] - rv[i - 1]):
                        rv[i] = alt[i]
                        n_flip += 1
                rv_full[sel0] = rv
                if n_flip:
                    print(f"  orientation rotvec unwrapped: {n_flip} frames "
                          f"were on the far side of +/-180 deg and were "
                          f"re-expressed continuously")
                # Robust pass: a minority of frames land on a different
                # (NOT symmetry-equivalent) local minimum tens of degrees
                # away.  The align step already made the symmetry branch
                # continuous, so what is left are isolated spikes -> remove
                # them with a short component-wise MEDIAN filter before the
                # averaging below (an average would smear a spike into its
                # neighbours instead of deleting it).
                sel = np.nonzero(cov)[0]
                if len(sel) >= 5:
                    rv = rv_full[sel]
                    k = max(int(getattr(args, "meas_median", 9)), 3)
                    if k % 2 == 0:
                        k += 1
                    pad = np.pad(rv, ((k // 2, k // 2), (0, 0)), mode="edge")
                    rv_med = np.stack([np.median(pad[i:i + k], axis=0)
                                       for i in range(len(rv))])
                    dev = np.degrees(np.linalg.norm(rv_med - rv, axis=1))
                    # ---- spike-vs-ramp decision --------------------------
                    # The blind median filter above flattens ANY monotone
                    # ramp: during a genuine fast turn every frame deviates
                    # from its short-window median, so the filter deletes the
                    # whole rotation (this is what made 165945's real box
                    # turn vanish).  Only replace a frame that is an ISOLATED
                    # spike -- its two neighbours agree with EACH OTHER but
                    # the frame sits far from both.  A ramp frame has
                    # neighbours that disagree with each other just as much,
                    # so it is preserved.  Opt-in via --meas_median_spike_only
                    # so the legacy behaviour is unchanged by default.
                    spike_only = bool(getattr(args, "meas_median_spike_only",
                                              False))
                    if spike_only and len(rv) >= 3:
                        nbr = np.degrees(np.linalg.norm(rv[1:-1] - rv[:-2],
                                                        axis=1))
                        nbr2 = np.degrees(np.linalg.norm(rv[2:] - rv[1:-1],
                                                         axis=1))
                        # neighbour-to-neighbour closeness
                        nn = np.degrees(np.linalg.norm(rv[2:] - rv[:-2],
                                                       axis=1))
                        thr = float(getattr(args, "meas_median_spike_thr",
                                            25.0))
                        is_spike = (nn < thr) & \
                            (dev[1:-1] > 5.0) & \
                            (nbr > 5.0) & (nbr2 > 5.0)
                        mask = np.zeros(len(rv), bool)
                        mask[1:-1] = is_spike
                        new_rv = rv.copy()
                        new_rv[mask] = rv_med[mask]
                        n_rep = int(mask.sum())
                        print(f"  orientation median filter (window {k}, "
                              f"spike-only thr {thr:.0f} deg): replaced "
                              f"{n_rep} ISOLATED spikes of {len(sel)} frames "
                              f"(ramps preserved; max dev {dev.max():.1f} deg)")
                        rv_full[sel] = new_rv
                    else:
                        rv_full[sel] = rv_med
                        print(f"  orientation median filter (window {k}): "
                              f"{int((dev > 5.0).sum())} of {len(sel)} frames "
                              f"deviated >5 deg from their local median and "
                              f"were replaced (max dev {dev.max():.1f} deg)")
                rv_s = _nan_ma(rv_full, SQ)
                ok = cov & np.isfinite(rv_s).all(axis=1)
                pR_new = pR.copy()
                pR_new[ok] = R.from_rotvec(rv_s[ok]).as_matrix()
                pR = pR_new
                print(f"  measured orientation smoothed (window {SQ} frames)")

            fallback_pos = box_pos.copy()
            fallback_quat = box_quat_xyzw.copy()

            # ---- anchor z on the two hard physical facts available ---------
            # The Umeyama alignment carries a systematic absolute-z bias that
            # is NOT a cuboid-fit error (see the checks below).  One constant
            # anchor on the FLOOR phase leaves the box sinking ~0.19 m too
            # low by the time it is lowered onto the stool (the bias grows
            # with real height) -- the box visibly clipped through the stool
            # during the drop and only sat right on the last frame.  Anchor
            # BOTH parked phases and interpolate the correction linearly over
            # the carry phase: bias(t) = (1-u)*bias_floor + u*bias_stool.
            z_bias_f = 0.0
            z_bias_s = 0.0
            fl = floor_meas & cov
            st = on_stool & cov
            if fl.sum() > 3:
                z_bias_f = float(np.median(pc[fl, 2]) - rest_floor)
                print(f"  z anchor: parked-on-floor centre measured at world z "
                      f"{np.median(pc[fl, 2]):+.3f}, but it rests at "
                      f"{rest_floor:+.3f} -> bias {z_bias_f * 1000:+.0f} mm "
                      f"({int(fl.sum())} frames)")
            else:
                print("  z anchor: no covered frame in the parked-on-floor "
                      "phase, so the alignment's absolute z is left as-is")
            if st.sum() > 3:
                z_bias_s = float(np.median(pc[st, 2]) - rest_stool)
                print(f"  z anchor: parked-on-stool centre measured at world z "
                      f"{np.median(pc[st, 2]):+.3f}, but it rests at "
                      f"{rest_stool:+.3f} -> bias {z_bias_s * 1000:+.0f} mm "
                      f"({int(st.sum())} frames)")
            pc = pc.copy()
            carry_ph = (idx >= carry_start) & (idx <= last_c)
            if st.sum() > 3 and carry_ph.any():
                u = np.clip((idx - carry_start) / max(last_c - carry_start, 1),
                            0.0, 1.0)
                # carry phase: interpolate floor->stool bias.
                # BEFORE the carry: floor bias.  AFTER it: STOOL bias, so a
                # parked-on-stool frame ends up sitting at its physical rest
                # height here too.  (The default --pose_z rest pins the parked
                # phases anyway, but using z_bias_f after the carry left this
                # intermediate value 189 mm low and made the "independent
                # check" below print a bogus error.)
                bias = np.where(carry_ph,
                                (1.0 - u) * z_bias_f + u * z_bias_s,
                                np.where(idx > last_c, z_bias_s, z_bias_f))
                print(f"  z anchor: TWO-anchor mode -- bias interpolated "
                      f"{z_bias_f * 1000:+.0f} mm (floor) -> "
                      f"{z_bias_s * 1000:+.0f} mm (stool) across the carry "
                      f"phase; parked phases get their rest heights anyway")
            else:
                bias = np.full(n_f, z_bias_f)
            pc[:, 2] -= bias

            # ---- which z to trust where -------------------------------------
            # The measured z is NOT consistent with the render world's object
            # metric: after anchoring, the box on the floor sits right, but the
            # box parked on the stool comes out ~0.24 m BELOW floor + stool +
            # half box, with only a 10 mm spread inside that phase.  Checked
            # against the old route: the 3-DoF run shows the same class of
            # absolute-z problem (+0.338) but never exposes it because it
            # SYNTHESISES the parked z.  Two candidate root causes remain, and
            # neither is decidable from this clip alone:
            #   * the world is not uniformly metric -- the Umeyama scale
            #     (s~0.71, "G1 world is 1.4x real metres") is applied to the
            #     box's TRANSLATION while the box/stool GEOMETRY is drawn at
            #     real sizes, so the box's own 0.45 m is compressed in z;
            #   * GVHMR translation-scale drift that the carry fit cannot see
            #     in the vertical direction.
            # Default is therefore the safe one: parked on the floor / on the
            # stool -> z from the physical rest condition (exactly what the
            # shipped route does), carried -> measured z, anchored so the two
            # meet continuously at pickup.  Use --pose_z measured to see the
            # raw measurement instead.
            carry_ph = (idx >= carry_start) & (idx <= last_c)
            if args.pose_z == "rest":
                pc_use = pc.copy()
                park = cov & ~carry_ph
                if on_floor.any():
                    pc_use[park & on_floor, 2] = rest_floor
                if on_stool.any():
                    pc_use[park & on_stool, 2] = rest_stool
                print(f"  z policy: parked phases pinned to the physical rest "
                      f"heights (floor {rest_floor:+.3f} / stool "
                      f"{rest_stool:+.3f}), carry phase uses the measured z "
                      f"(anchored).  --pose_z measured to override.")
            else:
                pc_use = pc.copy()
                print("  z policy: measured z everywhere (raw measurement, "
                      "see the stool check below for why that is not the "
                      "default)")

            # ---- the grasp wait leaves the box ON THE FLOOR at carry_start --
            # With the wait the box is still parked when the carry begins,
            # while the anchored measurement already has it ~0.15 m up (the
            # video's person had started lifting by then).  Taking the measured
            # z on that one frame popped the box up 127 mm; ramping the
            # measurement in over a short band instead made the lift hit
            # 64 mm/frame.  Pin the floor height for this single frame and let
            # the rate limiter after the grip ramp take the lift from there.
            # (The step INTO the carry stays 0, so the parked-phase statistics
            # the QA script checks are untouched.)
            if carry_start != first_c:
                pc_use[carry_start, 2] = rest_floor

            # ---- carry contact: low-pass the measured-vs-hands residual ----
            # The measured centre sits a retargeting+Sim3-error (~0.2-0.3 m)
            # away from the hand-rigid convention, and that residual jitters
            # frame to frame -> the box visibly slides on the hands.  During
            # the carry phase drive the box with the hand-rigid position plus
            # a LOW-PASSED version of the residual: gross measured drift is
            # kept, the fast hand-relative error is gone -> stable contact,
            # no wobble.
            LW = max(int(getattr(args, "carry_lpf", 15)), 1)
            G = float(getattr(args, "carry_meas_gain", 1.0))
            RG = float(getattr(args, "carry_grip_slack", 0.05))
            RB = float(getattr(args, "carry_body_clear", 0.23))
            # optional per-phase slack: after the lift the wandering
            # retargeted hands drift around the MEASURED box (detach look);
            # ramp the clamp down so the box returns between the hands for
            # the turn/walk (the v13-approved grip look)
            RGW = getattr(args, "carry_grip_slack_walk", None)
            RGW_win = getattr(args, "carry_grip_slack_walk_win", None)
            RG_t = np.full(n_f, RG)
            RGW_note = ""
            if RGW is not None:
                assert RGW_win, "--carry_grip_slack_walk_win F0 F1 is " \
                                "required with --carry_grip_slack_walk"
                w0w, w1w = int(RGW_win[0]), int(RGW_win[1])
                uu = np.clip((np.arange(n_f) - w0w)
                             / max(w1w - w0w, 1), 0.0, 1.0)
                uu = uu * uu * (3.0 - 2.0 * uu)
                RG_t = RG * (1.0 - uu) + float(RGW) * uu
                RGW_note = (f"; walk slack {float(RGW)*1000:.0f} mm "
                            f"ramped over f{w0w}..{w1w}")
            if carry_ph.any() and cov.any():
                # ---- grip clamp: keep the box STRICTLY between the hands --
                # The measured box centre sits 0.17-0.40 m from the hand
                # midpoint while the wrists are only ~0.35 m apart, so the
                # rendered box buried itself in the hands and the torso
                # (centre-to-spine down to 77 mm < the 212 mm box
                # half-diagonal).  A held box's centre belongs at the hand
                # midpoint: anchor it there and keep only a SMALL, low-passed
                # share of the measured drift (clamped into a ball of radius
                # `carry_grip_slack`) so gross measured motion survives but
                # contact can never break.
                selg0 = np.nonzero(carry_ph)[0][0]
                selg1 = np.nonzero(carry_ph)[0][-1]
                # NO pre-band: before the grasp the retargeted hands are
                # still HIGH (0.56 m vs the box resting at 0.22) -- blending
                # the grip in before the carry began lifted the box into the
                # air ahead of the hands, and it sagged back down as the hands
                # descended (a bump).  Instead ramp the grip in over the first
                # 25 carry frames, which coincide with the hand descent: the
                # box waits on the floor and starts following the hands as
                # they reach it.  The tail ramps out over 12 frames into the
                # drop-guarded touchdown.  (`carry_ph` above now starts at the
                # GRASP-WAIT frame, so this ramp begins when the hands have
                # actually arrived at the box, not on the video's schedule.)
                BW0, BW1 = 25, 12
                wcl = np.zeros(n_f)
                wcl[selg0 + BW0:selg1 + 1] = 1.0
                for jj, ii in enumerate(range(selg0, selg0 + BW0)):
                    uu = (jj + 1) / (BW0 + 1.0)
                    wcl[ii] = uu * uu * (3.0 - 2.0 * uu)
                for k in range(1, BW1 + 1):
                    uu = 1.0 - k / (BW1 + 1.0)
                    uu = uu * uu * (3.0 - 2.0 * uu)
                    if selg1 + k < n_f:
                        wcl[selg1 + k] = uu
                pre = pc_use.copy()
                res = pc - hand_mid                  # NaN where uncovered
                res_s = _nan_ma(res, LW)
                sel = ((wcl > 0) & cov
                       & np.isfinite(res_s).all(axis=1))
                edrift = G * res_s[sel]
                dn = np.linalg.norm(edrift, axis=1)
                rg_sel = RG_t[sel]
                over = dn > rg_sel
                edrift[over] *= (rg_sel[over]
                                 / np.maximum(dn[over], 1e-9))[:, None]
                grip = hand_mid[sel] + edrift                # ---- body keep-out: horizontal push out of the spine -----
                # Spine capsule pelvis -> torso (+0.2 m up to the shoulders);
                # the box's horizontal half-diagonal is ~0.212 m, so anything
                # closer than `carry_body_clear` to the spine clips the chest.
                if RB > 0:
                    _bn = KinematicsModel(
                        str(ROBOT_XML_DICT["unitree_g1"]),
                        device="cpu").body_names
                    pelB = body_pos[:, _bn.index("pelvis")]
                    torB = body_pos[:, _bn.index("torso_link")]
                    top = torB + 0.20 * (torB - pelB) / (
                        np.linalg.norm(torB - pelB, axis=1, keepdims=True)
                        + 1e-9)
                    ab = top - pelB
                    tt = np.clip(np.sum((grip - pelB[sel]) * ab[sel], axis=1,
                                        keepdims=True)
                                 / (np.sum(ab[sel] * ab[sel], axis=1,
                                           keepdims=True) + 1e-9), 0.0, 1.0)
                    proj = pelB[sel] + tt * ab[sel]
                    dh = grip[:, :2] - proj[:, :2]
                    dhn = np.linalg.norm(dh, axis=1, keepdims=True) + 1e-9
                    push = np.clip(RB - dhn, 0.0, None)
                    push3 = np.concatenate(
                        [push * dh / dhn, np.zeros((len(push), 1))], axis=1)
                    grip = grip + wcl[sel][:, None] * push3
                    if (push > 1e-4).any():
                        print(f"  body keep-out: pushed the box out of the "
                              f"spine capsule on {int((push > 1e-4).sum())} "
                              f"carry frames (max {push.max()*1000:.0f} mm)")
                # ---- stool capture -----------------------------------------
                # In the final approach the retargeted person bobs the hands
                # UP ~0.19 m before the last descent (checked the render:
                # box z 0.661 at f420 -> 0.848 at f438 -> 0.666).  Anchoring
                # the box to the hands made it ride up with the body and sag
                # back.  A stool is solid: once the box reaches the stool-top
                # level inside the release zone it STAYS there -- z is pinned
                # while the hands finish their bob above it.
                idxs = np.nonzero(sel)[0]
                zone = idxs[idxs >= selg1 - 60]
                if len(zone):
                    gpos = np.searchsorted(idxs, zone)
                    gz = grip[gpos, 2]
                    touched = zone[gz <= rest_stool + 0.02]
                else:
                    touched = np.array([], dtype=int)
                if len(touched):
                    f0t = int(touched[0])
                    cap = idxs >= f0t
                    grip[cap, 2] = np.minimum(grip[cap, 2], rest_stool + 0.01)
                    # Pin XY as well: the stool is placed under the MEASURED
                    # resting spot but the gripped box still follows the
                    # hands, and with the stool top only ~20 mm wider than
                    # the box any residual offset reads as the box hanging
                    # off the stool's front edge.  Slide the box centre onto
                    # the measured resting spot while the hands release.
                    stc = on_stool & cov
                    if stc.sum() > 3:
                        tgt = np.median(pc[stc, :2], axis=0)
                        CR = 15
                        k0 = int(np.nonzero(cap)[0][0])
                        for jj in range(len(idxs) - k0):
                            uu = min(1.0, (jj + 1) / (CR + 1.0))
                            uu = uu * uu * (3.0 - 2.0 * uu)
                            grip[k0 + jj, :2] = ((1.0 - uu) * grip[k0 + jj, :2]
                                                 + uu * tgt)
                    print(f"  stool capture: box pinned to the stool top "
                          f"from f{f0t} on ({int(cap.sum())} frames; z "
                          f"held, xy slid onto the measured resting spot "
                          f"over {CR} frames -- the hands may bob, the box "
                          f"does not)")
                pc_use[sel] = ((1.0 - wcl[sel])[:, None] * pre[sel]
                               + wcl[sel][:, None] * grip)
                print(f"  carry contact: GRIP CLAMP -- box centre anchored "
                      f"at the hand midpoint + <= {RG*1000:.0f} mm of "
                      f"low-passed measured drift (window {LW}, gain "
                      f"{G:.2f}; {int(sel.sum())} frames, ramps "
                      f"{BW0}/{BW1} frames at carry ends){RGW_note}")

            # ---- rate-limit the pickup lift --------------------------------
            # The grasp wait puts the box on the floor at carry_start while
            # the blended target is already climbing toward the hands (the
            # retargeted robot reaches the box late and lifts it from there).
            # A hard hand-over popped 127 mm in one frame.  Cap the per-frame
            # z step instead: the first steps are absorbed and the box rejoins
            # the target as soon as the target's own rate drops under the cap
            # (the measured lift is ~35 mm/frame, so this only ever smooths
            # the hand-over).  The cap stays under the QA jump threshold, so
            # the lift can never read as a teleport.
            zcap = float(getattr(args, "grasp_z_maxstep", 0.040))
            if carry_start != first_c and zcap > 0 and last_c > carry_start:
                prev, n_cap = rest_floor, 0
                for t in range(carry_start + 1, last_c + 1):
                    d = float(pc_use[t, 2] - prev)
                    if abs(d) > zcap:
                        pc_use[t, 2] = prev + np.sign(d) * zcap
                        n_cap += 1
                    prev = float(pc_use[t, 2])
                print(f"  grasp wait: pickup lift rate limited to "
                      f"{zcap * 1000:.0f} mm/frame ({n_cap} frames capped; "
                      f"the box rejoins the target once the target's own "
                      f"rate drops below the cap)")

            # ---- place descent: keep the hands ON the box -------------------
            # G1's arms are ~0.3 m shorter than the person's, so during the
            # final lowering the measured box z sinks away BELOW the hands
            # (measured on 165945: wrist midpoint 73-227 mm above the box top
            # through f240-304, and the hands dip to 120 mm BELOW the top at
            # f288-305) -> the rendered hands hover above / pass through the
            # box instead of resting on it.  Tie the box TOP to the wrist
            # midpoint over the descent window (top = mid_z - hand_len) with
            # smoothstep weight in/out; the drop guard after this still ramps
            # onto the physical stool rest height, so the landing is
            # unchanged.  Opt-in: --place_hand_z_follow F0 F1.
            phf = getattr(args, "place_hand_z_follow", None)
            if phf and args.pose_z == "rest":
                f0h, f1h = int(phf[0]), int(phf[1])
                handlen = float(getattr(args, "place_hand_z_drop", 0.06))
                hz = hand_mid[:, 2] - handlen - eff["box_size"][2] / 2.0
                # weight: ramp in over RW frames, hold, ramp out over RW
                RW = 10
                wh = np.zeros(n_f)
                lo, hi = max(f0h, 0), min(f1h, n_f - 1)
                wh[lo:hi + 1] = 1.0
                for k in range(1, RW + 1):
                    u = 1.0 - k / (RW + 1.0)
                    u = u * u * (3.0 - 2.0 * u)
                    if lo - k >= 0:
                        wh[lo - k] = u
                    if hi + k < n_f:
                        wh[hi + k] = u
                msk = (wh > 1e-6) & cov & np.isfinite(hz)
                if msk.any():
                    old_z = pc_use[msk, 2].copy()
                    pc_use[msk, 2] = ((1.0 - wh[msk]) * pc_use[msk, 2]
                                      + wh[msk] * hz[msk])
                    dz = (pc_use[msk, 2] - old_z) * 1000.0
                    print(f"  place hand-z follow: box top tied to the wrist "
                          f"midpoint - {handlen*1000:.0f} mm over f{f0h}.."
                          f"{f1h} ({int(msk.sum())} frames; z raised "
                          f"{dz.min():+.0f}..{dz.max():+.0f} mm so the hands "
                          f"stay on the box during the descent)")
                else:
                    print("  place hand-z follow: no covered frame in the "
                          "window; nothing changed")

            # ---- drop guard: land exactly on the stool top -----------------
            # The tail of the carry phase IS the box being lowered onto the
            # stool; the measured z there still wobbles a few cm and the LPF
            # above can undershoot -> visible clipping through the seat right
            # before touchdown.  Ramp z onto the physical rest height over
            # the last `drop_guard` frames of the carry phase so the box ends
            # exactly at stool-top + half box.
            GD = max(int(getattr(args, "drop_guard", 12)), 0)
            if args.pose_z == "rest" and GD > 0 and on_stool.any():
                g0 = last_c - GD
                selg = (idx >= g0) & (idx <= last_c) & cov
                wgt = np.clip((idx[selg] - g0) / float(GD), 0.0, 1.0)
                pc_use[selg, 2] = ((1.0 - wgt) * pc_use[selg, 2]
                                   + wgt * rest_stool)
                print(f"  drop guard: carry z ramped onto the stool rest "
                      f"height ({rest_stool:+.3f}) over the last {GD} "
                      f"carry frames ({int(selg.sum())} frames)")

            # ramp the weight in over `pb` frames at each end of every covered
            # run: a fallback stretch that differs from the measurement would
            # otherwise teleport the box on the boundary frame
            pb = max(int(args.pose_blend), 1)
            w = cov.astype(np.float64)
            starts = np.nonzero(cov & ~np.r_[False, cov[:-1]])[0]
            ends = np.nonzero(cov & ~np.r_[cov[1:], False])[0]
            for s0 in starts:
                for k in range(min(pb, n_f - s0)):
                    w[s0 + k] = min(w[s0 + k], (k + 1) / (pb + 1.0))
            for e0 in ends:
                for k in range(min(pb, e0 + 1)):
                    w[e0 - k] = min(w[e0 - k], (k + 1) / (pb + 1.0))
            w = w * w * (3.0 - 2.0 * w)          # smoothstep

            # A parked phase is STATIC and its measurement is trustworthy, so
            # the coverage-boundary blend must not bleed the SYNTHESISED
            # fallback pose into it.  Without this the first/last ~8 frames of
            # a covered run slerped the box toward the fallback's different
            # yaw (~45 deg away) -> a visible twist while the box was sitting
            # untouched on the floor / on the stool.  Forcing w=1 wherever
            # there IS a measurement inside a parked phase removes it; the
            # blend still covers the genuinely uncovered stretches.
            w[cov & ~carry_ph] = 1.0

            q_meas = fallback_quat.copy()
            q_meas[cov] = R.from_matrix(pR[cov]).as_quat()

            # ---- carried-box orientation source (optional) ------------------
            # A held box turns WITH the person: on this clip the source video
            # shows the body pivoting ~70-80 deg to face the walking direction
            # right after the stand-up (f192-212) with the box riding the
            # turn in front of the chest.  The depth fit's yaw turns ~30
            # frames early and wanders under self-occlusion (read as the box
            # "rotating abnormally in the hands"), while freezing the yaw to
            # the parked interpolation made the hands sweep across the box
            # faces as the body turned.  --carry_orient hand therefore drives
            # the carried box's orientation from the HAND-RIGID convention
            # (rigid to the wrists, so it turns exactly with the body),
            # blended in/out over short ramps; the parked phases keep the
            # measured orientation.  Pair with --carry_tilt_anchor_deg to
            # keep the box level, and raise --quat_max_step_deg above the
            # real turn rate (~4 deg/frame here) so the turn is not clipped.
            if getattr(args, "carry_orient", "measured") == "hand" \
                    and carry_ph.any():
                cs0, cs1 = int(carry_start), int(last_c)
                wh = np.zeros(n_f)
                wh[cs0:cs1 + 1] = 1.0
                RW = 10
                for k in range(RW):
                    uu = (k + 1) / (RW + 1.0)
                    uu = uu * uu * (3.0 - 2.0 * uu)
                    if cs0 + k < n_f:
                        wh[cs0 + k] = uu
                    if cs1 + 1 + k < n_f:
                        wh[cs1 + 1 + k] = 1.0 - uu
                for i in np.nonzero(wh > 0)[0]:
                    q_meas[i] = Slerp(
                        [0.0, 1.0],
                        R.from_quat([q_meas[i],
                                     fallback_quat[i]]))(wh[i]).as_quat()
                print(f"  carry orientation driven by the hand-rigid "
                      f"convention ({int((wh > 0).sum())} frames blended, "
                      f"+/-{RW}-frame ramps) -- the box turns with the "
                      f"body like in the source video")

            # ---- keep the CARRIED box's orientation physically plausible ----
            # A held box's orientation is tied to the hands, but the depth fit
            # still lands tens of degrees away on the frames where the arms /
            # body occlude the box (measured on this clip: the fit turned the
            # box ~150 deg while the source video shows it barely moving during
            # the lift).  Bound the deviation from the hand-rigid orientation
            # (symmetry-aware, a square box has 4 equivalent yaw branches) so
            # the real tilt is kept but a runaway branch is clipped instead of
            # being rendered as the box spinning in the hands.
            lim = float(getattr(args, "carry_quat_max_deg", 60.0))
            if lim > 0 and carry_ph.any():
                syms = []
                for k in range(4):
                    Rz = R.from_euler("z", 90.0 * k, degrees=True)
                    syms.append(Rz)
                    syms.append(Rz * R.from_euler("x", 180.0, degrees=True))
                selc = np.nonzero(carry_ph & cov)[0]
                over = 0
                for i in selc:
                    Rm = R.from_quat(q_meas[i])
                    Rf = R.from_quat(fallback_quat[i])
                    angles = [np.degrees((Rf.inv() * Rm * S).magnitude())
                              for S in syms]
                    k = int(np.argmin(angles))
                    a = angles[k]
                    if a > lim:
                        Rsel = Rm * syms[k]
                        q_meas[i] = Slerp(
                            [0.0, 1.0],
                            R.from_quat([fallback_quat[i],
                                         Rsel.as_quat()]))(lim / a).as_quat()
                        over += 1
                print(f"  carry orientation bounded to {lim:.0f} deg from the "
                      f"hand-rigid pose: {over} of {len(selc)} carry frames "
                      f"were clipped (occlusion-driven wrong branches)")
            # ---- carry tilt anchor (optional) -------------------------------
            # The fit's roll/pitch wanders while the box is carried close to
            # the body (self-occlusion), rendering as the box swaying ~15-20
            # deg in the hands while the source video shows it held flat.
            # Anchor the carry tilt (angle between the box's z-axis and world
            # up) to the interpolation between the two parked phases' median
            # tilts: a box picked up flat and put down flat is carried flat.
            # 0 disables.  (Yaw is preserved: the correction rotates about the
            # axis perpendicular to the tilt deviation.)
            tilt_tol = float(getattr(args, "carry_tilt_anchor_deg", 0.0))
            if tilt_tol > 0 and carry_ph.any():
                def _median_tilt(mask):
                    tt = []
                    for i in np.nonzero(mask & cov)[0]:
                        zi = R.from_quat(q_meas[i]).as_matrix()[:, 2]
                        tt.append(np.degrees(np.arccos(np.clip(zi[2],
                                                               -1, 1))))
                    return float(np.median(tt)) if tt else None

                a_floor = _median_tilt(idx < carry_start)
                a_stool = _median_tilt(idx > last_c)
                if a_floor is not None and a_stool is not None:
                    cs, lc = int(carry_start), int(last_c)
                    span = max(lc - cs, 1)
                    n_fixed = 0
                    for i in np.nonzero(carry_ph & cov)[0]:
                        zi = R.from_quat(q_meas[i]).as_matrix()[:, 2]
                        t_i = np.degrees(np.arccos(np.clip(zi[2], -1, 1)))
                        anchor = a_floor + (a_stool - a_floor) * \
                            (i - cs) / span
                        # wrap the anchor to the branch nearest the current
                        # tilt (a square box's up/down is a symmetry choice)
                        while anchor - t_i > 180.0:
                            anchor -= 360.0
                        while t_i - anchor > 180.0:
                            anchor += 360.0
                        t_d = np.clip(t_i, anchor - tilt_tol,
                                      anchor + tilt_tol)
                        if abs(t_i - t_d) < 1e-6:
                            continue
                        phi = np.arctan2(zi[1], zi[0])
                        z_new = np.array([np.sin(np.radians(t_d)) *
                                          np.cos(phi),
                                          np.sin(np.radians(t_d)) *
                                          np.sin(phi),
                                          np.cos(np.radians(t_d))])
                        axis = np.cross(zi, z_new)
                        nn = np.linalg.norm(axis)
                        if nn < 1e-6:
                            continue
                        # NOTE: rotating zi about axis=cross(zi,z_new) by
                        # +(t_d - t_i) takes zi -> z_new (verified vs
                        # Rodrigues); the previous t_i - t_d sign mirrored
                        # each corrected frame to the OTHER side and doubled
                        # the deviation (box tilted up to ~31 deg mid-carry).
                        ang = np.radians(t_d - t_i)
                        Rc = R.from_rotvec(axis / nn * ang)
                        q_meas[i] = (Rc * R.from_quat(q_meas[i])).as_quat()
                        n_fixed += 1
                    print(f"  carry tilt anchored to {tilt_tol:.0f} deg of "
                          f"the parked interpolation "
                          f"({a_floor:.1f} -> {a_stool:.1f} deg): "
                          f"{n_fixed} of "
                          f"{int((carry_ph & cov).sum())} carry frames "
                          f"corrected (self-occlusion roll/pitch wander)")
            # ---- carry yaw anchor (optional) ---------------------------------
            # The fit's yaw also wanders when the hands occlude the box
            # (grasp moment: the measured yaw can swing 30-70 deg out and
            # back, the source video shows the box keeping its heading).
            # Anchor the carry yaw to the interpolation between the two
            # parked phases' median yaws, wrapped to the nearest 90-deg
            # symmetry branch of the current yaw.  Applied AFTER the tilt
            # anchor and rotating about world Z, so it preserves the tilt.
            yaw_tol = float(getattr(args, "carry_yaw_anchor_deg", 0.0))
            if yaw_tol > 0 and carry_ph.any():
                def _median_yaw(mask):
                    yy = []
                    for i in np.nonzero(mask & cov)[0]:
                        Rm = R.from_quat(q_meas[i]).as_matrix()
                        yy.append(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])))
                    return float(np.median(yy)) if yy else None

                y_floor = _median_yaw(idx < carry_start)
                y_stool = _median_yaw(idx > last_c)
                if y_floor is not None and y_stool is not None:
                    cs, lc = int(carry_start), int(last_c)
                    span = max(lc - cs, 1)
                    n_fixed = 0
                    for i in np.nonzero(carry_ph & cov)[0]:
                        Rm = R.from_quat(q_meas[i]).as_matrix()
                        y_i = np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))
                        target = y_floor + (y_stool - y_floor) * \
                            (i - cs) / span
                        # wrap to the 90-deg branch nearest the current yaw
                        target += 90.0 * round((y_i - target) / 90.0)
                        dy = target - y_i
                        excess = abs(dy) - yaw_tol
                        if excess <= 1e-6:
                            continue
                        dy_c = dy - np.sign(dy) * yaw_tol
                        Rz = R.from_euler("z", dy_c, degrees=True)
                        q_meas[i] = (Rz * R.from_quat(q_meas[i])).as_quat()
                        n_fixed += 1
                    print(f"  carry yaw anchored to {yaw_tol:.0f} deg of "
                          f"the parked interpolation "
                          f"({y_floor:.1f} -> {y_stool:.1f} deg): "
                          f"{n_fixed} of "
                          f"{int((carry_ph & cov).sum())} carry frames "
                          f"corrected (self-occlusion yaw wander)")
            # ---- parked phases are STATIC by physics ------------------------
            # A box sitting untouched on the floor/stool does not move, but
            # the per-frame measurement still carries a couple of centimetres
            # of depth noise in xy and a degree or two in yaw -> the render
            # showed it twitching / rotating back and forth in place.  Freeze
            # both parked phases to their own median pose (position median,
            # chordal-mean rotation), ramping back to the measurement over a
            # short band next to the carry phase so pickup/dropoff stay
            # continuous.
            FR = max(int(getattr(args, "parked_freeze_band", 8)), 1)
            for pmask_full, pheight, plabel in ((idx < carry_start, rest_floor, "floor"),
                                                (idx > last_c, rest_stool, "stool")):
                pmask = pmask_full & cov
                if pmask.sum() <= 3:
                    continue
                med_xy = np.median(pc_use[pmask, :2], axis=0)
                q_med = R.from_matrix(chordal_mean(pR[pmask])).as_quat()
                for i in np.nonzero(pmask)[0]:
                    d = (carry_start - i) if i < carry_start \
                        else (i - last_c)
                    a = 1.0 if d >= FR else max(d, 0) / float(FR)
                    pc_use[i, :2] = a * med_xy + (1.0 - a) * pc_use[i, :2]
                    pc_use[i, 2] = a * pheight + (1.0 - a) * pc_use[i, 2]
                    q_meas[i] = Slerp(
                        [0.0, 1.0],
                        R.from_quat([q_meas[i], q_med]))(a).as_quat()
                # Extend the frozen pose over UNCOVERED frames of the same
                # parked phase: a box sitting untouched at the start/end has
                # no measurement there, and the fallback for those frames is
                # the SYNTHESISED grasp-median guess, which sits ~0.9 m away
                # from the real spot -> the box visibly teleported when the
                # measurement began (f10->11 jumped 887 mm).  Park it at the
                # frozen measured pose instead and drive it with w=1.
                ext = pmask_full & ~cov
                if ext.any():
                    pc_use[ext, :2] = med_xy
                    pc_use[ext, 2] = pheight
                    q_meas[ext] = q_med
                    w[ext] = 1.0
                    print(f"  parked freeze [{plabel}]: extended the frozen "
                          f"pose over {int(ext.sum())} uncovered parked "
                          f"frames (kills the measurement-start teleport)")
                print(f"  parked freeze [{plabel}]: {int(pmask.sum())} frames "
                      f"frozen to their median pose (xy "
                      f"[{med_xy[0]:+.3f} {med_xy[1]:+.3f}], z {pheight:+.3f}); "
                      f"ramped back to the measurement over the last {FR} "
                      f"frames before the carry phase")

            # ---- slew-rate limit the orientation --------------------------
            # Every REAL box motion in this clip is slow (measured median
            # angular step 0.4 deg/frame = 12 deg/s).  What is left after the
            # filters above are fit / retargeting steps of up to 12 deg/frame
            # -- a box that visibly twists in the hands and sways in place,
            # which the source video simply does not show.  Clamp the per-frame
            # angular step; slow real rotation passes through untouched.
            max_step = float(getattr(args, "quat_max_step_deg", 3.5))
            if max_step > 0:
                prev = None
                n_clip = 0
                for i in range(n_f):
                    if not cov[i]:
                        continue
                    Ri = R.from_quat(q_meas[i])
                    if prev is None:
                        prev = Ri
                        continue
                    d = np.degrees((prev.inv() * Ri).magnitude())
                    if d > max_step:
                        Ri = Slerp([0.0, 1.0],
                                   R.from_quat([prev.as_quat(),
                                                Ri.as_quat()]))(max_step / d)
                        n_clip += 1
                    q_meas[i] = Ri.as_quat()
                    prev = Ri
                print(f"  orientation slew-rate limited to "
                      f"{max_step:.1f} deg/frame: {n_clip} frames clipped "
                      f"(fast twists were fit/retarget noise, not real motion)")

            # DEBUG (env-gated): dump the stages so the yaw limit can be
            # attributed.  Set GMR_DBG_YAW=<path> to dump a npz with the
            # fallback (hand-rigid) yaw and the final q_meas yaw.
            if os.environ.get("GMR_DBG_YAW"):
                def _yawf(qarr):
                    out = np.full(len(qarr), np.nan)
                    for _i in range(len(qarr)):
                        _M = R.from_quat(qarr[_i]).as_matrix()
                        out[_i] = np.degrees(np.arctan2(_M[1, 0], _M[0, 0]))
                    return out
                np.savez(os.environ["GMR_DBG_YAW"],
                         fallback_yaw=_yawf(fallback_quat),
                         qmeas_yaw=_yawf(q_meas),
                         w=w, cov=cov, carry=carry_ph)
                print("  DBG yaw dumped -> %s" % os.environ["GMR_DBG_YAW"])

            # (w, not cov, decides: the parked-freeze extension writes the
            # frozen measured pose into pc_use for UNCOVERED parked frames
            # with w=1 -- selecting by cov here would bypass it and keep the
            # 0.9 m-off synthesised fallback -> the f10->11 teleport)
            pos_meas = np.where((w > 0)[:, None], pc_use, fallback_pos)
            box_pos = w[:, None] * pos_meas + (1.0 - w)[:, None] * fallback_pos
            box_quat_xyzw = np.empty((n_f, 4))
            for i in range(n_f):
                if w[i] <= 0.0:
                    box_quat_xyzw[i] = fallback_quat[i]
                elif w[i] >= 1.0:
                    box_quat_xyzw[i] = q_meas[i]
                else:
                    box_quat_xyzw[i] = Slerp(
                        [0.0, 1.0],
                        R.from_quat([fallback_quat[i],
                                     q_meas[i]]))(w[i]).as_quat()

            if getattr(args, "dump_box_pose", None):
                # QA hook: the box pose the renderer actually used, so static
                # phases can be checked numerically (std should be ~0) instead
                # of by eye.
                np.savez(args.dump_box_pose, box_pos=box_pos,
                         box_quat_xyzw=box_quat_xyzw, carry=carry_ph,
                         cov=cov, first_c=carry_start, last_c=last_c)
                print(f"  dumped the rendered box pose -> "
                      f"{args.dump_box_pose}")

            # How far is the hand-rigid convention from what the depth camera
            # actually saw?  This is the single number that says whether the
            # hips-to-hands retargeting put the box where it really was; it is
            # only measurable now that the box pose is measured independently.
            pure = cov & (idx >= carry_start) & (idx <= last_c)
            if pure.sum() > 3:
                dev = np.linalg.norm(box_pos[pure] - fallback_pos[pure],
                                     axis=1)
                print(f"6-DoF pose override: {int(cov.sum())}/{n_f} frames "
                      f"measured ({100.0 * cov.mean():.0f}%), "
                      f"{int(pure.sum())} of them in the carry phase")
                print(f"  measured box centre vs hand-rigid convention over "
                      f"those {int(pure.sum())} carry frames: median "
                      f"{np.median(dev) * 1000:.0f} mm, p90 "
                      f"{np.percentile(dev, 90) * 1000:.0f} mm "
                      f"(this IS the retargeting + Sim3 alignment error)")
            else:
                print(f"6-DoF pose override: {int(cov.sum())}/{n_f} frames "
                      f"measured ({100.0 * cov.mean():.0f}%)")
            if cov.any():
                zw = pR[cov][:, :, 2]
                tilt = np.degrees(np.arccos(np.clip(zw[:, 2], -1.0, 1.0)))
                print(f"  box z-axis vs world up: median {np.median(tilt):.1f} "
                      f"deg  p90 {np.percentile(tilt, 90):.1f} deg "
                      f"(0 = perfectly level; the box is carried, so a few "
                      f"degrees is expected)")
            if not cov.all():
                gaps = []
                run = 0
                for v in cov:
                    run = 0 if v else run + 1
                    gaps.append(run)
                print(f"  longest measurement gap: {max(gaps)} frames "
                      f"(hand-rigid fallback + {pb}-frame blend)")

            # the stool must sit under the box's FINAL spot.  With a measured
            # centre that is the median of the measured centres - no
            # front-face correction, that bias only exists for the centroid.
            st = on_stool & cov
            if st.sum() > 3:
                measured_chair_xy = np.median(pc[st, :2], axis=0)
                print(f"  stool placed under the measured final box centre "
                      f"XY {np.round(measured_chair_xy, 3)} "
                      f"({int(st.sum())} frames; no front-face "
                      f"correction needed)")
                # Independent test: the stool top height (chair_size[2]) was
                # NEVER used to derive the z anchor, so comparing the measured
                # final box height against floor + stool + half box is a real
                # check of both the anchor and the measured stool height.
                z_stool_now = float(np.median(pc[st, 2]))
                print(f"  independent check: box centre on the STOOL measured "
                      f"{z_stool_now:+.3f} vs floor + stool "
                      f"{eff['chair_size'][2]:.2f} + half box "
                      f"{motion['box_half_height']:.3f} = {rest_stool:+.3f}  "
                      f"(err {z_stool_now - rest_stool:+.3f} m)")
                if abs(z_stool_now - rest_stool) > 0.05:
                    print("    NOTE: this residual is an ANCHOR ARTEFACT, not a "
                          "fit error -- the parked-on-stool frames are pinned "
                          "to rest_stool by the z policy, so the rendered box "
                          "is correct.  Check the rendered rise instead: "
                          "STOOL z - FLOOR z should equal the stool height.")
                zc = pc[st, 2]
                print(f"    z spread within the stool phase: "
                      f"{(np.percentile(zc, 90) - np.percentile(zc, 10)) * 1000:.0f} "
                      f"mm")
                # the stool can now take its yaw from the MEASURED final box
                # orientation instead of the walk direction.
                bx = pR[st][:, :, 0].mean(axis=0)
                if float(np.hypot(bx[0], bx[1])) > 1e-6:
                    measured_yaw = float(np.arctan2(bx[1], bx[0]))
                    print(f"  stool yaw taken from the measured box x-axis: "
                          f"{np.degrees(measured_yaw):+.1f} deg (a square "
                          f"stool only defines this modulo 90 deg)")
            fc = floor_meas & cov
            if fc.sum() > 3:
                zf = pc[fc, 2]
                print(f"  z spread within the floor phase after anchoring: "
                      f"{(np.percentile(zf, 90) - np.percentile(zf, 10)) * 1000:.0f} "
                      f"mm (the guide's criterion is < 10 mm)")

    grasp_mid = np.median(hand_mid[max(eff["phase_lift"] - 5, 0):eff["phase_carry"] + 1, :2], axis=0)
    print(f"box ground pose auto: XY={np.round(grasp_mid, 2)}")
    if getattr(args, "box_pose_world", None):
        # in this mode the box follows the MEASUREMENT, so of course it is not
        # rigid in the hands -- reporting a "rigidity deviation" here would be
        # measuring the wrong thing.  The meaningful number was already
        # printed above: measured centre vs the hand-rigid convention.
        print("carry-phase rigidity check skipped: --box_pose_world drives the "
              "box from the measurement, not from the hand frame")
    else:
        # rigidity: box center expressed IN the hand frame must be constant
        rel = np.einsum("nji,nj->ni", hand_rot[phase == 2],
                        box_pos[phase == 2] - hand_mid[phase == 2])
        print(f"carry-phase rigidity check (max dev from offset): "
              f"{np.abs(rel - np.array(eff['box_offset'])).max() * 1000:.2f} mm")

    # ---- phase labels for the MEASURED routes ------------------------------
    # `phase` is created and filled inside compute_box_trajectory() and
    # describes the SYNTHESIZED 3-phase route (0 on-ground / 1 blend /
    # 2 carried) over the video's --phase_lift..--phase_carry schedule.  In
    # the --box_traj_world / --box_pose_world branches the box is NOT driven by
    # that route at all, so leaving `phase` as returned made both consumers
    # lie: the interactive label (render_interactive) and --save_box_traj
    # described the synthesis, not what was rendered.  Re-derive it from the
    # masks the measured branch actually rendered with, keeping the same
    # 3-value vocabulary:
    #     carry mask (rendered carried, from the measurement)      -> 2
    #     parked frames the measurement covers                     -> 0
    #     neither clearly parked nor carried (measurement gaps the
    #       hand-rigid fallback had to fill, and the parked<->carry
    #       blend bands)                                           -> 1
    # Placed AFTER the rigidity check above, which is the one consumer of the
    # synthesized labels outside this block (and is skipped on the pose route
    # anyway).  Length and dtype are unchanged (n, int) so no consumer changes
    # shape.  NOTE: this is diagnostics only -- nothing downstream of it feeds
    # box_pos/box_quat_xyzw or the model, so the renders are byte-identical.
    if getattr(args, "box_traj_world", None):
        _ph = np.ones(len(box_pos), dtype=int)          # default: blend
        if getattr(args, "box_pose_world", None):
            # pose route: exactly the masks that drove the render
            _cov = np.asarray(cov, dtype=bool)
            _carr = np.asarray(carry_ph, dtype=bool) & _cov
            _ph[_cov & ~_carr] = 0
        else:
            # centroid route: mcarry/mvalid are the render's own masks
            _mval = np.asarray(mvalid, dtype=bool)
            _carr = np.asarray(mcarry, dtype=bool) & _mval
            _ph[_mval & ~_carr] = 0
        _ph[_carr] = 2
        phase = _ph
        print("phase labels re-derived from the MEASURED masks "
              "(carry->2, covered parked->0, blend/gap->1): "
              f"{int((phase == 2).sum())} carried, "
              f"{int((phase == 0).sum())} on-ground, "
              f"{int((phase == 1).sum())} blend of {len(phase)} frames "
              "-- the synthesized --phase_lift/--phase_carry labels are not "
              "used in this mode")

    # ---- static stool, placed relative to the box's ground pose ----
    show_chair = (not args.no_chair) and eff.get("chair_visible", True)
    if show_chair:
        if measured_chair_xy is not None:
            # measured mode: sit the stool exactly under the box's final
            # spot, so the box visibly lands on it
            stool_pos = np.array([measured_chair_xy[0], measured_chair_xy[1], 0.0])
            stool_quat = np.array([np.cos(measured_yaw / 2.0), 0.0, 0.0,
                                   np.sin(measured_yaw / 2.0)])
            # ---- push the stool out of the robot's crouch volume ----------
            # The box's landing spot is BETWEEN the robot and the stool in the
            # real scene: the person bends forward and sets the box on the
            # stool's near half, with their legs BEHIND the stool's near face.
            # Putting the stool centre exactly at the measured box spot makes
            # the seat slice through the robot's crotch and its shins pass
            # under the seat (verified on the 20260917 clip: stool y=1.31 vs
            # feet y=1.04, so the stool sat 25 cm IN FRONT of the feet).
            # Fix: keep the box drop where it is (it is measured) and slide
            # the STOOL further away along the robot->box direction.
            if args.chair_push_out:
                # The box is 0.30 m and the stool seat 0.32 m, so the box is
                # seated ONLY while the stool stays near-concentric with it:
                # pushing the stool away by M slides the box M toward the
                # stool's near edge, and the overhang margin is just
                # (0.32 - 0.30) / 2 = 10 mm per side.  Warn (do NOT clamp --
                # the value may be deliberate for a one-off diagnostic).
                if args.chair_push_out > 0.010:
                    print(f"  [!] WARNING: --chair_push_out {args.chair_push_out} "
                          f"m = {args.chair_push_out * 1000:.0f} mm exceeds the "
                          f"~10 mm the seat can absorb: the box is 0.30 m wide "
                          f"and the stool seat is 0.32 m, so the overhang "
                          f"margin is only (0.32 - 0.30) / 2 = 10 mm per side. "
                          f"The stool is now pushed "
                          f"{args.chair_push_out * 1000 - 10:.0f} mm past that "
                          f"margin, so the box will hang off the seat edge and "
                          f"drop off it. Use <= 0.010 m, or move the stool with "
                          f"--chair_xy instead.")
                # reference = where the robot actually stands during the drop
                _j0, _j1 = max(last_c - 60, 0), last_c + 1
                _sel = body_pos[_j0:_j1][:, _LEG_IDS, :]
                ref_xy = np.median(_sel.reshape(-1, 2), axis=0)
                d = stool_pos[:2] - ref_xy
                dn = np.linalg.norm(d)
                if dn > 1e-6:
                    u = d / dn
                    stool_pos[:2] += u * args.chair_push_out
                    print(f"  stool pushed {args.chair_push_out:.3f} m away "
                          f"from the robot along the drop direction "
                          f"(robot legs XY {np.round(ref_xy, 3)} -> stool "
                          f"XY {np.round(stool_pos[:2], 3)})")
            # ---- report the resulting clearance ---------------------------
            if args.chair_min_clear > 0:
                _j0, _j1 = max(last_c - 80, 0), last_c + 1
                _legs = body_pos[_j0:_j1][:, _LEG_IDS, :]
                _hw = eff["chair_size"][0] / 2.0
                _hd = eff["chair_size"][1] / 2.0
                # distance from each leg point to the stool's footprint box
                _rel = _legs[:, :, :2] - stool_pos[None, None, :2]
                _cy, _sy = np.cos(measured_yaw), np.sin(measured_yaw)
                _lx = _rel[..., 0] * _cy + _rel[..., 1] * _sy
                _ly = -_rel[..., 0] * _sy + _rel[..., 1] * _cy
                _dx = np.maximum(np.abs(_lx) - _hw, 0.0)
                _dy = np.maximum(np.abs(_ly) - _hd, 0.0)
                _dist = np.hypot(_dx, _dy)
                _k = int(np.argmin(_dist))
                clear_m = float(_dist.reshape(-1)[_k])
                worst_t = _j0 + _k // len(_LEG_IDS)
                if clear_m < args.chair_min_clear:
                    print(f"  [!] stool-to-leg clearance {clear_m * 1000:.0f} mm "
                          f"< {args.chair_min_clear * 1000:.0f} mm at f{worst_t}"
                          f" -- consider a larger --chair_push_out")
                else:
                    print(f"  stool-to-leg clearance {clear_m * 1000:.0f} mm "
                          f"ok (>= {args.chair_min_clear * 1000:.0f} mm)")
        elif eff["chair_xy"] is not None:
            stool_pos, stool_quat = (
                np.array([eff["chair_xy"][0], eff["chair_xy"][1], 0.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
        else:
            stool_pos, stool_quat = compute_stool_pose(
                motion, box_pos, box_quat_xyzw, hand_mid, hand_rot,
                box_size=eff["box_size"],
                stool_size=eff["chair_size"],
                stool_offset=eff["chair_offset"],
                root_pos=root_pos)
        if eff["chair_yaw_deg"] is not None:
            yw = np.deg2rad(eff["chair_yaw_deg"])
            stool_quat = np.array([np.cos(yw / 2.0), 0.0, 0.0, np.sin(yw / 2.0)])
        print(f"stool placed at world XY=({stool_pos[0]:+.2f}, {stool_pos[1]:+.2f}), "
              f"yaw={np.degrees(2*np.arctan2(stool_quat[3], stool_quat[0])):+.1f} deg")
    else:
        stool_pos = stool_quat = None

    # ---- model with box, override box dims on the loaded model ----
    model = load_model(BOX_XML)
    half = np.array(eff["box_size"]) / 2.0
    gid = model.geom("carried_box_geom").id
    model.geom_size[gid] = half
    # two 40 mm top bosses, flush with the long edges, 40 mm clear gap
    # between their inner faces: half-x = half[0]/2 - 0.01 = 0.065 at
    # x-centre +/- (half[0]/2 + 0.01) = +/- 0.085, so the inner faces land at
    # x = +/- 0.02.  Size matches the real box (450 body + 2 x 20 boss =
    # 490 overall) -- the half[0]/2 - 0.01 form only reproduces the XML's
    # 0.065 because the CLI box_size defaults to the XML's 0.30.
    for k, sgn in ((1, -1.0), (2, 1.0)):
        g = model.geom(f"box_boss_{k}")
        g.size[:] = [half[0] / 2 - 0.01, half[1], 0.02]
        g.pos[:] = [sgn * (half[0] / 2 + 0.01), 0, half[2] + 0.02]
    for k in range(4):
        sid = model.site(f"box_corner_{k+1}").id
        model.site_pos[sid] = [
            half[0] * (1 if k in (1, 2) else -1),
            half[1] * (1 if k in (2, 3) else -1),
            -half[2],
        ]
    mocap_id = model.body("carried_box").mocapid
    data = mj.MjData(model)

    # optional: roll the rigid rubber-hand meshes about the forearm axis.  This
    # is what decides whether the PALM or the BACK of the hand sits on the box.
    # The two hands are mirror images: their thumbs live on -y (left) and +y
    # (right), so the SAME rotation about +x sends one thumb up and the other
    # down.  Hence opposite signs per hand, and optionally different magnitudes
    # (--hand_roll_left_deg / --hand_roll_right_deg) because the measured
    # optimal rolls for the two shells differ by ~26-37 deg.
    _rolls = {
        "left_rubber_hand": eff.get("hand_roll_left_deg"),
        "right_rubber_hand": eff.get("hand_roll_right_deg"),
    }
    for nm, sgn in (("left_rubber_hand", +1.0), ("right_rubber_hand", -1.0)):
        deg = _rolls[nm]
        if deg is None:
            deg = eff["hand_roll_deg"]
        if not deg:
            continue
        roll = R.from_rotvec([np.deg2rad(sgn * deg), 0, 0])
        bid = model.body(nm).id
        model.body_quat[bid] = roll.as_quat(scalar_first=True)

    # optional: shorten the rigid rubber-hand shell along its finger axis.
    # MuJoCo re-centres each STL and stores the alignment in mesh_quat, so the
    # mesh data is NOT in the geom frame. Work in the geom frame: the fingers
    # point along geom-local +x (the rubber_hand body sits at the end of the
    # wrist chain with an identity quat), so pull every vertex toward the
    # geom-local x minimum (= the wrist end) and push the normals through the
    # inverse-transpose so the shading stays right.
    if eff["hand_shrink"] != 1.0:
        s = float(eff["hand_shrink"])
        for nm in ("left_rubber_hand", "right_rubber_hand"):
            mesh_id = model.mesh(nm).id
            rq = R.from_quat(model.mesh_quat[mesh_id][[1, 2, 3, 0]]).as_matrix()
            adr = model.mesh_vertadr[mesh_id]
            num = model.mesh_vertnum[mesh_id]
            V = model.mesh_vert[adr:adr + num].reshape(-1, 3).copy()
            Vg = V @ rq                     # mesh frame -> geom-local
            x0 = Vg[:, 0].min()             # wrist end (fingers grow along +x)
            Vg[:, 0] = x0 + (Vg[:, 0] - x0) * s
            model.mesh_vert[adr:adr + num] = Vg @ rq.T
            if hasattr(model, "mesh_normal"):
                try:
                    adr_n = model.mesh_normaladr[mesh_id]
                    num_n = model.mesh_normalnum[mesh_id]
                    if num_n == num:
                        N = model.mesh_normal[adr_n:adr_n + num_n].reshape(-1, 3)
                        Ng = N @ rq
                        Ng[:, 0] /= max(s, 1e-6)    # inverse-transpose
                        Ng /= np.maximum(
                            np.linalg.norm(Ng, axis=1, keepdims=True), 1e-9)
                        model.mesh_normal[adr_n:adr_n + num_n] = Ng @ rq.T
                except (AttributeError, ValueError):
                    pass
        print(f"hand mesh shortened to {s:.2f} along the finger axis")

    # ---- resize the stool geoms from chair_size (W, D, H) ----
    if show_chair:
        sw, sd, sh = eff["chair_size"]
        t_seat = 0.030                      # seat plate thickness
        leg = 0.035                         # leg cross-section
        rail = 0.024                        # stretcher cross-section
        rail_z = max(sh * 0.42, 0.10)       # video: rungs sit in the lower half
        model.geom("stool_seat").size[:] = [sw / 2, sd / 2, t_seat / 2]
        model.geom("stool_seat").pos[:] = [0, 0, sh - t_seat / 2]
        hw, hd = sw / 2, sd / 2
        leg_h = (sh - t_seat) / 2.0
        for k, (lx, ly) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            g = model.geom(f"stool_leg_{k+1}")
            g.size[:] = [leg / 2, leg / 2, leg_h]
            g.pos[:] = [lx * (hw - leg / 2), ly * (hd - leg / 2), leg_h]
        for k, ly in enumerate([-1, 1]):        # stretchers on all four sides
            g = model.geom(f"stool_rail_x{k+1}")
            g.size[:] = [(sw - 2 * leg) / 2, rail / 2, rail / 2]
            g.pos[:] = [0, ly * (hd - leg / 2), rail_z]
        for k, lx in enumerate([-1, 1]):
            g = model.geom(f"stool_rail_y{k+1}")
            g.size[:] = [rail / 2, (sd - 2 * leg) / 2, rail / 2]
            g.pos[:] = [lx * (hw - leg / 2), 0, rail_z]
    stool_mocap_id = model.body("stool").mocapid

    def set_frame(t):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot_wxyz[t]  # wxyz for mujoco
        data.qpos[7:] = dof_pos[t]
        data.mocap_pos[mocap_id] = box_pos[t]
        data.mocap_quat[mocap_id] = R.from_quat(box_quat_xyzw[t]).as_quat(
            scalar_first=True)
        if show_chair:
            data.mocap_pos[stool_mocap_id] = stool_pos
            data.mocap_quat[stool_mocap_id] = stool_quat
        mj.mj_forward(model, data)

    # ---- camera follows the robot base like the repo viewer ----
    base_id = model.body("pelvis").id
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = args.cam_distance
    cam_az = args.cam_azimuth
    if cam_az is None:
        # auto: match the SOURCE VIDEO's viewpoint.  The world alignment
        # stores R_world_from_cam -- the real depth camera's orientation in
        # the render world.  MuJoCo's free-camera azimuth is the azimuth of
        # the camera's VIEWING direction (the camera sits lookat - dist *
        # forward), so the synthetic camera must look along the mapped real
        # camera forward: azimuth = fwd_az.  Verified empirically: with
        # fwd_az + 180 the 165945 render showed the robot's back while the
        # source video shows its front.
        cam_az = 315.0
        if args.box_pose_world:
            try:
                _wc = np.load(args.box_pose_world, allow_pickle=True)
                if "R_world_from_cam" in _wc.files:
                    _fwd = _wc["R_world_from_cam"] @ np.array(
                        [0.0, 0.0, 1.0])
                    _fwd_az = float(np.degrees(np.arctan2(_fwd[1], _fwd[0])))
                    if abs(_fwd[2]) < 0.95:
                        cam_az = _fwd_az
                        print(f"camera azimuth auto: real camera forward az "
                              f"{_fwd_az:+.1f} deg -> cam.azimuth "
                              f"{cam_az:+.1f} deg (matches the source "
                              f"video's viewpoint)")
                    else:
                        print(f"camera azimuth auto: mapped camera axis is "
                              f"near-vertical (z {_fwd[2]:+.2f}) -- keeping "
                              f"fallback 315 deg")
            except Exception as _e:
                print(f"camera azimuth auto failed ({_e}); using 315 deg")
        else:
            print("camera azimuth auto: no --box_pose_world, using 315 deg")
    cam.azimuth = cam_az
    cam.elevation = args.cam_elevation

    n = len(root_pos)

    # ---- shared epilogue: the render dispatch and the --save_box_traj write --
    # The diagnostic-only exits further down and the normal end of main() both
    # route through here, so --save_box_traj can no longer be skipped silently.
    # In window mode a diagnostic-only pass deliberately does NOT open the
    # interactive viewer (the user asked for numbers, not a window); it says so
    # out loud and then still writes the npz.
    def _finish(diag_reason=None):
        if diag_reason is not None:
            print(f"[{diag_reason}] diagnostics only: not opening the "
                  "interactive window (pass --record_video to render an mp4).")
        elif not args.record_video:
            render_interactive(model, data, set_frame, base_id, cam, box_pos,
                               args, n, fps, phase=phase)
        if args.save_box_traj:
            np.savez(args.save_box_traj,
                     box_pos=box_pos, box_quat_xyzw=box_quat_xyzw,
                     phase=phase, hand_mid=hand_mid,
                     stool_pos=(stool_pos if show_chair else np.zeros(3)),
                     stool_quat_xyzw=(stool_quat if show_chair
                                      else np.array([1.0, 0, 0, 0])),
                     meta=json.dumps({
                         **eff,
                         "source": args.robot_motion,
                     }))
            if diag_reason is not None:
                print(f"[{diag_reason}] still wrote the box trajectory "
                      f"(including the phase labels) -> {args.save_box_traj}")
            else:
                print(f"saved box trajectory: {args.save_box_traj}")
        elif diag_reason is not None:
            print(f"[{diag_reason}] no --save_box_traj requested: nothing "
                  "further to write; exiting without rendering.")

    # ---- optional: measure the real interpenetration at the two contact
    # surfaces the user cares about (hand<->box, box<->stool).  The rendered
    # geoms are contype=0 (pure visuals), so flip them collidable for a
    # throw-away pass that never touches the mp4.  data.contact[].dist is
    # negative when penetrating, in metres.
    if args.diag_contact:
        box_like = ("carried_box_geom", "box_boss_1", "box_boss_2")
        stool_like = [model.geom(i).name for i in range(model.ngeom)
                      if (model.geom(i).name or "").startswith("stool_")]
        for nm in list(box_like) + stool_like:
            try:
                gid_d = model.geom(nm).id
            except KeyError:
                continue
            model.geom_contype[gid_d] = 1
            model.geom_conaffinity[gid_d] = 1
        if os.environ.get("BOX_DIAG_VERBOSE"):
            hand_ids = [i for i in range(model.ngeom)
                        if "hand" in (model.geom(i).name or "")
                        or "wrist" in (model.geom(i).name or "")]
            for nm in box_like + tuple(stool_like):
                try:
                    i = model.geom(nm).id
                    print(f"[diag-dbg] {nm} id={i} "
                          f"contype={model.geom_contype[i]} "
                          f"conaff={model.geom_conaffinity[i]}")
                except KeyError:
                    print(f"[diag-dbg] {nm} MISSING")
            print(f"[diag-dbg] hand/wrist geoms: "
                  + str([(model.geom(i).name, model.geom_contype[i])
                         for i in hand_ids]))
            set_frame(120)
            print(f"[diag-dbg] frame120 ncon={data.ncon}")
            for ci in range(min(data.ncon, 40)):
                c = data.contact[ci]
                print(f"[diag-dbg]   {model.geom(c.geom1).name} <-> "
                      f"{model.geom(c.geom2).name}  dist={c.dist*1000:+.1f}mm")
        rows = []
        for t in range(n):
            set_frame(t)
            wh = ws = 0.0
            for ci in range(data.ncon):
                c = data.contact[ci]
                if c.dist >= 0:
                    continue
                a = model.geom(c.geom1).name or ""
                b = model.geom(c.geom2).name or ""
                pair = (a, b)
                is_box = any(p.startswith("carried_box") or
                             p.startswith("box_boss") for p in pair)
                is_hand = any(("hand" in p or "wrist" in p) for p in pair)
                is_stool = any(p.startswith("stool_") for p in pair)
                if is_box and is_hand:
                    wh = min(wh, c.dist)
                if is_box and is_stool:
                    ws = min(ws, c.dist)
            rows.append((t, wh * 1000.0, ws * 1000.0))
        arr = np.array(rows)
        for label, col in (("hand<->box", 1), ("box<->stool", 2)):
            bad = arr[arr[:, col] < -0.5]
            worst_i = int(arr[:, col].argmin())
            print(f"[diag] {label}: worst {arr[worst_i, col]:+.1f} mm "
                  f"@ frame {int(arr[worst_i, 0])}; "
                  f"{len(bad)} frames worse than -0.5 mm")
            if len(bad):
                top = bad[np.argsort(bad[:, col])[:8]]
                print("       deepest frames: " + ", ".join(
                    f"f{int(r[0])}:{r[col]:+.0f}mm" for r in top))
        if args.record_video is None:
            # --diag_contact is a read-only measurement pass (it only flips
            # contype/conaffinity on the throw-away model and calls
            # mj_forward), so with no --record_video there is nothing left to
            # render here.  Exit through the shared epilogue instead of
            # returning outright: the old bare `return` also skipped the
            # --save_box_traj write, silently.
            _finish("--diag_contact")
            return

    # ---- placement-window clipping diagnostic -----------------------------
    # The model registers no contacts (visual geoms), but data.xpos is exact:
    # transform the pelvis / torso / head / wrist body centres into the final
    # box frame and report how deep each sits inside the box OBB (inflated by
    # a body radius so the centre metric approximates the visible clip).
    # head_mocap (geom-less fixed +0.4 m child of torso_link) IS tracked --
    # it sits exactly where the rendered head hangs; the box/stool mocap
    # drivers and other geom-less frames are not.
    if getattr(args, "place_diag", False):
        half = np.asarray(eff["box_size"], dtype=float) / 2.0
        infl = (["pelvis", 0.10], ["torso_link", 0.12],
                ["head_mocap", 0.09], ["wrist", 0.05], ["rubber_hand", 0.05])
        picks = []
        for i in range(model.nbody):
            nm = model.body(i).name or ""
            if "mocap" in nm and "head_mocap" not in nm:
                continue          # box/stool mocap drivers, not robot bodies
            if model.body_geomnum[i] == 0 and "head_mocap" not in nm:
                continue          # geom-less frames carry no visible mesh,
                # EXCEPT head_mocap: a fixed +0.4 m child of torso_link --
                # exactly where the retargeted (rendered) head sits
            for key, r in infl:
                if key in nm:
                    picks.append((nm, i, r))
                    break
        lc = int(np.nonzero(np.asarray(
            np.load(args.box_pose_world, allow_pickle=True)["carry"]
        ).astype(bool))[0][-1])
        ws_, we_ = max(lc - 10, 0), n - 1
        print(f"[place_diag] bodies: {[p[0] for p in picks]}")
        print(f"[place_diag] window f{ws_}..f{we_}, box half {half}")
        rows = []
        for t in range(ws_, we_ + 1):
            set_frame(t)
            cbx = R.from_quat(box_quat_xyzw[t])
            for nm, bid, r in picks:
                loc = cbx.inv().apply(data.xpos[bid] - box_pos[t])
                lim = half + r
                if np.all(np.abs(loc) < lim):
                    depth = float(np.min(lim - np.abs(loc)))
                    rows.append((t, nm, depth * 1000.0))
        if not rows:
            print("[place_diag] no body centre inside the box OBB anywhere "
                  "in the window")
        else:
            rows.sort(key=lambda r: -r[2])
            print("[place_diag] deepest 12 body-in-box samples:")
            for t, nm, dep in rows[:12]:
                print(f"[place_diag]   f{t}  {nm:<26} {dep:+6.1f} mm")
            # per-body summary: worst + how many frames inside
            for nm in dict.fromkeys(r[1] for r in rows):
                sub = [r for r in rows if r[1] == nm]
                print(f"[place_diag]   {nm:<26} worst {max(s[2] for s in sub):+.0f} mm"
                      f"  inside on {len(sub)}/{we_ - ws_ + 1} frames")
            wf = sorted({r[0] for r in rows})
            print(f"[place_diag] frames with any clip: {wf}")
        # ---- exact helmet-mesh vs box OBB intersection --------------------
        # Body-centre metrics can miss a grazing helmet (mesh extends well
        # past head_mocap).  Transform the actual head-mesh vertices into
        # the box frame every frame and count the ones inside the RAW box.
        head_mesh_gid = None
        try:
            mid = model.mesh("head_link").id
            for g in range(model.ngeom):
                if model.geom_dataid[g] == mid and \
                        model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                    head_mesh_gid = g
                    break
        except (KeyError, AttributeError):
            pass
        if head_mesh_gid is not None:
            adr = model.mesh_vertadr[mid]
            num = model.mesh_vertnum[mid]
            Vl = model.mesh_vert[adr:adr + num].astype(float)
            worst = None
            for t in range(ws_, min(we_, ws_ + 40) + 1):
                set_frame(t)
                Vw = data.geom_xpos[head_mesh_gid] + \
                    Vl @ data.geom_xmat[head_mesh_gid].reshape(3, 3).T
                loc = R.from_quat(box_quat_xyzw[t]).inv().apply(
                    Vw - box_pos[t])
                ins = np.all(np.abs(loc) <= half[None, :], axis=1)
                if ins.any():
                    depth = float(np.min(
                        half[None, :] - np.abs(loc[ins]), axis=1).max())
                    if worst is None or depth > worst[2]:
                        worst = (t, int(ins.sum()), depth)
            if worst is None:
                print("[place_diag] helmet mesh: 0 vertices inside the raw "
                      "box in f%d..f%d (clean)" % (ws_, min(we_, ws_ + 40)))
            else:
                print("[place_diag] helmet mesh: worst f%d -- %d verts "
                      "inside, deepest %+.1f mm"
                      % (worst[0], worst[1], worst[2] * 1000))

    # ---- hand<->box mesh-vertex diagnostic + optional visual push-out ----
    # The rendered geoms are contype=0, so MuJoCo registers no hand<->box
    # contacts and --diag_contact can never see them.  Exact test instead:
    # transform every vertex of the rubber-hand / wrist meshes into the box
    # OBB frame each frame and count the ones inside the RAW box.  With
    # --hand_push_out_mm > 0 the same pass also yields, per frame and geom,
    # the minimal rigid translation (restricted to the 6 box face normals)
    # that would clear the penetrating vertices; it is clamped, zero-phase
    # EMA-smoothed and applied to data.geom_xpos AFTER mj_forward so the
    # render shows a hand pushed off the surface while every trajectory
    # (root, dof, box, stool) stays untouched.
    if args.hand_diag or args.hand_push_out_mm > 0:
        hand_geoms = []
        seen = set()
        for g in range(model.ngeom):
            if model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                continue
            bnm = model.body(model.geom_bodyid[g]).name or ""
            if "rubber_hand" in bnm or "wrist" in bnm:
                mid = model.geom_dataid[g]
                # each link usually has TWO geoms sharing one mesh (visual +
                # collision at the same pose) -> identical results; keep one
                key = (model.geom_bodyid[g], mid)
                if key in seen:
                    continue
                seen.add(key)
                adr = model.mesh_vertadr[mid]
                num = model.mesh_vertnum[mid]
                Vl = model.mesh_vert[adr:adr + num].astype(float)
                hand_geoms.append((g, bnm, Vl))
        print(f"[hand_diag] mesh geoms: {[(g, b, len(V)) for g, b, V in hand_geoms]}")
        want_push = args.hand_push_out_mm > 0
        dmax = args.hand_push_out_mm / 1000.0
        want_dump = getattr(args, "hand_diag_npz", None) is not None
        raw_push = [dict() for _ in range(n)]   # t -> {gid: world push (m)}
        stats = []                              # (t, gid, body, nv, depth_mm)
        if want_dump:
            depth_arr = np.zeros((n, len(hand_geoms)))
            cen_arr = np.zeros((n, len(hand_geoms), 3))
            push_arr = np.zeros((n, len(hand_geoms), 3))
        for t in range(n):
            set_frame(t)
            cbx = R.from_quat(box_quat_xyzw[t])
            for gi, (g, bnm, Vl) in enumerate(hand_geoms):
                Vw = data.geom_xpos[g] + \
                    Vl @ data.geom_xmat[g].reshape(3, 3).T
                loc = cbx.inv().apply(Vw - box_pos[t])
                if want_dump:
                    cen_arr[t, gi] = loc.mean(axis=0)
                ins = np.all(np.abs(loc) <= half[None, :], axis=1)
                if not ins.any():
                    continue
                depth = float(np.min(
                    half[None, :] - np.abs(loc[ins]), axis=1).max())
                stats.append((t, g, bnm, int(ins.sum()), depth * 1000.0))
                if want_dump:
                    depth_arr[t, gi] = depth * 1000.0
                # push ONLY the rubber-hand shells: the wrist links hide
                # behind the hands (invisible inside the box), and shoving
                # them would visibly detach the forearm from the hand
                if want_push and "rubber_hand" in bnm:
                    Li = loc[ins]
                    best = None
                    for ax in range(3):
                        dp = half[ax] - Li[:, ax].max()   # exit through +face
                        dn = Li[:, ax].min() + half[ax]   # exit through -face
                        for d, s in ((dp, 1.0), (dn, -1.0)):
                            if best is None or d < best[0]:
                                best = (d, ax, s)
                    d, ax, s = best
                    dvec = np.zeros(3)
                    dvec[ax] = s
                    raw_push[t][g] = cbx.apply(dvec) * min(d, dmax)
                    if want_dump:
                        push_arr[t, gi] = raw_push[t][g]
        if want_dump:
            np.savez(args.hand_diag_npz,
                     bodies=np.array([b for _, b, _ in hand_geoms]),
                     depth_mm=depth_arr, center_box=cen_arr,
                     push_world=push_arr, box_pos=np.asarray(box_pos),
                     box_quat=np.asarray(box_quat_xyzw))
            print(f"[hand_diag] per-frame dump -> {args.hand_diag_npz}")
        if args.hand_diag:
            if not stats:
                print("[hand_diag] no hand/wrist mesh vertex ever enters the "
                      "raw box OBB (clean)")
            else:
                stats.sort(key=lambda r: -r[4])
                print(f"[hand_diag] {len(stats)} frame/geom samples with "
                      f"vertices inside the raw box; worst 12:")
                for t, g, bnm, nv, dep in stats[:12]:
                    extra = ""
                    if want_push and g in raw_push[t]:
                        extra = (f"  push "
                                 f"{np.linalg.norm(raw_push[t][g]) * 1000:.1f}"
                                 f" mm")
                    print(f"[hand_diag]   f{t:<4d} {bnm:<20} {nv:>5} verts"
                          f"  deepest {dep:+6.1f} mm{extra}")
                for bnm in dict.fromkeys(r[2] for r in stats):
                    sub = [r for r in stats if r[2] == bnm]
                    print(f"[hand_diag]   {bnm:<20} worst "
                          f"{max(r[4] for r in sub):+.0f} mm  inside on "
                          f"{len(sub)}/{n} frames")
                print(f"[hand_diag] frames with any clip: "
                      f"{sorted({r[0] for r in stats})}")
            if args.record_video is None:
                # Same as --diag_contact above: read-only pass with nothing to
                # render, so leave through the shared epilogue -- the old bare
                # `return` also skipped --save_box_traj without saying so.
                _finish("--hand_diag")
                return
        if want_push and any(raw_push):
            gids = sorted({g for fr in raw_push for g in fr})
            a = 0.5                       # zero-phase EMA: forward + backward
            for g in gids:
                seq = np.array([raw_push[t].get(g, np.zeros(3))
                                for t in range(n)])
                fwd = np.zeros_like(seq)
                s = np.zeros(3)
                for i in range(n):
                    s = a * seq[i] + (1.0 - a) * s
                    fwd[i] = s
                bwd = np.zeros_like(seq)
                s = np.zeros(3)
                for i in range(n - 1, -1, -1):
                    s = a * seq[i] + (1.0 - a) * s
                    bwd[i] = s
                seq = 0.5 * (fwd + bwd)
                for t in range(n):
                    if seq[t].any():
                        raw_push[t][g] = seq[t]
            _sf0 = set_frame
            def set_frame(t, _f=_sf0, _raw=raw_push):
                _f(t)
                for g, v in _raw[t].items():
                    data.geom_xpos[g] += v
            names = dict.fromkeys(
                b for gg, b, _ in hand_geoms if gg in gids)
            print(f"[hand-push] active: {len(gids)} geoms {list(names)}, "
                  f"clamp {dmax * 1000:.0f} mm, zero-phase EMA a=0.5")
        elif want_push:
            print("[hand-push] nothing to push: no hand vertex ever enters "
                  "the box")

    if args.record_video:
        render_offscreen(model, data, set_frame, base_id, cam, box_pos,
                         args, n, fps)
        if args.save_box_traj:
            _finish()
    else:
        _finish()


def render_offscreen(model, data, set_frame, base_id, cam, box_pos, args, n, fps):
    import cv2

    renderer = mj.Renderer(model, height=args.height, width=args.width)
    os.makedirs(os.path.dirname(os.path.abspath(args.record_video)), exist_ok=True)
    writer = cv2.VideoWriter(args.record_video,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (args.width, args.height))
    try:
        for t in tqdm(range(n), desc="render"):
            set_frame(t)
            cam.lookat = np.r_[data.xpos[base_id][:2], 0.75]
            renderer.update_scene(data, camera=cam)
            up = trail_cut(box_pos, t, args.traj_skip_m)
            if up >= 0:
                draw_trajectory(renderer.scene, box_pos, every=args.traj_every,
                                ground_z=args.ground_z, upto=up)
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()
    print(f"rendered video: {args.record_video}")


def render_interactive(model, data, set_frame, base_id, cam, box_pos, args, n, fps,
                       phase=None):
    import mujoco.viewer as mjv
    from loop_rate_limiters import RateLimiter

    phase_names = ["on-ground", "blend", "carried"]
    rate = RateLimiter(frequency=fps, warn=False)
    viewer = mjv.launch_passive(model=model, data=data,
                                show_left_ui=False, show_right_ui=False)
    # mjVIS_LABEL was removed in MuJoCo >=3.10 (the label flag is gone from
    # mjtVisFlag). Set it only when the running build still exposes it, so the
    # script works on both old and new mujoco.
    _label_flag = getattr(mj.mjtVisFlag, "mjVIS_LABEL", None)
    if _label_flag is not None:
        viewer.opt.flags[_label_flag] = True
    else:
        print("[render] mjVIS_LABEL not present in this mujoco build; skipping "
              "label flag (harmless).")
    try:
        for t in range(n):
            if not viewer.is_running():
                break
            set_frame(t)
            viewer.user_scn.ngeom = 0
            up = trail_cut(box_pos, t, args.traj_skip_m)
            if up >= 0:
                draw_trajectory(viewer.user_scn, box_pos, every=args.traj_every,
                                ground_z=args.ground_z, upto=up)
            # floating frame counter so phase frames are easy to spot
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mj.mjv_initGeom(g, type=mj.mjtGeom.mjGEOM_SPHERE,
                            size=[0.01, 0, 0],
                            pos=data.xpos[base_id] + np.array([0, 0, 1.5]),
                            mat=np.eye(3).flatten(), rgba=[1, 0.3, 0.2, 0.5])
            g.label = f"frame {t}/{n - 1}  phase: {phase_names[phase[t]]}"
            viewer.user_scn.ngeom += 1
            cam.lookat = np.r_[data.xpos[base_id][:2], 0.75]
            if t == 0:  # initial framing only; afterwards the user can orbit
                viewer.cam.distance = cam.distance
                viewer.cam.azimuth = cam.azimuth
                viewer.cam.elevation = cam.elevation
            viewer.cam.lookat[:] = cam.lookat  # follow the robot base
            viewer.sync()
            rate.sleep()
        while viewer.is_running():  # keep window open at the end
            viewer.sync()
            rate.sleep()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

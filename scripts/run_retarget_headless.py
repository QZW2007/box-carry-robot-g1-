"""Headless retargeting: SMPL-X npz -> robot motion pkl, no MuJoCo GUI.

Same pipeline as scripts/smplx_to_robot.py but without the interactive viewer,
plus optional offscreen rendering to mp4 for quick visual checks.

Usage:
    python scripts/run_retarget_headless.py --smplx_file data/xxx_smplx.npz \
        --robot unitree_g1 --save_path data/xxx_g1.pkl \
        [--render_video video/xxx_g1.mp4] [--render_every 1]
"""

import argparse
import os
import pathlib
import pickle

import numpy as np
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)

HERE = pathlib.Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", required=True)
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"],
        default="unitree_g1",
    )
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--render_video", default=None, help="offscreen-render the robot motion to this mp4")
    parser.add_argument("--render_every", type=int, default=1, help="render every k-th frame")
    parser.add_argument("--tgt_fps", type=int, default=30)
    args = parser.parse_args()

    smplx_folder = HERE / ".." / "assets" / "body_models"
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, smplx_folder
    )
    print(f"human height estimate: {actual_human_height:.3f} m")

    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps
    )
    print(f"{len(smplx_data_frames)} frames @ {aligned_fps:.1f} fps")

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    qpos_list = []
    for frame in tqdm(smplx_data_frames, desc="retarget"):
        qpos_list.append(retarget.retarget(frame))

    qpos_arr = np.array(qpos_list)  # (N, 3+4+n_dof), quat wxyz
    root_pos = qpos_arr[:, :3]
    root_rot = qpos_arr[:, 3:7][:, [1, 2, 3, 0]]  # wxyz -> xyzw
    dof_pos = qpos_arr[:, 7:]

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(
            {
                "fps": aligned_fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
            },
            f,
        )
    print(f"Saved {args.save_path}: root_pos {root_pos.shape}, dof_pos {dof_pos.shape}")
    print(f"root height range: z = [{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}] m")

    # ---- optional offscreen render ----
    if args.render_video:
        render_mp4(retarget, qpos_arr, aligned_fps, args.render_video, args.render_every)


def render_mp4(retarget, qpos_arr, fps, video_path, render_every):
    import mujoco as mj
    import cv2

    model = retarget.configuration.model
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=720, width=1280)

    # Frame the robot by its own size: robots here range from a ~0.5 m toddler to
    # a ~1.3 m humanoid, so a fixed camera distance leaves the small ones tiny.
    rest = mj.MjData(model)
    mj.mj_forward(model, rest)
    top = max(rest.geom_xpos[g][2] + model.geom_size[g].max() for g in range(model.ngeom))
    eye_height = 0.6 * top

    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.6 * top
    cam.azimuth = 135.0
    cam.elevation = -8.0

    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
    )
    try:
        for qpos in tqdm(qpos_arr[::render_every], desc="render"):
            data.qpos[:] = qpos
            mj.mj_forward(model, data)
            # follow the base so the robot stays framed while it translates
            cam.lookat[:] = [qpos[0], qpos[1], eye_height]
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()  # (H, W, 3) uint8, rgb
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()
    print(f"Rendered video: {video_path}")


if __name__ == "__main__":
    main()

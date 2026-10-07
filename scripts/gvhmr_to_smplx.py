"""Convert GVHMR output (hmr4d_results.pt) into an AMASS-style SMPL-X npz for GMR.

GVHMR saves `smpl_params_global` as per-frame SMPL-X parameters (rotation
matrices) in a Y-up world (SMPL convention, gravity "ay", see
GVHMR hmr4d/utils/geo/hmr_global.py). GMR expects AMASS-style npz with
axis-angle rotations in a Z-up world (like MuJoCo).

The fix is one global rotation applied to the root and translation only:

    R_fix = Rx(+90 deg),  new = R_fix @ old

which maps the old +Y (up) onto the new +Z (up). Body joint rotations are
relative (parent frame) and stay untouched.

Usage:
    python scripts/gvhmr_to_smplx.py --input data/hmr4d_results.pt \
        --output data/20260911211441_smplx.npz \
        [--person 0] [--fps 30] [--video path.mp4] [--up y]
"""

import argparse
import pathlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

HERE = pathlib.Path(__file__).parent


def matrix_to_rotvec(mats: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 3) axis-angle, vectorized."""
    return Rot.from_matrix(mats).as_rotvec()


def extract_person(params: dict, person_idx: int) -> dict:
    """Squeeze the optional person dim P from every field."""
    out = {}
    for key in ("global_orient", "body_pose", "transl", "betas"):
        if key not in params:
            raise KeyError(f"smpl_params_global is missing '{key}'; keys={list(params)}")
        out[key] = torch.as_tensor(params[key]).float().cpu().numpy()
    # transl is (P, L, 3) when multiple people are tracked, else (L, 3)
    if out["transl"].ndim == 3:
        n_persons = out["transl"].shape[0]
        if person_idx >= n_persons:
            raise IndexError(f"--person {person_idx} out of range, found {n_persons} persons")
        for key in out:
            out[key] = out[key][person_idx]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="GVHMR hmr4d_results.pt")
    parser.add_argument("--output", required=True, help="output .npz for GMR")
    parser.add_argument("--person", type=int, default=0, help="person index for multi-person results")
    parser.add_argument("--fps", type=float, default=None, help="source fps; default: read from --video, else 30")
    parser.add_argument("--video", default=None, help="source RGB video, used to read the true fps")
    parser.add_argument("--up", default="y", choices=["y", "z"], help="up-axis of GVHMR global output")
    parser.add_argument("--no-check", action="store_true", help="skip the SMPL-X FK sanity check")
    args = parser.parse_args()

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    params = extract_person(data["smpl_params_global"], args.person)

    # GVHMR saves axis-angle by default: global_orient (L,3), body_pose (L,63);
    # older/other paths may save rotation matrices (L,1,3,3)/(L,21,3,3).
    transl = params["transl"].reshape(-1, 3)  # (L, 3)
    num_frames = transl.shape[0]

    go = params["global_orient"]
    if go.shape[-1] == 3 and go.shape[-2] != 3:  # axis-angle (..., 3)
        root_orient = go.reshape(num_frames, 3)
        pose_body = params["body_pose"].reshape(num_frames, 63)
    else:  # rotation matrices
        root_orient = matrix_to_rotvec(go.reshape(-1, 3, 3))  # (L, 3)
        pose_body = matrix_to_rotvec(params["body_pose"].reshape(-1, 21, 3, 3)).reshape(num_frames, 63)

    betas_dim = params["betas"].shape[-1]
    betas = params["betas"].reshape(-1, betas_dim).mean(axis=0)

    print(f"Loaded {num_frames} frames, betas dim {betas_dim}, mean betas: {np.round(betas, 3)}")

    # ---- world-frame rotation: y-up (SMPL) -> z-up (AMASS / MuJoCo) ----
    if args.up == "y":
        R_fix = Rot.from_euler("x", np.pi / 2)
        print("Applying Rx(+90 deg): y-up -> z-up")
    else:
        R_fix = Rot.identity()
        print("Input already z-up, no rotation applied")

    root_orient = (R_fix * Rot.from_rotvec(root_orient)).as_rotvec()  # R_fix @ R_old
    trans = R_fix.apply(transl)

    # ---- ground alignment: GVHMR's world origin is arbitrary (camera init);
    # AMASS-style data expects the lowest foot at z=0 ----
    import smplx

    model_folder = str(HERE / ".." / "assets" / "body_models")
    model = smplx.create(model_folder, "smplx", gender="neutral", use_pca=False)

    def fk(transl):
        return model(
            betas=torch.tensor(betas).float().view(1, -1).expand(num_frames, -1),
            global_orient=torch.tensor(root_orient).float(),
            body_pose=torch.tensor(pose_body).float(),
            transl=torch.tensor(transl).float(),
            jaw_pose=torch.zeros(num_frames, 3),
            leye_pose=torch.zeros(num_frames, 3),
            reye_pose=torch.zeros(num_frames, 3),
            left_hand_pose=torch.zeros(num_frames, 45),
            right_hand_pose=torch.zeros(num_frames, 45),
            expression=torch.zeros(num_frames, 10),
        )

    joints = fk(trans).joints.detach().numpy()  # (N, 127, 3)
    min_foot_z = joints[:, [10, 11], 2].min()   # SMPL-X: 10=left_foot, 11=right_foot
    print(f"Ground alignment: shifting motion down by {min_foot_z:.3f} m")
    trans = trans - min_foot_z

    # ---- fps ----
    fps = args.fps
    if fps is None and args.video is not None:
        import cv2

        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        cap.release()
    if fps is None:
        fps = 30.0
    print(f"Using source fps = {fps}")

    np.savez(
        args.output,
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(fps),
        root_orient=root_orient.astype(np.float64),
        pose_body=pose_body.astype(np.float64),
        trans=trans.astype(np.float64),
        betas=betas.astype(np.float64),
    )
    print(f"Saved: {args.output}")

    # ---- sanity check: FK a few frames, verify z is really up ----
    if not args.no_check:
        idx = np.linspace(0, num_frames - 1, min(10, num_frames)).astype(int)
        joints = fk(trans).joints.detach().numpy()
        pelvis_z = joints[idx, 0, 2]
        head_z = joints[idx, 15, 2]
        foot_z = joints[idx][:, [10, 11], 2].min(axis=1)
        print(f"[check] pelvis z: {pelvis_z.mean():.3f} m (expect ~0.8-1.1)")
        print(f"[check] head   z: {head_z.mean():.3f} m (expect ~1.4-1.8)")
        print(f"[check] foot min z: {foot_z.mean():+.3f} m (expect ~0 +-0.15)")
        if not (1.2 < head_z.mean() < 2.1):
            print("[check] WARNING: head height out of range - wrong up-axis? try --up z")
        else:
            print("[check] OK: z is the up-axis")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Align the depth-measured box trajectory (camera frame) into the G1 render
world using the carry phase as correspondence, then validate on the ground
phases.

Correspondence: while carried, box_cam(t) (real meters, OpenCV camera frame)
maps rigidly+scale to a point near hand_mid_world(t) (pkl world). Fit s,R,t
(Umeyama) on a seed window, refine with inlier iterations, then check:

  * ground phase (box parked far, start & end): transformed z should sit at
    floor + s * box_half_height  -> independent scale/tilt validation
  * fitted camera axes vs world: up(-y_cam)->+z_world, toward-cam(-z_cam)
    should match the robot's net walk direction

Gravity lock (--box_pose only): the retargeted hand trajectory can tilt the
Umeyama up-axis by several degrees, and a carry that runs mostly along the
camera depth axis turns that tilt directly into box-height error at the
parked phases (measured here: 6.6 deg of lean -> 0.12 m of missing floor->
stool rise).  A box resting on the floor has its height axis along gravity,
so the PARKED cuboid orientations give an independent camera-frame gravity
measurement; R is then locked so gravity maps onto world +z, and s,t are
re-solved with R fixed.  Disable with --no_gravity_lock.

Usage:
  python scripts/align_box_to_world.py \
      --robot_motion data/20260911211441_unitree_g1.pkl \
      --box_traj data/box_traj_depth.npz --out data/box_traj_world.npz
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from general_motion_retargeting import KinematicsModel, ROBOT_XML_DICT  # noqa: E402
from general_motion_retargeting.data_loader import load_robot_motion  # noqa: E402
from cuboid_fit import canonicalize_rotations, pose_err  # noqa: E402
from vis_robot_with_box import compute_hand_frames  # noqa: E402


def umeyama(X, Y):
    """Least-squares s, R, t with Y ~= s * R @ X + t. Self-tested below."""
    n = len(X)
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    H = Xc.T @ Yc / n                      # sum x y^T / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ S @ U.T
    var_x = (Xc ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_x
    t = my - s * R @ mx
    return float(s), R, t


def _selftest():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    th = 0.7
    R0 = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    s0, t0 = 1.7, np.array([0.3, -2.0, 0.5])
    Y = s0 * (X @ R0.T) + t0 + rng.normal(scale=1e-6, size=X.shape)
    s, R, t = umeyama(X, Y)
    assert abs(s - s0) < 1e-4 and np.abs(R - R0).max() < 1e-5 \
        and np.abs(t - t0).max() < 1e-4, "umeyama selftest FAILED"
    return True


def fit_sim3(X, Y, up_cam=None):
    """Umeyama s,R,t, optionally with R's up-axis locked onto gravity.

    up_cam: unit gravity-UP direction in CAMERA coords (measured from the
    parked box).  When given, the unconstrained Umeyama R is corrected by
    the minimal rotation that maps R @ up_cam onto world +z, and s,t are
    re-solved by least squares with R fixed.  Use case: the carry-phase
    correspondence (retargeted hands) can tilt the fitted up-axis; locking
    it to the independently measured gravity removes that tilt.
    """
    s, R, t = umeyama(X, Y)
    if up_cam is None:
        return s, R, t
    wu = R @ up_cam
    q = np.cross(wu, np.array([0.0, 0.0, 1.0]))
    n = float(np.linalg.norm(q))
    if n > 1e-9:
        axis = q / n
        th = np.arctan2(n, float(np.dot(wu, [0.0, 0.0, 1.0])))
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        Q = np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)
        R = Q @ R
    # re-solve s and t with R fixed: minimize ||Y - (s * X R^T + t)||^2
    Z = X @ R.T
    mz, my = Z.mean(axis=0), Y.mean(axis=0)
    Zc, Yc = Z - mz, Y - my
    s = float((Yc * Zc).sum() / (Zc ** 2).sum())
    t = my - s * mz
    return s, R, t


def best_frame_offset(curve_a, curve_b, span=40):
    """curve_a has N_a samples (pkl), curve_b N_b (bag). Find integer offset k
    so that a[i] pairs with b[i+k]; maximize Pearson correlation.
    Returns (k, r_best, r_runner_up): on flat-ground carries the peak is
    wide (1-frame ambiguity), and a peak pinned at +/-span means the search
    window was too narrow -- both are worth warning about."""
    rs = []
    n = min(len(curve_a), len(curve_b))
    for k in range(-span, span + 1):
        if k >= 0:
            ia, ib = slice(0, n - k), slice(k, n)
        else:
            ia, ib = slice(-k, n), slice(0, n + k)
        a, b = curve_a[ia], curve_b[ib]
        if len(a) < 30:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if np.isfinite(r):
            rs.append((float(r), k))
    rs.sort(reverse=True)
    if len(rs) < 2:
        return 0, -2.0, -2.0
    (r_best, k_best), (r_2, _) = rs[0], rs[1]
    return k_best, r_best, r_2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_motion", required=True)
    ap.add_argument("--box_traj", default="data/box_traj_depth.npz")
    ap.add_argument("--box_pose", default=None,
                    help="optional npz from box_traj_from_depth.py "
                         "--fit-cuboid; adds a real box ORIENTATION to the "
                         "world output instead of the hard-coded upright+"
                         "constant-yaw pose.  Incompatible with --flip_x.")
    ap.add_argument("--out", default="data/box_traj_world.npz")
    ap.add_argument("--seed_frac", nargs=2, type=float, default=[0.35, 0.85],
                    help="seed carry window as fraction of frames")
    ap.add_argument("--inlier_m", type=float, default=0.22,
                    help="carry inlier threshold in world meters")
    ap.add_argument("--box_half_h", type=float, default=0.225,
                    help="real box half height, meters (center when on floor). "
                         "Box origin = centre of the 450 mm MAIN BODY "
                         "(two 40 mm bosses on top excluded), so 0.225.")
    ap.add_argument("--stool_h", type=float, default=0.45,
                    help="real height of the stool the box is dropped on, "
                         "meters; used by the post-drop centre-height check")
    ap.add_argument("--no_gravity_lock", action="store_true",
                    help="disable the parked-box gravity lock on R (only "
                         "relevant with --box_pose)")
    ap.add_argument("--no_rot_canon", action="store_true",
                    help="disable canonicalising the fitted R sequence over "
                         "the box's 8-fold symmetry.  Only for debugging: "
                         "uncanonicalised output flickers between equivalent "
                         "solutions and makes any downstream smoothing or "
                         "static-phase averaging meaningless.")
    ap.add_argument("--flip_x", action="store_true",
                    help="mirror the camera-frame x axis before fitting. Use "
                         "when the axis check reports toward-cam vs walk ~180 "
                         "deg (depth trajectory alone has no orientation, so "
                         "the fit can converge to the mirrored solution)")
    args = ap.parse_args()

    _selftest()

    # ---- G1 hand midpoint per frame ----
    import torch
    motion_data, _, _, _, _, _, _ = load_robot_motion(args.robot_motion)
    km = KinematicsModel(str(ROBOT_XML_DICT["unitree_g1"]), device="cpu")
    root_pos = torch.tensor(motion_data["root_pos"], dtype=torch.float32)
    root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float32)
    dof_pos = torch.tensor(motion_data["dof_pos"], dtype=torch.float32)
    with torch.no_grad():
        body_pos, _ = km.forward_kinematics(root_pos, root_rot, dof_pos)
    body_pos = body_pos.numpy()
    hand_mid, _ = compute_hand_frames(body_pos, km.body_names)
    n_pkl = len(hand_mid)

    # ---- measured box trajectory in camera frame ----
    d = np.load(args.box_traj)
    pos_cam, valid = d["pos_cam"], d["valid"]
    if args.flip_x:
        pos_cam = pos_cam.copy()
        pos_cam[:, 0] *= -1.0
        print("flip_x: camera-frame x mirrored before fitting")
    n_bag = len(pos_cam)

    # ---- optional 6-DoF box pose (camera frame) ----
    pose = None
    pose_size = None
    if args.box_pose:
        if args.flip_x:
            raise SystemExit(
                "--flip_x mirrors the camera-frame cloud, so the fitted R is "
                "the mirrored solution and cannot be applied to a box "
                "rotation. Use one flag or the other, not both.")
        dp = np.load(args.box_pose)
        if "R_cam" not in dp.files or "t_cam" not in dp.files:
            raise SystemExit(
                f"{args.box_pose} has no R_cam/t_cam — re-run "
                "box_traj_from_depth.py with --fit-cuboid to produce them.")
        R_cam, t_cam = dp["R_cam"], dp["t_cam"]
        pv = (dp["pose_valid"] if "pose_valid" in dp.files
              else np.isfinite(t_cam[:, 0]))
        if len(t_cam) != n_bag:
            raise SystemExit(
                f"{args.box_pose} has {len(t_cam)} frames but "
                f"{args.box_traj} has {n_bag}; they must come from the same "
                "run of box_traj_from_depth.py.")
        # ---- make the R sequence continuous in time ------------------------
        # The fit resolves the box's 8-fold symmetry independently per frame,
        # so the raw sequence flickers between equivalent solutions.  Nothing
        # downstream (gravity lock, orientation smoothing, a parked-phase
        # median) can work on a sequence full of 90 deg teleports, and the
        # rendered box twitches in place even while it is untouched on the
        # floor.  Canonicalise along the timeline first.
        if not getattr(args, "no_rot_canon", False):
            R_cam = canonicalize_rotations(R_cam, valid=pv)
            jumps = 0
            sel = np.nonzero(pv)[0]
            for a, b in zip(sel[:-1], sel[1:]):
                if pose_err(R_cam[a], R_cam[b]) > 5.0:
                    jumps += 1
            print(f"rotation canonicalised over {int(pv.sum())} measured "
                  f"frames (symmetry-equivalent solutions made continuous; "
                  f"{jumps} remaining >5 deg steps)")
        pose = (R_cam, t_cam, pv)
        pose_size = (dp["box_size"] if "box_size" in dp.files else None)
        print(f"box_pose: {int(pv.sum())}/{len(pv)} frames carry a 6-DoF "
              f"measurement"
              + (f" (fitted box size {pose_size.round(3)} m)"
                 if pose_size is not None else ""))

    # ---- frame offset via height curves (pkl up = +z hand; bag up = -y) ----
    off, corr, corr2 = best_frame_offset(hand_mid[:, 2], -pos_cam[:, 1])
    print(f"frame offset: bag[i+({off})] <-> pkl[i]   (corr {corr:.3f}, "
          f"{n_pkl} pkl / {n_bag} bag frames)")
    if abs(off) >= 40:
        print("  WARNING: offset pinned at the +/-40 frame search boundary; "
              "the true peak may lie outside -- widen span in "
              "best_frame_offset()")
    if corr - corr2 < 0.005:
        print(f"  note: correlation peak is flat (best {corr:.4f} vs "
              f"runner-up {corr2:.4f}); a +/-1 frame (~33 ms) ambiguity "
              "exists and is left for the inlier gate to absorb")

    def pair(idx_pkl):
        """pkl frame indices -> (box_cam, hand_world) pairs."""
        ib = idx_pkl + off
        m = (ib >= 0) & (ib < n_bag) & valid[np.clip(ib, 0, n_bag - 1)]
        return pos_cam[ib[m]], hand_mid[idx_pkl[m]], m

    n = n_pkl
    paired_idx = np.nonzero(pair(np.arange(n))[2])[0]

    def run_fit(up_lock=None, tag=""):
        """Seed Umeyama + inlier refinement (iterate over PAIRED frames
        only: the frame offset makes valid frames arrive in fragments, so
        allidx (n) and the paired subset have different lengths)."""
        seed = np.arange(int(args.seed_frac[0] * n),
                         int(args.seed_frac[1] * n))
        X, Y, _ = pair(seed)
        s_, Rt_, t_ = fit_sim3(X, Y, up_lock)
        print(f"{tag}seed fit on {len(X)} frames: s={s_:.3f}")
        for it in range(4):
            Xa, Ya, _ = pair(paired_idx)
            Wp = s_ * (Xa @ Rt_.T) + t_
            dist = np.linalg.norm(Wp - Ya, axis=1)
            inl = dist < args.inlier_m
            if inl.sum() < 30:
                break
            X, Y, _ = pair(paired_idx[inl])
            s_, Rt_, t_ = fit_sim3(X, Y, up_lock)
            print(f"{tag}iter {it + 1}: inliers {inl.sum()}/{len(paired_idx)}"
                  f"  median {np.median(dist[inl]) * 1000:.0f} mm  "
                  f"p90 {np.percentile(dist[inl], 90) * 1000:.0f} mm  "
                  f"s={s_:.3f}")
        return s_, Rt_, t_

    def carry_split_gate(s_, Rt_, t_):
        """Final carry/ground split from the hand-distance gate. dist is indexed over PAIRED frames
        only; scatter it back to full pkl-frame length (valid frames can
        come in fragments when the frame offset shifts the bag timeline,
        and mixing the two lengths here silently corrupted the split).

        Temporal run logic: keep only the inlier RUN containing the seed
        window (bridging gaps of <=3 frames).  After the drop the robot
        stands next to the stool and its hands can idle within the inlier
        gate of the parked box -- that is a separate episode in time, not
        carry, and calling it carry would both corrupt the render (box
        glued to hands after the drop) and empty the post-drop validation
        set."""
        Xa, Ya, m = pair(np.arange(n))
        dist = np.linalg.norm((s_ * (Xa @ Rt_.T) + t_) - Ya, axis=1)
        pidx = np.nonzero(m)[0]
        inl = np.zeros(n, dtype=bool)
        inl[pidx[dist < args.inlier_m]] = True
        # bridge small gaps so a 1-3 frame dropout does not split the run
        ii = np.nonzero(inl)[0]
        for a, b in zip(ii[:-1], ii[1:]):
            if 2 <= b - a <= 4:
                inl[a:b] = True
        # contiguous runs; keep the one containing the seed midpoint
        runs = []
        i = 0
        while i < n:
            if inl[i]:
                j = i
                while j + 1 < n and inl[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        seed_mid = int(0.5 * (args.seed_frac[0] + args.seed_frac[1]) * n)
        keep = next((r for r in runs if r[0] <= seed_mid <= r[1]), None)
        if keep is None and runs:
            keep = max(runs, key=lambda r: r[1] - r[0])
        c = np.zeros(n, dtype=bool)
        if keep:
            a, b = keep
            # NOTE: this fallback deliberately does NOT try to walk the head
            # back to the true lift start.  A heuristic that did (extending
            # carry backwards to where the box's speed left the noise floor)
            # was tried and REMOVED: dragging rising frames out of the parked
            # set corrupts the gravity-lock axes, which re-solves s,t and moved
            # the measured stool rise from 11 mm to 44 mm.  Use motion_split()
            # -- it has the box's own height to work with and is Sim3-free.
            # Trim the run's tail: after the drop the hands can idle INSIDE
            # the inlier gate (the run stays contiguous, so the gap-based
            # split above never fires).  The box itself stops moving, so cut
            # where its own smoothed speed goes quiet and STAYS quiet to the
            # end of the run.  Speeds are in real metres per frame (Ya is in
            # world metres -> divide by the scale s_).
            # Ya is indexed over PAIRED frames; map pkl indices -> positions
            sel = [i for i in range(a, b + 1) if m[i]]
            if len(sel) >= 20:
                pidx = np.nonzero(m)[0]
                pos = np.searchsorted(pidx, np.array(sel))
                y = Ya[pos]
                sp = np.linalg.norm(np.diff(y, axis=0), axis=1) / max(s_, 1e-9)
                w = min(5, len(sp))
                sm = np.convolve(sp, np.ones(w) / w, mode="same")
                quiet = sm < 0.008          # <8 mm real per frame = parked
                cut = next((j for j in range(len(quiet) - 8)
                            if quiet[j:].all()), None)
                if cut is not None and cut > 0:
                    b = int(pidx[pos[cut]]) - 1
                    print(f"carry tail trimmed at pkl frame {b}: box speed "
                          f"quiet from there to the end (post-drop idle was "
                          f"inside the inlier gate)")
            c[a:b + 1] = True
        return m, c

    def motion_split():
        """carry/ground split from the BOX's OWN height (preferred).

        The hand-distance gate above opens only once the robot's hands are
        within inlier_m of the box CENTRE, and a person grips the box by its
        top corners -- so on a fast grab the box is already rising while
        carry is still 0.  The render's z policy pins every non-carry frame
        to the rest height and anchors the measured z back onto that height
        at first_c, so a late first_c makes the box snap up in one frame
        (20260922: 225 mm at f161->162, caught by check_render_quality).

        The box's camera-frame centre height and speed are INDEPENDENT of the
        Sim3 (world = s*(Rt @ x) + t, and Rt changes neither norms nor the
        direction "up"), so the split can be taken from the measured pose
        alone -- no circular dependence on the fit.  Start = the box leaves
        the floor (sustained height step); end = it stops moving after being
        set down.  The raw centre jitters ~20 mm/frame, so both tests run on
        a 9-frame running median.  Returns (covered, carry) or None."""
        if pose is None:
            return None
        R_cam, t_cam, pv = pose
        ib = np.arange(n) + off
        inb = (ib >= 0) & (ib < n_bag)
        cl = np.clip(ib, 0, n_bag - 1)
        cov = np.zeros(n, bool)
        cov[inb] = (np.asarray(valid)[cl][inb] > 0) & (pv[cl][inb] > 0)
        idxc = np.nonzero(cov)[0]
        if idxc.size < 60:
            return None
        C = np.asarray(t_cam, dtype=float)[ib[idxc], :]
        kw = 9
        pad = np.pad(C, ((kw // 2, kw // 2), (0, 0)), mode="edge")
        Cs = np.stack([np.median(pad[j:j + kw], axis=0)
                       for j in range(len(C))])
        up = -Cs[:, 1]                 # camera y points down -> -y is up
        m0 = max(int(0.10 * len(C)), 5)
        h_lo = float(np.median(up[:m0]))
        h_hi = float(np.median(up[-m0:]))
        if h_hi - h_lo < 0.15:
            return None                # no real lift in this clip
        # The running median leaves only ~4 mm of noise, so trigger the split
        # within 1.5 sigma of the parked floor level: at that instant the box
        # has risen just a few mm, so the renderer's z anchor (which forces
        # the carry z through rest_floor at first_c) meets the pinned floor
        # height with no visible step.  A plain percentage of the rise would
        # trigger 60-90 mm late and leave a 50 mm snap at pickup.
        sig = 1.4826 * float(np.median(np.abs(up[:m0] - h_lo)))
        tol = min(max(0.004, 1.5 * sig), 0.025)
        above = up > h_lo + tol        # the box has left the floor
        # Take the LONGEST sustained excursion above the floor band, not the
        # first one: a parked box still drifts a few mm (quantised depth), so
        # short spurious runs open early in the clip (20260922: f34..f40 sat
        # 6-13 mm high).  The real lift is the run that never comes back --
        # it ends only when the clip does, because the box finishes on the
        # stool, above the floor band.
        runs = []
        j = 0
        while j < len(C):
            if above[j]:
                k = j
                while k + 1 < len(C) and above[k + 1]:
                    k += 1
                runs.append((k - j + 1, j))
                j = k + 1
            else:
                j += 1
        if not runs or max(runs)[0] < 20:
            return None
        fm_sus = max(runs)[1]
        # Back off to the last frame still inside the floor band: the carry
        # then starts while the box has risen by at most `tol`, so the
        # renderer's z anchor meets the pinned floor height with only a
        # couple of mm of step (starting at the first frame ABOVE the band
        # instead leaves 12-60 mm of snap -- enough to trip the 10 mm
        # parked-phase continuity check on a noisier clip).
        fm = max(fm_sus - 1, 0)
        while fm > 0 and above[fm]:
            fm -= 1
        if os.environ.get("GMR_MOTION_DEBUG"):
            print("[dbg] h_lo=%.4f sig=%.4f tol=%.4f fm_sus=%d fm=%d"
                  % (h_lo, sig, tol, fm_sus, fm))
            lo_d = max(fm - 6, 0)
            for j in range(lo_d, min(fm + 8, len(C))):
                print("[dbg]   j=%3d pkl f%3d  up-h_lo=%+7.2f mm  above=%d"
                      % (j, idxc[j], (up[j] - h_lo) * 1000, bool(above[j])))
        sp = np.zeros(len(C))
        sp[1:] = np.linalg.norm(np.diff(Cs, axis=0), axis=1)
        w = 5
        sm = np.convolve(sp, np.ones(w) / w, mode="same")
        mov = sm > 0.008               # > 8 mm real/frame and sustained
        lm = next((j for j in range(len(C) - 1, 4, -1)
                   if mov[j - 4:j + 1].all()), None)
        if lm is None or lm - fm < 15:
            return None
        c = np.zeros(n, bool)
        c[idxc[fm]:idxc[lm] + 1] = True
        print(f"motion split: box leaves the floor at pkl f{idxc[fm]} "
              f"(height {h_lo:.3f}->{h_hi:.3f} m, tol {tol * 1000:.0f} mm), "
              f"set down and still at f{idxc[lm]}  [Sim3-free: the box's own "
              f"height/speed in the camera]")
        return cov, c

    def carry_split(s_, Rt_, t_):
        """Motion-based split when the 6-DoF pose is available, else the gate."""
        r = motion_split()
        if r is not None:
            return r
        print("motion split unavailable (no 6-DoF pose, or no clear motion "
              "run) -> falling back to the hand-distance gate")
        return carry_split_gate(s_, Rt_, t_)

    s, Rt, t = run_fit()

    # ---- gravity lock: measure camera-frame gravity from the PARKED box ----
    # The retargeted hand trajectory can tilt the Umeyama up-axis (on this
    # recording it leaned ~6.6 deg with a 0.11 component along camera depth,
    # and a carry running 1.4 m along that axis ate 0.12 m of the floor->
    # stool rise).  A box resting on the floor/stool has its height axis
    # along gravity, so the parked cuboid orientations give an independent
    # gravity measurement -- no Sim3 involved.
    if pose is not None and not args.no_gravity_lock:
        R_cam, t_cam, pv = pose
        if pose_size is None:
            print("gravity lock skipped: pose npz carries no box_size")
        else:
            msk1, carry1 = carry_split(s, Rt, t)
            hcol = int(np.argmax(pose_size))
            axes = []
            for i in np.nonzero(msk1 & ~carry1)[0]:
                ib = i + off
                if 0 <= ib < n_bag and pv[ib]:
                    a = np.asarray(R_cam[ib][:, hcol], dtype=float)
                    if abs(a[1]) < 0.5:
                        continue    # near-horizontal: per-frame sign unreliable
                    if a[1] > 0:
                        a = -a      # canonicalize to point up (y_cam is down)
                    axes.append(a)
            if len(axes) < 8:
                print(f"gravity lock skipped: only {len(axes)} parked "
                      "frames with a usable height axis")
            else:
                g = np.median(np.array(axes), axis=0)
                g /= np.linalg.norm(g)
                cam_pitch = np.degrees(np.arccos(np.clip(-g[1], -1, 1)))
                lean = float((Rt @ g)[2])
                corr_deg = np.degrees(np.arccos(np.clip(lean, -1, 1)))
                print(f"\ngravity from {len(axes)} parked frames: "
                      f"[{g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}] in camera "
                      f"coords (camera pitch vs -y_cam: {cam_pitch:.1f} deg)")
                if cam_pitch > 20:
                    print("  WARNING: camera looks strongly pitched; verify "
                          "the parked-box orientations before trusting this")
                print(f"  unconstrained fit leaned the up-axis {corr_deg:.1f} "
                      "deg off gravity -> locking R, re-solving s,t")
                s, Rt, t = run_fit(g, tag="[grav-locked] ")

    # ---- final transform + diagnostics ----
    world = np.full((n, 3), np.nan)
    Xa, Ya, msk = pair(np.arange(n))
    world[msk] = s * (Xa @ Rt.T) + t

    up_cam = Rt @ np.array([0.0, -1.0, 0.0])      # camera "up" in world
    toward_cam = Rt @ np.array([0.0, 0.0, -1.0])  # optical axis reversed
    walk = hand_mid[-1] - hand_mid[0]
    walk_xy = walk[:2] / (np.linalg.norm(walk[:2]) + 1e-9)
    ang = lambda a, b: np.degrees(np.arccos(np.clip(
        float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12),
        -1, 1)))
    print(f"\naxis check: cam-up vs world +z = {ang(up_cam, [0, 0, 1]):.1f} deg"
          f"   toward-cam vs walk dir = {ang(toward_cam[:2], walk_xy):.1f} deg")
    print(f"scale camera->G1 world: {s:.3f}  "
          f"(G1 world is {1 / s:.2f}x real metres)")
    print(f"inlier gate: {args.inlier_m:.2f} world m = "
          f"{args.inlier_m / s:.3f} real m")

    # carry set = final inliers (see carry_split for the length-mismatch note)
    msk, carry = carry_split(s, Rt, t)

    # ground phases: not carried and valid
    ground = msk & ~carry
    gz = world[ground, 2]
    gz_clean = gz[(gz > -1.0) & (gz < 2.0)]
    floor_z = np.min(body_pos[:, km.body_names.index("left_ankle_roll_link"), 2])
    floor_z = min(floor_z, np.min(body_pos[
        :, km.body_names.index("right_ankle_roll_link"), 2]))
    print(f"floor z (G1 world): {floor_z:+.3f}")
    if len(gz_clean):
        med = np.median(gz_clean)
        spread = np.percentile(gz_clean, 90) - np.percentile(gz_clean, 10)
        # NOTE: pos_cam is the HSV blob's visible centroid, NOT the box
        # centre. With the camera above the box the blob covers the front
        # AND top faces, so the centroid sits well above 0.225 m (view-
        # dependent bias, constant while the camera does not move). The
        # VALIDATION metric is therefore the SPREAD (rigidity / tilt), not
        # the absolute height.
        print(f"ground-phase box z: median {med:+.3f} "
              f"(p10 {np.percentile(gz_clean, 10):+.3f}, "
              f"p90 {np.percentile(gz_clean, 90):+.3f}, "
              f"spread {spread * 1000:.0f} mm)  "
              f"=> implied centroid height {med / s:.3f} m "
              f"(> box centre {args.box_half_h:.3f} by the front+top-face "
              f"centroid bias; spread is the real check)")
    print(f"phase split: carry {carry.sum()}  ground {ground.sum()}  "
          f"invalid {(~msk).sum()}")

    # ---- optional: box ORIENTATION carried into the world frame ----
    payload = {}
    if pose is not None:
        R_cam, t_cam, pv = pose
        box_R = np.full((n, 3, 3), np.nan)
        box_c = np.full((n, 3), np.nan)
        for i in range(n):
            ib = i + off
            if 0 <= ib < n_bag and pv[ib]:
                box_R[i] = Rt @ R_cam[ib]
                box_c[i] = s * (Rt @ t_cam[ib]) + t
        got = np.isfinite(box_c[:, 0])
        payload.update(box_R_world=box_R, box_center_world=box_c,
                       pose_valid=got)
        print(f"\n6-DoF box pose in world coords: {int(got.sum())} frames")

        # Physical check the centroid route simply cannot do: a box parked on
        # the floor has its CENTRE at floor + half height.  pos_cam is a
        # view-dependent surface average that sits above the centre, so only
        # the cuboid centre can be compared against this.
        #
        # IMPORTANT: "ground" is TWO parked phases at different heights --
        # before the pickup the box is on the floor, after the drop it is on
        # the STOOL.  Pooling them produces a bimodal z whose median and spread
        # describe the pool, not the accuracy, and that is exactly how a
        # healthy fit first looked broken here.  Split them.
        ci = np.nonzero(carry & got)[0]
        gsel = got & ~carry
        if ci.size:
            pre = gsel & (np.arange(n) < ci[0])
            post = gsel & (np.arange(n) > ci[-1])
        else:
            pre, post = gsel, np.zeros(n, bool)
        # the render world draws the box and stool at REAL sizes, so the
        # height the renderer can honour is floor + surface + box_half_h.
        # The alignment stores the trajectory scaled by s, so the two differ
        # by 1/s; print both rather than picking one and hiding a factor 1.4.
        for nm, sel, surf in (("on the FLOOR (pre-pickup)", pre, 0.0),
                              ("on the STOOL (post-drop)", post,
                               args.stool_h)):
            exp_render = floor_z + surf + args.box_half_h
            exp_scaled = floor_z + s * (surf + args.box_half_h)
            if sel.sum() < 4:
                print(f"  {nm}: only {int(sel.sum())} frames, skipped")
                continue
            obs = box_c[sel, 2]
            med = float(np.median(obs))
            spread = (np.percentile(obs, 90) - np.percentile(obs, 10)) * 1000
            print(f"  {nm}: box CENTRE z median {med:+.3f}  spread "
                  f"{spread:.0f} mm  ({int(sel.sum())} frames)")
            print(f"      vs floor + half box scaled by s = {exp_scaled:+.3f} "
                  f"(err {med - exp_scaled:+.3f}) ; unscaled (= what the "
                  f"renderer draws) = {exp_render:+.3f} "
                  f"(err {med - exp_render:+.3f})")
        if pre.sum() > 4 and post.sum() > 4:
            rise = float(np.median(box_c[post, 2]) - np.median(box_c[pre, 2]))
            print(f"  floor->stool RISE of the box centre: {rise:+.3f} world m "
                  f"= {rise / s:+.3f} real m ; a {args.stool_h:.3f} m stool "
                  f"needs {args.stool_h * s:+.3f} world m.  The data implies "
                  f"a stool top {rise / s:.3f} m above the floor -- if the "
                  "real stool does not match, the translation scale s and "
                  "the object metric disagree, which is a property of this "
                  "project's render world, not of the cuboid fit.")
        print("  NOTE: this absolute z is the Umeyama constant, and the shipped "
              "renderer never uses the measured z for the parked phases -- it "
              "sets floor + half box directly.  vis_robot_with_box.py "
              "--box_pose_world anchors it the same way by default "
              "(--pose_z rest).")

        zw = box_R[got][:, :, 2]
        tilt = np.degrees(np.arccos(np.clip(zw[:, 2], -1, 1)))
        print(f"  box z-axis vs world up: median {np.median(tilt):.1f} deg  "
              f"p90 {np.percentile(tilt, 90):.1f} deg")

    print(f"\nwritten -> {args.out}")
    np.savez(args.out,
             box_pos_world=world, hand_mid=hand_mid, carry=carry,
             valid=msk, frame_offset=off, scale=s,
             R_world_from_cam=Rt, t_world=t,
             meta=json.dumps({
                 "source_traj": args.box_traj,
                 "source_motion": args.robot_motion,
                 "source_pose": args.box_pose,
                 "frame_offset": off, "scale": s,
                 "inlier_m": args.inlier_m,
                 "stool_h": args.stool_h,
                 "gravity_lock": bool(pose is not None
                                      and not args.no_gravity_lock),
             }, ensure_ascii=False), **payload)


if __name__ == "__main__":
    main()

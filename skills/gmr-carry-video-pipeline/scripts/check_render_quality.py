"""QA check for the final G1 carry-box render.

Usage (from the GMR-master repo root, with the gmr conda python):
    python .workbuddy/skills/gmr-carry-video-pipeline/scripts/check_render_quality.py ^
        [--npz .workbuddy/boxpose_used.npz]

The npz is the one dumped by vis_robot_with_box.py --dump_box_pose and must
contain: box_pos (N,3), box_quat_xyzw (N,4), cov (N,), carry (N,),
first_c, last_c.

What it reports (mirrors the QA acceptance numbers in SKILL.md):
  1. Per-phase (FLOOR parked / STOOL parked / CARRY) visible rotation step and
     position step statistics. "Visible" rotation is measured modulo the
     square-section box symmetry group (8 elements: Rz(90k) x {I, Rx(180)}),
     so a 90-degree yaw step of a square box does not count as motion.
  2. The largest visible rotation steps, with frame indices.
  3. An UNMASKED position-jump scan over every frame -- this is what catches
     teleports at the measurement start/end boundaries (they hide outside the
     per-phase masks). Acceptance: no jump > 50 mm except the genuine
     lift/descent transitions (those are flagged with the phase context).
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Acceptance thresholds (mm / deg) -- keep in sync with SKILL.md.
JUMP_MM_WARN = 50.0
CARRY_ROT_MAX = 3.5
# Parked segments are frozen medians; a small residual (stool-capture slide
# tail, capture xy slide) is fine. 0.5 deg / frame is invisible in the video.
PARKED_ROT_P90 = 0.5
PARKED_POS_MAX = 10.0


def build_symmetry_group():
    syms = []
    for k in range(4):
        Rz = Rot.from_euler("z", 90.0 * k, degrees=True)
        syms.append(Rz)
        syms.append(Rz * Rot.from_euler("x", 180.0, degrees=True))
    return syms


def vis_step(Ra, Rb, syms):
    """Rotation difference the EYE can see, modulo the box symmetry."""
    best = np.pi
    for S in syms:
        best = min(best, (Ra.inv() * Rb * S).magnitude())
    return np.degrees(best)


def find_latest_npz():
    """No --npz given: pick the newest dumped box pose so batch runs never
    have to pass the path by hand.  Prefers data/<prefix>_boxpose_used.npz
    (per-clip dumps, 2026-09 convention), falls back to the legacy
    .workbuddy/boxpose_used.npz."""
    import glob
    cands = glob.glob(os.path.join("data", "*_boxpose_used.npz"))
    legacy = os.path.join(".workbuddy", "boxpose_used.npz")
    if os.path.exists(legacy):
        cands.append(legacy)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None,
                    help="path to the dumped box-pose npz; default = newest "
                         "data/*_boxpose_used.npz, else "
                         ".workbuddy/boxpose_used.npz")
    ap.add_argument("--carry_rot_max", type=float, default=CARRY_ROT_MAX,
                    help="acceptance cap (deg/frame) for the CARRY visible "
                         "rotation step.  Pass the renderer's "
                         "--quat_max_step_deg value: when a clip has a REAL "
                         "box turn (the measurement shows a smooth ramped "
                         "yaw excursion, not isolated spikes) the honest cap "
                         "is the renderer's own slew limit, not the 3.5 deg "
                         "default that assumes a frozen box.")
    args = ap.parse_args()
    carry_rot_max = float(args.carry_rot_max)

    if args.npz is None:
        args.npz = find_latest_npz()
        if args.npz is None:
            print("ERROR: no box-pose npz found. Render with --dump_box_pose "
                  "first, or pass --npz explicitly.")
            sys.exit(2)
        print("(no --npz given; using the newest dump: %s)" % args.npz)

    if not os.path.exists(args.npz):
        print("ERROR: npz not found: %s" % args.npz)
        sys.exit(2)
    d = np.load(args.npz)
    pos, q = d["box_pos"], d["box_quat_xyzw"]
    cov, carry = d["cov"].astype(bool), d["carry"].astype(bool)
    fc, lc = int(d["first_c"]), int(d["last_c"])
    n = len(pos)
    idx = np.arange(n)
    print("loaded %s : %d frames, first_c=%d last_c=%d, cov=%d carry=%d"
          % (args.npz, n, fc, lc, cov.sum(), carry.sum()))

    syms = build_symmetry_group()
    Rq = Rot.from_quat(q)
    steps = np.array([vis_step(Rq[i], Rq[i + 1], syms) for i in range(n - 1)])
    dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0

    groups = [("FLOOR parked", cov & (idx < fc)),
              ("STOOL parked", cov & (idx > lc)),
              ("CARRY", carry & cov)]
    fail = False
    print("\n[1] per-phase statistics")
    for name, m in groups:
        if m.sum() < 3:
            print("%-14s n=%3d  (too few frames, skipped)" % (name, m.sum()))
            continue
        ii = np.nonzero(m)[0]
        ii = ii[ii < n - 1]
        print("%-14s n=%3d | visible rot step: med %6.3f deg  p90 %6.3f  max %6.3f | "
              "pos step: med %5.1f mm max %5.1f mm"
              % (name, len(ii), np.median(steps[ii]), np.percentile(steps[ii], 90),
                 steps[ii].max(), np.median(dpos[ii]), dpos[ii].max()))
        if "parked" in name:
            if np.percentile(steps[ii], 90) > PARKED_ROT_P90:
                print("    WARN: parked rot p90 > %.2f deg -- parked frames not frozen" % PARKED_ROT_P90)
                fail = True
            if dpos[ii].max() > PARKED_POS_MAX:
                print("    WARN: parked pos jump %.1f mm > %.0f mm" % (dpos[ii].max(), PARKED_POS_MAX))
                fail = True
        else:
            if steps[ii].max() > carry_rot_max + 1e-6:
                print("    WARN: carry rot max %.2f deg > %.1f deg slew cap" % (steps[ii].max(), carry_rot_max))
                fail = True

    print("\n[2] frames where the VISIBLE rotation step is largest")
    order = np.argsort(-np.where(cov[:-1], steps, 0.0))[:15]
    for k in sorted(order):
        print("  f%3d->%3d  %6.3f deg   (carry=%d cov=%d)"
              % (k, k + 1, steps[k], int(carry[k]), int(cov[k])))

    # UNMASKED jump scan over every frame -- catches boundary teleports that
    # per-phase masks hide.
    print("\n[3] unmasked position-jump scan (> %.0f mm)" % JUMP_MM_WARN)
    bad = np.nonzero(dpos > JUMP_MM_WARN)[0]
    if len(bad) == 0:
        print("  none -- clean")
    else:
        # lift/descent phase transitions are expected; everything else is a
        # teleport. A jump is "expected" only if carry/cov state differs.
        n_teleport = 0
        for k in bad:
            phase_change = (carry[k] != carry[k + 1]) or (cov[k] != cov[k + 1])
            tag = "lift/transition, expected" if phase_change else "TELEPORT"
            print("  %-24s f%3d->%3d  %6.1f mm  (carry=%d->%d cov=%d->%d)"
                  % (tag, k, k + 1, dpos[k], carry[k], carry[k + 1], cov[k], cov[k + 1]))
            if not phase_change:
                n_teleport += 1
        if n_teleport:
            fail = True

    print("\nRESULT: %s" % ("PASS" if not fail else "FAIL -- see WARN lines above"))
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()

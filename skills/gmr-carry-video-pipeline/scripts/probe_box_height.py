# -*- coding: utf-8 -*-
"""Probe the box-centre height signal used by align_box_to_world.motion_split.

Usage (repo root, gmr conda python):
    python .workbuddy/skills/gmr-carry-video-pipeline/scripts/probe_box_height.py
        [data/<prefix>_6dof.npz] [frame_offset]

Prints the 9-frame-median-filtered camera-frame height, its robust noise in
the floor phase, and the carry start (`first_c`) a range of thresholds would
pick -- the fastest way to see why the align step chose the frame it did.
Default path = the newest data/*_6dof.npz that is not a *_world_6dof.npz or
*_boxpose_used.npz (i.e. a cuboid-fit output). The frame offset defaults to 2
and only shifts the reported pkl indices; it must match the align log.
"""
import glob
import os
import sys

import numpy as np


def newest_fit_npz():
    c = [p for p in glob.glob(os.path.join("data", "*_6dof.npz"))
         if not p.endswith(("_world_6dof.npz", "_boxpose_used.npz"))]
    return max(c, key=os.path.getmtime) if c else None


path = sys.argv[1] if len(sys.argv) > 1 else newest_fit_npz()
if not path or not os.path.exists(path):
    raise SystemExit("no cuboid-fit npz found; pass data/<prefix>_6dof.npz")
print("probing %s" % path)
off = int(sys.argv[2]) if len(sys.argv) > 2 else 2
d = np.load(path, allow_pickle=True)
pv = np.asarray(d["pose_valid"]) > 0
t_cam = np.asarray(d["t_cam"], dtype=float)
valid = np.asarray(d["valid"]) > 0
n_bag = len(t_cam)
print("npz: %d bag frames, pose_valid %.0f%%, valid %.0f%%"
      % (n_bag, pv.mean() * 100, valid.mean() * 100))

n = n_bag - off if off >= 0 else n_bag + off
ib = np.arange(n) + off
inb = (ib >= 0) & (ib < n_bag)
cl = np.clip(ib, 0, n_bag - 1)
cov = np.zeros(n, bool)
cov[inb] = (valid[cl][inb] > 0) & (pv[cl][inb] > 0)
idxc = np.nonzero(cov)[0]
C = t_cam[ib[idxc], :]
kw = 9
pad = np.pad(C, ((kw // 2, kw // 2), (0, 0)), mode="edge")
Cs = np.stack([np.median(pad[j:j + kw], axis=0) for j in range(len(C))])
up = -Cs[:, 1]
m0 = max(int(0.10 * len(C)), 5)
h_lo = float(np.median(up[:m0]))
h_hi = float(np.median(up[-m0:]))
print("h_lo=%.4f  h_hi=%.4f  rise=%.4f real m" % (h_lo, h_hi, h_hi - h_lo))

# robust noise of the filtered height, measured on the clearly-parked floor band
floor = up[:m0]
print("raw    floor-phase height: median %.4f  MAD %.4f"
      % (float(np.median(floor)), float(np.median(np.abs(floor - np.median(floor))))))
sig = 1.4826 * float(np.median(np.abs(up[:m0] - h_lo)))
print("filtered floor-phase robust sigma = %.4f m (%.1f mm)" % (sig, sig * 1000))


def split_for(tol):
    """Reproduce align_box_to_world.motion_split's trigger for one tolerance:
    longest contiguous run above the band, then back off to the last frame
    still inside it."""
    above = up > h_lo + tol
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
        return None, None, None
    n_run, fm_sus = max(runs)[0], max(runs)[1]
    fm = max(fm_sus - 1, 0)
    while fm > 0 and above[fm]:
        fm -= 1
    return fm, fm_sus, n_run


tol = min(max(0.004, 1.5 * sig), 0.025)
fm, fm_sus, run = split_for(tol)
print("\nmotion_split would use tol = max(4 mm, 1.5 sigma) = %.1f mm" % (tol * 1000))
print("  -> longest above-band run %s frames, first_c = pkl f%s, "
      "height above floor %s"
      % (run, idxc[fm] if fm is not None else "-",
         "%.1f mm" % ((up[fm] - h_lo) * 1000) if fm is not None else "-"))
print("\nother thresholds (5-frame sustained, the OLD rule, shown for contrast):")
for thr in (0.008, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.0675):
    above = up > h_lo + thr
    f5 = next((j for j in range(len(C) - 4) if above[j:j + 5].all()), None)
    _, f_sus, n_run = split_for(thr)
    print("  thr %5.1f mm -> 5-frame rule pkl f%-5s | longest-run rule pkl f%-5s"
          " (run %s f)"
          % (thr * 1000, f5 if f5 is None else idxc[f5],
             f_sus if f_sus is None else idxc[f_sus], n_run))
print()
show = max(idxc[0], (idxc[fm] - 8) if fm is not None else 145)
print("filtered height around that frame (pkl frame : height / above-floor):")
for i in range(show, min(show + 20, n)):
    if cov[i]:
        j = int(np.searchsorted(idxc, i))
        print("  f%3d  %.4f  %+7.1f mm  %s"
              % (i, up[j], (up[j] - h_lo) * 1000,
                 "<- first_c" if j == fm else ""))

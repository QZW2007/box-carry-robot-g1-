# -*- coding: utf-8 -*-
"""Rigorous half-way timing inside RENDER vs RAW.
For each curve: find start plateau, end plateau, and the frames at
25/50/75% of the transition (for the OUT half and the BACK half separately).
This gives a like-for-like 'when did the rotation actually happen'."""
import numpy as np, pickle, os
from scipy.spatial.transform import Rotation as R
os.chdir(r".")

def uw(a, per):
    o = a.copy()
    for i in range(1, len(o)):
        if not (np.isfinite(o[i]) and np.isfinite(o[i-1])): continue
        while o[i]-o[i-1] > per/2: o[i] -= per
        while o[i]-o[i-1] < -per/2: o[i] += per
    return o

p = pickle.load(open('example/20260922_162859/data/20260922_162859_g1.pkl','rb'))
rr = p['root_rot']
body = uw(np.degrees(np.arctan2(2*(rr[:,3]*rr[:,2]+rr[:,0]*rr[:,1]),
                                1-2*(rr[:,1]**2+rr[:,2]**2))), 360)
n = len(body)
w = np.load('example/20260922_162859/data/20260922_162859_world_6dof.npz'); B = w['box_R_world']
raw = np.full(n, np.nan)
for i in range(B.shape[0]):
    if np.all(np.isfinite(B[i])): raw[i] = np.degrees(np.arctan2(B[i][1,0], B[i][0,0]))
rawu = uw(raw, 90)
d = np.load('example/20260922_162859/data/20260922_162859_boxpose_used.npz'); q = d['box_quat_xyzw']
ren = np.full(n, np.nan)
for i in range(len(q)):
    if d['cov'][i]:
        M = R.from_quat(q[i]).as_matrix(); ren[i] = np.degrees(np.arctan2(M[1,0], M[0,0]))
renu = uw(ren, 90)

bodN = body - np.nanmedian(body[146:152])
rawN = rawu - np.nanmedian(rawu[146:152])
renN = renu - np.nanmedian(renu[146:152])

def frac_frames(y, span, start, end, which):
    """Return frame where y reaches which% of the way from start-val to end-val."""
    a = np.nanmedian(y[span[0]:span[1]])
    b = np.nanmedian(y[span[2]:span[3]])
    tgt = a + (b-a)*which
    rng = range(span[0], span[3]) if b < a else range(span[0], span[3])
    prev = None
    for f in rng:
        if not np.isfinite(y[f]): continue
        if prev is not None:
            if (b < a and prev >= tgt >= y[f]) or (b > a and prev <= tgt <= y[f]):
                return f
        prev = y[f]
    return None

print("=== OUT half: from plateau(f146-152) to the extremum plateau ===")
# extremum plateau: body around f176-180, box around f180-184
print("curve   25%   50%   75%   (frames where the out-turn crosses 25/50/75%)")
for k, y, ext in (("BODY", bodN, (176,181)), ("RAWbox", rawN, (180,185)), ("RENbox", renN, (180,185))):
    a = np.nanmedian(y[146:152]); b = np.nanmedian(y[ext[0]:ext[1]])
    res = []
    for wq in (0.25, 0.5, 0.75):
        tgt = a + (b-a)*wq
        found = None
        for f in range(152, 190):
            if not np.isfinite(y[f]): continue
            if y[f] <= tgt: found = f; break
        res.append(found)
    print(f"{k:<8} {res[0]}   {res[1]}   {res[2]}     (a={a:.1f} b={b:.1f})")

print()
print("=== BACK half: from the extremum plateau back to the end plateau(f236-248) ===")
print("curve   25%   50%   75%")
for k, y, ext in (("BODY", bodN, (176,181)), ("RAWbox", rawN, (180,185)), ("RENbox", renN, (180,185))):
    a = np.nanmedian(y[ext[0]:ext[1]]); b = np.nanmedian(y[236:248])
    res = []
    for wq in (0.25, 0.5, 0.75):
        tgt = a + (b-a)*wq
        found = None
        for f in range(185, 250):
            if not np.isfinite(y[f]): continue
            if y[f] >= tgt: found = f; break
        res.append(found)
    print(f"{k:<8} {res[0]}   {res[1]}   {res[2]}     (a={a:.1f} b={b:.1f})")

# -*- coding: utf-8 -*-
"""For limiter 4.5 vs 20: list every frame where the OUTPUT yaw steps > 6 deg,
to see whether raising the limit reintroduces isolated 'twist' spikes
(a spike = isolated high step with low steps on both sides)."""
import numpy as np, pickle, os
from scipy.spatial.transform import Rotation as R, Slerp
os.chdir(r".")
w=np.load("data/20260922_165945_world_6dof.npz")
pR0=w["box_R_world"].copy(); cov=w["carry"].astype(bool); n=pR0.shape[0]
covered=np.nonzero(cov)[0]
syms8=[]
for k in range(4):
    Rz=R.from_euler("z",90.0*k,degrees=True)
    syms8.append(Rz); syms8.append(Rz*R.from_euler("x",180.0,degrees=True))
def snap(M):
    P=M.copy(); ref=R.from_matrix(P[covered[0]]); prev=None
    for j in covered:
        Rj=R.from_matrix(P[j])
        angs=[np.degrees((ref.inv()*Rj*S_).magnitude()) for S_ in syms8]
        k=int(np.argmin(angs))
        if k: Rj=Rj*syms8[k]
        P[j]=Rj.as_matrix()
        if prev is None or j==prev+1: ref=Rj
        prev=j
    return P
def nan_ma(a,ww):
    out=np.full_like(a,np.nan); hh=ww//2
    for i in range(len(a)):
        seg=a[max(0,i-hh):i+hh+1]
        fin=np.isfinite(seg).all(axis=-1)
        if fin.any(): out[i]=seg[fin].mean(axis=0)
    return out
def chain(P,mk=3,sq=5,step_max=4.5):
    rv_full=np.full((n,3),np.nan); rv_full[cov]=R.from_matrix(P[cov]).as_rotvec()
    sel=covered; rv=rv_full[sel]
    th=np.linalg.norm(rv,axis=1,keepdims=True)
    alt=rv*np.where(th>1e-9,(th-2*np.pi)/np.maximum(th,1e-9),1.0)
    for i in range(1,len(rv)):
        if np.linalg.norm(alt[i]-rv[i-1])<np.linalg.norm(rv[i]-rv[i-1]): rv[i]=alt[i]
    rv_full[sel]=rv
    if mk>=3:
        k=mk if mk%2==1 else mk+1
        pad=np.pad(rv_full[sel],((k//2,k//2),(0,0)),mode='edge')
        rv_full[sel]=np.stack([np.median(pad[i:i+k],axis=0) for i in range(len(sel))])
    if sq>1:
        rv_s=nan_ma(rv_full,sq); ok=cov&np.isfinite(rv_s).all(axis=1)
        rv_full=rv_full.copy(); rv_full[ok]=rv_s[ok]
    q=np.full((n,4),np.nan)
    for i in np.nonzero(cov)[0]:
        if np.isfinite(rv_full[i]).all(): q[i]=R.from_rotvec(rv_full[i]).as_quat()
    if step_max>0:
        prev=None
        for i in np.nonzero(cov)[0]:
            Ri=R.from_quat(q[i])
            if prev is None: prev=Ri; continue
            d=np.degrees((prev.inv()*Ri).magnitude())
            if d>step_max:
                tgt=R.from_quat(np.vstack([prev.as_quat(),Ri.as_quat()]))
                Ri=Slerp([0.,1.],tgt)(step_max/d)
            q[i]=Ri.as_quat(); prev=Ri
    return q
def yaw_of(q):
    out=np.full(n,np.nan)
    for i in range(n):
        if np.isfinite(q[i]).all():
            M=R.from_quat(q[i]).as_matrix(); out[i]=np.degrees(np.arctan2(M[1,0],M[0,0]))
    o=out.copy()
    for i in range(1,n):
        if not(np.isfinite(o[i]) and np.isfinite(o[i-1])): continue
        while o[i]-o[i-1]>45: o[i]-=90
        while o[i]-o[i-1]<-45: o[i]+=90
    return o
P=snap(pR0)
for lim in (4.5, 20.0):
    y=yaw_of(chain(P,3,5,lim))
    print(f"--- limiter {lim}: carry frames with |dyaw| > 6 ---")
    for f in range(146,281):
        if np.isfinite(y[f]) and np.isfinite(y[f-1]):
            d=abs(y[f]-y[f-1])
            if d>6:
                dprev=abs(y[f-1]-y[f-2]) if np.isfinite(y[f-2]) else np.nan
                dnext=abs(y[f+1]-y[f]) if np.isfinite(y[f+1]) else np.nan
                spike = (np.isfinite(dprev) and dprev<3 and np.isfinite(dnext) and dnext<3)
                print(f"   f{f}: {d:5.1f}  (prev {dprev:4.1f} next {dnext:4.1f}){ '   <-- ISOLATED SPIKE' if spike else ''}")
    print()

# -*- coding: utf-8 -*-
"""Full-range evidence: minAreaRect aspect + angle, and label offset, f150-330."""
import numpy as np, os, cv2
os.chdir(r".")

w = np.load("data/20260922_165945_world_6dof.npz")
B = w["box_R_world"]; n = B.shape[0]
def face_az(M):
    x = M[:,0]; return np.degrees(np.arctan2(x[1], x[0]))
az = np.full(n, np.nan)
for i in range(n):
    if np.all(np.isfinite(B[i])): az[i] = face_az(B[i])
def unwrap360(a):
    o = a.copy()
    for i in range(1, len(o)):
        if np.isfinite(o[i]) and np.isfinite(o[i-1]):
            while o[i]-o[i-1] > 180: o[i] -= 360
            while o[i]-o[i-1] < -180: o[i] += 360
    return o
au = unwrap360(az)
print("=== fitted box face azimuth (unwrap360), f150-330 step4 ===")
for f in range(150, 331, 4):
    if f < len(au) and np.isfinite(au[f]):
        print(f"  f{f:<4} {au[f]:8.1f}")

cap = cv2.VideoCapture("video/20260922_165945.mp4")
print("\n=== frame | box rect w,h,ang | aspect w/h | label x-frac (off from box cx / box w) ===")
f = 0
while True:
    ok, im = cap.read()
    if not ok: break
    if 150 <= f <= 330 and f % 3 == 0:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        mo = cv2.inRange(hsv, (5,110,120), (28,255,255))
        mo = cv2.morphologyEx(mo, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        cnts,_ = cv2.findContours(mo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line = f"  f{f:<4}"
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 1500:
                (cx,cy),(bw,bh),ang = cv2.minAreaRect(c)
                asp = bw/max(bh,1)
                line += f" rect({bw:5.1f},{bh:5.1f}) ang{ang:6.1f} asp{asp:5.2f}"
                # white label
                mw = cv2.inRange(hsv,(0,0,180),(180,60,255))
                cw,_ = cv2.findContours(mw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cw:
                    c2 = max(cw, key=cv2.contourArea)
                    if cv2.contourArea(c2) > 300:
                        M = cv2.moments(c2); lx = M["m10"]/M["m00"]
                        off = (lx-cx)/max(bw,1)
                        line += f"  labelx {off:+.2f}"
        print(line)
    f += 1
cap.release()

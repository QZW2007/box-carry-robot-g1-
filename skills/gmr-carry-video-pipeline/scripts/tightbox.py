# -*- coding: utf-8 -*-
"""Determine the true 3D rotation axis. The label face stays visible (area const),
so the label face normal stays roughly toward camera. But the box SILHOUETTE flips
wide->tall->wide. 

KEY IDEA: for a CUBOID 0.30 x 0.30 x 0.45 (taller than wide), rotating about the
VERTICAL axis by theta changes the projected width from 0.30 (face-on) to 0.42
(edge-on, diagonal). The projected HEIGHT stays ~0.45. So aspect w/h goes 0.67->0.93.
NOT enough to flip to 0.66.

But observed: f198 w=100 h=63 (aspect 1.59!) -> that's WIDE and SHORT. A 0.30x0.45
box CANNOT project as 100 wide x 63 tall unless it's being held at an angle, OR the
person is holding it tilted, OR it's a different orientation.

Let me just directly measure: take the box's 4 corner points via the fitted world pose
and the camera matrix, and see the projected shape. And ALSO track the TOP EDGE line.

Better: look at the actual images once more, zoomed on just the top face."""
import cv2, os, numpy as np
os.chdir(r".")
cap=cv2.VideoCapture("example/20260922_162859/video/20260922_162859.mp4")
frames=[176,182,188,192,196,200,204,208,214,220,240,252,258,264,270]
tiles=[]
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read()
    if not ok: continue
    h,w=im.shape[:2]
    # tighter: just the box
    cx,cy=int(w*0.47),int(h*0.52); cw,ch=int(w*0.17),int(h*0.34)
    x0,y0=max(cx-cw//2,0),max(cy-ch//2,0)
    c=im[y0:y0+ch,x0:x0+cw].copy()
    c=cv2.resize(c,(240,int(240*ch/cw)),interpolation=cv2.INTER_CUBIC)
    cv2.putText(c,"f%d"%f,(6,26),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    tiles.append(c)
H=max(t.shape[0] for t in tiles)
tiles=[np.vstack([t,np.zeros((H-t.shape[0],t.shape[1],3),np.uint8)]) if t.shape[0]<H else t for t in tiles]
rows=[np.hstack(tiles[i:i+5]) for i in range(0,len(tiles),5)]
sheet=np.vstack(rows)
ok,buf=cv2.imencode(".jpg",sheet,[cv2.IMWRITE_JPEG_QUALITY,94]); buf.tofile("agent/out/tightbox.jpg")
print("wrote",sheet.shape)

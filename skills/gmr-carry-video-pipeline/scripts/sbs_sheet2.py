# -*- coding: utf-8 -*-
"""Side-by-side REAL|REND sheets, jpg output + dedup-busting pixel tweak.

Usage: python sbs_sheet2.py <out_name> f1,f2,... [x0,y0,x1,y1]  (crop normalized 0-1)
REAL = video/<prefix>.mp4 at f+off, REND = video/<prefix>_g1_measured_box_6dof.mp4 at f
"""
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

prefix = "20260922_165945"
out_name = sys.argv[1]
frames = [int(x) for x in sys.argv[2].split(",")]
crop = None
if len(sys.argv) > 3 and sys.argv[3] != "-":
    crop = [float(x) for x in sys.argv[3].split(",")]
rend_override = sys.argv[4] if len(sys.argv) > 4 else None

worldp = "data/%s_world_6dof.npz" % prefix
d = np.load(worldp, allow_pickle=True)
off = int(json.loads(str(d["meta"]))["frame_offset"])

real_mp4 = "video/%s.mp4" % prefix
rend_mp4 = rend_override or "video/%s_g1_measured_box_6dof.mp4" % prefix

TW = 470
TH = 352


def grab(path, f):
    cap = cv2.VideoCapture(path)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(max(f, 0), nf - 1)))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


rows = []
for f in frames:
    tiles = []
    for path, ff, tag in ((real_mp4, f + off, "REAL"),
                          (rend_mp4, f, "REND")):
        im = grab(path, ff)
        if im is None:
            im = np.full((480, 640, 3), 30, np.uint8)
        if crop is not None:
            h, w = im.shape[:2]
            x0, y0, x1, y1 = crop
            im = im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        im = cv2.resize(im, (TW, TH))
        cv2.putText(im, "%s r%d" % (tag, ff), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        tiles.append(im)
    rows.append(np.hstack(tiles))

sheet = np.vstack(rows)
# dedup-busting: nudge one pixel
sheet[0, 0, 0] = (int(sheet[0, 0, 0]) + 1) % 255
out = os.path.join(REPO, ".workbuddy", "out", out_name)
ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
buf.tofile(out)
print("offset %+d -> %s %s" % (off, out, sheet.shape))

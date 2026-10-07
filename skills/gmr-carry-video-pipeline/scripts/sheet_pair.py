# -*- coding: utf-8 -*-
"""Source-vs-render paired contact sheet, zoomed.

For each requested frame: source frame on top, render frame below, both cropped
to the region of interest and scaled to the same width, tiled across the sheet.
Writes the PNG with cv2.imencode+tofile (cv2.imwrite dies on this CN path).

    python .workbuddy/sheet_pair.py SRC.mp4 REN.mp4 OUT.png 140,148,156 \
        [--cx 0.5 --cy 0.5 --w 0.45 --h 0.75 --scale 1.4]
"""
import argparse

import cv2
import numpy as np


def crop_scale(img, cx, cy, w, h, scale):
    H, W = img.shape[:2]
    w0, h0 = int(W * w), int(H * h)
    x0 = int(np.clip(cx * W - w0 / 2, 0, W - w0))
    y0 = int(np.clip(cy * H - h0 / 2, 0, H - h0))
    c = img[y0:y0 + h0, x0:x0 + w0]
    return cv2.resize(c, (0, 0), fx=scale, fy=scale,
                      interpolation=cv2.INTER_LANCZOS4)


ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("ren")
ap.add_argument("out")
ap.add_argument("frames")
ap.add_argument("--cx", type=float, default=0.5)
ap.add_argument("--cy", type=float, default=0.5)
ap.add_argument("--w", type=float, default=0.45)
ap.add_argument("--h", type=float, default=0.75)
ap.add_argument("--scale", type=float, default=1.4)
ap.add_argument("--src_off", type=int, default=0,
                help="src frame = f + src_off, ren frame = f "
                     "(bag[i+off] <-> pkl[i], so pass the world_6dof frame_offset)")
a = ap.parse_args()

frames = [int(x) for x in a.frames.split(",")]
caps = [cv2.VideoCapture(a.src), cv2.VideoCapture(a.ren)]
for cap, nm in zip(caps, ("src", "ren")):
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % nm)
print("src frames=%d  ren frames=%d"
      % (int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT)),
         int(caps[1].get(cv2.CAP_PROP_FRAME_COUNT))))

cols = []
for f in frames:
    tiles = []
    for k, cap in enumerate(caps):
        want = f + a.src_off if k == 0 else f
        cap.set(cv2.CAP_PROP_POS_FRAMES, want)
        ok, img = cap.read()
        if not ok:
            tiles = []
            break
        t = crop_scale(img, a.cx, a.cy, a.w, a.h, a.scale)
        lbl = "%s f%d" % ("SRC" if k == 0 else "REN", want)
        cv2.putText(t, lbl, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(t, lbl, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    if tiles:
        cols.append(np.vstack(tiles))
for cap in caps:
    cap.release()
if not cols:
    raise SystemExit("no frames read")
W = max(c.shape[1] for c in cols)
cols = [np.pad(c, ((0, 0), (0, W - c.shape[1]), (0, 0))) for c in cols]
sheet = np.hstack(cols)
ok, buf = cv2.imencode(".png", sheet)
buf.tofile(a.out)
print("wrote %s  %dx%d" % (a.out, sheet.shape[1], sheet.shape[0]))

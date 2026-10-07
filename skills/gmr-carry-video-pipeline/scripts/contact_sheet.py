# -*- coding: utf-8 -*-
"""Extract chosen frames from the rendered mp4 into a contact sheet PNG.

cv2.imwrite silently fails on the Chinese project path, so write the PNG via
imencode + tofile (project convention).
"""
import sys

import cv2
import numpy as np

src = sys.argv[1]
out = sys.argv[2]
frames = [int(x) for x in sys.argv[3].split(",")]
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    raise SystemExit("cannot open %s" % src)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
tiles = []
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    if not ok:
        continue
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.putText(img, "f%d" % f, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "f%d" % f, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 1, cv2.LINE_AA)
    tiles.append(img)
cap.release()
print("%s: %d frames available" % (src, n))
if tiles:
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".png", sheet)
    buf.tofile(out)
    print("wrote %s  %dx%d" % (out, sheet.shape[1], sheet.shape[0]))

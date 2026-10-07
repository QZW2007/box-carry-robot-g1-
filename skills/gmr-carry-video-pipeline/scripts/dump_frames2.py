# -*- coding: utf-8 -*-
"""Dump specific full-res frames from a video. Usage:
   python dump_frames2.py <video> <out_dir> f1,f2,f3,..."""
import cv2
import os
import sys

video = sys.argv[1]
outdir = sys.argv[2]
frames = [int(s) for s in sys.argv[3].split(",")]
os.makedirs(outdir, exist_ok=True)
tag = os.path.splitext(os.path.basename(video))[0]
cap = cv2.VideoCapture(video)
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, im = cap.read()
    if ok:
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        buf.tofile(os.path.join(outdir, "%s_f%03d.jpg" % (tag, f)))
    else:
        print("MISS f%d" % f)
cap.release()
print("done %d frames from %s" % (len(frames), video))

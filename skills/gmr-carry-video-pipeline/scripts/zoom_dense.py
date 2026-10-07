# -*- coding: utf-8 -*-
"""Very tight zoom on the box during the turn f175-f215 to read the box face angle
and the label side. Also f230-f290 for the 'turn back'."""
import cv2, os, numpy as np
os.chdir(r".")

def strip(frames, out, cw_frac=0.19, ch_frac=0.40, cx_frac=0.47, cy_frac=0.50, tw=170, cols=8):
    cap = cv2.VideoCapture("example/20260922_162859/video/20260922_162859.mp4")
    tiles = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
        if not ok: continue
        h, w = im.shape[:2]
        cx, cy = int(w*cx_frac), int(h*cy_frac)
        cw, ch = int(w*cw_frac), int(h*ch_frac)
        x0, y0 = max(cx-cw//2,0), max(cy-ch//2,0)
        c = im[y0:y0+ch, x0:x0+cw].copy()
        c = cv2.resize(c, (tw, int(tw*ch/cw)), interpolation=cv2.INTER_CUBIC)
        cv2.putText(c, "f%d"%f, (5,22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
        tiles.append(c)
    cap.release()
    H = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((H-t.shape[0], t.shape[1],3), np.uint8)]) if t.shape[0]<H else t for t in tiles]
    rows=[]
    for i in range(0, len(tiles), cols):
        r = tiles[i:i+cols]
        if len(r)<cols:
            r = r + [np.zeros((H, tw, 3), np.uint8)]*(cols-len(r))
        rows.append(np.hstack(r))
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 93]); buf.tofile(out)
    print("wrote", out, sheet.shape)

# turn-out段, dense
strip(list(range(176, 216, 2)), "agent/out/turnA.jpg")
# later段 (turn back?), sparse
strip(list(range(216, 292, 4)), "agent/out/turnB.jpg")

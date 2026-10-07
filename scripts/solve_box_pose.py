"""Per-frame box 6D pose from ArUco markers (re-shoot pipeline, step B).

Input:
  * video with ArUco-tagged box (front + top faces tagged),
  * camera_calib.json from calibrate_camera.py,
  * markers.json mapping marker id -> its 4 corner positions in the BOX frame
    (see --make-template), measured with a ruler.

Corner order everywhere (must match cv2.aruco detection order):
  topLeft, topRight, bottomRight, bottomLeft - as the marker looks when its
  printed side faces you upright.

Output (--out prefix):
  <out>.npz  per-frame box pose in the CAMERA frame (meters),
  <out>.csv  quick tabular view,
  report printed to console (detection rate, gaps, reprojection errors),
  optional --preview annotated mp4.

Gaps (frames with no visible marker, e.g. arms occluding while carried) are
fine: the rigid hand-box binding covers them; the report tells you where they are.

Quick pipeline sanity check without any footage:
    python scripts/solve_box_pose.py --selftest
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# core: one frame -> box pose
# --------------------------------------------------------------------------- #

def solve_frame(corners, ids, marker_defs, K, dist):
    """All visible markers pooled into one PnP. Returns dict or None."""
    obj_pts, img_pts = [], []
    for c, mid in zip(corners, ids.flatten()):
        key = str(int(mid))
        if key not in marker_defs:
            continue
        obj_pts.extend(marker_defs[key]["corners_3d"])
        img_pts.extend(c.reshape(-1, 2))
    if len(obj_pts) < 4:
        return None
    obj_pts = np.array(obj_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)

    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        obj_pts, img_pts, K, dist,
        reprojectionError=4.0, confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.linalg.norm(img_pts - proj.reshape(-1, 2), axis=1).mean())

    R, _ = cv2.Rodrigues(rvec)
    return {"R": R, "t": tvec.ravel(), "n_markers": len(obj_pts) // 4,
            "err_px": err}


def draw_box(frame, K, dist, R, t, dims):
    """Wireframe the box in its own frame, so the pose can be judged by eye."""
    w, d, h = (v / 2.0 for v in dims)
    keys = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    pts = np.array([[a * w, b * d, c * h] for a, b, c in keys], dtype=np.float32)
    img = cv2.projectPoints(pts, cv2.Rodrigues(R)[0], t, K, dist)[0]
    img = img.reshape(-1, 2).astype(int)
    i = {k: n for n, k in enumerate(keys)}
    edges = [((sx, sy, -1), (sx, sy, 1)) for sx in (-1, 1) for sy in (-1, 1)]
    edges += [((sx, -1, sz), (sx, 1, sz)) for sx in (-1, 1) for sz in (-1, 1)]
    edges += [((-1, sy, sz), (1, sy, sz)) for sy in (-1, 1) for sz in (-1, 1)]
    for a, b in edges:
        cv2.line(frame, tuple(img[i[a]]), tuple(img[i[b]]), (0, 255, 255), 1)


def process_video(args):
    with open(args.calib, encoding="utf-8") as fh:
        calib = json.load(fh)
    K = np.array(calib["K"])
    dist = np.array(calib["dist"])
    with open(args.markers_json, encoding="utf-8") as fh:
        marker_defs = json.load(fh)["markers"]
    for k, v in marker_defs.items():
        v["corners_3d"] = np.array(v["corners_3d"], dtype=np.float32)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != tuple(calib["width_height"]):
        print(f"WARNING: video {W}x{H} != calibration "
              f"{calib['width_height']}: K is NOT valid for this video")

    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dict))
    # OpenCV 5 removed the functional detectMarkers; use the class API.
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX",
                                            params.cornerRefinementMethod)
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    poses, writer = [], None
    if args.preview:
        path = args.preview
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W, H))

    up = max(1, int(args.upscale))
    box_dims = ([float(v) for v in args.box_dims.split(",")]
                if args.box_dims else None)
    if up > 1:
        print(f"detecting on a {up}x upscaled copy (markers are small); the "
              f"corners are scaled back so K stays valid as-is")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # A small marker binarises below the decoder's floor at native size and
        # is never detected at all.  Upscaling first recovers those frames.
        src = frame if up == 1 else cv2.resize(
            frame, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if up > 1 and ids is not None and len(ids):
            corners = [np.asarray(c, np.float32) / up for c in corners]
        result = None
        if ids is not None and len(ids):
            result = solve_frame(corners, ids, marker_defs, K, dist)
        poses.append(result)

        if writer is not None:
            if ids is not None and len(ids):
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if result:
                cv2.drawFrameAxes(frame, K, dist,
                                  cv2.Rodrigues(result["R"])[0], result["t"], 0.12)
                if box_dims:
                    draw_box(frame, K, dist, result["R"], result["t"], box_dims)
                cv2.putText(frame, f"{result['n_markers']} mk, {result['err_px']:.2f} px",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "no markers", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"preview: {args.preview}")

    # ---- save + report ----
    n = len(poses)
    valid = np.array([p is not None for p in poses])
    report(poses, valid, fps)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             valid=valid,
             R=np.array([p["R"] if p else np.eye(3) for p in poses]),
             t=np.array([p["t"] if p else np.full(3, np.nan) for p in poses]),
             err_px=np.array([p["err_px"] if p else np.nan for p in poses]),
             n_markers=np.array([p["n_markers"] if p else 0 for p in poses]),
             fps=fps)
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "valid", "n_markers", "err_px", "x", "y", "z"])
        for i, p in enumerate(poses):
            if p:
                w.writerow([i, 1, p["n_markers"], round(p["err_px"], 3),
                            *(round(v, 5) for v in p["t"])])
            else:
                w.writerow([i, 0, 0, "", "", "", ""])
    print(f"saved: {args.out}.npz / .csv")


def report(poses, valid, fps):
    n = len(poses)
    errs = np.array([p["err_px"] for p in poses if p])
    print(f"\n=== box pose report ===")
    print(f"frames: {n} ({n / fps:.1f} s) | with pose: {valid.sum()} "
          f"({100 * valid.mean():.0f}%)")
    if len(errs):
        print(f"reproj err px: median {np.median(errs):.2f} | "
              f"p95 {np.percentile(errs, 95):.2f} | max {errs.max():.2f}")
    gaps, start = [], None
    for i, v in enumerate(valid):
        if not v and start is None:
            start = i
        elif v and start is not None:
            gaps.append((start, i - 1))
            start = None
    if start is not None:
        gaps.append((start, n - 1))
    if gaps:
        big = [g for g in gaps if g[1] - g[0] >= 3]
        print(f"gaps >=4 frames (need binding to fill): "
              f"{[(f'{a}-{b}', round((b-a+1)/fps,1)) for a, b in big] or 'none'}")


# --------------------------------------------------------------------------- #
# selftest: synthetic render -> detect -> PnP -> compare against ground truth
# --------------------------------------------------------------------------- #

def selftest():
    """Renders markers at known box-frame spots with a synthetic camera and
    checks the recovered pose. Catches corner-order / axis-convention bugs."""
    s = 0.06           # marker side, m
    hx, hy, hz = 0.15, 0.12, 0.15
    marker_defs = {"0": {"corners_3d": [
                       [-s/2, -hy, +s/2], [ s/2, -hy, +s/2],      # TL, TR (front face)
                       [ s/2, -hy, -s/2], [-s/2, -hy, -s/2]]},    # BR, BL
                   "1": {"corners_3d": [                          # top face, print-up
                       [-s/2, +s/2, hz], [ s/2, +s/2, hz],        #   toward BACK edge
                       [ s/2, -s/2, hz], [-s/2, -s/2, hz]]}}      # BR, BL

    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    # Camera pose built physically: camera sits 0.6 m in front and 0.8 m above
    # the box center, looking straight at it -> both the front face (~50 deg
    # incidence) and the top face (~49 deg) are well within detection range.
    C = np.array([0.10, -0.60, 0.80])            # camera position, box frame
    zc = -C / np.linalg.norm(-C)                 # camera forward (toward box)
    xc = np.cross(zc, [0, 0, 1.0]); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R_gt = np.stack([xc, yc, zc])                # rows = cam axes in box frame
    t_gt = -R_gt @ C                             # box origin in camera frame

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # render 4x supersampled, then downscale - simulates real photo edges so
    # corner quantization does not dominate the PnP error
    SS = 4
    W, H = 640 * SS, 480 * SS
    K_ss = K.copy(); K_ss[:2] *= SS
    canvas = np.full((H, W), 245, np.uint8)
    for mid, spec in marker_defs.items():
        img = cv2.aruco.generateImageMarker(dictionary, int(mid), 400 * SS)
        src = np.float32([[0, 0], [400 * SS, 0], [400 * SS, 400 * SS], [0, 400 * SS]])
        obj = np.array(spec["corners_3d"], np.float32).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(obj, R_gt, t_gt, K_ss, dist)
        M = cv2.getPerspectiveTransform(src, proj.reshape(-1, 2).astype(np.float32))
        warped = cv2.warpPerspective(img, M, (W, H))
        mask = cv2.warpPerspective(np.full((400 * SS, 400 * SS), 255, np.uint8),
                                   M, (W, H))
        canvas[mask > 0] = warped[mask > 0]
    canvas = cv2.resize(canvas, (640, 480), interpolation=cv2.INTER_AREA)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX",
                                            params.cornerRefinementMethod)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(canvas)
    assert ids is not None and set(ids.flatten().tolist()) == {0, 1}, \
        f"detection failed, saw {ids}"

    result = solve_frame(corners, ids, marker_defs, K, dist)
    t_err = np.linalg.norm(result["t"] - t_gt) * 1000
    R_err = np.degrees(np.linalg.norm(
        cv2.Rodrigues(result["R"] @ R_gt.T)[0]))
    print(f"selftest: detected {result['n_markers']} markers, "
          f"reproj {result['err_px']:.2f} px")
    print(f"translation error: {t_err:.2f} mm | rotation error: {R_err:.3f} deg")
    ok = t_err < 2.0 and R_err < 0.3 and result["err_px"] < 1.0
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)


def make_template(path):
    s = 0.06
    hx, hy, hz = 0.15, 0.12, 0.15   # box half sizes - EDIT after measuring
    template = {
        "_readme": [
            "Box frame: origin at box CENTER, x = width (right), y = depth "
            "(toward back), z = up, meters.",
            "Corner order must be topLeft, topRight, bottomRight, bottomLeft "
            "as the marker appears upright to the camera.",
            "This template assumes a 30x24x30 cm box and 6 cm markers: "
            "marker 0 on the FRONT face centered, marker 1 on the TOP face "
            "centered. Measure your real box and adjust ALL numbers.",
        ],
        "markers": {
            "0": {"corners_3d": [[-s/2, -hy, +s/2], [s/2, -hy, +s/2],
                                  [s/2, -hy, -s/2], [-s/2, -hy, -s/2]],
                  "desc": "front face, upright"},
            "1": {"corners_3d": [[-s/2, s/2, hz], [s/2, s/2, hz],
                                  [s/2, -s/2, hz], [-s/2, -s/2, hz]],
                  "desc": "top face, print-up edge toward the BACK (+y); "
                          "TL,TR,BR,BL clockwise seen from above"},
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(template, fh, indent=2)
    print(f"template written: {path} - edit the numbers after measuring your box")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--calib", default="data/camera_calib.json")
    parser.add_argument("--markers_json", default="data/box_markers.json")
    parser.add_argument("--dict", default="DICT_4X4_50")
    parser.add_argument("--out", default="data/box_pose")
    parser.add_argument("--preview", default=None, help="annotated mp4 path")
    parser.add_argument("--upscale", type=int, default=1,
                        help="detect on an N-times upscaled copy; recovers small "
                             "markers that never binarise at native size")
    parser.add_argument("--box_dims", default=None,
                        help="draw a box wireframe, e.g. 0.30,0.30,0.45 (W,D,H m)")
    parser.add_argument("--make_template", default=None,
                        help="write a markers.json template to this path")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
    elif args.make_template:
        make_template(args.make_template)
    else:
        for f in (args.video, args.calib, args.markers_json):
            if not os.path.exists(f):
                raise SystemExit(f"missing input: {f}")
        process_video(args)


if __name__ == "__main__":
    main()

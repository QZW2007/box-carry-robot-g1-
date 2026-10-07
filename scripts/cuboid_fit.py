#!/usr/bin/env python
"""Marker-free 6-DoF cuboid pose from a segmented RGB-D point cloud.

Why this is not a plane-fitting problem
---------------------------------------
The obvious approach — per-pixel surface normals from the depth grid, then
cluster them into the front and top faces — was tried on the real
``20260911_211441.bag`` footage and it does not work there.  The box is only
30-70 px wide at 2-3.5 m on a 640x480 stream, so the depth gradient across one
pixel step is comparable to the depth noise; the estimated normals are close to
random.  Measured symptoms: recovered front-face width 0.43 m for a 0.30 m box,
and a fitted silhouette that missed the box entirely (IoU 0.00).

What works instead: the box has KNOWN dimensions, so its projected silhouette
is fully determined by its 6-DoF pose.  The colour mask, by contrast, is
sub-pixel accurate.  So the pose is recovered by maximising the silhouette IoU
against the mask, plus a weak depth-consistency term that removes the residual
sliding ambiguity when only one face is visible.

The plane-based orientation route was ABANDONED.  `normals_from_depth`,
`dominant_normals`, `_trimmed_plane` and `orient_from_planes` below have ZERO
call sites -- nothing in this module, and nothing anywhere in the repo, invokes
them (`orient_from_planes` itself only calls the other three, so the whole
chain is unreachable).  They are kept for REFERENCE ONLY, as the record of the
approach that this docstring explains why we do not use.  Do not "wire them up"
as an initialiser without re-measuring: they are exactly the estimator whose
front-face width came out 0.43 m for a 0.30 m box.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CAM_UP = np.array([0.0, -1.0, 0.0])     # image up in the OpenCV camera frame
CAM_BACK = np.array([0.0, 0.0, -1.0])   # outward normal of a camera-facing face

CORNERS8 = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                    dtype=np.float64)


def project(P, K):
    """(N,3) camera-frame points -> (N,2) pixels."""
    z = np.maximum(P[:, 2], 1e-6)
    return np.stack([K[0, 0] * P[:, 0] / z + K[0, 2],
                     K[1, 1] * P[:, 1] / z + K[1, 2]], axis=1)


def rotvec_to_R(v):
    """Rodrigues: rotation vector -> rotation matrix."""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v, dtype=np.float64) / th
    Kx = np.array([[0.0, -k[2], k[1]],
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def hull_mask(R, C, half, K, shape):
    """Filled silhouette of the cuboid, or None if any corner is behind us."""
    P = C + (CORNERS8 * half) @ R.T
    if np.any(P[:, 2] <= 0.05):
        return None
    uv = cv2.convexHull(project(P, K).astype(np.int32))
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [uv], 1)
    return m


def ray_depth(R, C, half, K, uv):
    """Depth at which the ray through each pixel first meets the cuboid.

    Slab test done in the box frame, so it is 3 divisions and 2 reductions per
    pixel, fully vectorised.  ``K`` has no distortion term because the bag's
    depth is already aligned to the colour stream, whose intrinsics are near
    pinhole (the fisheye coefficients in camera_calib.json are tiny).

    Returns nan where the ray misses the box.
    """
    n = len(uv)
    if n == 0:
        return np.zeros(0)
    KINV = np.linalg.inv(K)
    hom = np.concatenate([uv.astype(np.float64), np.ones((n, 1))], axis=1)
    d = hom @ KINV.T              # (n,3) unnormalised rays, d[:,2] == 1
    D = d @ R                     # box-frame ray directions
    O = -(R.T @ C)                # camera origin in the box frame
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - O) / D
        t2 = (half - O) / D
    lo = np.minimum(t1, t2)
    hi = np.maximum(t1, t2)
    par = np.abs(D) < 1e-9        # (n,3) ray parallel to this slab pair
    inside_axis = np.abs(O) <= half          # (3,) origin within this slab?
    lo = np.where(par, np.where(inside_axis[None, :], -np.inf, np.inf), lo)
    hi = np.where(par, np.where(inside_axis[None, :], np.inf, -np.inf), hi)
    t_enter = lo.max(axis=1)
    t_exit = hi.min(axis=1)
    hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 0.0)
    t = np.where(t_enter > 0.0, t_enter, t_exit)
    return np.where(hit, t, np.nan)


def mask_pixels(blob, step=3):
    """Subsampled mask pixel list + the observed depth at those pixels."""
    ys, xs = np.nonzero(blob > 0)
    if len(ys) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    sl = slice(None, None, max(1, int(step)))
    return np.stack([xs[sl], ys[sl]], axis=1), (ys[sl], xs[sl])


def depth_median_err(R, C, half, K, uv, depth_img, yx, free_m=0.06):
    """Median |predicted - observed| depth over the mask, ignoring a noise floor.

    This is what breaks the cuboid's silhouette ambiguity.  A parallelepiped
    seen in perspective can match the observed OUTLINE just as well tilted
    toward the camera as away from it, so IoU alone happily returns a box
    leaning 45 deg backwards (observed: IoU 0.89 with z_box tilted 45 deg).
    The depth pattern cannot be fooled the same way.
    """
    if len(uv) < 20:
        return 0.0
    zp = ray_depth(R, C, half, K, uv)
    py, px = yx
    zo = depth_img[py, px].astype(np.float64)
    ok = np.isfinite(zp) & (zo > 0.1) & (zo < 12.0)
    if ok.sum() < 20:
        return 0.0
    return float(np.median(np.abs(zp[ok] - zo[ok])))


def silhouette_iou(R, C, half, K, blob, n_obs=None, ob=None):
    """IoU between the projected silhouette and the colour mask.

    ``ob`` / ``n_obs`` let the caller hand in the mask already thresholded.
    This matters: the search evaluates this hundreds of times per frame and
    ``blob > 0`` over a 640x480 array dominates the run time otherwise.
    """
    m = hull_mask(R, C, half, K, blob.shape)
    if m is None:
        return -1.0
    if ob is None:
        ob = blob > 0
    if n_obs is None:
        n_obs = int(ob.sum())
    fb = m > 0
    inter = int(np.logical_and(ob, fb).sum())
    union = n_obs + int(fb.sum()) - inter
    return (inter / union) if union else 0.0


# --------------------------------------------------------------------------
# point-cloud orientation: ABANDONED route, kept for reference only.
# Nothing below is called -- see the module docstring.
# --------------------------------------------------------------------------

def normals_from_depth(Z, K, valid, smooth=2):
    """Surface normals via depth-grid differences, smoothed over a window.

    REFERENCE ONLY -- zero call sites (see the module docstring).

    ``smooth`` widens the finite-difference stencil AND the averaging window;
    both are needed because on this footage a 1-pixel stencil is dominated by
    sensor noise.
    """
    H, W = Z.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = np.stack([(uu - K[0, 2]) / K[0, 0] * Z,
                  (vv - K[1, 2]) / K[1, 1] * Z, Z], axis=2)

    s = max(1, int(smooth))
    Pu = np.zeros_like(P)
    Pv = np.zeros_like(P)
    Pu[:, s:-s] = P[:, 2 * s:] - P[:, :-2 * s]
    Pv[s:-s, :] = P[2 * s:, :] - P[:-2 * s, :]

    vu = np.zeros_like(valid)
    vu[:, s:-s] = valid[:, s:-s] & valid[:, 2 * s:] & valid[:, :-2 * s]
    vv2 = np.zeros_like(valid)
    vv2[s:-s, :] = valid[s:-s, :] & valid[2 * s:, :] & valid[:-2 * s, :]

    N = np.cross(Pu, Pv)
    nn = np.linalg.norm(N, axis=2)
    ok = vu & vv2 & (nn > 1e-9)
    N = np.where(ok[..., None], N / np.maximum(nn, 1e-12)[..., None], 0.0)

    # average over a neighbourhood to suppress what is left of the noise
    k = 2 * s + 1
    Ns = cv2.boxFilter(N, -1, (k, k), normalize=True, borderType=cv2.BORDER_REPLICATE)
    okf = cv2.erode(ok.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    nn2 = np.linalg.norm(Ns, axis=2)
    okf &= nn2 > 0.5
    Ns = np.where(okf[..., None], Ns / np.maximum(nn2, 1e-12)[..., None], 0.0)

    # a visible surface normal points at the camera origin: dot(N, P) < 0
    sgn = np.sign(np.sum(Ns * P, axis=2))
    sgn[sgn == 0] = 1.0
    Ns = Ns * (-sgn)[..., None]
    return P, Ns, okf


def dominant_normals(n, ang_deg=25.0, sep_deg=55.0, max_pts=1500):
    """REFERENCE ONLY -- zero call sites outside orient_from_planes (itself
    abandoned); see the module docstring."""
    if len(n) > max_pts:
        idx = np.linspace(0, len(n) - 1, max_pts).astype(int)
        n = n[idx]
    if len(n) < 30:
        return None, None
    S = n @ n.T
    cos_a = math.cos(math.radians(ang_deg))
    cnt = (S > cos_a).sum(axis=1)
    i1 = int(np.argmax(cnt))
    n1 = n[i1]
    far = S[i1] < math.cos(math.radians(sep_deg))
    if far.sum() < 20:
        return n1, None
    S2 = S[np.ix_(far, far)]
    n2 = n[far][int(np.argmax((S2 > cos_a).sum(axis=1)))]
    return n1, n2


def _trimmed_plane(pts, n_ref, keep=0.7, rounds=3):
    """Plane fit with outlier trimming (a plain SVD is dragged by the
    silhouette boundary, where the depth jumps to the background).

    REFERENCE ONLY -- zero call sites outside orient_from_planes (itself
    abandoned); see the module docstring."""
    if len(pts) < 12:
        return None, None
    sel = np.ones(len(pts), bool)
    for _ in range(rounds):
        c = pts[sel].mean(axis=0)
        _, _, vt = np.linalg.svd(pts[sel] - c, full_matrices=False)
        nr = vt[2]
        if float(nr @ n_ref) < 0:
            nr = -nr
        res = np.abs((pts - c) @ nr)
        thr = np.percentile(res[sel], 100.0 * keep)
        new = res <= max(thr, 1e-4)
        if new.sum() < 12:
            break
        sel = new
    c = pts[sel].mean(axis=0)
    _, _, vt = np.linalg.svd(pts[sel] - c, full_matrices=False)
    nr = vt[2]
    if float(nr @ n_ref) < 0:
        nr = -nr
    return nr, float(nr @ c)


def orient_from_planes(pts, nrms, half):
    """Plane-based orientation + centre.  Returns (R, C, info) or (None,None,info).

    ABANDONED -- zero call sites repo-wide; this is the estimator that measured
    a 0.43 m front face on a 0.30 m box and produced IoU 0.00.  Kept for
    reference only, together with the three helpers it calls.  See the module
    docstring before considering it again.
    """
    info = {}
    n1, n2 = dominant_normals(nrms)
    if n1 is None:
        info["reason"] = "too few valid normals"
        return None, None, info
    if n2 is None:
        info["reason"] = "single face visible"
        return None, None, info

    if float(n1 @ CAM_UP) >= float(n2 @ CAM_UP):
        n_top_raw, n_front_raw = n1, n2
    else:
        n_top_raw, n_front_raw = n2, n1

    ortho = math.degrees(math.acos(min(1.0, abs(float(n_top_raw @ n_front_raw)))))
    info["ortho_deg"] = ortho
    if not (55.0 <= ortho <= 125.0):
        info["reason"] = f"faces not orthogonal ({ortho:.0f} deg)"
        return None, None, info

    a_t = nrms @ n_top_raw
    a_f = nrms @ n_front_raw
    thr = math.cos(math.radians(25.0))
    sel_t = (a_t > thr) & (a_t >= a_f)
    sel_f = (a_f > thr) & (a_f > a_t)
    if sel_t.sum() < 30 or sel_f.sum() < 30:
        info["reason"] = (f"too few inliers (top {int(sel_t.sum())}, "
                          f"front {int(sel_f.sum())})")
        return None, None, info

    n_top, d_top = _trimmed_plane(pts[sel_t], n_top_raw)
    n_front, d_front = _trimmed_plane(pts[sel_f], n_front_raw)
    if n_top is None or n_front is None:
        info["reason"] = "plane fit failed"
        return None, None, info

    if float(n_top @ CAM_UP) < 0:
        n_top, d_top = -n_top, -d_top
    if float(n_front @ CAM_BACK) < 0:
        n_front, d_front = -n_front, -d_front

    # Rebuild the frame ORTHONORMALLY, pinning z to the top normal and
    # projecting the front normal into the plane perpendicular to it.  The
    # previous code let SVD average the two non-orthogonal normals, which
    # rotated the frame by tens of degrees when the top face was barely
    # visible -- that is what put the fitted box at z=5.2 m for a box that was
    # actually at 3.5 m.
    z_box = n_top / np.linalg.norm(n_top)
    y_raw = -n_front
    y_box = y_raw - (y_raw @ z_box) * z_box
    ny = np.linalg.norm(y_box)
    if ny < 1e-6:
        info["reason"] = "degenerate axes"
        return None, None, info
    y_box /= ny
    x_box = np.cross(y_box, z_box)
    R = np.stack([x_box, y_box, z_box], axis=1)

    # distances recomputed against the axes actually used
    d_front = float((-y_box) @ pts[sel_f].mean(axis=0))
    d_top = float(z_box @ pts[sel_t].mean(axis=0))

    proj_f = pts[sel_f] @ x_box
    rhs = np.array([float(proj_f.mean()),
                    half[1] - d_front,
                    d_top - half[2]])
    C = R @ rhs

    info["w_m"] = float(proj_f.max() - proj_f.min())
    proj_t = pts[sel_t] @ x_box
    info["d_m"] = float(proj_t.max() - proj_t.min())
    info["n_top"] = int(sel_t.sum())
    info["n_front"] = int(sel_f.sum())
    return R, C, info


# --------------------------------------------------------------------------
# silhouette-IoU refinement (the real estimator)
# --------------------------------------------------------------------------

def make_objective(half, K, blob, n_obs, ob, depth_img=None, uv=None, yx=None,
                   weight_d=1.0, depth_scale_m=0.15):
    """The single objective used everywhere: IoU minus a depth penalty.

    It MUST be shared by the candidate ranking and by the refinement.  They
    used to differ -- candidates were ranked by silhouette IoU alone while
    refinement optimised IoU minus depth -- and that handed the refinement the
    wrong three starting points on most of the parked-box frames.  Concrete
    case, frame 0 of the real bag: the correct pose scored IoU 0.886 with a
    25 mm depth residual, while three wrong-depth poses scored IoU 0.931 /
    0.921 / 0.919 with 411 / 323 / 316 mm residuals.  Ranked by IoU the truth
    came seventh and was never refined; the run reported 0/20 for that block.
    """
    use_depth = (depth_img is not None and uv is not None and len(uv) >= 20)

    def score(R, C):
        s = silhouette_iou(R, C, half, K, blob, n_obs, ob=ob)
        if s < 0:
            return -1.0
        if use_depth:
            e = depth_median_err(R, C, half, K, uv, depth_img, yx)
            s -= weight_d * min(1.0, max(0.0, e) / depth_scale_m)
        return s

    return score


def refine(R0, C0, half, K, blob, depth_img=None, uv=None, yx=None,
           rounds=8, weight_d=1.0, depth_scale_m=0.15, ob=None,
           score_fn=None):
    """Pattern search on (3 translation, 3 rotation).

    Objective = silhouette IoU minus a depth-agreement penalty.  IoU alone
    cannot resolve the cuboid's toward/away tilt ambiguity; the depth term is
    what makes the answer unique.
    """
    n_obs = int(ob.sum()) if ob is not None else int((blob > 0).sum())
    if n_obs == 0:
        return R0, C0, -1.0
    if score_fn is None:
        score_fn = make_objective(half, K, blob, n_obs, ob, depth_img, uv, yx,
                                  weight_d, depth_scale_m)
    score = score_fn

    R, C = R0.copy(), C0.copy()
    best = score(R, C)
    step = np.array([0.02, 0.02, 0.04, 0.12, 0.12, 0.20])
    for _ in range(rounds):
        improved = False
        for i in range(6):
            for sgn in (1.0, -1.0):
                d = np.zeros(6)
                d[i] = sgn * step[i]
                Rt = rotvec_to_R(d[3:]) @ R
                Ct = C + d[:3]
                v = score(Rt, Ct)
                if v > best + 1e-6:
                    best, R, C, improved = v, Rt, Ct, True
        if not improved:
            step = step * 0.5
            if step.max() < 1e-3:
                break
    return R, C, best


def initialisers(half, K, blob, pos_cam):
    """Candidate (R, C) starts covering the physically plausible orientations.

    The family is the one this project actually meets: the box stands UPRIGHT
    on the ground and the camera looks DOWN at it, so the box's up-axis lands
    near the camera's -y, tipped further by the camera pitch (measured ~15.6 deg
    on the real footage).

    An earlier version swept ``rotvec([pitch, -yaw, 0])`` with pitch in
    [-30, 30].  That family structurally CANNOT express an upright box -- in
    that parameterisation upright needs pitch ~= 90 deg -- so every candidate
    started ~60 deg away from the truth and refinement (max step 11.5 deg per
    round) could never close the gap.  That is why the synthetic self-test went
    to 0/12: the search was not merely unlucky, it was looking in the wrong
    place.  Sweeping camera pitch, box yaw and a small box tilt explicitly
    removes the blind spot.

    ``half``/``K``/``blob`` are unused, kept so the call site stays unchanged.
    """
    C = np.asarray(pos_cam, dtype=np.float64)
    cands = []
    for cam_pitch in (0.0, 10.0, 20.0, 30.0):
        for yaw in (0.0, 30.0, 60.0):
            for tilt in (-25.0, 0.0, 25.0):
                cands.append((realistic_pose(yaw, tilt, cam_pitch), C.copy()))
    return cands


def scan_center(R, seed, half, K, blob, steps=11, span=0.5, ob=None,
                score_fn=None):
    """Slide the centre along the ray from the camera through ``seed``.

    The mask centroid used as a position seed is biased toward the camera (it
    is an average over the VISIBLE surface, ~0.15-0.2 m in front of the true
    centre).  Leaving that bias in place poisons orientation scoring: with the
    centre 0.18 m off, a box rolled onto its side can out-score the correct
    pose, which is precisely how a third of the synthetic cases failed.  Fixing
    the depth first makes the orientation comparison meaningful.
    """
    d = float(np.linalg.norm(seed))
    if d < 1e-6:
        return np.asarray(seed, dtype=np.float64)
    u = np.asarray(seed, dtype=np.float64) / d
    best_v, best_C = -2.0, np.asarray(seed, dtype=np.float64)
    for t in np.linspace(-span, span, int(steps)):
        C = best_C if t == 0 else np.asarray(seed, dtype=np.float64) + t * u
        if C[2] < 0.4:
            continue
        v = (score_fn(R, C) if score_fn is not None
             else silhouette_iou(R, C, half, K, blob, ob=ob))
        if v > best_v:
            best_v, best_C = v, C
    return best_C


def solve(half, K, blob, pos_cam, depth_img=None, uv=None, yx=None,
          depth_step=3, R_prev=None, C_prev=None, n_starts=3):
    """Best cuboid pose for one frame.  Returns (R, C, info).

    ``R_prev``/``C_prev`` (the previous frame's pose) is a strong initialiser
    for video, where the pose moves smoothly.  ``n_starts`` refinements are run
    from the best few distinct initialisers and the highest-scoring result
    wins: a single pattern search of +/-7..11 deg steps cannot climb out of a
    90 deg local minimum, which is exactly how the earlier single-start version
    failed on roughly a third of synthetic cases.
    """
    info = {}
    if uv is None and depth_img is not None:
        uv, yx = mask_pixels(blob, depth_step)

    ob = blob > 0
    n_obs = int(ob.sum())
    if n_obs == 0:
        info["reason"] = "empty mask"
        return None, None, info

    cands = []
    if R_prev is not None and C_prev is not None and np.all(np.isfinite(C_prev)):
        cands.append((np.asarray(R_prev, dtype=np.float64),
                      np.asarray(C_prev, dtype=np.float64)))
    cands.extend(initialisers(half, K, blob, pos_cam))

    score_fn = make_objective(half, K, blob, n_obs, ob, depth_img, uv, yx)

    # rank the starts with the SAME objective the refinement optimises.  Doing
    # this by IoU alone is what made the parked-box frames fail: it preferred
    # three wrong-depth poses whose outlines happened to match a little better.
    scored = []
    for R0, C0 in cands:
        # the slide position is chosen by the FULL objective too: at a wrong
        # depth the outline alone is perfectly happy (frame 0: IoU 0.931 at a
        # 411 mm depth error), and this is the step that gets to fix it
        C1 = scan_center(R0, C0, half, K, blob, ob=ob, score_fn=score_fn)
        v = score_fn(R0, C1)
        if v > -1.0:
            scored.append((v, R0, C1))
    if not scored:
        info["reason"] = "no usable initialiser"
        return None, None, info
    scored.sort(key=lambda z: -z[0])

    best_sc, best_R, best_C = -2.0, None, None
    for _v, R0, C0 in scored[:max(1, int(n_starts))]:
        R, C, sc = refine(R0, C0, half, K, blob,
                          depth_img=depth_img, uv=uv, yx=yx, ob=ob,
                          score_fn=score_fn)
        if sc > best_sc:
            best_sc, best_R, best_C = sc, R, C
    if best_R is None:
        info["reason"] = "refinement failed"
        return None, None, info

    info["iou"] = silhouette_iou(best_R, best_C, half, K, blob, n_obs, ob=ob)
    info["score"] = best_sc
    if depth_img is not None and uv is not None and len(uv) >= 20:
        info["depth_err_m"] = depth_median_err(best_R, best_C, half, K, uv,
                                              depth_img, yx)
        # a large residual means the search settled on the wrong side of the
        # cuboid's toward/away ambiguity: report no measurement rather than a
        # confidently wrong pose
        if info["depth_err_m"] > 0.15:
            info["reason"] = (f"depth mismatch {info['depth_err_m']:.2f} m "
                              "(silhouette-only solution)")
            return None, None, info
    return best_R, best_C, info


# --------------------------------------------------------------------------
# self-test: synthesise a box, render mask + depth from the known pose, solve
# --------------------------------------------------------------------------

def box_symmetries():
    """The 8 rotations that map a (0.15, 0.15, 0.225) box onto itself.

    x and y share a half-extent while z is unique, so z survives as an axis
    (up to sign) and {x, y} may be swapped -- giving the four Rz(k*90) plus the
    180 deg flip about x, which sends y -> -y and z -> -z.  90 deg alone is not
    enough: a plain box has no marked top, so an "upside down" fit is the same
    physical object and must not be scored as an error.
    """
    out = []
    for k in range(4):
        th = k * math.pi / 2
        Rz = np.array([[math.cos(th), -math.sin(th), 0.0],
                       [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
        out.append(Rz)
        out.append(Rz @ np.diag([1.0, -1.0, -1.0]))
    return out


_BOX_SYM = None


def pose_err(Ra, Rb):
    """Rotation difference modulo the box's own symmetry group.

    ``R`` and ``R @ S`` (S a symmetry) describe the SAME physical box, so
    scoring against one representative alone reports false failures -- observed
    as "rot 175 deg BAD" for a fit whose silhouette IoU was 0.955 and depth
    residual 15 mm, i.e. a good fit penalised for choosing the flipped twin.
    """
    global _BOX_SYM
    if _BOX_SYM is None:
        _BOX_SYM = box_symmetries()
    best = 180.0
    for S in _BOX_SYM:
        best = min(best, math.degrees(math.acos(np.clip(
            (np.trace(Ra.T @ Rb @ S) - 1) / 2, -1, 1))))
    return best


def canonicalize_rotations(mats, valid=None, ref=None):
    """Walk a per-frame rotation sequence and pick, on EVERY frame, the
    symmetry-equivalent representative that is closest to the previous frame's
    choice.

    Why this is needed: ``solve()`` resolves the box's 8-fold symmetry on each
    frame independently, so the raw R sequence flickers between equivalent
    solutions (~90 deg yaw steps, and the 180 deg flip).  Measured on the
    20260917 clip: 183 of 485 consecutive pairs jumped more than 5 deg in
    symmetry-canonical yaw.  Any downstream smoothing, averaging or "freeze a
    parked phase to its median pose" is meaningless across those flips -- the
    average of two equivalent poses is a third, wrong one, which is exactly
    what made a box sitting untouched on the floor twitch and rotate in place.

    ``ref`` (3x3) optionally seeds the walk (e.g. the parked-phase mean), so
    the whole sequence is expressed in one consistent branch.
    """
    out = np.array(mats, dtype=float, copy=True)
    if valid is None:
        valid = np.isfinite(out).all(axis=(1, 2))
    syms = box_symmetries()
    prev = None if ref is None else np.asarray(ref, dtype=float)
    for i in range(len(out)):
        if not valid[i]:
            continue
        cands = [out[i] @ S for S in syms]
        if prev is None:
            # start from the most upright representative: world +z of the box
            # z-axis is the last column's z component
            k = int(np.argmax([c[2, 2] for c in cands]))
        else:
            k = int(np.argmin([math.degrees(math.acos(np.clip(
                (np.trace(prev.T @ c) - 1) / 2, -1, 1))) for c in cands]))
        out[i] = cands[k]
        prev = out[i]
    return out


def _cam_Rx(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, math.cos(a), -math.sin(a)],
                     [0.0, math.sin(a), math.cos(a)]])


def realistic_pose(yaw_deg, tilt_deg, cam_pitch_deg=15.0):
    """A pose like the ones this project actually has to solve.

    The box stands UPRIGHT (up-axis roughly along the camera's -y) and the
    camera is pitched down.  Building the box any other way makes the test
    unrealistically easy to fool: a box lying face-on presents a square
    silhouette with its depth axis along the view ray, which is the one
    configuration where a 90 deg rotation is genuinely indistinguishable, so
    such cases measure the test harness rather than the estimator.
    """
    R0 = np.array([[1.0, 0.0, 0.0],    # box x -> camera x (right)
                   [0.0, 0.0, -1.0],   # box y (to back) -> camera +z (away)
                   [0.0, 1.0, 0.0]])   # box z (up) -> camera -y (image up)
    y = math.radians(yaw_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0.0],
                   [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    return _cam_Rx(cam_pitch_deg) @ R0 @ Rz @ _cam_Rx(tilt_deg)


def selftest(seed=11, trials=12, verbose=True):
    """Returns (n_ok, n_trials).  Raises AssertionError on a hard failure."""
    half = np.array([0.15, 0.15, 0.225])
    K = np.array([[386.8, 0.0, 321.9], [0.0, 386.1, 244.8], [0.0, 0.0, 1.0]])
    shape = (480, 640)
    rng = np.random.default_rng(seed)
    ok = tot = 0
    for t in range(trials):
        R = realistic_pose(rng.uniform(-40, 40), rng.uniform(-15, 15))
        C = np.array([rng.uniform(-0.6, 0.6), rng.uniform(0.2, 1.2),
                      rng.uniform(1.5, 3.6)])
        mask = hull_mask(R, C, half, K, shape)
        if mask is None or (mask > 0).sum() < 250:
            continue
        tot += 1
        blob = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(mask > 0)
        uv = np.stack([xs, ys], axis=1).astype(np.float64)
        zt = ray_depth(R, C, half, K, uv)
        keep = np.isfinite(zt)
        ys, xs, zt = ys[keep], xs[keep], zt[keep]
        dimg = np.zeros(shape, np.float32)
        dimg[ys, xs] = zt + rng.normal(scale=0.012, size=len(zt))
        zmed = float(np.median(dimg[ys, xs]))
        seed_pos = np.array([(xs.mean() - K[0, 2]) / K[0, 0] * zmed,
                             (ys.mean() - K[1, 2]) / K[1, 1] * zmed, zmed])
        mu, my = mask_pixels(blob, 3)
        Rr, Cr, info = solve(half, K, blob, seed_pos, depth_img=dimg,
                             uv=mu, yx=my)
        if Rr is None:
            if verbose:
                print(f"  trial {t}: no solution ({info.get('reason')})")
            continue
        dR = pose_err(Rr, R)
        dC = float(np.linalg.norm(Cr - C))
        good = dR < 7.0 and dC < 0.06
        ok += good
        if verbose:
            print(f"  trial {t}: rot {dR:5.2f} deg  centre {dC*1000:5.1f} mm  "
                  f"IoU {info['iou']:.3f}  depth "
                  f"{info.get('depth_err_m', float('nan'))*1000:5.1f} mm  "
                  f"{'OK' if good else 'BAD'}")
    print(f"selftest: {ok}/{tot} within tolerance (rot <7 deg, centre <60 mm)")
    assert tot >= 6, "selftest: too few usable synthetic cases"
    assert ok >= int(0.8 * tot), f"selftest FAILED: only {ok}/{tot} passed"
    return ok, tot


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="synthesise boxes at known 6-DoF poses, render their "
                         "silhouette and a noisy depth map, solve, and check "
                         "the recovered pose.  Run this before trusting a fit "
                         "on real footage.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--trials", type=int, default=12)
    args = ap.parse_args()
    if args.selftest:
        selftest(seed=args.seed, trials=args.trials)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

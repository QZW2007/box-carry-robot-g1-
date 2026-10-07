---
name: cuboid-6dof-from-rgbd
description: Recover a known-size box's 6-DoF pose (position AND orientation) from a segmented RGB-D stream with no fiducial markers, by fitting its projected silhouette to a colour mask plus a per-pixel depth-consistency term. Use when a depth camera must track a rigid box/block/cuboid, when ArUco or checkerboard markers are unavailable, when a per-pixel depth-normal approach has failed, or when a single-point depth centroid gives position but no orientation. Covers the synthetic self-test, the orientation-symmetry traps, and the debug order.
agent_created: true
---

# 6-DoF cuboid pose from segmented RGB-D (marker-free)

## When this applies

A rigid box of KNOWN dimensions seen by an aligned RGB-D camera, where you need
its full pose and cannot (or do not want to) place fiducial markers.

Key insight: **because the dimensions are known, the projected silhouette is
fully determined by the 6-DoF pose.** So the pose can be recovered by matching a
rendered silhouette against a colour mask — and colour segmentation is
sub-pixel accurate at any distance, unlike depth.

## Do NOT start with per-pixel depth normals

The obvious first attempt — normals from the depth grid, clustered into faces —
fails whenever the object is small in the image. Quantify before committing:
if the object is under ~100 px wide, the depth gradient across ONE pixel step
is comparable to the depth noise, so the normals come out near-random.

Measured failure signature on a 0.30 m box at 2–3.5 m on a 640×480 stream:
recovered front-face width 0.43 m (true 0.30) and silhouette IoU 0.00. If you
see a face measured larger than the object itself, stop and switch method.

## The working objective

```
score(R, C) = silhouette_IoU(R, C) - min(1, median_depth_residual / 0.15 m)
```

- `silhouette_IoU`: render the cuboid's convex hull, compare to the mask.
- `median_depth_residual`: for the mask pixels, compute where the ray first
  meets the cuboid (a **slab test in the box frame** — 3 divisions and 2
  reductions per pixel, fully vectorisable) and take the median
  `|predicted - observed|`, ignoring a small noise floor.

**The depth term is not optional.** A parallelepiped's outline matches almost
equally well tilted toward the camera as away from it, so IoU alone converges
to a box leaning ~45° backwards with a great-looking IoU (~0.89). Depth is the
only thing that breaks that ambiguity.

## Trap list (each of these cost real debugging time)

1. **Rank candidate starts and refine with the SAME objective.** Ranking by IoU
   while refining on IoU−depth silently hands refinement the wrong starts. Real
   case: the correct pose scored IoU 0.886 / depth 25 mm and ranked 7th, behind
   three wrong-depth poses at IoU 0.931/0.921/0.919 with 411/323/316 mm depth
   residuals. Those frames came out 0/20; fixing this alone made them 20/20.
   Factor the objective into one shared function.
2. **Score rotation error modulo the object's FULL symmetry group.** A plain
   box has no marked top, so it is invariant under 180° flips too — the
   symmetry group is order 8 (`Rz(k·90°)` × `{I, Rx(180)}`) for a square
   cross-section, order 4 for a fully asymmetric box. Checking only 90° yaw
   reported an IoU-0.955 / depth-15 mm fit as a 175° failure.
3. **Initialiser families must be able to express the true pose.** Sweeping
   `rotvec([pitch, -yaw, 0])` with pitch ∈ [-30°, 30°] *cannot* express an
   upright box on the ground (that needs ~90° in that parameterisation), so the
   search started ~60° away and every trial failed. Sweep the physically
   meaningful axes directly: camera pitch × box yaw × small box tilt.
4. **A tilted and an upright object can share a silhouette**, so build synthetic
   test cases from PHYSICALLY REALISTIC poses (upright box + pitched camera).
   Face-on degenerate cases are genuinely 90°-ambiguous, so they measure the
   test harness rather than the estimator.
5. **Fix the centre before comparing orientations.** The mask centroid is
   biased toward the camera by ~0.15–0.2 m (it averages the VISIBLE surface).
   Slide the centre along the view ray first, scoring with the full objective.
6. **One pattern search cannot climb out of a 90° local minimum.** Run several
   refinements from the best distinct starts and keep the highest score.
7. **Reject rather than guess.** If the best depth residual still exceeds the
   sensor noise by a wide margin, return "no measurement". A confidently wrong
   pose is far worse downstream than a gap.
8. **Report statistics over ACCEPTED frames only.** If the fit info struct
   keeps its IoU/depth numbers even when the pose is rejected, averaging the
   raw arrays describes a mixture of two populations. This produced a reported
   "median depth residual 204 mm" on a run whose rejection gate was 150 mm.

## Sensor-noise floor

RealSense-class depth noise is ~1% of range. At 2–3.5 m that is 25–35 mm. A
median residual at or below that is the sensor floor, NOT fit error — do not
tune against it. Check the residual against this number before "improving" the
fit.

## Verification recipe (in this order)

1. **Synthetic self-test.** Render a silhouette + noisy depth from a known
   pose, solve, check recovered rotation/centre. Gate the pipeline on it.
   A pass rate of 80%+ with rotation <7° and centre <60 mm is a reasonable bar.
2. **Look at the rendered silhouette.** Annotate the fitted outline over the
   image. It must trace the object's 3D structure (e.g. the fold of a visible
   top face), not just a 2D rectangle.
3. **Check against a plane you can measure independently.** For a box resting
   on a floor, fit the floor plane in the CAMERA frame and measure the box
   centre's perpendicular distance to it. For a 0.45 m box that must be
   ~0.225 m. This validates absolute height without involving any downstream
   alignment, and it is what localises a height error to the fit or to the
   alignment.
4. **Sanity-check the implied pose physically.** A level (upright) object's
   local z-axis in the camera frame should point near the image "up" direction
   (image up is `(0,-1,0)` in OpenCV convention — NOT world vertical).

## Coordinate-convention pitfalls

- `(0,-1,0)` is IMAGE up in the OpenCV camera frame, not world vertical. A
  downward-pitched camera makes a level object's z-axis differ from it by the
  camera pitch.
- Determine the camera's actual pitch from the data (fit the floor plane and
  look at its normal) rather than assuming it. A "camera pitched 15° down"
  assumption was wrong here — the floor plane normal showed the camera was
  essentially LEVEL, which changes which initialisers matter.
- A camera-frame pose maps to a world frame as `R_world = R_world_from_cam @ R_cam`
  (rotation composes on the LEFT), while positions follow the full Sim3
  `s * (R_world_from_cam @ t_cam) + t`.

## Feeding the fit into a renderer (temporal consistency)

A frame-wise fit that is *accurate in isolation* is still not *usable as a motion
trajectory*. The failure mode is specific: on a square-cross-section box the
solver hops between **symmetry-equivalent branches** (`Rz(90k)`, and
`Rx(180)` flips), so a few frames land ~90°/180° away and back. IoU/depth stay
high (the pose is genuinely good) while the rendered object appears to spin in
the hands. Order of operations that fixed it, cheapest first:

0. **Snap the symmetry branch to a FIXED anchor before anything else.** Pick a
   stable reference (e.g. the chordal mean of the parked/first-phase frames)
   and, for every frame, re-express the measurement on the symmetry-equivalent
   branch closest to that anchor: `R_i ← R_i · argmin_S angle(R_anchor⁻¹ R_i S)`.
   Real numbers: on one clip the fit's yaw drifted a full **180°** during the
   pickup (f148–f204 sat ~170° from the floor median while tilt stayed <10°)
   and *stayed* there — invisible on a square box, but ~150° of rendered spin.
   **Do NOT snap to the previous frame (chained):** the wrong-branch yaw drifts
   *continuously* (deltas < 45°/frame), so a chained reference just follows the
   drift branch by branch — on the real clip it fired **0** times. A fixed
   anchor snapped 350/486 frames and eliminated every fast twist (the
   downstream slew limiter went from clipping 91 frames to 0).
1. **Unwrap the orientation in rotvec domain before any filtering.** Quaternion
   double-cover means a smooth physical motion shows up as ±180° jumps. Real
   numbers: **130 of 486** frames were on the far side of ±180°. Filtering
   before unwrapping averages across the discontinuity and destroys the signal.
   Never average raw quaternions; canonicalise/unwrap first.
2. **Median-filter the rotvec, not the quaternion, and not the rotation matrix.**
   Window 9 on a 30 fps stream: **225 of 486** frames deviated >5° from their
   local median (max dev 87.6°) and were replaced. Median (not mean) because the
   noise is *branch hops*, i.e. outliers, not Gaussian jitter.
3. **Add the physical prior if the object is held.** While carried, the object's
   orientation is set by the gripper/hand: bound the deviation from the
   hand-rigid orientation (60° was generous enough to never clip a real motion,
   here 0 of 316 carry frames). Tune the bound so it only ever rescues
   wrong-branch frames.
4. **Slew-rate limit as the last resort.** Angular velocity cap per frame
   (3.5°/frame here) clipped 91 frames. **This one is dangerous** — it also
   destroys genuinely fast real motion — so only apply it after you have
   *verified against the source video* that the real object never rotates that
   fast. Measure the real rate first; here the median was 0.4°/frame.
5. **Freeze parked phases to their median pose** and ramp back to the
   measurement over ~8 frames at the transition. A stationary object should be
   numerically stationary (rot step p90 → 0.000°), and the ramp avoids a step.

### The diagnostic that finds this

Extract the same frame indices from the rendered output **and the source video**
and look at them side by side. "The fit says the box turned 100° between f172
and f182" vs "the source video shows the box barely moving" is what localises
the bug to the estimator/pipeline rather than to the retargeting. Do this before
tuning any filter — otherwise you are tuning blind.

### Two more leaks that masquerade as fit noise

- **The `A + LPF(B − A)` identity.** Driving the object with
  `hand_rigid + LPF(measured − hand_rigid)` rewrites (gain=1) as
  `LPF(measured) + (hand_rigid − LPF(hand_rigid))` — the **unfiltered**
  hand-rigid term leaks its high-frequency jitter at FULL amplitude. Smooth the
  hand-rigid term with the same window, or the object keeps bouncing no matter
  how much you smooth the measurement. Real numbers: vertical jolts of
  +58 mm dropped to 0 only after both terms were filtered.
- **Retargeted-body jitter feeding the object.** If the object position is
  anchored to body landmarks (hands), a single jittery joint (e.g. an elbow
  inventing a 4→60→−3° excursion in 0.7 s because the arm was occluded during
  pose estimation) jolts the object. Check the source video: if the real limb
  is steady, despike the joint — per-dof step stats (median/p90/max) locate the
  offender in one pass. A 20–30-frame "bump" is NOT removable by a short
  median filter or a slew limiter (both just delay/stretch it); use a
  rolling-median baseline with window > 2× the bump and clamp the deviation
  (61-frame window, ±12° cap worked; genuine motion stayed ≤5.9°/frame).
  **The clamp is only half the cure:** the same artifact also produces
  *below-threshold* dips-and-recoveries (a 13° wrist-roll snap, a 10°
  wrist-yaw dip) that pass straight through the clamp. Finish with a plain
  median filter (window ~51) on the joint trajectories — median filters
  PRESERVE monotone ramps and step edges (the shapes of genuine motion) while
  deleting dips shorter than half the window, unlike a mean filter which
  delays and flattens real ramps.
- **A steadily-wrong limb pose is a different disease from twitching.** If one
  limb consistently renders in an unnatural pose (e.g. a wrist dorsiflexed so
  the hand reads as "pried upward" against the object) while the other side
  looks right, no amount of filtering fixes it — the pose estimate itself is
  biased (typical when the limb is partially occluded in the pose-estimation
  input). If the subject's motion is near-symmetric, drive the bad limb with
  the MIRROR of the good one during the contact phase (sign map: pitch/elbow
  keep sign, roll/yaw negate — verify against the medians of a
  symmetric-looking stretch), with short ramps at the phase boundaries, and
  apply it BEFORE forward kinematics so derived landmarks stay consistent.
  Quantify first with per-frame |left − mirror(right)| stats: medians near
  zero confirm both the bias is episodic and the sign convention.
- **Contact artifacts call for geometric constraints, not blend tuning.** When
  the held object must not interpenetrate the hands/body, replace the
  measured-vs-hand blend with a stack of hard constraints: anchor the object
  centre at the hand midpoint + a low-passed measured drift clamped into a
  small ball (50 mm) — the measured trajectory deviates from the hands by the
  whole retargeting error (0.17–0.40 m on the real clip, wrist spacing only
  0.35 m); push the centre horizontally out of a spine capsule with radius ≥
  the object's half-diagonal + margin (the measured centre came within 77 mm
  of the spine vs 212 mm half-diagonal — deep torso clipping); and CAPTURE the
  object on its support surface: in the release zone, once the anchored z
  reaches the support height, pin it there — the retargeted body bobs the
  hands up ~0.19 m before the final descent, and a hand-anchored object rides
  up with it. Watch the phase boundary: the align split can precede the actual
  grasp, so ramp the anchor in over the hand-descent frames (anchoring at the
  split yanked the box off the floor +341 mm in one frame).

## Deliverable shape

Keep the pose output as a separate, parallel array set alongside whatever
single-point estimate already exists (`R_cam`, `t_cam`, `pose_valid`,
`fit_iou`, `fit_depth_err_m`) so nothing downstream breaks, and record a
`fit_reason` per frame — a bare "49% failed" is not actionable, "106 frames
rejected on depth mismatch, longest stretch 95 frames starting at frame 0" is.
Print a per-block coverage timeline too; it immediately distinguishes an
occluded stretch from a systematically broken estimator.

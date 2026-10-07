# Troubleshooting — symptoms → cause → fix

Each entry was a real failure on real footage; the numbers are from the
20260917 clip (497 frames, 30 fps, D455, 1280×720).

## Box

### Box spins / turns in place while parked or carried
- **Cause**: the cuboid fit resolves pose only modulo the box's symmetry
  group (square section: `Rz(90k)` × `Rx(180)`, order 8). Under arm occlusion
  the solver re-locks onto an equivalent branch — on the real clip the yaw
  drifted a CONTINUOUS 180° during pickup (each frame delta <45°, so chained
  prev-frame snapping fired 0 times) and stayed there.
- **Fix order**: snap every frame to the symmetry branch closest to a FIXED
  anchor (floor-parked median) → unwrap rotvec along the timeline → median
  filter (window 9) → mean smooth (window 15) → slew limit 3.5°/frame.
  Real numbers: snap 350/486 frames, unwrap 130, slew clips 91→0.
- **Trap**: never average raw quaternions; a chained previous-frame reference
  follows gradual drift branch by branch and does nothing.

### Box bobs vertically during carry
- Cause 1: a jittery retargeted joint (left elbow invented a 4→60→−3° excursion
  in 0.7 s) jolts the hand midpoint → despike the arm dofs (below).
- Cause 2: `A + LPF(B − A)` rewrites as `LPF(B) + (A − LPF(A))` — the
  UNFILTERED hand term leaks its high-frequency jitter at full amplitude even
  with gain 1. Smooth both terms with the same window.
- Accept: carry z step med ~0.6 mm, max ≈ lift speed only.

### Box clips through the hands or the torso
- Cause: measured centre vs hand-rigid convention differs by the whole
  retargeting+Sim3 error (0.17–0.40 m; wrist spacing only 0.35 m; box centre
  came within 77 mm of the spine vs 212 mm half-diagonal).
- **Fix (in order, all automatic in the current render)**:
  1. GRIP CLAMP — carry box centre = hand midpoint + low-passed measured
     drift clamped into a 50 mm ball; ramp 25 frames in at carry start
     (the align split precedes the actual grasp — anchoring at the split
     yanked the box off the floor +341 mm in one frame), 12 frames out.
  2. BODY KEEP-OUT — push the centre horizontally out of a spine capsule
     (pelvis→torso+0.2 m) with radius = 0.23 m.
  3. STOOL CAPTURE — in the last ~60 carry frames, once z reaches the stool
     top: pin z there and slide xy onto the measured resting spot over 15
     frames, then HOLD (a sliding-then-releasing version let the hands drag
     the box 93 mm off the stool).

### Box teleports at the very start/end of the clip
- Cause: uncovered frames fall back to the SYNTHESISED parked pose, which
  sits ~0.9 m from the real measured spot; `w=1` on covered parked frames
  and `pos_meas = where(cov, ...)` both bypass the blend. This hid from QA
  for days because the check masked by cov.
- **Fix**: extend the frozen floor-parked median over uncovered parked
  frames (w=1), and select `pos_meas` by `w > 0`, not by cov.
- **Rule**: the QA jump list must scan ALL frames unmasked.

### Box placed off the stool (overhangs the front edge)
- Cause: stool placed under the MEASURED resting spot, but the gripped box
  still follows the hands; stool top is only ~20 mm wider than the box.
- **Fix**: the stool-capture xy slide (above). Verify with the box centre's
  offset from its own final resting spot (must be ~0 from capture on).

### Box snaps upward in one frame at pickup (QA: `parked pos jump > 10 mm`)
- Cause: `first_c` (the carry flag) arrived LATE. The old split opened the
  carry gate when the robot's hands came within `inlier_m` of the box CENTRE,
  but a person grips the box by its top corners — so on a fast grab the box
  had already risen 0.2-0.5 m by the time the gate opened (20260922: gate at
  f197, box left the floor at f148). The renderer pins every non-carry frame
  to the rest height and anchors the measured z onto that height at `first_c`,
  so the first carry frame jumped 225 mm.
- **Fix** (already in `align_box_to_world.py`): `carry_split()` now takes the
  split from the BOX's own camera-frame height/speed (Sim3-free), choosing the
  longest excursion above the floor band and backing off to the last frame
  still inside it. See "How the carry/ground split is decided" in SKILL.md.
  Result: 225 mm → 5.0 mm (20260922), 11.7 mm → 7.5 mm (20260917).
- **Watch out** when tuning the trigger: a parked box drifts a few mm
  (quantised depth), so a 5-frame "sustained above" test fires early in the
  clip (20260922 f34-f40 sat 6-13 mm high, which dragged `first_c` to f33 and
  wrecked every downstream set). Require a LONG run, not a short one.
- Diagnostic: `python .dsh/skills/gmr-carry-video-pipeline/scripts/probe_box_height.py data/<prefix>_6dof.npz 2`
  prints the filtered height, its parked-phase σ, and the `first_c` a range of
  thresholds would pick. `GMR_MOTION_DEBUG=1` makes the align step dump the
  height trace around the chosen frame.

### The robot crouches before it reaches the box / its hands never touch it

- Symptom: in the frames around the pickup the box rises and drifts toward the
  robot while the hands are visibly NOT on it — the robot appears to squat down
  before it has walked up to the box.
- **Cause**: `first_c` is when the **box** really left the floor *in the video*.
  The **retargeted robot** does not reach the box on that frame. Measured on
  20260922: at `first_c` (f148) the robot's hand midpoint was still **0.32 m**
  short of the parked spot horizontally, and only came level with it at f166.
  The renderer was starting the measured lift anyway, so the box flew to the
  hands instead of the hands reaching the box. It is a retargeting/reach
  residual, not a measurement error — the box's own spot is right (its final
  landing was 0.052 m from the stool centre).
- **Fix** (already in `vis_robot_with_box.py`): the **grasp wait**. The box is
  held on its parked spot until the hands actually arrive (horizontally within
  `--grasp_reach`, default 0.24 m, and low enough to be gripping), then handed
  over to the rigid grip. The z hand-over is rate-limited
  (`--grasp_z_maxstep`, default 40 mm/frame) so the lift starts from the floor
  smoothly. The gate only ever DELAYS, never advances a pickup, so clips where
  the hands do arrive on time are bit-for-bit unchanged (verified on
  20260917: "no wait needed", carry stayed 188..391).
- Failure-mode log, both of which are real QA failures — do not reinvent them:
  pinning/replacing z per frame gave a **127 mm** pop at the handover frame;
  ramping the measurement in over a short band gave a **64 mm/frame** lift.
  See "When the robot reaches the box LATE" in SKILL.md.
- Diagnostic: `.workbuddy/diag_crouch.py <g1.pkl> <boxpose_used.npz> [f0 f1]`
  prints root/pelvis/hand heights against the box pose per frame;
  `.workbuddy/diag_lag.py` reports whether the robot is *uniformly* late (if the
  best global lag `tau` is 0, the problem is local to the pickup, not a time
  offset — do not time-shift the box). `.workbuddy/sheet_pair.py SRC.mp4 REN.mp4
  OUT.png 148,156,164 --cx 0.5 --cy 0.58 --w 0.22 --h 0.62 --scale 1.5` puts
  the same source and render frames side by side, zoomed.
- **NOT a bug — a measurement you must not eyeball**: it is tempting to conclude
  the robot starts its deep hip-fold ~20 frames before the person. That was
  measured and is **false**: the retarget tracks the SMPL-X human at **0-1 frame**
  (corr >= 0.98 on pelvis / chest / knee-flexion / torso-lean), GVHMR leads the
  source video by **+2 frames**, and the box mapping is **+1 frame**. The
  impression comes from (a) the G1 being 1.32 m vs a 1.615 m human, (b) a
  **5-15° surplus knee bend** at equal pelvis height, and (c) the source video's
  chest-height frontal camera vs the render's oblique top-down one, which
  flattens a lean in one and opens it up in the other. Two traps when comparing
  frames by eye: **source frame = pkl frame + `frame_offset`** (+2 on 20260922,
  **-11** on 20260917), and never compare leans across two different camera
  setups. Use `scripts/check_timing_sync.py` instead — see SKILL.md, "Timing sync:
  measure it, never eyeball it". **Do not time-shift the dof trajectory.**

### The arm vanishes INTO the box during the lift (hand<->box interpenetration)

- Symptom (20260922 v13, correct camera finally made it obvious): around the
  lift the whole right forearm+hand disappears into the box side, and the box
  top cuts the chin. The user reports "hand penetrates the box".
- **Cause**: the retargeted G1 GRIP COLLAPSES during the lift — wrist span
  shrinks 213 -> 140 mm while the box is 300 mm wide, so the hands are
  physically inside the box volume, and the box centre is anchored at the hand
  midpoint (GRIP CLAMP). Anchoring the box to the MEASUREMENT instead buries
  the hands too (105/136 carry frames have a rubber-hand centre inside the
  measured box) AND used to bury the torso (centre-to-spine 77 mm < 212 mm
  half-diagonal — the reason GRIP CLAMP exists). There is NO box placement
  that looks like a proper two-hand side grip: the G1 hand pose simply does
  not match the human's (median offset 148 mm). Pick which artefact to hide.
- **Fix (v15, 165945 — v14's box-side tweaks were REJECTED by the user)**:
  fix the ARM AMPLITUDE, not the box:
  1. `--carry_abduct_deg 25 --carry_abduct_win F0 F1` — abduct both shoulders
     over the collapse window (left dof16 += deg, right dof23 −= deg, 12-frame
     smoothstep ramps, applied AFTER the carry mirror). At d=25° the hands
     swing laterally OUT of the box volume to its side faces at top-edge
     height — exactly the human grip; d=15° is not enough. Measured effect
     (165945): bend-down/approach rubber-hand penetration 20–102 mm → 0,
     lift peak +124 → +53 mm. Keep `--carry_grip_slack 0.45` (box follows
     the measurement; the EXISTING spine keep-out protects the torso,
     75 frames max 157 mm).
  2. **Place phase: mirror the LEFT arm onto the RIGHT** (default-on,
     `--no_mirror_right_place` to disable). The retargeted right hand dives
     into the placed box near its centre (up to +123 mm deep — v13's
     hand-following box hid this; v14's stool-pinned box exposed it, which is
     why v14 looked WORSE) while the raw left hand lands on the box edge
     (+3 mm). Post-carry, right dofs 22–28 := mirrored left dofs 15–21,
     12-frame ramp from the carry end; the ramp is self-consistent (while
     left ≈ mirror(right), mirror(left) ≈ right itself, so nothing moves
     until the left arm returns to its raw pose). Known limit: the dof mirror
     reflects across the ROBOT's sagittal plane, not the box axes (18.8° apart
     on 165945), so the reflected hand still sits ~100 mm deep — but the
     elbow drops with it, the "arm plunging into the box top" silhouette
     disappears, and the box occludes the rest (reads as edge-gripping).
  3. `--hand_push_out_mm 30` stays as a BACKSTOP only. It is a depth-REDUCER,
     not a true exit (MTV = min over faces of half−max; a centre-buried
     vertex whose nearest faces are y/z gains nothing from an x push), and a
     clamp large enough to fully exit a centred hand (~150 mm+) would detach
     it from the wrist. Never rely on it as the primary fix.
- Diagnose first, and trust the RENDER over the numbers:
  `--hand_diag` (vis_robot_with_box.py) counts, per frame, every hand/wrist
  mesh vertex inside the raw box OBB and prints the MTV at the worst frames
  (each link has TWO geoms sharing one mesh — deduped); add
  `--hand_diag_npz out.npz` for the per-frame depth/centre/MTV dump, then
  split by phase (approach <first_c, lift, carry, place) — the top-12 alone
  hides WHERE the penetration lives. Deep wrist-link numbers are usually
  invisible; the rubber-hand ones are what the user sees.
  `.workbuddy/probe_handgrip.py` / `probe_abduct*.py` print wrist span +
  hand centres in the measured box frame per frame and verify the abduction
  magnitude BEFORE rendering. Zoomed REAL|REND sheets decide what actually
  needs treating.

### An arm detaches from the box / a hand pokes through the box face in the walk (turn/carry phase)

- Symptom (20260922 v15 REJECTED): mid-walk one arm hangs fully detached off
  the box's side; a few steps later a hand silhouette shows through the
  middle of the camera-facing box face. Grasp and place look fine.
- **Cause**: with `--carry_grip_slack 0.45` the box stays on the MEASURED
  trajectory for the whole carry, but the retargeted hands wander around it
  once the box is airborne (on 165945: left hand 180–230 mm beyond the side
  face f197–209, then half-buried in the front-face corner f213–245; right
  hand hovering outside the front face). It is NOT introduced by the
  abduction/mirror fixes — a v14-vs-v15 hand-position diff is frame-identical
  once the lift ends. v13 (slack 0.05, box-follows-hands) never showed it.
- **Fix (v19, 165945)**: phase the slack — keep 0.45 for the grasp/lift
  (validated there), then ramp down to hand-following:
  `--carry_grip_slack_walk <s> --carry_grip_slack_walk_win F0 F1`
  (smoothstep on top of the existing clamp) + `--mirror_left_end F` (hand
  the left arm back to the raw re-gripping retarget once the lift is done).
  Window MUST cover the whole pivot: the drift starts when the body begins
  turning (165945: f185), a ramp that only completes mid-pivot (f196–216
  tried first) leaves the early detach visible. v19/v20 ran **0.06 over
  f172–192** (`render_v19.log`: "walk slack 60 mm ramped over f172..192").
  ⚠ **This is NOT what got delivered.** From v23 through the delivered v29 the
  value was tightened to **0.02 over f150–165** (`run_v23.ps1` … `run_v29.ps1`;
  `render_v29.log`: "walk slack **20 mm** ramped over **f150..165**"). The
  change is recorded in no log. **Reproduce from `run_v29.ps1`, not from prose.**
- **What does NOT work**: tightening the slack alone (v18) turns the
  one-arm detach into BOTH hands floating ~10–15 cm off the box edges
  during the pivot — the retarget spreads the hands ~520 mm apart (raw AND
  mirrored alike; `--mirror_left_end` changes it by <40 mm) while the body
  rotates ~60° around the yaw-anchored box re-gripping, and a 300 mm box
  centred between them cannot reach either hand. This residual pivot float
  matches the v13-accepted baseline; only arm-level IK / box-frame mirroring
  could do better, and it has not been needed so far.
- Side effect to expect: with the box riding at the hands, the body
  keep-out stops triggering (box centre stays ≥230 mm from the spine) —
  don't assert its log line in delivery checklists. QA stays PASS (carry
  rot step unchanged, no >50 mm jumps).
- Diagnose: `--hand_diag_npz` dump + `.workbuddy/hd_walk.py <npz> <out.txt>`
  (hand centres in the box frame over the walk, box yaw trace);
  `dump_frames2.py <video> <outdir> f1,f2,...` for full-res frames of ANY
  video (dbg renders included); `sbs_sheet2.py <out.jpg> frames - <rend.mp4>`
  for REAL|REND sheets against a dbg render.

### The wrist/hand faces OUTWARD (back of the hand on the box)

- Symptom (20260922 v19, spotted by the advisor): in the walk the hand
  pressed on the box shows its BACK — fingers point away from the box.
  "The wrist should face inward."
- **Cause**: `--hand_roll_deg` defaults to 85 (tuned on the 0911 clip: the
  rigid shell's thumb sticks out ~48 mm sideways and poked into the 0.30 m
  box side, so the hand was rolled until the thumb pointed up over the top
  edge). The side effect is the palm turning OUTWARD. It is a per-clip
  cosmetic, not a universal default.
- **Fix (v20, 165945)**: `--hand_roll_deg 0` — the stock orientation is the
  correct grip on this clip (palm on the box, fingers wrapping the edge;
  grasp-on-top and place-edge also read naturally). Cost: fingertip depth
  into the pressed face grows ~5–20 mm — invisible against the face the
  hand is pressing on. Decide per clip with `.workbuddy/probe_wristroll.py`
  (rebuilds the delivered scene offline, renders a roll-candidate x frame
  grid in seconds — far cheaper than full renders; compare penetration with
  `.workbuddy/hd_rollcmp.py` on two `--hand_diag_npz` dumps).

- **★ Why it came back on 13 of 14 clips (found 2026-10-06).** The v20 fix
  was applied ONLY to 165945, and only through its hand-written run script
  (`run_v29.ps1` passes `--hand_roll_deg 0`). `batch_pipeline.py`'s render
  command passes **no** `--hand_roll_deg`, so every other clip silently fell
  back to `BUILTIN_DEFAULTS["hand_roll_deg"] = 85` — the value known to roll
  the palm OUTWARD. Net effect: 165945 correct, all 13 other delivered clips
  shipped back-to-front, which is the reported "almost every video has the
  hands facing the wrong way".
  **Lesson**: a per-clip cosmetic that lives in an invisible builtin default
  is a trap. The default is now `0.0` (palm on the box), and the value is
  recordable per clip in the sidecar `<pkl>_box.json`.
- **Per-hand control**: the two shells are mirror images and their measured
  optimal rolls sit **26–37° apart**, so one scalar may not suit both. Use
  `--hand_roll_left_deg` / `--hand_roll_right_deg` to override each hand
  independently; both default to `None` → fall back to `--hand_roll_deg`.
  The renderer applies `+deg` to the LEFT hand and `−deg` to the RIGHT.
- **Before blaming the roll, check for burial.** The hands sit INSIDE the box
  volume on **39–71 %** of carry frames (measured over 5 clips; the box is
  0.30 m wide while the retargeted wrist span is ~0.28 m and the rigid shell
  reaches ~0.12–0.22 m past the wrist). When a hand is embedded, no roll can
  make the grip read correctly — fix the width first (`--hand_shrink`,
  `--carry_abduct_deg`, or skip the clip).
- **Verification that a roll change is cosmetic** (do this before re-rendering
  a delivery): the dumped `--dump_box_pose` npz must be **bit-identical**
  (`box_pos` / `box_quat_xyzw` max abs diff `0.0`) between the two roll
  values, because the grip clamp is driven by `hand_mid` (a body position),
  not by the rolled mesh. Do NOT use mp4 md5 for this — see the WGL note
  above; compare pixels with a noise-floor tolerance instead.

### The hands are NOT touching the box while setting it down

- Symptom (20260922 v20, user: "the hand orientation is fine now, but in many
  stretches the hands still aren't touching the box"). Several seconds of the
  place phase show the box with only a hovering hand near it.
- **Cause**: the SET-DOWN BOW was removed by the helmet-clearance fix.
  `--place_waist_lift_deg 20` SUBTRACTS 20 deg from waist pitch (dof 14) from
  the carry end on — but the source person BENDS right over the box at that
  moment (measured dof 14 = +29.8 deg on 165945 f290-300). Net effect: the G1
  stands upright, and the retargeted right hand is flung up and BACK off the
  box — on 165945 it sat 84 mm INSIDE the box's front face (reads as "the
  hand is poking through the middle of the box"), the left hand 80 mm off the
  left side. Diagnosis: dump the source frames at the place phase
  (`.workbuddy/dump_frames2.py video/<prefix>.mp4 out f294,f300`) and compare
  with a REAL|REND sheet (`.workbuddy/sbs_sheet2.py`) — REAL folds, REND is
  bolt upright.
- **Fix (v26, 165945)**: `--place_bow_deg 15 --place_bow_win 280 316` — a
  FLOOR (max, applied AFTER the lift) on dof 14, smoothstep-ramped 12 frames.
  Pick the value by A/B on the RENDERED hand diag (the offline
  `.workbuddy/hd_sgap.py` rebuilds the arm chain and IGNORES bow, so it will
  report "no change" — use `--hand_diag_npz` and read `center_box`/`depth_mm`):
  12 = right hand still on the front face (helmet clean); **15 = right hand
  up on the box's right edge, helmet mesh 22 mm in (adopted)**; 18 = right
  hand properly on the right SIDE face but the helmet sinks 35 mm into the
  box top, which IS visible. The helmet can't be rescued independently —
  `head_mocap` is a fixed +0.4 m child of `torso_link` (no dof), and raising
  `--place_root_clear_m` does nothing because the folded head reaches
  forward-and-down, not backward.
- **Dead ends on this clip**: `--place_hand_z_follow` (tying the box top to
  the wrist midpoint so the BOX rises to the hand) is a no-op in the place
  phase — carry ends at last_c (280 here) and from there the box is pinned to
  the stool rest height, so every carry-side z rule is overwritten by the
  drop guard; it only distorted the still-attached frames into a V. And
  `--no_mirror_right_place` is WORSE than the default mirror (right hand
  +33…+55 mm into the front face vs +17 with the mirror) — the place-phase
  right-arm mirror is doing real work here.

## Robot body

### One hand/elbow twitches (fast oscillation)
- **Cause**: GVHMR loses an occluded arm and invents a 20–30-frame "bump"
  (4→60→−3° in 0.7 s) that the source video does not show. A short median
  filter or a slew limiter only stretches it.
- **Fix**: two-stage despike on the 14 arm dofs (legs/waist untouched — gait
  is trustworthy): rolling-median baseline (window 61, > 2× bump length) with
  deviation clamped to ±12°, THEN median filter window 51 for the
  below-threshold dips (median filters preserve monotone ramps and step
  edges — the shapes of genuine motion; a mean filter does not).

### Left forearm/hand steadily held in a wrong pose ("pried upward")
- **Cause**: systematically biased estimate of the occluded limb — filtering
  cannot fix a steady wrong pose.
- **Fix**: drive the LEFT arm with the mirrored RIGHT arm during carry
  (pitch/elbow keep sign, roll/yaw negate — verify the sign map against the
  medians of a symmetric stretch), 12-frame ramps at both ends, applied
  BEFORE forward kinematics. A person carrying a box grips symmetrically.

## Fit / alignment

### Cuboid fit 0% or mask tracks the person
- Wrong HSV for the box colour; the mask chases clothing. Dump a preview
  frame and look at where the yellow mask + green outline are.

### Parallelepiped leaning 45° backwards with great IoU
- IoU alone cannot distinguish tilt toward vs away from the camera; the
  per-pixel depth-consistency term is the only thing that breaks it. Rank
  candidates and refine with the SAME objective (IoU + depth).

### The box only turns once / a real box rotation is missing
- **Symptom**: the source video shows the box being turned in the hands
  (front face → side → back again, ~90° out and ~90° back), but the render
  shows the box keeping one heading the whole carry ("only turns once" /
  not at all). 165945 turned ~5° in v26, ~40° in v27 — both far short of
  the real ~90°.
- **Cause (165945) — TWO layers, not one**:
  - **Layer A (fixed in v27): three temporal filters** tuned on a 0911 clip
    where the box genuinely did not rotate:
    1. `--carry_yaw_anchor_deg D` — pulls the carry yaw toward the linear
       interpolation of the two PARKED-phase median yaws, wrapped to the
       nearest 90° symmetry branch (98 of 136 carry frames corrected).
       **Main culprit of the v26 "5°" case.**
    2. `--meas_median K` — replaces ANY frame deviating >5° from its local
       median with that median; a genuine monotone turn has every frame
       deviating, so it deletes the rotation.
    3. `--meas_smooth_quat W` — a W-frame moving average rounding the ramp.
    Setting `--carry_yaw_anchor_deg 0 --meas_median 3 --meas_smooth_quat 5`
    lifted the swing to ~40° but **could not go past ~67°**.
  - **Layer B (fixed in v28 — the real reason the turn capped at ~67°):
    the `symmetry branch snap` used a FIXED anchor.** A square-section box
    is identical modulo 90°, so a GENUINE 90° turn lands on a
    symmetry-EQUIVALENT branch. The snap maps every frame to the
    representative closest to a fixed anchor (the floor-parked median) —
    which folds the real 90° turn back to ~0°. No downstream filter change
    can recover it; the information is thrown away at the snap step.
- **The proof the measurement DOES contain the turn**: unwrap the fitted
  `box_R_world` yaw **mod 90** (i.e. allow ±90° branch hops, chain from the
  previous frame) and the true trajectory appears: 165945 goes −15° (f145)
  → −96° (f186, turned right ~80°) → back to ~0° (f206) → settled. Script:
  `.workbuddy/unwrap90.py`.
- **Fix (v28)**: new CLI `--box_sym_snap continuity`. Instead of snapping to
  a fixed anchor it **chains the branch from the previous frame** (seeded by
  the anchor). This follows a genuine turn while still removing the isolated
  per-frame branch flips the fit makes under self-occlusion (those sit far
  from BOTH neighbours, so the previous-frame reference rejects them).
  Keep the v27 filter settings (`--carry_yaw_anchor_deg 0 --meas_median 3
  --meas_smooth_quat 5`). On 165945 this restored the amplitude from 40°
  (v27) to **107°** with a clean monotone out-and-back ramp
  (f152–f168 out at 9°/frame, f196–f214 back at 9°/frame, peak at f180–f184).
  The rendered yaw now tracks the raw-unwrap90 measurement exactly (a
  constant ~90° branch offset, invisible on a square box).
- **How to confirm a REAL turn vs fit noise** (use all four):
  1. Source-video strip: crop the box region over the window and look for the
     face going edge-on (narrow) then fronto-parallel (wide). Script:
     `.workbuddy/zoom_dense.py` / `.workbuddy/tightbox.py`.
  2. `minAreaRect` on the box mask: the rectangle's long axis FLIPS 90°
     across a real turn (165945: long-axis horizontal through f201, vertical
     from f202). Script: `.workbuddy/wide_turn.py`.
  3. Label tracking: the white sticker lives on ONE face and (unlike the
     box) is symmetry-free — its offset from the box centre slides edge to
     edge across a real turn.
  4. **mod-90 unwrap of `box_R_world`** (`.workbuddy/unwrap90.py`): the
     decisive test. A real turn is a smooth multi-frame ramp in the
     mod-90-unwrapped series; noise is an isolated single-frame jump that
     immediately returns. Do NOT judge by the RAW amplitude alone — a
     fixed-anchor snap hides the ramp, and an euler/branch wrap can inflate
     the raw number.
- **Side effects**: QA is byte-for-byte identical to v27 (only the accepted
  pickup TELEPORT at f159-f164), hand-box penetration is unchanged
  (walk phase identical; a +17 mm wrist-link blip at f212 is occluded by the
  box face), the parked phases stay frozen. QA: pass `--carry_rot_max 4.5`
  to `check_render_quality.py` — a real turn legitimately rides the
  renderer's own `--quat_max_step_deg` cap.

### The box turn LAGS the robot body ("箱子转的相对于机器人来说有一些延迟")
- **Symptom**: the amplitude is right (the box does turn ~90° out and back),
  but its rotation happens **late relative to the robot's own body turn** —
  the body has already pivoted while the box is still finishing its turn, and
  the box "catches up" several frames behind. Reported by the user on v28:
  "感觉箱子转的相对于机器人来说有一些延迟吧".
- **Cause (165945, fixed in v29): the slew-rate limiter
  `--quat_max_step_deg` was set BELOW the real box turn rate.** It defaults
  to 3.5 and v28 ran it at 4.5 (chosen in the v8 era, where the comment even
  says "raise it above the real turn rate (~4 deg/frame)"). The limiter is a
  **hard clip with `prev` feedback** — it compares each frame to the
  already-CLIPPED previous frame, so once it clips it keeps clipping until the
  target is caught. A genuine burst therefore becomes a long ramp:
  - the measurement's true back-turn is a **~34 deg/frame** burst
    (f196) settling in ~8 frames;
  - clipped to 4.5 deg/frame it becomes a **22-frame plateau (f194–f215)**;
  - so the render's back-turn 50 % crossing slid from f200 (raw) to f204 (v28),
    and the 75 % crossing from f202 to f209 — **+4 to +7 frames late**.
- **Quantify it — the halfway metric** (`.workbuddy/halfway.py` or
  `.workbuddy/verify_timing.py`): for each of {BODY yaw (g1 `root_rot`),
  RAW box yaw, REN box yaw} find the frames where it crosses 25/50/75 % of the
  way from the pre-turn plateau to the far plateau, separately for the OUT
  half and the BACK half. On 165945 (render indices):
  | | OUT 25/50/75 | BACK 25/50/75 | BACK lag vs body |
  |---|---|---|---|
  | BODY | 160/165/170 | 190/194/199 | — |
  | RAW  | 157/160/164 | 196/200/202 | 6/6/3 |
  | v28  | 157/161/166 | 198/204/209 | **8/10/10** |
  | v29  | 157/160/164 | 197/199/202 | **7/5/3** ✓ |
  The OUT half passes even with a tight cap (its rate is lower); **the BACK
  half is where the clip bites** — always measure both halves, and always
  compare against the RAW box, not just against the body (the raw box itself
  genuinely trails the body by ~6 frames here; a faithful render must
  reproduce THAT, no more).
- **Size the cap from the data, not from a default.**
  `.workbuddy/raw_steps.py` prints the symmetric-aware 3-D angular step of the
  raw fit over the carry (nearest of the 8 box symmetries, so a 90° branch hop
  does not read as motion). 165945: **median 4.7 / p95 25.9 / max 38.5
  deg/frame** — the 4.5 cap was clipping the majority of the real rotation.
  Set `--quat_max_step_deg` at or just above p95 (20 worked: the log then
  says "slew-rate limited to 20.0 deg/frame: **0 frames clipped**", i.e. the
  limiter is a pure safety net and the measured motion passes untouched).
- **Do NOT just disable the limiter** if you still distrust the fit — and if
  you do raise it, prove no spike came back: with 20 the >6 deg/frame output
  steps form two smooth ramps (f158–160 and f195–203, neighbours always
  comparable), no isolated single-frame spike
  (`.workbuddy/spike_chk.py`). An isolated spike = high step with low steps on
  BOTH sides; a real fast turn = a ramp.
- **Side effects on 165945 (v29)**: amplitude preserved (107.2°); QA diffs
  from v28 ONLY in the rotation-step lines — position steps, parked phases and
  the accepted pickup TELEPORT (f159–164) are byte-identical. Hand-box
  penetration: the peak is unchanged (v28 72.8 mm, v29 66.5 mm at f207); the
  per-frame overlap simply **shifts ~4 frames earlier** to follow the box
  (f199–204 +up to 25 mm worse, f206–211 up to −18 mm better) — that is the
  correct behaviour, not a regression. QA: `--carry_rot_max 20`.
- **The general lesson**: any temporal filter/limiter added to stop noise will
  bake in a PHASE LAG proportional to how much of the real signal it clips.
  Always re-derive the limiter's threshold from the clip's own measured rate
  before trusting a default tuned on a different (quieter) clip, and verify
  with the halfway metric rather than by eye.

### Floor→stool rise wrong
- Measure the box centre's height above the fitted floor plane IN CAMERA
  FRAME first (0.225 m expected for a 0.45 m box) — that localises the error
  to the fit vs the Sim3 alignment. Pitch leakage in Umeyama scales with the
  travel distance; gravity-lock + enough parked frames fixed it (rise error
  3 mm on the real clip).

### Log says `independent check ... (err -0.189 m)` — is the render broken?
- **No.** Verify the render, not the log: read the dumped npz and compare
  STOOL-parked median z − FLOOR-parked median z. It must equal the stool
  height (0.4500 m measured, exact). If that is right, the render is right.
- **Cause**: the two-anchor z bias built its per-frame `bias` array with
  `np.where(carry_ph, interpolated, z_bias_f)` — frames AFTER the carry got
  the FLOOR bias instead of the STOOL bias, so the parked-on-stool z came out
  189 mm low *in that intermediate array only*. The `--pose_z rest` policy
  then pins the parked phases to their physical rest heights, so the shipped
  video was always correct; only the printed "independent check" was wrong.
- **Fix** (already in): after `last_c`, use `z_bias_s`; the check now prints
  `err +0.000 m`. General rule — when a QA number looks alarming, first
  confirm which *stage* of the pipeline it was computed from; a mid-pipeline
  diagnostic can disagree while the final output is fine.

### Many clips BAD at world with riseE 13..140 mm (batch, far parking spot)
- **Mechanism**: the rise gate (`check_world` in batch_pipeline.py, ±12 mm)
  compares the *measured* floor→stool rise against the real stool height. The
  measured rise is purely the box 6-DoF depth measurement (scale and the
  Umeyama constant cancel in the difference), so a range-dependent depth bias
  at the far parking spot directly inflates it. 20260922 batch: box parked
  ~4 m away → +13..+140 mm error on most clips; the two clips parked closer
  (≤3.4 m) passed.
- **Not a fit bug**: fit metrics are excellent (IoU 0.90+, dep 20–35 mm) and
  the wireframe hugs the box in the fit preview.
- **Render is unaffected**: the renderer pins parked phases to floor+half-box
  (`--pose_z rest`); the measured z never reaches the picture. So a BAD world
  only *blocks* render/qa/sync in the pipeline — it does not corrupt anything.
- **Fix (20260922)**: two switches in batch_pipeline.py —
  `--world_rise_tol_mm 60` (relax the gate; table then shows OK) or
  `--allow_bad_world` (BAD no longer blocks render/qa/sync; the table keeps
  showing BAD + rise_err, which is the honest option). After a full batch,
  re-advance everything with:
  `python scripts/batch_pipeline.py --bag-root <数据根> --run --allow_bad_world`
- Related marginal QA fail: 162859 FAILed only on "parked pos jump 11.6 mm >
  10 mm" (1.6 mm over) — same far-range noise, jump scan clean; acceptable.

## Diagnostics that localise any of the above

1. Extract the SAME frame indices from the rendered video AND the source
   video, side by side — "fit says 100°, source says nothing moved" localises
   the bug instantly. Tool:
   `python .dsh/skills/gmr-carry-video-pipeline/scripts/contact_sheet.py <video.mp4> <out.png> 0,120,149,155,200,240`
   (writes a tiled PNG; the raw frame plus the render at the same index is the
   fastest way to see whether an artefact is in the data or in the render).
   `.workbuddy/sheet_pair.py SRC.mp4 REN.mp4 OUT.png 148,156,164 --cx 0.5
   --cy 0.58 --w 0.22 --h 0.62 --scale 1.5` is the zoomed variant: it stacks
   source over render per frame, cropped, so you can actually see hands vs box.
2. Per-dof step stats (median/p90/max, sorted) find the twitching joint in
   one pass.
3. Unmasked top-jump list over the rendered pose finds teleports at
   coverage boundaries that masked statistics hide.
4. Ground-truth the RENDER, not the logs: load `boxpose_used.npz` and check
   (a) STOOL z − FLOOR z == stool height, (b) parked rot step p90 ≈ 0,
   (c) no >50 mm jump outside the lift. A log line can be a mid-pipeline
   artefact (see the `err -0.189 m` entry above).
5. Split body-vs-box timing (the "robot reaches the box late" class of bug):
   `.workbuddy/diag_crouch.py data/<prefix>_g1.pkl data/<prefix>_boxpose_used.npz 120 178`
   prints root/pelvis/hand heights next to the box pose per frame;
   `.workbuddy/diag_gap.py` prints the hand-to-box gap and the per-frame box
   step through the pickup (this is what catches a >50 mm lift step);
   `.workbuddy/diag_lag.py` answers "is the robot uniformly late?" — if the
   best global lag is `tau=0`, the problem is LOCAL to the pickup and shifting
   the box timeline is the wrong fix.


## The hands stab INTO the box (clipping / 穿模)

- **Symptom**: fingers visible poking out through the box faces during the
  carry; the hands look buried in the box.  Measured on all 14 delivered
  clips: 32–56 % of hand-shell vertices sit INSIDE the box volume, up to
  150 mm deep, on up to 100 % of carry frames.
- **Why `--diag_contact` never sees it**: it measures the wrist COLLISION
  CAPSULE, and `rubber_hand` is a contype=0 pure-visual mesh.  MuJoCo also
  bakes the collision hull at compile time, so the runtime `--hand_shrink`
  edit of `mesh_vert` is invisible to it.  Contact-based checks can never
  catch this — measure at VERTEX level (`.workbuddy/hand_clip_measure.py`).
- **Cause**: the retargeted wrist span (~280 mm) is NARROWER than the 300 mm
  box while the rigid shell reaches 120–220 mm past the wrist, so the shells
  must stab into the box.  This is the retargeting's AMPLITUDE error
  (G1 1.32 m vs a 1.615 m human) — the same class the left-arm mirror fixes.
- **Fix**: `--carry_abduct_deg` (default **30**) with an AUTO window.
  - The angle is measured, not guessed: worst penetration 82 mm at 0 deg,
    4.1 mm at 26 deg, **exactly 0 at 27 deg**; 30 leaves margin.
  - The window defaults to AUTO = the measured carry span (`carry` in the
    world npz) grown by 8 frames each side.  A hand-picked window LEAKS:
    161727 still clipped 24.9 % with 148..195, and only the full-span window
    reaches 0 — which is why the old 25 deg / f148-195 setting never cured it.
  - Per clip: 9/14 clips go to 0.0 % with the default; 162931/170444 need
    ~50 deg, 173139 ~40, 0917 ~45 (set per clip in the sidecar json).
- **The abduction must stay cosmetic.**  `compute_box_trajectory` anchors the
  carried box on the hand midpoint, which the abduction shifts: with a single
  dof array a 35 deg swing moved the delivered box trajectory by **80 mm**.
  The renderer therefore keeps the RAW (pre-abduction) dofs for the box /
  clamp / contact math and only renders the abducted arms — verified: the
  delivered trajectory then differs by just 0.33 mm and QA still PASSes.
- **Known boundary**: 165945 keeps 5.6 % / 48 mm even at 50 deg, all on the
  RIGHT hand at the box's front-bottom corner entering from BELOW (box-frame
  z ≈ −0.8, f166–185).  That is the documented "right hand dives" defect, not
  a grip-width problem — abduction cannot reach it.
- **mesh→world convention for vertex measurements** (easy to get wrong):
  `v_geom = rq @ v_mesh`, i.e. batch `Vg = V @ rq` — the same convention the
  renderer's shrink block uses.  Verified empirically: it puts the shell
  0.127–0.139 m past the wrist_yaw origin (matches the documented
  0.121–0.136 m); the inverse (`Vg = V @ rq.T`) gives an implausible
  0.159–0.165 m.  `mesh_quat` here is ~114–129 deg from identity, so the two
  conventions differ a lot.  See `.workbuddy/probe_mesh_frame.py`.

## Script / environment errors (not data problems)

### ★ mp4 md5 is NOT a valid "nothing changed" oracle on this machine
- **Symptom**: you re-render the same clip with the same arguments (even with an
  *unmodified* script) and get a different `md5` for the mp4. It looks like the
  change you just made altered the render.
- **Cause**: MuJoCo 3.13.0 here renders through the **WGL software path**
  (no `libOSMesa` installed, `MUJOCO_GL` unset), which is only ~1-bit stable.
  Measured: two runs of identical code differ by **up to 2 LSB on ~203 of 354
  frames**; decoded-video max |Δpixel| = 33 — the *same* magnitude as the
  pre-vs-post spread, so a code change contributes nothing detectable.
- **What to use instead** (in order of strength):
  1. **Compare the dumped NPZ.** `--dump_box_pose` and `--hand_diag_npz` come
     from the same numbers that drive the render and are exactly reproducible
     (measured max abs diff **0.0** across independent runs). This is the real
     oracle. `boxpose_v29.npz` reproduced **byte-identical** (`63473c15f0cc`)
     a week after the original run.
  2. **Fingerprint the renderer input** (`geom_*`, `body_*`, `mesh_vert`,
     `qpos`, `mocap_pos/quat`, `cam_lookat`) — 21 arrays, max abs diff 0.0.
  3. Only then eyeball frames.
- **Corollary for QA**: never gate a change on "the mp4 hash is unchanged".
  A hash mismatch here is expected noise; a *numerical* mismatch is the signal.
- Note this does not affect the delivered files: the shipped v29 mp4 is the
  original artifact (`md5 79945c372a766fc8d42bf88ec897a291`), and the shipped
  `boxpose_v29.npz` reproduces exactly, so the delivered pair is self-consistent.

### Window render dies with `mjtVisFlag has no attribute 'mjVIS_LABEL'`
- **Cause**: MuJoCo >=3.10 (confirmed on 3.13.0) removed `mjVIS_LABEL` from
  `mjtVisFlag`. Old code sets `viewer.opt.flags[mj.mjtVisFlag.mjVIS_LABEL]`
  unconditionally, and it raises AFTER the whole render is computed — the
  traceback looks scary but nothing is wrong with your data.
- **Fix** (already in the script): look the flag up with
  `getattr(mj.mjtVisFlag, "mjVIS_LABEL", None)` and only set it if present.
  Never hard-code a version-specific enum name.

### The render looks like an OLD version (box not measured / wrong trajectory)
- **Cause**: `跑搬运渲染窗口.bat` / `跑搬运渲染-出视频.bat` originally passed
  ONLY `--robot_motion`, so the box came from the SYNTHESISED 3-phase
  trajectory, not the 6-DoF measurement. Easy to miss — the video still renders.
- **Fix** (already in the bats): auto-detect `data\<pklstem>_world_6dof.npz`
  (fallback `_world.npz`) and pass `--box_pose_world` + `--dump_box_pose`.
  Missing file → the bat prints a warning instead of silently degrading.
  Naming: pkl is `<prefix>_g1.pkl`, the world npz is `<prefix>_world_6dof.npz`.

### batch_pipeline.py says "no bag found for the fit step -- skipped"
- **Cause**: `record_rgbd.py` writes into `video/bag/rgbd_<timestamp>/x.bag`
  (TWO levels under `video/`), while the original `find_bag` only globbed
  `video/<prefix>.bag` and `video/*/<prefix>.bag`. The bag existed, the fit
  was skipped, and the clip stayed stuck on `need-fit` forever.
- **Fix** (already in the script): `video/**/<name>.bag` with
  `glob.glob(..., recursive=True)`.
- **Related trap**: a failed step used to print nothing at all, so a failed
  retarget was indistinguishable from a successful one. `run_step()` now
  prints `ok` or `FAILED rc=N (log: ...)` plus the log tail — always read it.

### My bags are on another drive (`<数据根>\rgbd_<ts>\realsense.bag`) — how do I batch them?

- **Do not copy ~20 GB into the repo.** Add a search root:
  `python scripts/batch_pipeline.py --bag-root <数据根> --run`. The prefix still
  comes from the `rgbd_<timestamp>` folder, so every downstream path
  (`video/<p>.mp4`, `data/<p>_*.npz`) lands in the repo as usual and the bag stays
  put. Equivalent to the env var `GMR_BAG_ROOTS` (`;`-separated), which is what
  `clip_naming.bag_roots()` reads; `--bag-root` may be given more than once.
- Check the discovery first by running the command **without** `--run` — the table
  plus the `bag -> canonical prefix` map shows whether every folder resolved.
- **The `mp4` step is the one that needs no GPU**, and it must be checked before
  the GVHMR gate: `--run` skips any clip reporting `need-GVHMR`, so if
  `blocker()` checked the GPU first, a fresh recording would never get converted
  and you would have to run `bag_to_mp4.py` by hand once per bag. With the
  ordering in place, one `--run` converts all of them and the clips correctly
  report `need-GVHMR(GPU)` on the next pass.
- A recording present BOTH in `video/` and in the external root is reported as
  `[!] 2 bags map to the same prefix`; `find_bag` prefers the shallowest path, so
  the repo copy wins. That is a real duplicate to delete, not a bug.

### GVHMR dies in `run_preprocess` → `Tracker()` with `pickle.UnpicklingError`

- **Symptom**: `demo.py` gets past `[Preprocess] Start!`, then the first thing it
  builds — `tracker = Tracker()` — raises
  `pickle.UnpicklingError: Weights only load failed` /
  `Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel`. Every clip in
  the batch dies the same way, so the run summary reads `ok (0)`.
- **Cause**: torch >= 2.6 flipped `torch.load`'s `weights_only` default to
  `True`, but ultralytics 8.2.42's `torch_safe_load` never passes it, so loading
  its own `yoloV3.pt` (which pickles a `DetectionModel` class) fails. Nothing to
  do with the footage, the script or the paths.
- **Fix**: add `weights_only=False` to the three calls that matter (the rest of
  the package's occurrences are in `models/sam`, `models/nas`, `engine/tuner`
  and are never reached here):
  | file | line | call |
  |---|---|---|
  | `ultralytics/nn/tasks.py` | 775 | `torch.load(file, map_location="cpu")` |
  | `ultralytics/nn/tasks.py` | 795 | same, in the `check_requirements` retry |
  | `ultralytics/utils/torch_utils.py` | 513 | `torch.load(f, map_location=torch.device("cpu"))` |
  ```bash
  $HOME/miniconda3/envs/gvhmr/bin/python - <<'PY'
  import pathlib
  sp = pathlib.Path.home()/"miniconda3/envs/gvhmr/lib/python3.10/site-packages/ultralytics"
  for rel, old, new in [
      ("nn/tasks.py", 'torch.load(file, map_location="cpu")',
                      'torch.load(file, map_location="cpu", weights_only=False)'),
      ("utils/torch_utils.py", 'torch.load(f, map_location=torch.device("cpu"))',
                               'torch.load(f, map_location=torch.device("cpu"), weights_only=False)'),
  ]:
      p = sp/rel; s = p.read_text(); n = s.count(old)
      p.write_text(s.replace(old, new)); print(rel, "patched", n)
  PY
  ```
  Prefer this over `pip install -U ultralytics` (an upgrade can move torch too,
  against GVHMR's pins). Do **not** use a `sitecustomize.py` that monkeypatches
  `torch.load` — it makes every python start in that env import torch.
- Verify the installed pair before guessing:
  `site-packages/ultralytics/__init__.py` → `__version__`,
  `site-packages/torch/version.py` → `__version__`.

### Batch GVHMR over a whole folder of mp4s (stage 2, GPU machine)

**There is a ready-made runner: `scripts/gvhmr_batch.sh`.**

```bash
bash gvhmr_batch.sh [mp4_dir] [out_root] [--fast] [--no-copy]
```

- No arguments? It auto-detects both the GVHMR checkout (`$GVHMR_REPO`, `$PWD`,
  `~/gvhmr`, …) and the mp4 folder: it scans 3 levels under `$HOME`, prunes
  `miniconda3`/`.cache`/`.conda`/`.nv`/`.git`/`node_modules`/`cuda_home`/
  `third_party`/`**outputs**` (GVHMR's own renders), then **picks the directory
  with the most `*.mp4`** — a real recording batch has 10+, GVHMR leftovers have
  1–2. Other candidates are listed; a tie asks you to choose.
- Skips clips whose `hmr4d_results.pt` already `torch.load`s (a truncated file
  from a killed run is detected, deleted and redone), keeps going after a
  failure, and copies results back as `data/<prefix>.pt`.
- `--fast` stops each job once the `.pt` has been size-stable for ~9 s, skipping
  the two preview renders `demo.py` does afterwards.
- **Interpreter**: it defaults to `~/miniconda3/envs/gvhmr/bin/python` (also
  `~/anaconda3/envs/gvhmr`, then `conda run -n gvhmr`), so `conda activate` is
  **not** required. That matters on this box: `.bashrc` has no `conda init`
  block, so `conda` is not even on PATH and `conda activate gvhmr` fails with
  `command not found`. The env is named `gvhmr`; the proven invocation is
  `/home/qzw/miniconda3/envs/gvhmr/bin/python -m tools.demo.demo --video=<mp4> -s`
  — note `-m tools.demo.demo` (cwd = repo root), which is what the history shows
  working. Override with `PYTHON=/path/to/python`. If `import torch` fails it
  lists every `~/miniconda3/envs/*/bin/python` and exits before doing any work.
- **Path gotcha**: the folder the user created was `9.22-mp4` (hyphen); Windows
  Explorer renders the tab title such that it reads like `9.22.mp4` (dot). Always
  confirm a WSL path over the UNC share
  (`\\wsl.localhost\Ubuntu\home\<user>\...`) before handing it to someone.

The manual equivalent, and why GVHMR's own folder script is the wrong tool:
The official entry point is `tools/demo/demo_folder.py -f <folder> -d <out_root> -s`.
It is a thin loop, and reading its source is worth the 30 seconds:

```python
mp4_paths = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.MP4")))
for mp4_path in mp4_paths:
    subprocess.run(["python", "tools/demo/demo.py", "--video", str(mp4_path), ...],
                   check=True)
```

Three consequences that bite on a 12-clip batch:

1. **Not recursive.** Only `*.mp4` sitting *directly* in `-f` is found — stage the
   videos flat in one folder first.
2. **One process per video.** The model and its checkpoints are re-loaded for every
   clip, so the load cost is paid N times. That is the feature you actually wanted
   a batch runner for.
3. **`check=True` aborts the whole batch** on the first failure — the remaining
   clips never run and there is no list of what was skipped.

So drive it yourself (skips finished work, survives failures, and writes the
result straight to the name the pipeline expects):

```bash
cd ~/gvhmr                                    # the GVHMR repo root
PY=$HOME/miniconda3/envs/gvhmr/bin/python     # no `conda activate` needed
mkdir -p ~/gvhmr_in ~/gvhmr_out
cp /mnt/c/Users/QZW/Desktop/<...>/GMR-master/video/20260922_*.mp4 ~/gvhmr_in/

for f in ~/gvhmr_in/*.mp4; do
  p=$(basename "$f" .mp4)                        # already the canonical prefix
  [ -f ~/gvhmr_out/$p/hmr4d_results.pt ] && { echo "skip $p"; continue; }
  echo "=== $p ==="
  "$PY" -m tools.demo.demo --video "$f" --output_root ~/gvhmr_out -s \
    || { echo "FAIL $p -- continuing"; continue; }
done

for d in ~/gvhmr_out/*/; do                      # copy the .pt back, renamed
  p=$(basename "$d")
  cp "$d/hmr4d_results.pt" "/mnt/c/Users/QZW/Desktop/<...>/GMR-master/data/$p.pt"
done
```

- `-s` (`--static_cam`) is **mandatory** for this rig. Without it `demo.py` runs
  SimpleVO (or DPVO with `--use_dpvo`) over a camera that never moves.
- Keep the videos on a **native, ASCII** path (`~/gvhmr_in`), not
  `/mnt/c/.../中文/...`: much faster I/O, and it dodges the non-ASCII `cv2` writer
  problems the Windows side already suffers from.
- **Do not pass `--f_mm`.** It would change GVHMR's internal focal and therefore
  the world scale, and every existing clip was produced without it. The scale error
  is already absorbed downstream by the Umeyama scale in `align_box_to_world.py`,
  which every clip goes through anyway — mixing the two halves the batch.
- Results land in `<output_root>/<video_stem>/hmr4d_results.pt`. Because the mp4s
  were exported under the canonical prefix, `<video_stem>` **is** the prefix:
  `cp` it to `data/<prefix>.pt` and the status table flips to the local steps.
- Only the `.pt` is needed downstream. `demo.py` saves it **before** rendering the
  two preview videos, and the preprocess caches (`bbx`, `vitpose`, `vit_features`)
  are reused on a re-run, so a re-run is cheap. If render time dominates, poll for
  the `.pt`, wait until its byte size is stable, then kill the process:
  ```bash
  "$PY" -m tools.demo.demo --video "$f" --output_root ~/gvhmr_out -s & pid=$!
  stab=0; last=-1
  while [ $stab -lt 2 ]; do
    s=$(stat -c%s ~/gvhmr_out/$p/hmr4d_results.pt 2>/dev/null || echo -1)
    if [ "$s" = "$last" ] && [ "$s" != "-1" ]; then stab=$((stab+1)); else stab=0; fi
    last=$s; sleep 3
  done
  kill $pid 2>/dev/null
  ```

### Bags named freely (`9.22.1.bag`, `9.22.2.bag`) — do they resolve?
- **Yes, by design.** The canonical prefix is the RECORDING FOLDER's timestamp
  (`video/bag/rgbd_20260922_181000/9.22.2.bag` → `20260922_181000`), never the
  bag's own file name. `scripts/clip_naming.py` walks: `configs/clips.json`
  override → `<timestamp>` stem → nearest ancestor `rgbd_<ts>`/`<ts>` folder →
  `metadata.json` `recorded_at` → else **unmatched, reported, never guessed**.
- The status table prints a `bag -> canonical prefix` map with a
  `[need data/<prefix>.pt]` marker, so the GVHMR output can be named correctly
  without guessing. An unmatched bag is listed under
  `bags with no timestamp in the name, folder or metadata.json`.
- Pin anything the folder cannot explain (a bag moved out of its folder, or a
  folder with no timestamp):
  `configs/clips.json` = `{"9.22.3.bag": "20260922_183000"}`.
- `batch_pipeline.py` chdirs to the repo root at startup, so a double-clicked
  `.bat` or a scheduled task cannot silently scan the wrong directory.

### Two prefixes for the same take (e.g. `9.22.1` AND `20260922_161727`)
- **Cause**: the take was exported/GVHMR'd under an ad-hoc name before the bag
  was renamed to the recording folder's timestamp. The status table then shows
  one row per artifact family and you can't tell which chain is the good one.
- **Fix**: rename the whole `data/` family to the canonical prefix
  (`data/9.22.1.pt` → `data/20260922_161727.pt`, likewise `_smplx.npz`,
  `_g1.pkl`, and the preview mp4). Record once with `<prefix> = the folder
  timestamp` and never look back.
- If the duplicate is a stray COPY of the bag, move it to
  `.workbuddy/_trash/<date>/...` rather than deleting it (reversible), after
  confirming the MD5 matches the canonical bag.


### `batch_pipeline.py --run` does nothing after I edited a pipeline script
- **Cause**: `--run` only executes steps that are MISSING or whose acceptance
  failed. A clip with `world OK` and an mp4 on disk is considered done, so
  re-running after editing `align_box_to_world.py` (or the renderer) is a
  no-op — and if the clip fails only QA, the `blocker(r)` filter skips it too.
- **Fix**: `--force fit|world|render|qa` re-runs that step and every later one
  regardless, and processes every non-GVHMR-blocked clip:
  `python scripts/batch_pipeline.py --run --only 20260922 --force world`.
- Note the render is gated on `world OK`; if a forced realign comes out BAD
  the render is skipped and the QA verdict you see is the OLD mp4's. Fix the
  alignment (or force `render` again after it passes) rather than trusting it.

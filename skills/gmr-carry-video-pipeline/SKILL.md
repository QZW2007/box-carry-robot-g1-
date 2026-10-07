---
name: gmr-carry-video-pipeline
description: Run the full GMR-master pipeline that turns a new RGB-D recording of a person carrying a box onto a stool into the final G1 robot render (colour+depth bag → GVHMR → retargeting → 6-DoF box fit → Sim3 alignment → constrained render). Use when re-rendering a new take, when a render shows clipping / box spin / jitter, or when asked how the box and stool poses are produced. Contains the environment rules for this Windows machine (conda python, PowerShell quirks, Chinese-path pitfalls), the exact command sequence, per-step acceptance numbers, and the QA script.
agent_created: true
---

# GMR carry-video pipeline (bag → final G1 render)

## How to read the paths in this document

Two different roots appear in the commands below, and mixing them up is the most
common way to "lose" a script:

- **Repo-relative** (`scripts/box_traj_from_depth.py`, `data/<prefix>_g1.pkl`,
  `video/<prefix>.bag`) — relative to `GMR-master/`. **`cd` there first**; most
  scripts anchor themselves to that root.
- **Skill-relative** (`scripts/probe_box_height.py`, `scripts/unwrap90.py`, …) —
  Bundled with **this** skill (in `scripts/` next to this file).

`troubleshooting.md` and older memory notes sometimes write the skill scripts as
`.workbuddy/<name>.py`. Those were their original home. **They are now bundled in
this skill's `scripts/` directory**, and all 25 of them are still present in
`GMR-master/.workbuddy/` too, so both spellings work. If a path in the prose does
not exist, look in the skill's `scripts/` folder before assuming it was deleted.

Purpose: reproduce the current best render (`video/<prefix>_g1_measured_box_6dof.mp4`)
from a new recording, with all the temporal-consistency and contact fixes baked
in. Written for the GMR-master repo on the lab Windows machine.

## Environment rules (read first — every one of these cost real time)

1. Run ALL python with `python`
   (absolute path). Other pythons lack numpy/torch/mujoco/pyrealsense2.
2. Bash tool is broken on this machine — use PowerShell. PowerShell does not
   echo stdout: end commands with `*> log.txt`, then Read the file. `*>`
   writes UTF-16; convert with `Get-Content log | Out-File -Encoding utf8 out.log`
   before reading, or read via python printing UTF-8.
3. Chinese paths: `cv2.imread`/`cv2.imwrite` silently fail → use
   `cv2.imencode + ndarray.tofile` (and `np.fromfile + cv2.imdecode` to read
   back). `cv2.VideoWriter`/`cv2.VideoCapture` are FINE (ffmpeg backend
   handles UTF-8 paths) — verified on both the fit preview and the final
   render under the Chinese project root.
   `MjModel.from_xml_path` fails → use
   `general_motion_retargeting.mj_compat.load_model`.
4. Compare pyrealsense2 enums with `==`, never `is`.
5. After every render, run the QA script (below) — it scans ALL frames for
   jumps with no mask, which is how the hidden 0.9 m teleport was found.

## Running on a NEW venue / box / camera setting (scene-agnostic operation)

The pipeline was tuned around one venue, and that shows up as literal numbers
scattered across scripts: a 0.30×0.30×0.45 box, an orange box, 1280×720,
`--roi-pad 280 --min-area 400`. When a new recording differs — different room or
time of day (lighting), different box colour or size, different resolution,
different camera height — the failure is typically **silent and far downstream**
("the fit is 0 %", "the mask is tracking a jumper"), with nothing pointing back
at the assumption. These affordances exist to make that explicit.

### Step 0 — always run the scene profile first (cheap, read-only)

```bash
python scripts/scene_profile.py --bag video/<prefix>.bag
```

It reads the bag and its `metadata.json` and prints, for every parameter the
pipeline needs, **the value, where it came from, and how much to trust it**
(`measured` / `derived` / `assumed` / `user-given`), plus warnings for each
assumption still in force. It also samples ONE frame and reports colour and
box-size *candidates*. It never writes anything and never runs the fit.

What it catches for you:
- **Auto-exposure left on** (`enable_auto_exposure: 1`) — depth and colour drift
  during the take. This is exactly what ruined 20260917's HSV calibration, and it
  is invisible in the rendered output.
- **A colour-only recording** (no depth stream) — blocks the whole depth route.
- **The resolution**, which then drives `--roi-pad` / `--min-area`.
- Whether an HSV range for this session exists in `configs/boxes.json`, or
  whether you are about to run on the built-in default.

### Colour: three ways, least to most manual

| how | when | what it does |
|---|---|---|
| `--hsv auto` | new room, unknown box colour | finds the largest **solid, box-like** blob on a sampled frame and measures its range with the *same* estimator as `calib_hsv.py` (circular-hue mean ± max(10, 3σ), S/V = median − 80 … 255). Prints **every** candidate so you can pick a different one. Falling back loudly if nothing credible is found. |
| `configs/boxes.json` | same venue, later takes | automatic — matched by clip prefix / `session`. This is what makes a re-shoot "just work". |
| `scripts/calib_hsv.py` | you want it exact | drag a box over the box face; prints the range and can `--save` it into `configs/boxes.json` for the whole session. |

Validated on the real material: on `20260922_161727` (`--hsv auto`) the tool
measured **`4,77,77:24,255,255`** versus the hand-calibrated
`4,75,79:24,255,255`, and the full fit then ran **IoU 0.911, depth 31 mm** with
no human input. On the 9.17 clip it proposes `5,78,69:25,255,255` against the
known-good `5,90,80:28,255,255`.

`auto` is a *seed*, not a replacement: if the mask looks wrong, run
`calib_hsv.py` and pass the printed string. A wrong range is the single most
common reason for a 0 % fit.

### Box size: the fit's only scale source

- **Best**: caliper the box and pass `--box-size W,D,H` (metres). The run now
  prints the value it will use so a typo is visible.
- **Fallback**: `--box-size auto` estimates the visible face from one frame,
  using the **measured depth** at the blob when available (otherwise it assumes
  a 3 m standoff and says so). Validated on `20260922_161727`: it estimated
  **0.282 × 0.435 m** against a true 0.30 × 0.45 m, and the fit still reached
  IoU 0.917.
- It is reported as a **lower bound** (foreshortening only shrinks a face) with
  its standoff, and the run says in the log that it is *not* a caliper
  measurement. A wrong size moves the box **and its orientation systematically**,
  not slightly — so verify before quoting any metric.

### Resolution and pixel-space gates

`--roi-pad` and `--min-area` are now **derived from the frame size** unless you
pass them explicitly, against the 1280×720 baseline (280 px / 400 px²):

| resolution | `--roi-pad` | `--min-area` |
|---|---|---|
| 640×480 | 140 | 133 |
| 1280×720 | 280 | 400 |
| 1920×1080 | 420 | 900 |

The old defaults were 140/120 (a 640×480 take) while the batch passed 280/400 (a
720p take) — handing either pair to the other resolution breaks the tracker. Both
values are printed with their provenance. Explicit flags always win.

### The depth gate is now range-aware

The depth residual is *sensor noise*, and stereo noise scales with distance
(~1 % of range), so a fixed millimetre threshold is really a fixed standoff in
disguise. Measured across all 14 clips: **Pearson r = 0.944** between the median
box standoff and the median residual, and the residual is **0.66–0.93 % of range
(mean 0.76 %)**. The gate is therefore
`residual <= max(40 mm, 1.1 % × standoff)`.

Why it matters: the old fixed 40 mm is "1 % of 4 m", so it **rejected good far
shots and would have accepted bad near ones** (at 2 m, 40 mm is 2 % of range —
twice the sensor floor). The clip it rejected, `20260922_165811` (5.25 m → 42.6 mm),
was in fact a perfectly good fit (IoU 0.897, valid 100 %, render QA PASS) and is
now delivered.

### What still is NOT automatic (be honest about the edges)

- `--stool_h` and `--box_half_h` must be given when the furniture changes; the
  rise check is graded against them.
- The **camera must stay static** for a take — every downstream stage assumes one
  rigid transform.
- `s` (Umeyama scale) is fitted per clip and varies widely (0.549–1.025 over the
  13 older clips). Never carry one clip's value to another.
- Lighting changes *within* a take (a cloud, someone switching a light) are not
  modelled; that is what locking exposure is for.

## Pipeline (bag → video)

Prefix convention: `<prefix>` = e.g. `20260917_212026`, used by every step.

| # | Step | Command | Accept |
|---|------|---------|--------|
| 1 | record | `record_rgbd.py --exposure <us>` (locks exposure!) | both streams present in metadata |
| 1b | pre-flight | `python scripts/probe_bag.py --bag video/<prefix>.bag` | streams 1280x720@30, `enable_auto_exposure: 0` in the session's `metadata.json` |
| 1c | naming | rename the bag to the recording folder's timestamp and put it at `video/<prefix>.bag` | prefix == folder timestamp (`rgbd_20260922_161727/` → `20260922_161727`); never keep ad-hoc names like `9.22.1.bag` — the batch runner and bats pair by timestamp prefix |
| 2 | export colour | `python scripts/bag_to_mp4.py --bag video/<x>.bag --out video/<prefix>.mp4` | frame count ≈ frameset count |
| 3 | human pose | GVHMR in WSL Ubuntu: `demo.py --video video/<prefix>.mp4 -s` — for a whole folder use `bash scripts/gvhmr_batch.sh <mp4_dir>` | `data/<prefix>.pt` |
| 4 | retarget | `跑新视频.bat <prefix>` | `_g1.pkl` + preview mp4 |
| 5 | box 6-DoF fit | `python scripts/box_traj_from_depth.py --bag <bag> --fit-cuboid --hsv <H> --roi-pad 280 --min-area 400 --out data/<prefix>_6dof.npz --preview video/<prefix>_6dof.mp4` | valid 3D ≥95%, IoU med ≥0.85, depth residual ≤40 mm (`TH_DEPTH_M = 0.040`; the 35 mm figure still quoted elsewhere is stale — the 20260922 batch is shot farther away and 170444 measured 38 mm with IoU 0.899) |

Step 5 runtime: ~10 min for a 494-frame 720p bag. stdout is block-buffered
under `*>`, so the log stays empty until the process exits — to watch
progress, check the growing file size of the preview mp4 (VideoWriter writes
it incrementally, roughly proportional to frames done).
| 6 | align | `python scripts/align_box_to_world.py --robot_motion data/<prefix>_g1.pkl --box_traj data/<prefix>_6dof.npz --box_pose data/<prefix>_6dof.npz --out data/<prefix>_world_6dof.npz --stool_h <h>` — BOTH `--box_traj` and `--box_pose` are required and point at the SAME 6dof npz (position vs orientation); passing only `--box_pose` silently falls back to the legacy default `data/box_traj_depth.npz` and dies with FileNotFoundError | stool-rise check: measured floor→stool centre lift vs stool height within ~10 mm |
| 7 | render | `python scripts/vis_robot_with_box.py --robot_motion data/<prefix>_g1.pkl --box_pose_world data/<prefix>_world_6dof.npz --record_video video/<prefix>_g1_measured_box_6dof.mp4 --dump_box_pose data/<prefix>_boxpose_used.npz` | QA script below |
| 8 | QA | `python .dsh/skills/gmr-carry-video-pipeline/scripts/check_render_quality.py` (auto-uses the newest `data/*_boxpose_used.npz`; pass `--npz` when several clips coexist). The skill moved from `.workbuddy/skills/` to `.dsh/skills/`; `batch_pipeline.py` auto-probes both. | see numbers below |
| 9 | timing sync | `python scripts/check_timing_sync.py --prefix <prefix>` | hop 1 lag <= 3 frames and corr >= 0.95 (pelvis/chest) / 0.90 (knee/lean) on all four channels — see "Timing sync" below |

### How the carry/ground split is decided (step 6)

#### ★ The rise check does NOT validate the scale — do not let it reassure you

`rise` is computed as `rise_w / s` where `rise_w = s·(R·Δx)_z`, so **`s` cancels
out of the ratio**. A clip can pass the ±12 mm rise gate while its Umeyama scale
is still drifting. Real example: **20260922_161727 passed rise at −7 mm while `s`
drifted from 0.758 all the way to 0.549** — the gate that was supposed to catch a
bad alignment was blind to it by construction.

So:
- The rise gate validates the **box's own depth measurement consistency** (floor
  vs stool), nothing more. It is a good check that the depth route found the box
  and that the parked phases are sane.
- To validate `s` you need an **independent second path**, not the rise. The
  available one is the double-link ratio: G1 foot→head in the XML rest pose
  (1.1979 m) ÷ human height (`1.66 + 0.1·β₀` = 1.6609 m) = **0.7212**, versus the
  Umeyama value **0.7155** — a 0.80 % agreement, and the two share no input
  (one comes from robot geometry + body-shape params through the IK, the other
  from camera-frame box poses ↔ robot wrist midpoints).
- `s` is **per-clip** and varies a lot across the repo's 13 clips
  (**0.549 … 1.025**, a 1.87× spread). Never reuse a previous clip's scale.

`carry_split()` now prefers a **motion split taken from the box itself**: the
box's camera-frame centre height and speed. Both are Sim3-free (world =
`s*(Rt @ x) + t`, and `Rt` changes neither norms nor the "up" direction), so
the split no longer depends on the fit and cannot chase it.

- `first_c` = the last frame *inside* the floor band, where "band" = parked
  floor height + `max(4 mm, 1.5σ)` of the 9-frame-median-filtered height
  (σ ≈ 3-5 mm, so the trigger is a few mm of real rise).
- `last_c` = the last frame whose filtered centre still moved > 8 mm/frame.
- It takes the **longest** contiguous excursion above the floor band, not the
  first: a parked box drifts a few mm (quantised depth), which opens short
  spurious runs early on (20260922: f34-f40 sat 6-13 mm high).
- Falls back to the old hand-distance gate when there is no 6-DoF pose or no
  clear lift, and prints which route it took.

Why this matters: the old hand-distance gate opened only when the robot's
hands came within `inlier_m` of the box CENTRE. A person grips the box by its
top corners, so on a fast grab the gate opened 10-45 frames AFTER the box
started rising. The renderer pins every non-carry frame to the rest height and
anchors the measured z onto that height at `first_c`, so a late `first_c` made
the box snap upward in one frame (20260922: 225 mm at f161→f162 — a hard QA
FAIL). The motion split puts `first_c` where the box has risen a few mm, so the
anchor meets the pinned height with a ~3-5 mm step. Verified: 20260922
225 mm → 5.0 mm, 20260917 11.7 mm → 7.5 mm; both QA PASS, rise 7 / 12 mm.

### When the robot reaches the box LATE: the grasp wait (step 7)

`first_c` (above) is when the **box** really left the floor in the video. The
**retargeted robot** does not necessarily arrive at the box on that frame — on
20260922 the box starts rising at f148 while the robot's hand midpoint is still
0.32 m short of the parked spot horizontally, and only comes level with it
around f166. Rendering the measured lift on the measured schedule therefore
showed the box rising and drifting toward a robot whose hands were visibly NOT
on it — the robot "squats down before it has reached the box".

`vis_robot_with_box.py` therefore *waits*: the box is held on its parked spot
until the hands have actually arrived, then handed to the rigid grip.

- `--grasp_reach` (default **0.24** m) — the horizontal hand-to-parked-box
  distance that counts as "the hands are at the box". `0` disables the wait and
  restores the old frame-count behaviour.
- `--grasp_dz` (default **0.12** m) — extra metres above the box TOP that still
  count as a grip (a hand passing above the box is not one).
- `--grasp_z_maxstep` (default **0.040** m) — cap on the box's per-frame z step
  through the delayed pickup.

The search runs from `first_c` **onward only**: the wait may delay a pickup the
robot cannot make in time, never advance one. When the hands are already at the
box on `first_c` the gate does not fire and nothing changes.

Two seams have to be closed, and both were real QA failures first:

1. **The handover frame.** With the wait, the box is on the floor at
   `carry_start` while the anchored measurement already has it ~0.15 m up (the
   person was lifting by then). Taking the measured z on that one frame popped
   the box **127 mm**. Ramping the measurement in over a short band instead made
   the lift hit **64 mm/frame** — also a FAIL. The shipped fix pins the floor
   height for that single frame (so the step *into* the carry is 0 and the
   parked-phase statistics are untouched) and then **rate-limits** the box's z
   over the carry. The box rejoins the target as soon as the target's own rate
   drops under the cap (the measured lift is ~35 mm/frame, so this only ever
   smooths the hand-over).
2. **The carry group's z profile.** CARRY has no position-step acceptance in
   the QA script, but the **unmasked** > 50 mm scan does apply and only forgives
   a big step when `carry`/`cov` change state. That is why the cap must stay
   under 50 mm: a 60 mm z step mid-carry reads as a TELEPORT. 40 mm leaves room
   for the xy contribution (observed max 44 mm).

Verified on 20260922: `carry 158..240` (wait = 10 frames), floor parked pos
step max 2.7 mm, CARRY pos step max 44 mm, jump scan **clean**, QA PASS. The
horizontal behaviour is the important part: the box barely moves in xy —
the **hands** travel the remaining 0.21 m to it — so the pickup reads as the
robot reaching the box rather than the box flying into the hands.

The robot's early deep hip-fold — which *looks* like it starts ~20 frames before
the person does — was measured and is **NOT a timing error**. See "Timing sync:
measure it, never eyeball it" below. The impression comes from body proportions,
a 5-15° surplus knee bend, and the two cameras' very different viewpoints.
**Do not time-shift the dof trajectory**: the retarget is already faithful to
within 0-1 frame, so shifting would break a correct relationship (and foot
contact with it).



## Timing sync: measure it, never eyeball it (step 9)

The single most expensive wrong conclusion in this pipeline was "the robot bends
down ~20 frames before the real person does". It was reached by putting the
render next to the source video and looking — and it is **wrong**. Verify with:

```bash
python scripts/check_timing_sync.py --prefix <prefix>   # or --all
```

It runs three INDEPENDENT hops and cross-correlates each:

1. **SMPL-X human -> robot motion (the retarget).** Runs the `smplx` body model
   on `<p>_smplx.npz` and the G1 forward kinematics on `<p>_g1.pkl`, then
   correlates four channels: pelvis height, chest height, knee flexion, torso
   lean, over the pickup window (pelvis minimum +/- 60 frames — walking dominates
   the rest and inflates the score).
2. **Source video human -> SMPL-X (GVHMR).** The RealSense is static, so
   median-frame background subtraction recovers the real person's silhouette and
   its head-top row is compared with the SMPL-X head height onset.
3. **Video frame axis -> bag/box axis.** The orange box is HSV-segmented; its rise
   frame is compared with the depth-measured carry start + `frame_offset`. This
   validates the frame mapping everything else depends on.

Measured (2026-09-22): hop 1 gives **0-1 frame lag, corr >= 0.977 on all four
channels, on both clips**. Hop 2 and 3 on 20260922: **+2** and **+1** frames.
Conclusion: **the retarget is temporally faithful. Do not time-warp it.**

⚠ **Hop 3 (the mapping hop) is currently OUT OF TOLERANCE on most clips — and
that is a real, unresolved observation, not a false alarm.** Re-measured
2026-10-06 on every clip with a `data/<p>_timing_sync.json`: hop 3 passes only
on **20260922_161727** (delta +1). The rest fail by **+6 … +15** frames:

| clip | frame_offset | carry start (pkl) | box rise in video | predicted | delta |
|---|---|---|---|---|---|
| 20260922_161727 | +2 | 148 | 151 | 150 | **+1  OK** |
| 20260922_165230 | −6 | 114 | 114 | 108 | +6 |
| 20260922_165945 | −6 | 145 | 154 | 139 | **+15** |
| 20260917_212026 | −11 | 188 | 185 | 177 | +8 |

(All four rows are read straight from the `data/<p>_timing_sync.json` files
already on disk — this table just collects them in one place for the first time.
Reproduce per clip with `python scripts/check_timing_sync.py --prefix <p>`.)

Two things this does and does not mean:

- **It does not invalidate the delivered clips.** Hop 3 is evidence-only; hop 1
  is the gate, and hop 1 is excellent on all of them (lag 0-1, corr ≥0.979 on
  four channels). The frames themselves line up — `mp4`, `g1.pkl` and
  `world_6dof.npz` all have identical length on these clips (verified), so this
  is not an off-by-N frame-index bug.
- **It does mean the `frame_offset` is not a pure frame index for every clip.**
  A consistent early `box_rise` vs `carry_start + offset` (the box is measured as
  rising LATER in the video than the depth split predicts, by a clip-dependent
  6-15 frames) is what you would expect if the depth-derived carry start comes
  from the box's *centre height*, while the video hop detects the box's
  *silhouette top/mean row* — the two features cross their thresholds at
  different moments. The discrepancy is therefore best read as a **definition
  mismatch between the two "rise" estimators**, possibly plus a real
  exposure/frame-drop effect on 20260917 (which was recorded with
  auto-exposure ON).

**Do not "fix" this by adjusting `frame_offset`** — `frame_offset` is fitted
from the body-motion cross-correlation and validated by hop 1; bending it to
make hop 3 pass would break the relationship hop 1 proves. **Do not gate on
hop 3.** If you need the video↔bag axis trusted for a new clip, check it by
hand with `contact_sheet.py` on the same frame indices from the mp4 and the
render, which is the method that actually settled the timing question before.

Also note the tooling that produced the table above: `check_timing_sync.py` no
longer hard-codes a `frame_offset` of 2 / a pickup frame of 148 when a clip has
no `world_6dof.npz` (that made it silently reuse 20260922_161727's numbers — the
one clip whose hop 3 passes). It now says why it skipped the hop instead.

The gate is hop 1 only. Hops 2-3 are reported as *evidence*, because a flaky
source video (unlocked auto-exposure, dropped frames — 20260917's hop 3 is off by
+8 and its hop 2 is therefore marked `INCONCLUSIVE`) must not fail a clip whose
motion is provably in sync.

Two model quirks you need to measure this at all:

- `head_link` in this G1 XML is a **dummy frame level with the pelvis**; the real
  head is `head_mocap`.
- `torso_link` sits only **44 mm above the pelvis**, so leaning it is
  noise-dominated and will give you the opposite answer. Use the **shoulder
  midpoint** — that lever is 0.291 m on the human side and 0.291 m on the robot
  side, so the two are finally comparable.

`batch_pipeline.py` carries a `sync` column and a `need-timing-sync` /
`sync-FAIL` blocker; `--force sync` re-runs it. Results land in
`data/<prefix>_timing_sync.json`.

If you want the robot to *look* closer to the person, the knob is **amplitude,
not time**: pull the grasp-wait box lift back toward f148-150 (the person lifts
at ~f150) and/or down-weight the surplus 5-15 deg of knee flexion in the IK.
Deliberately making the robot squat slower than the person is a **different**
feature (a retarget-side time warp — requires re-running the IK and handling foot
slip), not a fix for a bug.

- `scripts/probe_box_height.py <data/<prefix>_6dof.npz> [frame_offset]` —
  prints the filtered box height, the parked-phase σ, the frame a range of
  thresholds would pick, and the height trace around the lift. Use it when
  `first_c` looks wrong; `GMR_MOTION_DEBUG=1` on the align command dumps the
  trace from inside the real run.
- `scripts/contact_sheet.py <video.mp4> <out.png> f1,f2,...` — tiles chosen
  frames into one PNG (cv2.imencode + tofile, so it survives the Chinese
  project path). Feed it BOTH the source and the rendered mp4 at the same
  indices to separate a data problem from a render problem.
- `scripts/unwrap90.py <world_6dof.npz>` — mod-90 unwrap of the fitted box
  yaw: the decisive test for whether a box "turn" is real. Use before choosing
  `--box_sym_snap`.
- `scripts/raw_steps.py <world_6dof.npz>` — symmetric-aware per-frame angular
  step over the carry (median/p95/max) and a suggested
  `--quat_max_step_deg`. Run it BEFORE fixing the slew cap: a default 3.5/4.5
  silently clips real rotation and makes the box lag the body.
- `scripts/verify_timing.py <g1.pkl> <world_6dof.npz> <boxpose_used.npz>` — the
  halfway metric: 25/50/75 % crossing frames of the BODY / RAW box / REN box
  yaw, for the OUT and BACK halves. Tells you whether the rendered box timing
  is faithful and, if not, by how many frames and in which half. Run it after
  every change to the orientation filters or the slew cap.

Or just double-click `跑搬运渲染-出视频.bat` (headless, writes the mp4) or
`跑搬运渲染窗口.bat` (interactive window). Both auto-detect
`data/<pkl stem>_world_6dof.npz` and pass `--box_pose_world` for you; if that
file is missing they warn instead of silently falling back to the synthesised
trajectory. Drag a different `_g1.pkl` onto the bat to switch clips.

`--hsv`: ORANGE box `5,90,80:28,255,255`; BLUE box default
`100,70,40:130,255,255`. Wrong HSV shows as the mask tracking clothing —
valid% collapses immediately. Non-720p resolutions: scale `--roi-pad` and
`--min-area` proportionally.

## Never guess HSV again: calib_hsv.py

`python scripts/calib_hsv.py --bag video/<x>.bag` extracts one frame
(middle by default, `--frame N` / `--image` override), you drag a rectangle
over ONLY the box face, ENTER, and it prints a measured `--hsv` string
(circular-hue mean ± max(10, 3σ); S/V floors = median − 80; ceilings 255)
plus a 3-panel preview (original | mask | kept pixels). It warns when
coverage is >40% (too wide) or <0.2% (too tight) and when the S floor is
below 60. `--save --name <colour> --prefix <prefix>` merges the range into
`configs/boxes.json`, which `batch_pipeline.py` then picks up automatically
(name substring or exact `prefix` match). Headless: `--roi x,y,w,h
--no-show` writes the preview to `configs/calib_preview.png` instead of
opening windows. Validated on the 9.17 orange bag: measured
`5,57,123:25,255,255` from a tiny ROI, full fit re-run confirmed the same
acceptance numbers as the hand-tuned range.

**Calibrate once per venue, not once per session.** Same box, a different
room gives a different range: the 9.22 venue (bright hall, mint-tinted
walls) measured `4,75,79:24,255,255` where the 9.17 room gave
`5,57,123:25,255,255`. The tool's `--frame` default is the MIDDLE frame —
pass `--frame N` pointing at a moment where the box is parked and
unoccluded, otherwise the ROI lands on whatever is at the frame centre.

## Batch mode: scripts/batch_pipeline.py

For converting many bags, drive everything from one command instead of
typing per-clip commands:

```
python scripts/batch_pipeline.py                  # status table: where each clip is stuck
python scripts/batch_pipeline.py --run            # execute the missing steps (GVHMR excepted)
python scripts/batch_pipeline.py --run --only 20260917_212026
python scripts/batch_pipeline.py --run --only 20260922 --force world   # re-run world+render+qa
python scripts/batch_pipeline.py --csv status.csv

# recordings that live OUTSIDE the repo (e.g. <数据根>\rgbd_<ts>\realsense.bag):
python scripts/batch_pipeline.py --bag-root <数据根>                  # just show the table
python scripts/batch_pipeline.py --bag-root <数据根> --run --steps mp4  # convert only
python scripts/batch_pipeline.py --bag-root <数据根> --run            # then everything local
```

`--steps S1,S2` restricts which steps run (`mp4|retarget|fit|world|render|sync`);
omit it to run everything. Use it when you want one stage — `--steps mp4` turns a
directory of bags into mp4s and stops, without also kicking off the ~minutes-per-
clip 6-DoF fit and world alignment.

**Bags outside the repo — nothing is copied.** `--bag-root DIR` (repeatable) adds a
search root; the prefix still comes from the `rgbd_<timestamp>` folder, so a whole
drive of recordings can be batched in place. It is equivalent to setting the env
var `GMR_BAG_ROOTS` (`;`-separated), which is what `clip_naming.bag_roots()`
reads. The only cost is that the ~1.6 GB bags stay where they are, which is the
point.

**`mp4` runs first and is not GPU-gated.** `bag_to_mp4.py` needs no GPU, and every
later step reads `video/<prefix>.mp4`, so `blocker()` checks `need-mp4` *before*
`need-GVHMR(GPU)`. Otherwise `--run` would skip a fresh recording entirely (it
skips anything reporting `need-GVHMR`) and you would have to convert by hand. The
table still shows `pt=-` and the bag map still prints
`[need data/<prefix>.pt]`, so the GPU trip is not hidden — it just is not the
next local step.

**Stage 3 (GVHMR) has its own batch runner:** `scripts/gvhmr_batch.sh`, to be
run on the GPU box. `bash gvhmr_batch.sh <mp4_dir> [out_root] [--fast] [--no-copy]`
— it auto-detects the GVHMR checkout (`$GVHMR_REPO`, `$PWD`, `~/gvhmr`, …),
skips clips whose `hmr4d_results.pt` already loads, survives a failing clip
instead of aborting, optionally stops each job the moment the `.pt` is written
(`--fast`, skipping the two preview renders that `demo.py` does afterwards),
and copies the results back as `data/<prefix>.pt`. Do **not** use GVHMR's own
`tools/demo/demo_folder.py` for a real batch — it is a thin `for` loop over
`folder.glob("*.mp4")` with `subprocess.run(..., check=True)`, so it is
non-recursive, re-loads the checkpoints for every clip, and aborts the whole
batch on the first failure (details in `references/troubleshooting.md`).

- Prefixes are collected from existing artifacts (`data/*.pt`, `*_g1.pkl`,
  `*_6dof.npz`, `*_world_6dof.npz`, `*_boxpose_used.npz`, timestamped bags) —
  no registration.
- Acceptance is ENFORCED between steps (fit thresholds, rise ±12 mm, QA
  PASS): a clip never advances on bad metrics.
- Failure isolation: one broken clip is only flagged, others continue; per-
  step logs land in `.workbuddy/batch_logs/<prefix>_<step>.log`. Every step
  prints `ok` or `FAILED rc=N` plus the tail of its log — a step is never
  silently skipped.
- `--force mp4|fit|world|render|qa|sync`: re-run that step AND every later one even
  though its output already exists. Use it after editing any pipeline script
  — without it `--run` sees `world OK` / the mp4 on disk and does nothing.
  It also lifts the "only blocked clips" filter, so a clip that merely FAILs
  QA gets re-processed.
- HSV chain: `--hsv` arg → `configs/boxes.json` → orange default.
- **Prefix = the RECORDING FOLDER's timestamp, never the bag's file name.**
  `record_rgbd.py` makes `video/bag/rgbd_<YYYYmmdd_HHMMSS>/` per recording and
  the `.bag`/`.db3` inside is free-form, so you may name them anything
  (`9.22.1.bag`, `9.22.2.bag`, `realsense.db3`). `scripts/clip_naming.py`
  resolves each bag in this order: explicit `configs/clips.json` override →
  file stem if it is `<timestamp>` (+suffix) → nearest ancestor `rgbd_<ts>` /
  `<ts>` directory → `metadata.json` `recorded_at` → otherwise **reported as
  unmatched, never guessed**. The status table always prints the resulting
  `bag -> canonical prefix` map, so you know exactly which
  `data/<prefix>.pt` to produce on the GPU machine.
  Pin anything the folder cannot explain (e.g. a bag moved out of its folder):
  `{"9.22.3.bag": "20260922_183000"}` in `configs/clips.json`.
- Legacy `data/<bag_ts>_unitree_g1.pkl` prefixes still resolve (the `_unitree`
  suffix is stripped).
- `batch_pipeline.py` anchors itself to the repo root at startup, so a
  double-clicked `.bat` or a scheduled task cannot scan the wrong directory.
- Regression tests for the naming scheme (run them after touching
  `clip_naming.py` or `collect_prefixes`):
  `python .workbuddy/test_clip_naming.py` (unit, 7 cases) and
  `python .workbuddy/test_batch_naming_e2e.py` (builds a fake
  `video/bag/rgbd_<ts>/9.22.2.bag`, asserts the status table resolves the
  FOLDER timestamp and names the `.pt` to produce, then cleans up).
- `need-GVHMR(GPU)` clips (no `.pt`) must go to the GPU machine first; their
  `mp4` is still produced locally first (see above). Everything else
  (need-mp4 / need-retarget / need-fit / need-align / need-render / qa-FAIL /
  sync-FAIL) is handled by `--run` locally.

## QA acceptance (run on the dumped `boxpose_used.npz`)

- FLOOR/STOOL parked: rot step p90 ≈ 0.000 deg (frozen), pos step ≈ 0 mm.
- CARRY: rot step med ≲0.6 deg, max ≤3.5 deg (slew cap) **when the source
  clip shows no box turn**. If the measurement shows a REAL, SMOOTH ramped
  yaw excursion (the box is being turned in the hands), the honest cap is the
  renderer's own `--quat_max_step_deg`, not 3.5; pass it as
  `--carry_rot_max <cap>` to `check_render_quality.py`. The decisive test is
  the **mod-90 unwrap** of `box_R_world` (`.workbuddy/unwrap90.py`): a real
  turn is a smooth multi-frame ramp (4-9 deg/frame, one direction for many
  frames); noise is an isolated single-frame jump of 20-80 deg that
  immediately returns. Do NOT read the raw amplitude — a fixed-anchor snap
  hides the ramp, and an euler/branch wrap can inflate it. If a real ramp is
  present you MUST render with `--box_sym_snap continuity` or the turn is
  folded away; see `references/troubleshooting.md` "the box only turns once".
  ★ **AND raise `--quat_max_step_deg` above the real turn rate, or the slew
  limiter stretches the turn and the box visibly LAGS the body** (the "箱子转
  的相对于机器人有一些延迟" symptom). The limiter is a HARD clip with `prev`
  feedback, so once it clips it keeps clipping until it catches up: a genuine
  30 deg/frame burst becomes a 20-frame 4.5 deg/frame ramp. Size the cap from
  the measurement's own step statistics (symmetric-aware 3-D step, NOT yaw
  only): `.workbuddy/raw_steps.py` prints median/p95/max over the carry — on
  165945 that is med 4.7 / p95 25.9 / max 38.5 deg/frame, so the old 4.5 cap
  clipped the majority of the real rotation. A cap of 20 (≈p95, above the real
  peak) clips NOTHING on 165945 (log: "0 frames clipped") yet still catches
  degenerate branch flips; verify with the log line and with the QA
  `--carry_rot_max 20`. The lag is asymmetric in general — the OUT half is
  short and passes, the BACK half is where the clip bites — so always measure
  the BACK half's 25/50/75 % crossing frames (see references).
  pos max ≲45 mm (the genuine lift; 20260922 44 mm with the grasp wait,
  20260917 40 mm).
- Top-jump list: no jump >50 mm anywhere except the lift frames; no
  teleport at the measurement start/end boundaries.
- Grip geometry (rebuild FK if needed): |box centre − hand midpoint| ≈ slack
  (50 mm); box centre ≥0.23 m from the spine capsule; last frames: box xy at
  the stool centre, z at stool top.

## Rendering-side constraint stack (what step 7 does automatically)

In order, inside the `--box_pose_world` branch: **GRASP WAIT** (hold the box on
its parked spot until the retargeted hands have actually arrived at it, then a
rate-limited z hand-over — see above) → **symmetry-branch snap** (square box:
8-element group; `--box_sym_snap anchor` = snap every frame to the rep closest
to a FIXED floor-median anchor — removes spurious wrong-branch drift but FOLDS
AWAY a genuine 90° box turn, since a square box is identical modulo 90°;
`--box_sym_snap continuity` = chain the branch from the previous frame, seeded
by the anchor — follows a real turn while still killing isolated per-frame
branch flips. Use `continuity` whenever the source clips shows the box being
turned in the hands; the "real turn capped at ~67°" symptom is this snap) →
rotvec unwrap → median(9)
+ MA(9/15) filters → z two-anchor bias (floor/stool) → carry GRIP CLAMP
(box centre = hand midpoint + ≤`--carry_grip_slack` (default 0.05) low-passed
drift; ramps 25 frames in / 12 out — raise the slack to ~0.45 when the
retargeted grip collapses narrower than the box and the hands end up INSIDE
it; the box then follows the measurement and the spine keep-out still
applies). With a large slack the box stays on the measurement while the
retargeted hands WANDER around it in the turn/walk (one arm detaches, a
hand ends up half-buried in the camera-facing face) — ramp the slack back
down once the box is airborne: `--carry_grip_slack_walk <s>
--carry_grip_slack_walk_win F0 F1` (smoothstep; cover the whole turn — the
hand drift starts with the pivot, a window that starts mid-pivot leaves the
detach visible).

  ★ **The shipped value is NOT the value in the v19/v20 narrative.** The v19/v20
  logs read `walk slack 60 mm ramped over f172..192` (``0.06`` / ``172 192``), and
  earlier drafts of this file quoted those. But **every run script from v23
  through the delivered v29 uses `0.02` / `150 165`**, and the delivered log
  confirms it: ``.workbuddy/render_v29.log`` → "walk slack **20 mm** ramped over
  **f150..165**". The tightening happened between v20 and v23 and is recorded in
  no log — see `.workbuddy/run_v23.ps1`. **When reproducing the delivered clip,
  copy `run_v29.ps1` verbatim rather than reading parameters out of this prose.**
  Both values are legitimate per-clip tunings; the point is that this knob is
  per-clip and the authoritative record is the run script. The left-arm mirror can also be ended early
(`--mirror_left_end F`) so the raw re-gripping retarget takes over during
the turn; note the mirror is NOT what spreads the hands — the raw retarget
spreads them just as wide (hands ~520 mm apart vs the 300 mm box while the
body pivots ~60 deg around the yaw-anchored box), so during the pivot both
hands float ~10-15 cm off the edges no matter which box-side lever you
pull (v13-accepted baseline; only arm-level IK could do better)
→ body keep-out (spine capsule, 0.23 m) → stool capture (z pinned at
stool top, xy slid onto the measured resting spot over 15 frames and HELD) →
parked freeze (floor/stool medians, extended over uncovered frames) → slew
limit `--quat_max_step_deg` (default 3.5 — RAISE it whenever the measurement
has a real turn; see the QA section: a cap below the real rate stretches the
turn and the box visibly lags the body). In the PLACE phase two waist levers
trade off against
each other on the SAME dof (14 = waist pitch, POSITIVE = fold forward):
`--place_waist_lift_deg D` SUBTRACTS up to D deg from the carry end on (the
G1 stays folded ~30 deg after the box lands and the helmet grazes the box
top-back corner; the real person is taller so needs less fold) — but that is
exactly the fold that carries the hands DOWN the box, so a large lift stands
the torso up and pulls the hands off it (on 165945 the raw right hand ended
up 84 mm INSIDE the box's front face). `--place_bow_deg D --place_bow_win F0
F1` applies a FLOOR (max, after the lift) so the fold never drops below D
deg over the window — use the source video as the arbiter (REAL 165945 f294-
311 folds right over the box, both hands on the upper sides) and A/B the
value: on 165945 bow 12 left the right hand on the front face (helmet clean),
bow 15 lifted it to the box's right edge (helmet mesh 22 mm in — adopted),
bow 18 seated it on the right SIDE face but pushed the helmet 35 mm into the
box top (visible; head_mocap is a fixed torso child, NOT controllable, and
raising --place_root_clear_m does not help because the folded head reaches
forward-down, not back). Measure the hands with the RENDERED
`--hand_diag_npz` (`center_box`/`depth_mm` = true geometry), NOT with
`.workbuddy/hd_sgap.py` — the latter rebuilds the arm chain itself and
ignores bow. The raw-vs-mirrored right arm must be A/B'd too: on 165945 the
place-phase right-arm MIRROR (not `--no_mirror_right_place`) is what keeps
the right hand off the front face. Arm dofs are despiked (rolling-median clamp 61/12° +
median 51), the left arm is mirrored from the right during carry
(`--no_arm_despike`, `--no_mirror_left`), and — when the retargeted grip
collapses — both shoulders are abducted outward over the collapse window
(`--carry_abduct_deg 25 --carry_abduct_win F0 F1`: left dof16 += deg,
right dof23 −= deg, 12-frame smoothstep ramps; swings the hands to the
box SIDES at the top-edge height the human uses) and after the carry the
RIGHT arm is mirrored from the left (`--no_mirror_right_place`), because
the raw right hand dives into the placed box while the raw left lands on
its edge. Hand-mesh cosmetics on top: `--hand_roll_deg D` rolls the rigid
rubber-hand shells about the forearm axis — this is the **palm-facing**
parameter, so it decides whether the palm or the BACK of the hand ends up on
the box. 85 was tuned on 0911 to lift the 48 mm-thumb clear of the box side,
but it turns the PALM OUTWARD; on 165945 an A/B probe grid
(`.workbuddy/probe_wristroll.py`) showed 0 is correct: palm on the box,
fingers wrapping the edge, only +5-20 mm fingertip depth hidden against
the pressed face. **The default is now 0.0** (it used to be 85, and because
`batch_pipeline.py` passes no `--hand_roll_deg`, that stale default silently
gave every batch clip the palm-outward hand — 13 of the 14 delivered clips).
Per-hand overrides `--hand_roll_left_deg` / `--hand_roll_right_deg` exist
because the mirror-image shells want rolls 26-37 deg apart; the renderer
applies `+D` to the left hand and `−D` to the right. A wrongly-rolled hand
reads as "the wrist faces the
wrong way" — check with the probe, not by eyeballing one render),
`--hand_shrink` shortens the shell along the finger axis, plus
`--hand_diag`
(+ optional `--hand_diag_npz out.npz` per-frame dump: penetration depth of
every hand/wrist mesh, hand centres in the box frame, MTV — the visual
geoms register no MuJoCo contacts) and `--hand_push_out_mm N` (visually
shove the rubber-hand geoms out of the box along that direction, clamped +
zero-phase-EMA smoothed; visual only, trajectories untouched; it is a
depth-REDUCER, not a true exit — treat as a backstop, fix the arm pose
instead).

Symptoms and their fixes live in `references/troubleshooting.md` — load it
when a render looks wrong. It also covers two script-level traps that look
like data bugs: the `mjVIS_LABEL` AttributeError (MuJoCo >=3.10 removed that
enum; the render is already finished when it fires) and the bats silently
using the synthesised box trajectory when `_world_6dof.npz` is absent.

## Recording requirements that decide success (see 重拍指南.md / RealSense深度实测拍摄流程.md)

Exposure LOCKED; five-phase structure (empty scene → box parked ≥5 s → walk
up and grasp → carry → place and hold ≥3 s); camera near-level (pitch ≤15°);
box front+top faces visible throughout; flat floor visible at the frame
bottom; measure the box's three edges (the fit's only scale source).

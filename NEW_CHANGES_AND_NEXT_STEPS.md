# New changes and next steps

Session log for the camera-calibration work (2026-07-23): what changed in the repo, why,
and what's left to actually run on hardware. See [CLAUDE.md](CLAUDE.md) for the
architecture these pieces plug into and [README.md](README.md) for the project overview.

## Why this session happened

Merge Path steps 1-2 (camera intrinsics, hand-eye calibration) were still placeholders —
`data/camera_intrinsics.json` and `data/hand_eye.json` don't exist yet, so
`PixelToWorldCalibrator` silently falls back to a guessed focal length and an identity
hand-eye transform (camera == gripper origin, no rotation). The pipeline runs and returns
plausible-looking `PickTarget`s, but the numbers are meaningless — the dangerous kind of
broken, because nothing fails loudly. This session built and fixed the tooling to replace
both with real measurements; it does **not** itself contain a completed calibration run
(that needs the physical arm + camera + MATLAB server — see "Next steps" below).

## What changed

### Environment fix: `opencv-python` version pin

`requirements.txt` now pins `opencv-python>=4.8,<5.0`. The unpinned dependency was
resolving to `5.0.0.93`, whose wheel is **missing the `cv2.calibrateHandEye` Python
binding entirely** — the `CALIB_HAND_EYE_*` enum constants exist, the function itself
does not. This silently broke the pre-existing `scripts/calibrate_hand_eye.py`
independent of anything else in this session. Confirmed `4.10.0.84` has both
`calibrateHandEye` and the modern ChArUco API (`CharucoDetector`, `matchImagePoints`) this
project already depends on. Re-check this bound if a newer 5.x wheel restores the binding.

If you recreate `.venv` from scratch, `pip install -r requirements.txt` now gets the right
version automatically — no manual step needed.

### New: `src/vision_pipeline/calibration/hand_eye.py`

Shared hand-eye sampling + solve logic, used by both the terminal script and the web
dashboard's calibration panel so the math exists exactly once.

- `detect_board_poses(frame_gray, intrinsics, detectors)` — per-board `solvePnP`, built on
  `charuco.detect` (not `detector.detectBoard` directly) so a frame whose corner/ID counts
  disagree is dropped rather than silently mispaired.
- `HandEyeAccumulator` — buckets `(T_base_gripper, T_cam_board)` samples **per board**
  (the workspace tiles 6 distinct ChArUco boards, so one frame can see several) and solves
  each board independently with `cv2.calibrateHandEye`. Two boards converging on the same
  camera offset is a stronger trust signal than the existing TSAI-vs-PARK check alone,
  since TSAI/PARK share their input data and can't catch a systematically bad sample set.
- `select_best` / `cross_board_agreement_mm` — picks the most-sampled board to save, and
  reports the worst pairwise disagreement between boards.
- Tested fully synthetically in `tests/test_hand_eye.py` (9 tests): plants a known
  `T_gripper_camera`, generates the samples a real session would record (varied rotation
  *and* translation — `calibrateHandEye` needs rotation diversity to be well-conditioned),
  and confirms the solve recovers it to sub-millimetre precision.

### Fixed: `scripts/calibrate_hand_eye.py`

Rewritten. The previous version had two silent-wrong-answer defects and one design choice
that would have failed most of its own sweep:

1. **Only ever saw board 1.** It built a `CharucoBoard` with no explicit marker-ID list,
   defaulting to IDs 0-29 — invisible to boards 2-6. Now uses
   `charuco.build_detectors()` / `charuco.detect()`, matching the intrinsics script.
2. **Gripper pose round-tripped through `Pose` (RPY).** `geometry.transform_to_pose`
   forces `roll=0` near pitch = ±90°, and a top-down tool orientation sits close to that
   gimbal-lock singularity — exactly the rotation error that poisons `calibrateHandEye`.
   Now takes the FK 4x4 straight from `MatlabIKClient.request_fk`, never through `Pose`.
3. **Cartesian IK sweep → joint-space jogs.** The arm works close in (tip x ≈ 75 mm),
   where small Cartesian moves demand large J1/J5 swings that `SERVO_MAX_MOVE_DELTA_TICKS`
   would refuse (see CLAUDE.md's "back probe" note). Jogging J4/J5 directly by raw ticks
   (same primitive as `jog_joint.py`) also gives far better **wrist rotation diversity**,
   which is what the solve actually needs — translation-only sampling leaves rotation
   underconstrained even when every move succeeds.
4. Samples now bucket per board and solve independently (see `HandEyeAccumulator` above).

New controls: digit keys `1`-`5` select the active joint, `a`/`d` jog it, `[`/`]` change
step size, `r` records a sample from every board currently visible, `c` computes & saves.

### New: web dashboard calibration panel

`scripts/run_arm_ui.py --hardware --calibrate` now connects a `MatlabIKClient` and enables
a hand-eye capture panel alongside the existing joint-jog dashboard — the same
`calibration/hand_eye.py` logic as the terminal script, driven by clicking instead of
keyboard shortcuts:

- **`src/vision_pipeline/webui/camera_stream.py`** — the overlay is now pluggable (a
  `frame -> frame` callable) instead of hardcoded to brick detection. `--overlay` behavior
  is unchanged; calibration mode passes a ChArUco-corner overlay instead.
- **`src/vision_pipeline/webui/app.py`** — `create_app` takes an optional `ik_client`.
  Without it, four new routes (`/api/calib/detect|sample|samples|solve`) return **501**
  (not 404 — the routes exist, calibration mode just isn't on), so the default dashboard
  still needs no MATLAB server.
- **`templates/index.html`** — a calibration panel (sample counts per board, Record
  Sample, Solve & Save) that only renders when calibration mode is on.
- Covered by 6 new tests in `tests/test_webui.py`, including the 501-without-`ik_client`
  path and a full sample→solve round trip through the real Flask routes (a fake IK client
  + a monkeypatched `detect_board_poses` sequence, `config.HAND_EYE_PATH` monkeypatched to
  a `tmp_path` so tests never touch the real `data/hand_eye.json`).

### New: `scripts/validate_pixel_to_world.py`

The script that answers "does this actually work, in millimetres?" Picks one ChArUco
corner (true position known from the board's own measured geometry), observes it from
≥2 arm poses, and cross-checks three independent answers:

1. **Single-view** — back-projects the corner's pixel via `PixelToWorldCalibrator.pixel_to_world`
   (assumes `TABLE_Z_IN_BASE`) against an independent answer computed the other way: full
   FK → hand-eye → `solvePnP` board pose → board-local corner geometry, no plane assumption.
2. **Two-view** — `triangulate_pixels` on the same views, also no plane assumption.
   Agreeing with (1) validates both paths independently.
3. **Cross-pose spread** — the board never moved, so every view's `T_base_board` should
   agree; the spread is a direct mm-level quality number, and its z re-measures the table
   plane for free (compare against the current `TABLE_Z_IN_BASE = -0.074`).
4. **`--goto-corner`** — commands the claw tip to the recovered point via `HardwareRobot`
   for a final ruler-against-reality check.

The compute-section math (independent-answer formula, triangulation, spread) was
hand-verified against synthetic ground truth during this session and recovered the true
values to floating-point precision — see git history for the scratch check.

### Docs

`CLAUDE.md` updated: the calibration-scripts section now describes the fixed
`calibrate_hand_eye.py` behavior and the new `hand_eye.py` module and
`validate_pixel_to_world.py`; the web-UI section documents `--calibrate`; the Tests
section documents the new test files.

## Session 2 (2026-07-26): camera intrinsics CALIBRATED on hardware

Merge Path step 1 is **done**. This was the first time any of the calibration tooling above
ran against the real camera.

### Result

| Metric | Value |
|---|---|
| Views captured | **56** (minimum is `CALIB_INTRINSICS_MIN_SAMPLES` = 15) |
| Boards contributing | all 6 |
| RMS reprojection error | **0.324 px** (accept threshold ~1 px) |
| Resolution | 640x480 |

```
fx = 660.9045852787211      cx = 316.54256156175046
fy = 661.1259343533698      cy = 230.38470664487326
distortion = [0.04948341884997802, -0.0928735686760541, -0.007186978169095462,
              -0.0008801535848997029, 0.060709883927266166]
```

Three sanity checks beyond the RMS, all of which pass — worth repeating on any re-calibration,
because a low RMS alone can hide a bad view set:
- **`fx` ~= `fy`** (660.90 vs 661.13, 0.03% apart) — square pixels, correct for this sensor.
- **`cx, cy` near the image centre** (316.5, 230.4) vs (320, 240) — off by 3.5 / 9.6 px, i.e.
  plausible lens decentering. A result tens of px away would mean the solve fit noise.
- **Small distortion coefficients** — normal for this class of webcam.

**This step mattered more than it looks.** The `config.py` placeholder was `CAMERA_FX = 550`;
the truth is 661 — a **20% error**, which every lateral world coordinate inherited.

### Where the data lives (and why it is NOT in git)

`data/camera_intrinsics.json`, written by `save_intrinsics`. It is **gitignored on purpose**
(`.gitignore`: "Local calibration data ... machine-specific"), alongside `data/hand_eye.json`
and `data/servo_calibration.json`. So **cloning this repo does not get you a calibrated
camera** — the numbers above are recorded here precisely because the file itself never leaves
this machine. Backed up locally to `data/camera_intrinsics.json.bak-20260726-184843`
(`data/*.bak-*` is also gitignored).

### Camera identity: the NexiGo is OpenCV index 1

This machine exposes three capture devices, and only one is the arm camera. Determined by
probing resolution + grabbing frames, since OpenCV indices carry no device names:

| Index | Device | Signature |
|---|---|---|
| 0 | EOS Webcam Utility | caps at 1024x576; dark when the DSLR is off |
| **1** | **NexiGo N930AF** | **the arm camera — FHD-capable** |
| 2 | OBS Virtual Camera | FHD, shows the OBS placeholder logo |

`config.CAMERA_INDEX = 1` is therefore already correct. If a device is added or unplugged
these indices can renumber — re-probe rather than assuming.

### Why 640x480, not the NexiGo's native 1080p

**Intrinsics are resolution-specific** — `fx/fy/cx/cy` are in pixels, so the same lens gives
`fx ~= 661` at 640x480 and `fx ~= 1650` at 1920x1080. `Camera` opens at
`config.FRAME_WIDTH/HEIGHT` (640x480), which is what the pick pipeline, demo scripts, and web
UI all run at, so calibrating at that size is the correct and self-consistent choice.
**Never change resolution partway through a capture run** — the views become mutually
inconsistent and `calibrateCamera` fits garbage.

Moving to 1080p later is a real but bounded change (est. 1.5-3 h). An audit of `config.py`
found only four genuinely pixel-pinned values — `MIN_CONTOUR_AREA` (area, ~6.75x),
`MORPH_KERNEL_SIZE`, `STUD_REGION_CLOSE_MIN/MAX_PX`, `SPECULAR_INPAINT_RADIUS`. Everything
else is either a fraction, normalised against `STUD_ROI_REFERENCE_PX`, or derived
(`CAMERA_CX/CY` follow `FRAME_WIDTH/HEIGHT` automatically). Evidence it is low-risk:
`tests/sample_images/` already spans 540x360 to 3024x3024 with `MIN_CONTOUR_AREA = 350`
unchanged, because that constant is a *noise floor*, not a "bricks are this big" threshold.
The genuine unknown is the **aspect-ratio change** (4:3 -> 16:9), which is a different sensor
crop and not a resampling, so the field of view changes shape — verify empirically before
assuming 1080p is strictly a superset. Recommendation: get the full chain validated at
640x480 first, so a bad result later can't be blamed on two variables at once.

### Operational notes learned the hard way

- **Re-running intrinsics starts from scratch AND overwrites.** Views accumulate in in-memory
  lists only — nothing is persisted between runs — and `save_intrinsics` is a plain
  `write_text` with no backup. A second run capturing 20 mediocre views would silently
  replace this 56-view / 0.324 px result, unrecoverably. Back the JSON up before re-running.
- **`no board with at least 6 corners - not captured` is the guard working, not an error.**
  It fired 3 times out of ~59 `c` presses (pointed off the boards / motion blur / too
  oblique). A refused press banks nothing, so it cannot corrupt the set. Only worry if it
  fires on *every* press and the run ends with 0 views.
- **The interactive OpenCV scripts need a real console window.** Launching
  `calibrate_camera_intrinsics.py` as a detached background process made it exit immediately
  with 0 views — a detached process has no interactive desktop session, so `cv2.waitKey`
  never receives keys. Launch via `Start-Process powershell -ArgumentList '-NoExit',...`
  (a helper script doing this is not committed; it is two lines). **Click the video window
  before pressing keys** — `waitKey` only sees input directed at that window, not the console.
- **`.venv` had to be recreated** on this machine (`pyvenv.cfg` pointed at a nonexistent
  interpreter, so `import cv2` failed). `Remove-Item -Recurse -Force .venv;
  python -m venv .venv; .venv\Scripts\pip install -r requirements.txt` — as CLAUDE.md's
  Commands section already documents. Resolved to `opencv-python 4.13.0`, which satisfies the
  `>=4.8,<5.0` pin and has both `calibrateHandEye` and the modern ChArUco API.
- `pytest`: **129 passed** after the environment rebuild.

## Next steps (needs the physical arm + camera + MATLAB server)

Step 2 below is **DONE** (see Session 2 above). Steps 1 and 3-5 are still outstanding — note
that step 1 was *not* needed for intrinsics (no arm involved) but **is** required before
hand-eye. Do the rest **in order**:

1. **Re-verify the frame seam is intact.** `python scripts/check_servo_health.py` — the
   B4 block must show the claw tip *below* the wrist. If not, stop: the physical/model
   frame conversion has been broken somewhere and every downstream number is garbage (see
   CLAUDE.md's "COORDINATE FRAMES" section).
2. ~~**Camera intrinsics.**~~ **DONE 2026-07-26** — 56 views, RMS 0.324 px, written to
   `data/camera_intrinsics.json`. See "Session 2" above for the values, the camera-index
   mapping, and why it was done at 640x480. **Still to do before step 3: tape all 6 boards
   flat and rigid across the tabletop** — hand-eye requires them stationary for the whole
   session, and they were handheld/loose for the intrinsics run.
3. **Hand-eye calibration.** Either `python scripts/calibrate_hand_eye.py` or
   `python scripts/run_arm_ui.py --hardware --calibrate` (MATLAB server must be running:
   `matlab/` → `ik_fk_server`). Record ≥12 samples per board across varied jogged poses.
   Accept on: TSAI-vs-PARK disagreement < 5 mm, board spread < 10 mm, and — new this
   session — per-board results agreeing with each other. Sanity-check the saved
   translation against a rough ruler measurement of the camera's physical offset from the
   wrist; a result that disagrees by centimetres is wrong regardless of its internal stats.
4. **End-to-end validation.** `python scripts/validate_pixel_to_world.py --board 1
   --corner-id 0` — single-view vs. independent-answer agreement, two-view triangulation
   gates, cross-pose board spread, and a `TABLE_Z_IN_BASE` cross-check against the current
   -0.074 m figure.
5. **Physical truth test.** Re-run step 4 with `--goto-corner` and measure the actual
   miss with a ruler. This is the number that says the system works — expect a few mm of
   open-loop servo error on top of calibration error (README's Stage D log recorded ~4 mm).

Record the resulting numbers in `CLAUDE.md` the way the Stage B/C/D bring-up results
already are (see `README.md`'s bring-up log and `CLAUDE.md`'s "Bring-up status" section).

### Also still outstanding (pre-existing, not part of this session)

Carried over from `README.md`'s Stage D log — these block a fully-trusted `HardwareRobot`
independent of the camera work above:

- Stage D lateral probes (`forward`/`left`/`right`) — vertical motion alone barely
  exercises J1/J5, so a sign or scale error there wouldn't have shown up yet.
- J6 (gripper) direction is confirmed but its open/closed tick range hasn't been
  calibrated against the physical claw's actual travel.
- `TABLE_Z_IN_BASE` is still a ruler-to-eye estimate (±2 mm) — step 4 above will either
  corroborate it via the camera or point at a real discrepancy.

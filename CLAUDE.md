# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Computer vision pipeline (OpenCV, classical CV — no ML yet) for a **5-DOF** robotic arm that picks up a single red Lego brick from a table. The camera is **eye-in-hand** (mounted on the arm). The arm's own code does not exist yet; this repo is the vision side, built so the robot code merges in by implementing one interface. Development is on Windows / PowerShell.

The full path is implemented end-to-end against a **simulated** robot: detect brick → back-project its pixel to a table coordinate in the robot base frame → plan a top-down grasp → command the arm. What is *not* real yet: the calibration numbers (camera intrinsics + hand-eye transform are placeholders) and the robot backend (only `SimRobot` exists). See **Merge path** below — those are the two things to fill in.

See [README.md](README.md) for a quick project overview, setup, and the command list; this file goes deeper on architecture, the data flow, and the merge path onto real hardware.

## Commands

```powershell
# Environment (venv already exists at .venv/)
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# If .venv was copied from another machine, .venv\pyvenv.cfg will point at a
# nonexistent python.exe and every command below fails with "did not find
# executable". Recreate it locally: Remove-Item -Recurse -Force .venv;
# python -m venv .venv; then reinstall requirements.txt.

# Tests — run from the repo root (pytest auto-discovers tests/)
pytest
pytest tests/test_lego_detector.py                          # one file
pytest tests/test_lego_detector.py::test_rejects_synthetic_red_shape_without_studs  # one test
pytest -k "synthetic"                                        # by name substring

# Full pick pipeline end-to-end against the SIMULATED robot (no hardware, no camera):
python scripts/run_pick_demo.py                                     # synthetic brick frame
python scripts/run_pick_demo.py --image "tests/sample_images/red lego brick 2.jpg"
python scripts/run_pick_demo.py --camera                            # one live frame, still sim arm

# Interactive tools (need a webcam, or pass a static image)
python scripts/run_live_detection.py          # live webcam detection, 'q' to quit
python scripts/tune_hsv.py --image tests/sample_images/"red lego brick 2.jpg"   # dial in HSV thresholds
python scripts/review_images.py               # step through tests/sample_images/ showing pass/fail

# Arm observation / jog web dashboard (live camera + per-joint J1-J6 controls):
python scripts/run_arm_ui.py                  # mock joints, no hardware needed
python scripts/run_arm_ui.py --hardware       # drives the real servo bus
python scripts/run_arm_ui.py --no-camera --overlay   # flags compose freely
```

There is no build step and no linter configured.

## Import model (important, and easy to trip on)

The package is **not installed** — there is no `pyproject.toml`, `setup.py`, `conftest.py`, or `pytest.ini`. Every test and script makes `vision_pipeline` importable by inserting `src/` onto `sys.path` at the top of the file:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

Consequences:
- Always run `pytest` and the scripts **from the repo root**.
- Any new test or script under `tests/` or `scripts/` must repeat that `sys.path.insert` line before importing `vision_pipeline`, or the import fails.
- Inside `src/vision_pipeline/`, modules import each other by absolute package path (`from vision_pipeline import config`), never by relative import.

## Architecture

Two-stage detection, deliberately kept as separate classes so the plain color stage stays reusable:

1. **Color stage** — [`ColorDetector`](src/vision_pipeline/detection/color_detector.py): HSV threshold → mask → morphological cleanup → contours. Finds *anything red* (brick, cup, hand). Red needs **two** HSV ranges because red's hue wraps around 0/179; the detector OR-combines them. Returns `Detection`s sorted largest-area first.
2. **Stud + shape stage** — [`LegoBrickDetector`](src/vision_pipeline/detection/lego_detector.py) wraps `ColorDetector`, then for each red region first repairs its contour with [`color_detector.close_contour_gaps`](src/vision_pipeline/detection/color_detector.py) (a specular highlight on a glossy stud can dip below the HSV saturation floor and punch a hole clean through the color mask, fragmenting the silhouette both signals below rely on — closes it with a kernel proportional to the candidate's own size), then blends two independent signals into one confidence score: [`stud_detector.count_studs`](src/vision_pipeline/detection/stud_detector.py) (Hough circles on the grayscale ROI, upscaling a too-small ROI to a standard reference resolution with a correspondingly relaxed accumulator threshold, and suppressing specular highlights on glossy studs) and [`shape_detector.score_shape`](src/vision_pipeline/detection/shape_detector.py) (rectangularity × aspect-ratio plausibility, purely geometric, multiplicative so a non-rectangular blob can't get credit just from a plausible elongation). A candidate is **kept if `STUD_WEIGHT * stud_score + SHAPE_WEIGHT * shape_score >= DETECTION_CONFIDENCE_THRESHOLD`**. This is what rejects a red cup or a hand (red but smooth AND round) while still accepting a brick whose studs are legitimately unresolvable — too far from the camera, or glare that survived suppression — because shape evidence, or a handful of recovered studs even without full resolution, can still carry it. All of the size/distance-scoped thresholds here (`STUD_ROI_REFERENCE_PX`, `STUD_HOUGH_PARAM2_UPSCALED`, `MIN_CONTOUR_AREA`) were calibrated against real photos, not just synthetic test fixtures — a hand and a genuinely distant brick can land in overlapping ranges on any single classical-CV signal (fill ratio, raw Hough permissiveness), so the real regression guard is `tests/sample_images/` + `SAMPLE_TRUTH`, not the synthetic tests alone.

[`Detection`](src/vision_pipeline/detection/types.py) is a plain dataclass that acts as the **contract** between detection and the downstream stages — it carries `centroid_px`, `area`, `bbox`, `angle_deg`, `contour`, `num_studs`, `shape_score`, `confidence`. Downstream code consumes these and never touches OpenCV.

### The full pick flow (this is the part to understand)

[`PickPipeline`](src/vision_pipeline/pipeline.py) orchestrates everything and is the class the robot code merges into. Two entry points:
- `locate_brick(frame, ee_pose) -> PickTarget | None` — vision only, no robot movement, fully testable.
- `run_once(frame) -> PickTarget | None` — reads the arm's current pose from the robot, locates the brick, and executes the grasp.

Data flow for one frame:

```
frame ──LegoBrickDetector──▶ Detection (pixel centroid + angle)
                                   │  + ee_pose (arm forward kinematics, at capture time)
                                   ▼
              PixelToWorldCalibrator ── pixel ▶ ray(camera) ▶ ray(base) ▶ ∩ table plane
                                   ▼
                              PickTarget (base-frame x,y,z + yaw)
                                   ▼
              plan_pick_sequence ─▶ [hover+open, descend, close, lift]  ──▶ RobotInterface
```

**Why the eye-in-hand camera drives the whole design:** a pixel only becomes a world point once you know where the camera was when the frame was shot. That pose comes from the arm's forward kinematics at capture time, so the pipeline *asks the robot* for it (`RobotInterface.get_end_effector_pose`) and feeds it into calibration. The transform chain is `T_base_camera = T_base_gripper (from the arm, runtime) @ T_gripper_camera (fixed hand-eye calibration)`; the brick's ray is back-projected through that and intersected with the table plane `z = TABLE_Z_IN_BASE` to recover depth from a single camera. This is why calibration takes a **4x4 gripper pose**, not just a pixel.

### Calibration internals
- [`geometry.py`](src/vision_pipeline/calibration/geometry.py) — pure-numpy rigid transforms + ray/plane intersection + `triangulate_rays` (least-squares closest point to N rays; the two-view depth primitive). Defines the project's RPY convention (`R = Rz(yaw)·Ry(pitch)·Rx(roll)`); change it here if the arm uses a different one. `Pose.to_matrix()` in `robot_interface/base.py` routes through this.
- [`camera_model.py`](src/vision_pipeline/calibration/camera_model.py) — `CameraIntrinsics` (fx,fy,cx,cy,distortion) + `pixel_to_ray` / `project_point`. `load_intrinsics`/`load_hand_eye` read JSON from `data/` and **fall back to config placeholders when the file is absent** (missing calibration is not a crash).
- [`pixel_to_world.py`](src/vision_pipeline/calibration/pixel_to_world.py) — `PixelToWorldCalibrator`, the chain above. Kept independent of `robot_interface` (takes a matrix, not a `Pose`) so calibration never depends on the robot package. **Two depth paths:** `pixel_to_world` (single view, assumes the table plane) and `triangulate_pixels` (two-or-more views → `TriangulationResult` with the point plus `residual_m`/`parallax_deg` quality numbers; recovers true depth for tilted/stacked/unknown-height bricks, no plane assumption).

### Two-view (moving-camera stereo) depth
The eye-in-hand camera moves with the arm and its pose is known every instant (FK @ hand-eye), so two shots of the same brick from two arm poses form a stereo pair with a known baseline — triangulating removes the "brick lies flat on `TABLE_Z_IN_BASE`" assumption. This is **purely additive**: the single-view flat-table path is unchanged. `PickPipeline.locate_brick_two_view(captures)` detects the brick in each `(frame, ee_pose)` capture and triangulates the centroids, gating on `TWO_VIEW_MIN_PARALLAX_DEG` / `TWO_VIEW_MAX_RESIDUAL_M` before trusting the result. Driven end-to-end by [`scripts/run_two_view_pick.py`](scripts/run_two_view_pick.py) (`--dry-run` locates without grasping). The math is covered by [test_triangulation.py](tests/test_triangulation.py) — exact-answer ray cases + a full project→triangulate→recover round-trip (incl. non-identity hand-eye), all synthetic (no hardware).

### Camera calibration scripts (run-by-hand, Merge Path steps 1–2)
Interactive, hardware-dependent tools, not part of pytest. They use a **ChArUco** board (chessboard + a unique ArUco marker per square), not a plain chessboard — every corner is individually identified, so detection survives partial/angled views, which matters because the eye-in-hand camera's view of the board swings around as the arm moves (OpenCV 5.0 `cv2.aruco.CharucoDetector` + `board.matchImagePoints`). The board geometry lives in `config.CALIB_ARUCO_DICT` / `CALIB_CHARUCO_SQUARES_X/Y` / `CALIB_SQUARE_SIZE_M` / `CALIB_MARKER_SIZE_M` — the **single source of truth**: generator and both detectors read the same values, so the printed board and the detector's expectation can't drift. **Never substitute a board from elsewhere** (a random online image): a mismatched dictionary/geometry doesn't just fail to detect, it can silently feed wrong 3D coordinates and yield a confidently-wrong calibration.
- [`scripts/generate_charuco_board.py`](scripts/generate_charuco_board.py) — renders the printable board to `data/charuco_board.png` at a known DPI (print at 100%, measure a square, correct `CALIB_SQUARE_SIZE_M` if the printer rescaled). Run this **first**; regenerate whenever the config geometry changes.
- [`scripts/calibrate_camera_intrinsics.py`](scripts/calibrate_camera_intrinsics.py) — `cv2.calibrateCamera` (fed by `matchImagePoints`), camera only, writes `data/camera_intrinsics.json` via `save_intrinsics`. Do this before hand-eye.
- [`scripts/calibrate_hand_eye.py`](scripts/calibrate_hand_eye.py) — `cv2.calibrateHandEye` (eye-in-hand); **drives the real arm** through a workspace sweep. `get_end_effector_pose().to_matrix()` is gripper2base and `solvePnP` (on `matchImagePoints` output) gives target2cam directly, so OpenCV's returned cam2gripper *is* `T_gripper_camera` — written to `data/hand_eye.json` via `save_hand_eye`. Reports a TSAI-vs-PARK cross-check and the stationary-board position spread as trust signals. Needs intrinsics first. **Board must stay stationary**; if using multiple identical printed copies to cover area, keep only one in the camera's view (duplicate marker IDs across sheets would make the "stationary target" ambiguous).

### Robot seam
- [`RobotInterface`](src/vision_pipeline/robot_interface/base.py) — the entire contract to the arm: `get_end_effector_pose` (FK), `send_target_pose`, `set_gripper`. `Pose` (base-frame, meters + degrees) is the shared type.
- [`SimRobot`](src/vision_pipeline/robot_interface/sim.py) — in-memory backend: reports a fixed gripper pose and records every command. This is what makes the pipeline testable/demoable with no hardware.
- [`PickTarget` + `plan_pick_sequence`](src/vision_pipeline/planning/pick.py) — `PickTarget` is the vision→robot handoff (x,y,z,yaw; roll/pitch are *not* here because a 5-DOF top-down pick keeps the tool pointing down — those fixed angles come from config). `plan_pick_sequence` turns one target into the backend-agnostic hover→descend→close→lift step list, so every backend runs the same grasp and only implements the primitive moves.

- [`Camera`](src/vision_pipeline/capture/camera.py) — thin `cv2.VideoCapture` wrapper (context manager + frame iterator) so detection depends on this interface, not OpenCV's camera API directly.

### Arm observation / jog web UI (bring-up tool)

[`scripts/run_arm_ui.py`](scripts/run_arm_ui.py) launches a Flask dashboard (`src/vision_pipeline/webui/`) showing the live camera feed and letting you jog joints J1–J6 by hand — built for hardware bring-up, since `ServoBus` (above) hasn't been exercised on physical hardware yet. It sits *beside* `PickPipeline`, not inside it — nothing here is on the pick pipeline's path.

- [`JointController`](src/vision_pipeline/robot_interface/joint_controller.py) is the joint-space seam this fills in: `RobotInterface`/`HardwareRobot` only expose Cartesian `Pose` (xyz, solved through IK), and `SimRobot` has no joint model at all — individual-joint control previously existed only at the raw-tick `ServoBus` level. `MockJointController` (in-memory, seeded at each joint's `home_tick`) and `ServoJointController` (thin wrapper over a real `ServoBus`) both implement `read_joint`/`jog`/`set_gripper`/`degrees_per_tick`; `scripts/run_arm_ui.py --hardware` selects which one backs the dashboard.
- [`servo_calibration.py`](src/vision_pipeline/robot_interface/servo_calibration.py) factors the JSON-or-`config.SERVO_CALIBRATION_FALLBACK` loading and tick↔radian conversion out of `ServoBus` into serial-free functions, so `MockJointController` converts ticks↔degrees using the exact same calibration `ServoBus` would, without opening a serial port. `ServoBus` keeps its own copy (its byte-level tests pin that behavior) — this is a standalone twin, not a replacement.
- [`webui/app.py`](src/vision_pipeline/webui/app.py) (`create_app`) wires one `JointController` and a background-threaded [`CameraStreamer`](src/vision_pipeline/webui/camera_stream.py) (owns the one `Camera`, degrades to a placeholder frame if no webcam) into Flask routes: `/` (dashboard), `/video_feed` (MJPEG), `/api/joints`, `/api/joints/<id>/jog`, `/api/gripper`. Every controller call goes through one lock — `ServoBus` isn't thread-safe, and Flask's dev server is multithreaded.
- The dashboard shows each joint's position in both degrees and raw ticks, with jog buttons whose step size is set by a per-joint tick slider; the slider also previews its size in degrees (`JointController.degrees_per_tick`) so a tick count is legible without doing the conversion by hand.

## Merge path (what's left to make this drive a real arm)

1. **Calibrate the camera intrinsics** (chessboard + `cv2.calibrateCamera`), write `data/camera_intrinsics.json`. Until then `config.CAMERA_*` placeholders are used and world coordinates are only roughly right.
2. **Hand-eye calibrate** (`cv2.calibrateHandEye`) once the arm exists, write the 4x4 gripper→camera transform to `data/hand_eye.json`. Placeholder is identity (camera == gripper), which is wrong for any real mount.
3. **Measure table geometry** into config: `TABLE_Z_IN_BASE`, `PICK_Z_OFFSET`, `APPROACH_HEIGHT`, and confirm `PICK_ROLL_DEG`/`PICK_PITCH_DEG` match the arm's "tool pointing down" convention.

   **5-DOF orientation caveat:** this specific arm is genuinely 5-DOF and its IK (`matlab/`, below) is **position-only** (`weights = [0 0 0 1 1 1]`) — there is no spare joint for wrist yaw/roll/pitch. `PickTarget.yaw_deg` is still *computed* by vision, but `HardwareRobot` **silently drops** all orientation (`roll/pitch/yaw`) and commands `(x, y, z)` only; it logs the dropped angles at debug level. `PICK_ROLL_DEG`/`PICK_PITCH_DEG` describe the arm's *fixed* mechanical top-down pose, they are not commanded. Accepted because the grasp is top-down on a near-symmetric brick; revisit only if orientation-dependent grasping becomes a requirement.
4. **Write a real `RobotInterface` backend** (sibling to `SimRobot`) against your ROS/serial/TCP arm, implementing the three methods. Swap it in wherever `SimRobot()` is constructed (`scripts/run_pick_demo.py`, or your own entry point). Nothing else changes. **One such backend now exists** — see "Real hardware backend" below.

The agreement point with the robot code is the `Pose`/`PickTarget` convention: base-frame origin, meters, degrees, RPY as defined in `geometry.py`. Align units/axes with the kinematics code there.

### Real hardware backend (MATLAB IK/FK + Feetech servos)

[`HardwareRobot`](src/vision_pipeline/robot_interface/hardware.py) is a concrete `RobotInterface` backend for the physical arm, composed of two seams that are tested independently before being combined:

- **MATLAB IK/FK bridge** — the validated position-only IK solver (`matlab/IKtrials_v2.m`, kept verbatim as reference) is hosted as a persistent TCP JSON server, [`matlab/ik_fk_server.m`](matlab/ik_fk_server.m), on `localhost:9999`. Shared setup (`importrobot` of `Robomainassemjoints.slx`, servo→joint mapping, idler freeze, `ClawTip` end effector, IK solver) lives in the `matlab/init_arm.m` **script** — deliberately a script, not a function, because `importrobot`'s compile step reads base-workspace variables (`smiData` from `Robomainassem_DataFile.m`, and the `j1..j6` "From Workspace" placeholder signals) that must exist *before* the import. FK returns the **wrist (Body08)**, not the claw tip, so it composes with the separate hand-eye calibration. [`MatlabIKClient`](src/vision_pipeline/robot_interface/matlab_client.py) is the stdlib-socket Python client; `IKUnreachableError` surfaces an out-of-workspace target. Verify with `matlab/test_ik_fk.m` (offline, MATLAB-side) and `scripts/test_matlab_bridge.py` (against a running server).
- **Servo bus** — [`ServoBus`](src/vision_pipeline/robot_interface/servo_driver.py) speaks the Feetech STS3215 (Dynamixel-1.0-compatible) serial protocol over a Waveshare adapter, with **write-then-read-back verification** on every move. Per-servo calibration (`home_tick`, `ticks_per_rad`, `dir_sign`) loads from `data/servo_calibration.json` with a `config.SERVO_CALIBRATION_FALLBACK` placeholder (same pattern as camera intrinsics). J1–J5 are IK-driven; **J6 is the gripper** (open/close), mapped to `set_gripper`. Also exposes `ping`/`scan_ids`/`write_servo_id` for one-time bus setup — Feetech servos ship at a shared factory ID, so each must be assigned a unique ID (1..6 for J1..J6) before wiring the chain; [`scripts/set_servo_id.py`](scripts/set_servo_id.py) drives that one servo at a time. `write_servo_id` does the Feetech EEPROM dance (unlock→write ID→re-lock, re-lock addressed to the *new* ID). The move/read/ID-change protocol is unit-tested against a fake-serial servo emulator ([test_servo_driver.py](tests/test_servo_driver.py), exact-byte assertions), but has NOT been exercised against physical hardware yet — that's the remaining Stage-2 bring-up.

`HardwareRobot` owns the joint-angle state in Python (seeded from a servo read-back at init, refreshed from the **verified** read-back after every move) and treats MATLAB as a pure stateless math service: `get_end_effector_pose` = live FK of the last verified angles; `send_target_pose` = IK on `(x,y,z)` → `move_and_verify` J1–J5. The MATLAB server and servo driver are the two placeholders left before this drives real hardware (calibration numbers + a live server + a wired bus), mirroring the intrinsics/hand-eye placeholders on the vision side.

## Tuning is centralized in config.py

All thresholds live in [`config.py`](src/vision_pipeline/config.py) — detection tunables (HSV bounds, `MIN_CONTOUR_AREA`, `MORPH_KERNEL_SIZE`, Hough parameters, `STUD_ROI_REFERENCE_PX`/`STUD_ROI_MAX_UPSCALE`/`STUD_HOUGH_PARAM2_UPSCALED`, `STUD_REGION_CLOSE_*`, `SPECULAR_*`, `SHAPE_*`, `STUD_WEIGHT`/`SHAPE_WEIGHT`/`DETECTION_CONFIDENCE_THRESHOLD`) **and** the geometry/calibration tunables (`CAMERA_*` intrinsics, `HAND_EYE_PATH`, `TABLE_Z_IN_BASE`, `PICK_Z_OFFSET`, `APPROACH_HEIGHT`, `PICK_ROLL_DEG`/`PICK_PITCH_DEG`). Detection and calibration classes read these as **constructor defaults**, so tests/callers override per-instance without touching config. When behaviour is wrong, retune here rather than editing logic. Key relationships baked into current values:
- High saturation floor (150) deliberately rejects skin tones (low-saturation orange-red) so the detector doesn't fire on a hand. Raise it if background skin/wood leaks in; lower the value floor (90) first if a real brick is missed in dim light.
- `MIN_CONTOUR_AREA` (350) is deliberately well above single-digit-pixel JPEG/lighting noise speckles (seen as low as ~30-300px² in real photos) — those tiny blobs are pixel-grid-quantized near-rectangles almost by accident and can otherwise slip through on shape score alone. Don't lower this much without also re-checking the noise-speck real-photo regression case.
- `STUD_WEIGHT`/`SHAPE_WEIGHT`/`DETECTION_CONFIDENCE_THRESHOLD` trade false positives vs. missed bricks the same way `MIN_STUDS` used to alone: raise the threshold or `STUD_WEIGHT` if smooth red objects get misclassified; lower the threshold, or raise `STUD_ROI_REFERENCE_PX`/`STUD_REGION_CLOSE_FRAC` if a real brick far from the camera (or at a bad, glare-prone angle) is being missed. Be cautious raising `SHAPE_WEIGHT` or loosening `SHAPE_RECT_SCORE_LOW`/`STUD_HOUGH_PARAM2_UPSCALED` much further — real photos show a hand's silhouette and a genuinely distant brick's silhouette land in overlapping ranges on fill ratio and on Hough's own permissiveness, so those two knobs don't have a clean global setting that helps one without also helping the other; tune against `tests/sample_images/`, not intuition.

## Tests

Everything runs without a camera or robot. The suite has five parts:
- **Detection** ([test_color_detector.py](tests/test_color_detector.py), [test_lego_detector.py](tests/test_lego_detector.py), [test_shape_detector.py](tests/test_shape_detector.py), [test_stud_detector.py](tests/test_stud_detector.py)) — synthetic frames (red square; red rectangle with/without drawn "studs"; a smooth red oval) pin the core rule *red + (studs or brick-shaped) = brick, red alone (round, no studs) = not a brick* — including a dedicated case proving a rectangular region with zero resolvable studs is still accepted on shape confidence alone (the far-away/glare-corrupted case). Plus real-photo regression tests parametrized over `tests/sample_images/`, keyed by the `SAMPLE_TRUTH` dict (`True` = must detect, `False` = must reject; missing files skip, not fail). Add a photo → add it to `SAMPLE_TRUTH`. The `False` entries (red cup, hand) guard the false positives the stud+shape stage exists to fix.
- **Geometry** ([test_geometry.py](tests/test_geometry.py)) — exact-answer math tests for transforms and ray/plane intersection; they lock the RPY convention.
- **Calibration** ([test_pixel_to_world.py](tests/test_pixel_to_world.py)) — the load-bearing one: a **projection round-trip**. It projects a known table point through the camera to a pixel, then back-projects through `PixelToWorldCalibrator` and asserts recovery. If projection and back-projection agree, the whole chain is self-consistent.
- **Pipeline** ([test_pipeline.py](tests/test_pipeline.py)) — end-to-end against `SimRobot`: a synthetic brick frame yields a `PickTarget` and the exact hover→descend→close→lift command sequence (4 poses, gripper `[open, close]`). This is the "vision side is merge-ready" regression guard.
- **Joint control & web UI** ([test_joint_controller.py](tests/test_joint_controller.py), [test_webui.py](tests/test_webui.py)) — `MockJointController` jog/clamp/degree-conversion correctness, and the Flask dashboard's REST endpoints via `app.test_client()` wired to the mock (no camera or hardware needed). `ServoJointController` is checked against a small duck-typed fake bus, since `ServoBus`'s own wire protocol is already exhaustively covered by [test_servo_driver.py](tests/test_servo_driver.py).

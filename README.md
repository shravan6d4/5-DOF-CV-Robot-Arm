# Robotic Arm Vision

Computer vision pipeline for a 6-DOF robotic arm to detect and pick up a single Lego brick.

Pipeline phases:
1. **Phase 1** (current): Camera capture + HSV color thresholding + contour detection + centroid.
2. **Phase 2**: Pixel -> robot world-coordinate calibration.
3. **Phase 3**: Integration with the arm's control interface.
4. **Phase 4** (stretch): ML-based detector.

## Setup

```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Confirm OpenCV imports correctly
python -c "import cv2, numpy; print(cv2.__version__)"
```

## Project Layout

- `src/vision_pipeline/capture/` — webcam frame capture
- `src/vision_pipeline/detection/` — brick detection (Phase 1). Two stages:
  `color_detector.py` finds red regions; `lego_detector.py` then verifies each
  region has Lego *studs* (via `stud_detector.py`) so it only accepts real
  bricks, not other red objects like a cup or a hand.
- `src/vision_pipeline/calibration/` — pixel-to-world coordinate mapping (Phase 2, stub for now)
- `src/vision_pipeline/robot_interface/` — abstract interface for the arm's control code (Phase 3, stub for now)
- `scripts/` — runnable demos and tools (live detection viewer, HSV tuning tool)
- `tests/` — unit-style tests, run against static images in `tests/sample_images/`

## Running things

```powershell
# Tune HSV thresholds interactively (needs a webcam or a sample image)
python scripts/tune_hsv.py

# Run live detection with a webcam
python scripts/run_live_detection.py

# Run tests (works even without a camera, using tests/sample_images/)
pytest
```

## Testing without a camera

Drop a photo of a Lego brick on a plain background into `tests/sample_images/` (e.g. `brick1.jpg`)
and `pytest` will pick it up automatically. Until an image is added there, the detection test is
skipped with a clear message instead of failing.

## Hardware bring-up log — servo calibration (2026-07-22)

First physical bring-up of the 6-servo Feetech STS3215 bus (Waveshare adapter). Kept here as a
record of what was done and what went wrong, since some of it changes how the arm must be driven
going forward. See `CLAUDE.md` for the architecture this hardware plugs into.

**Servo ID assignment.** Servos ship from the factory all sharing the same default ID, so each was
connected alone to the bus and assigned a unique ID (1-6, matching J1-J6) via
`scripts/set_servo_id.py`. Confirmed afterward with a full read-only bus scan: all six IDs present
and answering.

**Power.** The servo bus needs its own supply (6-12.6V depending on servo variant) separate from
USB — USB only powers the adapter's own USB-serial chip, not the servos, and a lit power LED on
the adapter does not by itself confirm the servo rail is live. There is no way to power down a
single servo on the bus independently; cutting power drops holding torque on *every* joint at
once.

**Direction/scale verification.** Each joint was jogged a small number of ticks (~100-150) one at a
time, with a human watching and confirming the physical motion, to determine its real-world
rotation direction before trusting any automated move:

| Joint | Role | +ticks direction |
|---|---|---|
| J1 | base yaw | counterclockwise (viewed from above) |
| J2 | shoulder | tilts up |
| J3 | elbow | folds down, toward the table |
| J4 | wrist pitch | down/toward base (ccw from the side) |
| J5 | wrist roll | ccw (viewed from behind the robot) |
| J6 | gripper | opens (+ticks); confirmed by jogging toward the known-closed reference |

`ticks_per_rad` (~651.89 for J1-J5, ~325.95 for J6, the config placeholder values) held up as a
reasonable approximation across all these jogs — no joint's observed motion was wildly off from
the expected angle for a given tick delta.

**Incident 1 — J1 encoder wrap-seam runaway.** J1's captured home (4086 ticks) sat only 9 ticks
below the encoder's 0/4095 wrap point. After a power cycle, J1 drifted slightly and its reading
flipped across the seam (4086 -> 17) even though it had barely moved physically. A "return to home"
script read that 17 and blindly commanded a return to 4086 — in single-turn position mode the
servo cannot cross the dead zone at the seam, so instead of a ~2 degree correction it tried to
travel the *long way around* (~4069 ticks, ~358 degrees) at speed. Power was cut before it reached
a hard limit.

Root cause: no cap on commanded move size relative to current position, combined with a home
position parked right on the worst possible spot on the encoder. Fixed by re-centering J1 via the
Feetech one-key midpoint calibration (write `128` to the Torque Enable register, address 40) with
J1 physically at its home pose — this re-labels the servo's current physical position as tick
2048, dead center, as far from the wrap seam as possible. J1's home is now 2048 and has read
consistently stable since (no more seam flips).

**Incident 2 — J3/table collision (near miss).** During the same direction-verification jogging,
commanding J3 (elbow) further in the "+" direction attempted to fold the arm down into the table
the claw was already resting on. Caught immediately by the operator watching the physical arm;
power was cut before any damage. J3's "+" direction was already confirmed unsafe from that pose as
a result — future jogs from a similar starting position should use "-" (fold up, away from the
table) or a smaller step, not assume clearance.

**Home re-capture.** After the above, the arm was manually repositioned to a home pose with more
table clearance, and all six joints' positions were read and immediately commanded to *hold at
that same reading* (goal = present position, so effectively zero net motion) via a small
read-then-hold script, then written to `data/servo_calibration.json` as the new `home_tick` per
joint. This is the same file format `scripts/capture_servo_home.py` produces and is *not* committed
to git (machine-specific, see `.gitignore`).

**Known-good process for returning to home after a power cycle**, learned from the above: never
blindly command a stored home tick. Always (1) read-only pass across all joints first, (2) compute
the delta from current position to stored home for each, (3) refuse/skip any joint whose delta
exceeds a small safety cap (a few hundred ticks) rather than executing it, (4) move only the joints
within cap, one at a time, with someone watching.

**Outstanding before the MATLAB-driven pipeline (`HardwareRobot`) drives the arm for real:**
- The safety cap and refuse-on-large-delta behavior above was only applied in throwaway bring-up
  scripts, not in `ServoBus.move_and_verify()` itself (the actual method `HardwareRobot` calls).
  This needs to land in the real driver before the first live IK-driven move, not just in scratch
  scripts.
- `move_and_verify()` currently reads back the servo's position *immediately* after commanding a
  move, before it has physically arrived — this throws on essentially every real move today. Bring-up
  scripts worked around it with a settle delay (~1.5s) before reading back; the real driver needs
  the same fix.
- `dir_sign` in `data/servo_calibration.json` is still the placeholder (`+1` for every joint). The
  physical directions confirmed in the table above still need to be reconciled against MATLAB's own
  positive-rotation convention (`matlab/init_arm.m`) before they can be trusted for real IK moves —
  a desk exercise, not something that needs the arm powered.
- J6 (gripper) direction is confirmed but its open/closed tick range (vs. `config.SERVO_GRIPPER_OPEN_RAD`
  / `SERVO_GRIPPER_CLOSE_RAD`) has not been calibrated against the physical claw's actual travel.

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

## `dir_sign` reconciliation (2026-07-22, later session — REDONE after the frame flip was found)

The physical `+ticks` directions in the table above describe what the arm *does*; MATLAB's IK/FK
works in its own angle convention. `dir_sign` reconciles the two, and getting it wrong means a
commanded angle drives a joint the wrong way with nothing to catch it.

**A first reconciliation pass this session got J1–J4 exactly backwards**, because it interpreted
the model's FK axes as physical directions. The operator then identified the root fact, now
documented as the canonical frame section in `CLAUDE.md` ("COORDINATE FRAMES"): **the imported
model's base frame is upside-down relative to the physical robot** — in the model the arm reaches
*upward* off the base; physically the claw hangs down toward the table (the plane motor 1 sits
on). `physical (x,y,z) = model (x,−y,−z)`. `MatlabIKClient` now converts at the wire, so all
Python-side code and scripts speak the physical frame; MATLAB-side code (and the historical
target names in `test_ik_fk.m` — its "low" is physically high) stay in the model frame.

Final values, physical frame throughout, **all confirmed by physical jog except J5**:

| Joint | `dir_sign` | Status | Notes |
|---|---|---|---|
| J1 | **+1** | confirmed by jog | derivation agreed |
| J2 | **−1** | confirmed by jog ×2 | **derivation said +1 and was wrong** |
| J3 | **−1** | derivation only | anchored by the table-strike incident (`+ticks` physically drove the claw *down* into the table) |
| J4 | **−1** | confirmed by jog | derivation agreed; first jog attempt was void (see below) |
| J5 | +1 | **unconfirmed — blocks IK moves** | see below; **not** position-irrelevant |

**J2 is the cautionary one.** The desk derivation concluded `+1` — MATLAB's `+angle` moves the
wrist physically up, and the bring-up log recorded `+ticks` = shoulder tilts up. Two independent
physical jogs both showed the arm moving *down*. Either the bring-up note was taken from a
different vantage, or the derivation chain has an error not yet isolated. Physical evidence won;
the wrong derivation is recorded rather than quietly deleted, because it's the one that would
otherwise be re-derived the same way next time.

**J5 is unconfirmed and that blocks IK-driven motion.** Its one "confirmation" was read against a
model-frame prediction, so the frame flip invalidates it. It was then dismissed as unimportant on
the grounds that the tip sits on J5's rotation axis — **wrong, and measurably so.** That confused
the *wrist* (which J5 barely moves, ~0.0 mm) with the *tip*, which hangs `CLAW_LEN` = 70 mm out on
the lever: measured **12.6 mm of tip travel for 30°, 38.5 mm for 104°**. J5 is a major positional
contributor and the IK solver leans on it heavily for lateral targets, so a wrong sign moves the
claw tens of mm the wrong way. Confirm it with a jog (the tip motion is large and easy to see)
before Stage D.

**J4's first confirm jog was void by construction**, worth knowing before trusting any future jog:
`jog_joint.py` predicted **wrist** motion while the operator naturally watched the **claw**. For
the big arm joints those move together, but J4 is wrist pitch — the wrist sits near its own
rotation axis and barely translates (~4.5 mm, lateral) while the tip swings ~70 mm out on the
lever (~8.5 mm, vertical). The prediction and the observation described different axes. The script
now predicts **claw-tip** motion and flags when wrist and claw diverge; the re-run matched.

**Two driver bugs found and fixed during this work:**
- `move_and_verify()` read back position *immediately* after commanding a move, catching the servo
  mid-travel. Now polls until the servo settles, with a stall check so an obstructed joint is
  reported rather than held against the obstruction.
- The stall check then false-triggered on a servo's normal acceleration ramp-up (only 0.3s of
  apparent stillness was enough). Caught on hardware: an 80-tick J5 move was reported as stalled
  near its start position, but a later read-only check found it had fully arrived. Fixed with a
  grace period before stall-counting begins.
- Also added: a bounded read-retry, after a single dropped serial byte mid-poll crashed an
  otherwise-successful move. The goal write happens *before* polling, so the servo is already
  driving toward its target regardless of whether our verification read succeeds.

**Outstanding before the MATLAB-driven pipeline (`HardwareRobot`) drives the arm for real:**
- **Stage D — IK round-trip validation.** `dir_sign` being correct only settles *direction*;
  `ticks_per_rad`, backlash, and overall model fit are still unvalidated against physical reality.
  Command a target → IK → move → read back → FK, and compare against a ruler.
- `TABLE_Z_IN_BASE = -0.073` is an estimate (frame-corrected FK tip at z = −63mm + the operator's
  ~10mm measured clearance). Replace with a touch-probe measurement: hand-position the claw to
  touch the table, read the servos, run the angles through `request_fk_tip`.
- J6 (gripper) direction is confirmed but its open/closed tick range (vs. `config.SERVO_GRIPPER_OPEN_RAD`
  / `SERVO_GRIPPER_CLOSE_RAD`) has not been calibrated against the physical claw's actual travel.
- Camera intrinsics and hand-eye calibration have not been run against the real camera/arm. Note
  hand-eye **drives a full IK-commanded workspace sweep**, so it wants Stage D passing first.

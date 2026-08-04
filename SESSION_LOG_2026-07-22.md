# Session Log — 2026-07-22

A record of a single working session covering: finishing the servo bus driver
(ID assignment), ChArUco board sizing, a physical bring-up of all 6 arm servos
(including two real incidents and their fixes), hardware research (power specs,
Waveshare tooling), a git push of the session's code, and misc project questions.
Kept as a narrative log rather than a transcript — organized by topic, in the
order it happened, with what was asked, what was found, and what was done.

See also: `CLAUDE.md` (architecture), `README.md` "Hardware bring-up log" section
(a condensed version of the calibration/incident portion of this log).

---

## 1. Servo bus driver — ID assignment tooling

**Ask:** finish the servo bus driver so individual Feetech STS3215 servos (which
ship from the factory all sharing one default ID) can be given unique IDs before
being wired into the shared 6-servo bus.

**Research first, then build:** checked Waveshare's official resources page for a
ready-made Windows configuration tool for the "Bus Servo Adapter (A)" board — none
exists (only source-code SDK bundles for Python/Arduino/Linux, no GUI). Decided to
build ID-writing directly into the project's own driver rather than pull in a
separate SDK dependency.

**Register addresses verified from source, not memory** (a wrong EEPROM address on
real hardware is not something to guess at): pulled from a public STS3215 register
reference — `ID` register = address 5, EEPROM `Lock` register = address 55
(0 = unlocked, 1 = locked).

**Added to `src/vision_pipeline/robot_interface/servo_driver.py`:**
- `ping(servo_id)` — reads the ID register, returns whether anything answers.
- `scan_ids(id_range)` — pings a range, returns which IDs are present.
- `write_servo_id(old_id, new_id, verify=True)` — the Feetech EEPROM sequence:
  unlock (write 0 to Lock) → write new ID → **re-lock addressed to the NEW id**
  (the servo starts answering to the new ID the instant it's written — locking
  the old ID would silently fail). Optionally pings the new ID to verify.

**Added `scripts/set_servo_id.py`** — guided one-servo-at-a-time CLI:
`--scan` to list what's on the bus, `--new-id N` to assign (auto-detects the
single connected servo's current ID, refuses if more than one is present).

**Tested without hardware:** `tests/test_servo_driver.py` uses a `FakeServoSerial`
class that emulates one STS3215 servo (decodes WRITE/READ packets, replies with
correctly-checksummed status packets), so exact wire bytes and the ID-change
sequence (including the new-ID re-lock addressing) are asserted directly. All
tests passed before any hardware was involved. (This file has since grown further
test fixtures — `MovingFakeServo`, `FlakyReadFakeServo` — added outside this
session; see the file itself for current coverage.)

---

## 2. ChArUco calibration board sizing

**Ask, in stages:** started with "is 40mm/square reasonable?" then, after the
user's own print attempt didn't fill the page, settled on a wider board with
smaller squares.

**40mm verdict:** at the board's grid then (7×5 squares), 40mm/square produced a
280×200mm footprint — too wide for US Letter even in landscape (279.4mm long
edge), and only barely fit A4 landscape with almost no trim margin. Also flagged a
detection-side concern: for this arm's short working distance, oversized squares
risk not fitting in frame at close range.

**Final sizing:** grid increased from 7×5 to **10×6** squares (more corners, wider
aspect ratio as asked), square size initially set to 25mm, then corrected to
**26.7mm** once the user measured the actual printout (measuring across multiple
squares and dividing, to average out ruler error — arrived at "2.6666 repeating"
→ 26.7mm per square). Marker size was scaled by the same factor the printer
apparently applied (18mm → 19.2mm), preserving the original 18/25 = 0.72 design
ratio rather than leaving it mismatched to the corrected square size.

**Files touched:** `src/vision_pipeline/config.py` (`CALIB_CHARUCO_SQUARES_X/Y`,
`CALIB_SQUARE_SIZE_M`, `CALIB_MARKER_SIZE_M`), `scripts/generate_charuco_board.py`
(fit-check logic made orientation-aware, since the new board is wider than tall
and needs to be checked against paper in landscape, not assumed portrait).
Regenerated `data/charuco_board.png`; full test suite stayed green throughout.

---

## 3. Camera mounting placement (design discussion, no code)

Three related questions, answered from how the pipeline's math actually works
(`PixelToWorldCalibrator`, hand-eye calibration) rather than general robotics
advice:

- **"Before or after the wrist rotation joint (J4 vs. past J5)?"** — must be
  *past* J5, rigidly fixed to whatever body FK reports (the wrist, Body08).
  Hand-eye calibration solves for one *constant* `T_gripper_camera`; if a joint
  sits between the camera and that reported frame, the true transform changes
  with that joint's angle and a fixed calibration silently corrupts pixel→world
  accuracy as the joint moves away from wherever it was during calibration.
- **"So, right on the gripper motor?"** — yes, but specifically the gripper's
  **static housing**, not the moving jaw. J6 only opens/closes the jaws; the
  housing itself doesn't move relative to the wrist, so mounting there keeps the
  transform constant regardless of gripper state. Mounting on a jaw would
  reintroduce the same problem as J4, just gated on gripper-open/closed instead
  of joint angle.
- **"Side of the gripper, or does it have to be on top?"** — kinematically no
  constraint either way (hand-eye calibration handles any fixed offset/angle).
  Practically, it matters for *detection*: `stud_detector.py` uses
  `cv2.HoughCircles` to find the studs on a brick's top face, which only reads as
  circular from a close-to-fronto-parallel view. A side-mounted camera looking
  mostly horizontally would see the brick's smooth side face — no studs, and the
  ≥`MIN_STUDS` gate would reject every real brick, same as it's designed to
  reject a red cup. Verdict: side placement is fine as long as it's angled down
  enough to keep a steep view of the top face — which also happens to help dodge
  the jaws blocking the view at close range, a legitimate reason to prefer some
  side offset over dead-center-top.

---

## 4. Repo folder rename (KrishOPENCV → 5DOFPYTHONPIPELINE) — inconclusive

Checked for hardcoded path dependencies before considering the rename: grep found
no references to the folder name anywhere in the repo; `.venv`'s `activate.ps1`
resolves its own location dynamically at runtime (the absolute path in
`pyvenv.cfg` is just provenance, unused); MATLAB `.m` files have no hardcoded
absolute paths. Code-wise, safe.

Was in the middle of checking the local `.claude/settings.local.json` for
anything path-sensitive when the exchange got interrupted by an unrelated request
(the reminder recall, then plan mode). **Never reached a conclusion or performed
the rename.** One real risk flagged before the interruption: Claude Code's
project memory/session-history identity is keyed off the absolute working
directory path, so renaming would likely start a fresh, disconnected project
identity rather than carrying over the memory already built up this session —
worth deciding on deliberately if this comes up again, not something to do
casually.

---

## 5. Session-start memory saved

Per an explicit ask to be reminded "before any conversation starts," wrote a
persistent memory (outside this repo, in Claude Code's own memory store) that
surfaces at the top of future sessions: share the MATLAB session
(`shareMATLABSession()`) and start `ik_fk_server` before any `HardwareRobot` /
MATLAB-bridge work, plus the (at-the-time) still-pending hardware bring-up
checklist. That checklist is now largely superseded by what actually happened in
sections 6-8 below.

---

## 6. Physical bring-up: servo pairing, power, webcam

- **All 6 servo IDs assigned** one at a time via `set_servo_id.py`; final
  read-only bus scan confirmed `[1, 2, 3, 4, 5, 6]` all present.
- **Power research, done properly (not from memory):** looked up actual
  datasheet numbers rather than guess, since wrong voltage risks damaging
  hardware. Waveshare ST3215 servo: 6-12.6V input (torque scales with voltage,
  ~19.5kg·cm @7.4V vs ~30kg·cm @12V), idle current ~180mA, stall current ~2.7A
  per servo. Waveshare Bus Servo Adapter (A): accepts 5-12.6V via a 5.5×2.1mm DC
  jack, must match the servo's chosen voltage; no published max-current rating.
  For all 6 servos together, worst-case theoretical simultaneous-stall draw is
  ~16A, but a practical target of 8-10A continuous is reasonable. For the
  ID-pairing step specifically (one unloaded servo, no motion), even ~1A of
  headroom is far more than the ~180mA actually drawn.
- **Clarified a real point of confusion:** a lit "power" LED on the adapter with
  only USB connected does not confirm the servo bus itself is powered — that LED
  likely just reflects the adapter's own USB-fed logic circuitry. Recommended
  testing directly (`--scan` with only USB connected) rather than trusting the
  LED.
- **Hot-swap safety:** recommended cutting power before disconnecting/
  reconnecting each servo during one-at-a-time pairing — these connectors aren't
  designed for hot-swapping under load, and reconnecting live risks a brief
  pin-bridging short or a voltage-spike kickback.
- **Webcam confirmed detected** at `config.CAMERA_INDEX = 0` via the project's
  own `Camera` wrapper — opened and captured a real frame at 1024×576.

---

## 7. Direction/scale verification jogs — J1 through J6

**Method, established and held to throughout:** move one joint a small, bounded
number of ticks (~100-150), with the operator physically watching, report back
direction and rough angle, then move to the next joint. Never jogged more than
one joint per confirmation.

Confirmed physical behavior:

> **CORRECTION (2026-08-04): the J2 row below is INVERTED.** A raw-tick jog
> during camera-based joint validation showed `-ticks` tilts the shoulder UP,
> i.e. `+ticks` tilts it DOWN. MATLAB FK independently predicts `+100 ticks ->
> wrist z -5.2 mm (down)`, so FK and the physical arm agree and
> `data/servo_calibration.json`'s `dir_sign = -1` for J2 is CORRECT — only this
> table is wrong. Do not re-derive anything from the J2 row; verify by jog.

| Joint | Role | +ticks direction |
|---|---|---|
| J1 | base yaw | counterclockwise (viewed from above) |
| J2 | shoulder | tilts up  *(WRONG — see correction above; +ticks tilts DOWN)* |
| J3 | elbow | folds down, toward the table |
| J4 | wrist pitch | down/toward base (ccw from the side) |
| J5 | wrist roll | ccw (viewed from behind the robot) |
| J6 | gripper | opens |

`ticks_per_rad` (the config placeholder, ~651.89 for J1-J5 / ~325.95 for J6) held
up as roughly correct throughout — no joint's observed rotation was wildly off
from the angle its tick delta implied.

**A process bug found along the way:** `ServoBus.move_and_verify()` reads back
the servo's position *immediately* after commanding a move, with no settle
delay — it threw `RuntimeError` on essentially every real move during these
tests (the servo hadn't physically arrived yet). Worked around in every ad-hoc
bring-up script with `time.sleep(1.5)` between the goal write and the
verification read. **Not yet fixed in the actual driver method** — see Outstanding
Items.

---

## 8. Two real incidents during bring-up

### Incident A — J3 / table near-collision

While jogging J3 (elbow) in the "+" direction (already known, per the table
above, to fold the arm down), the claw — already resting near the table — was
driven further down into it. Caught immediately by the operator watching
physically; power was cut before any damage. Confirmed J3's "+" direction is
unsafe from that starting pose. This was the first concrete case for the
one-joint-at-a-time, watch-and-confirm process actually paying off.

**Recovery:** with power off, the operator manually repositioned the whole arm to
a pose with more table clearance and asked to capture that as the new home. Built
a read-then-hold script: read each servo's current (manually-held) position,
immediately command it to hold exactly there (goal = present position, so ~zero
net travel — safe even with unverified calibration, since it's not traveling
anywhere), then write those readings into `data/servo_calibration.json` as the
new `home_tick` per joint. One transient no-response read was hit on every joint
except J1 during this — consistently resolved on a single retry, treated as a
benign timing quirk rather than a wiring fault (it never failed a second time on
any joint).

New home positions captured: J1=4086, J2=2951, J3=1054, J4=1673, J5=2494,
J6=2741 (this was *before* the J1 fix in Incident B — J1's home moved again after
that).

### Incident B — J1 encoder wrap-seam runaway

While returning J1-J5 to their just-captured home positions, J1 was commanded
from a fresh read of **17 ticks** to its stored home of **4086** — a difference
the return-home script didn't check before writing the goal. J1 began a fast,
large rotation and the operator had to cut power before it reached a mechanical
limit.

**Root-caused, with the operator's own observation ("J1 barely moved") supplying
the key clue** that the initial explanation (position had genuinely drifted a lot)
was wrong. Correct diagnosis: J1's home (4086) sat only 9 ticks below the
encoder's 0/4095 wrap point. A physically tiny move (~26 ticks, ~2.3°) crossed
that seam and the *reading* flipped from 4086 to 17 — a huge apparent jump for a
tiny real one. The servo, in single-turn position mode, cannot cross the
dead-zone at the seam directly; commanded to "return" to 4086 from a reading of
17, it instead took the long way around: ~4069 ticks, ~358°, at speed. Confirmed
this wasn't an encoder reset by pointing to earlier-session evidence: other
joints read their real non-zero home values after previous power cycles, not
zero — ruling out "power-off resets position" as an explanation.

Two compounding causes identified: (1) home parked on the worst possible spot on
the encoder, and (2) the return-home script had no cap on commanded move size —
unlike the jog script used for section 7, which had refused any move over 300
ticks without an explicit override.

**Fix — re-centering, not just re-homing:** used the Feetech "one-key midpoint
calibration" (write `128` to the Torque Enable register, address 40, while the
EEPROM is unlocked) with J1 physically at its home pose. This redefines the
servo's *current physical position* as tick 2048 — dead center, as far from the
wrap seam as geometrically possible — without commanding any motion (verified:
before/after reads showed no physical movement, only the reported tick value
changed). J1 before: 4052 (itself ~34 ticks off the earlier 4086 capture — the
seam instability in action). After: exactly 2048. `data/servo_calibration.json`
updated, J1 `home_tick` 4086 → 2048.

**Recovery of J2-J5:** built a homing script with an explicit hard safety cap
(350 ticks) that reads every joint first, computes the delta to its stored home,
and *skips* (does not command) any joint whose delta exceeds the cap rather than
executing it — the exact protection missing during the incident. All of J2-J5
were within cap and homed cleanly; largest correction was J4 at ~268 ticks
(had drooped while unpowered during the J6 claw handling in section 9 below).
J1 was left untouched (already re-centered) and read a stable 2047 (home ±1) —
direct confirmation the seam problem is gone.

---

## 9. J6 (gripper) testing

Requested cutting power specifically to J6 to manually open the claw — clarified
there's no way to power down a single servo on a shared bus (would cut power to
all six); the operator handled this as a physical action on their end (power off
the whole bus briefly, open the claw by hand, power back on), consistent with
what was already established in section 6.

Jogged J6 a small amount (-100 ticks, toward the previously-known closed
reference at 2741) to confirm direction. First attempt wasn't visually observed;
repeated identically (return to start, then redo the same jog) so it could be
watched. Confirmed: **+ticks opens the claw**, motion was a small proportional
nudge, consistent with the operator's own earlier hypothesis (if oriented like
J4, the same-direction tick sign would open it).

---

## 10. Return-to-home procedure, hardened

Combining the lessons from Incident B, built and ran a script that:
1. Reads all 6 joints first (read-only, no motion).
2. For J2-J5 only, computes each one's delta from current position to its stored
   home tick.
3. Refuses (skips) any joint whose delta exceeds 350 ticks rather than moving it.
4. Moves only the joints within cap, with a settle delay before the verification
   read-back.
5. **J1 and J6 are read-only in this pass, never commanded** — J1 because it was
   just re-centered and any further correction should be deliberate, not
   automatic; J6 per explicit instruction, since its claw state was intentionally
   left as manually set.

Result: all four eligible joints (J2-J5) homed cleanly, largest move 271 ticks
(J4), well inside the cap — no repeat of Incident B.

---

## 11. Session's code pushed to GitHub

Before pushing, verified rather than assumed: located `git.exe` (not on PATH but
installed at `C:\Program Files\Git\cmd\git.exe`), confirmed the remote
(`https://github.com/shravan6d4/5-DOF-CV-Robot-Arm.git`, branch `main`, already
0 ahead/0 behind before this session's commits), reviewed the exact staged diff,
and specifically confirmed the machine-specific calibration files
(`data/servo_calibration.json` — today's measured home positions — plus the
placeholder `camera_intrinsics.json`/`hand_eye.json`) are excluded by
`.gitignore` and did **not** get staged, by design (they're specific to this
physical arm, not meant to travel with the repo).

**Committed and pushed** (`6818625`, 11 files changed): the servo bus ID-write
tooling and its tests (section 1), the ChArUco board generator/config changes
and calibration scripts (section 2), `run_two_view_pick.py` and its config gates,
`capture_servo_home.py`, and the `CLAUDE.md` updates describing all of it.

**Explicitly not pushed** (by existing, deliberate `.gitignore` policy): the
calibration JSON with today's measured servo homes and the J1 re-center. That
data currently lives only on this machine.

---

## 12. Interrupted: designing a safety cap into the real driver

Immediately after the J1/J3 incidents, the explicit follow-up ask was to design
the missing safety cap and settle-delay fix directly into
`ServoBus.move_and_verify()` (the method `HardwareRobot` actually calls for real
IK-driven moves) rather than leaving those protections only in throwaway
bring-up scripts. Entered plan mode for this; drafted a detailed Plan-agent brief
covering: a configurable per-command tick cap with a refuse-not-clamp exception,
a staged/incremental-move path for legitimate large moves, the settle-delay +
read-retry fix, and test coverage using the existing `FakeServoSerial` harness.

**This was interrupted before the agent ran and before any plan was finalized.**
No code changes were made for this item. It remains the most important piece of
unfinished business from the session — see Outstanding Items.

---

## 13. README updated with a condensed bring-up log

Added a "Hardware bring-up log — servo calibration (2026-07-22)" section to
`README.md` covering the pairing, power findings, direction/scale table, both
incidents and their fixes, the home-capture process, and the outstanding items
below. Left the rest of the (already-known-stale, per `CLAUDE.md`) README
untouched, as that's a separate cleanup from what was asked.

---

## Outstanding items (not yet done)

1. **Safety cap + settle-delay fix, in the real driver.** `ServoBus.move_and_verify()`
   still has neither protection today — only the ad-hoc scratchpad scripts used
   during bring-up did. This needs to land before `HardwareRobot` ever drives the
   arm via the MATLAB IK bridge for real. (Section 12 — was interrupted mid-design.)
2. **`dir_sign` reconciliation.** Every joint's calibration still has the
   placeholder `dir_sign = +1`. The physical directions confirmed in section 7's
   table need to be checked against MATLAB's own positive-rotation convention
   (`matlab/init_arm.m`, `homeAngles`) before any real IK-computed angle can be
   trusted to drive the right physical direction. Desk exercise, no hardware
   needed.
3. **J6 travel range calibration.** Direction confirmed, but the actual
   open/closed tick range hasn't been measured against `config.SERVO_GRIPPER_OPEN_RAD`
   / `SERVO_GRIPPER_CLOSE_RAD`.
4. **Repo folder rename** — investigated, not decided or performed (section 4).
5. **camera/hand-eye calibration** — scripts exist and were written/tested this
   session (see `CLAUDE.md`), but have not been run against the real camera/arm
   yet; needs the arm to actually be drivable first (item 1-2 above).

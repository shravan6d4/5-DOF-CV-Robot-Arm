# Session Log — 2026-08-04

A record of a single working session that set out to finish hand-eye calibration
(Merge Path step 2) and instead spent itself diagnosing why hand-eye kept failing,
then pivoted to a hardware-only pick attempt. Ends with the hand-eye failure
traced to a probable servo sign error, the pick blocked on joint travel limits,
and four new safety mechanisms in the driver after an arm drop damaged nothing
but cost a servo its torque.

Kept as a narrative log rather than a transcript — organized by topic, in the
order it happened, with what was asked, what was found, and what was done.

See also: `CLAUDE.md` (architecture, safety conventions), `SESSION_LOG_2026-07-22.md`
(the previous bring-up session).

---

## 1. Hand-eye calibration attempts — repeated failure

Two full capture sessions produced solves whose camera offsets were nonsense
(cross-board spread 260–460 mm where <10 mm is needed). Fixes applied along the
way, each a real defect:

- **The TSAI-vs-PARK agreement gate only compared translation.** Both methods
  could converge on the same badly-wrong *orientation* and report "ok".
  Rotation disagreement is now checked too (`tsai_park_rotation_deg`).
- **No raw-sample persistence.** A session's work was lost if the solve was bad.
  `save_samples` / `load_samples` plus `--samples` / `--fresh`.
- **Stale frames and an unsettled arm.** Samples were recorded before the arm
  stopped moving and from buffered frames. Now: settle, drain 6 frames, detect,
  re-read joints, discard if anything moved. Data quality went from ~3°/68°
  gripper-vs-camera rotation mismatch to **median 0.3°, max 1.8°**.
- **Autofocus was enabled.** Refocusing moves the lens and changes the focal
  length, which invalidates fixed intrinsics. Disabled and focus pinned
  (`config.CAMERA_FOCUS`).

**None of it fixed the solve.** Quantifying the autofocus effect afterwards
showed only ~1% fx change over the 200–500 mm working range, ~3–8 mm of error
against the 45–75 mm observed — roughly an order of magnitude too small to be
the cause. It was retained as a genuine but minor defect.

### The anomaly that resisted everything

> FK and the camera agreed on rotation **magnitude** (0.3° median) but disagreed
> on rotation **axis** (Kabsch residual 25–40°).

For a rigidly-mounted camera both must hold. Eliminated by measurement, each in
turn: board square size (ruler-confirmed 26.7 mm), camera resolution, planar pose
ambiguity (board normals parallel to 0.9°, IPPE ≡ ITERATIVE), outlier samples,
stale frames, MATLAB model joint axes (J2∥J3∥J4 to 0.0°), `ticks_per_rad`
(FK 8.61° vs camera 8.43°), and joint zero offsets (two separate objectives, both
failed).

## 2. Joint travel limits — measured and enforced

Two servos jammed during the session (J3, then J4), each ending with a stalled
servo straining against a hard stop until power was cut. Root cause: `ServoBus`
capped how FAR one command travelled but had no idea WHERE a joint could go, so
small legal steps could walk a joint into a stop.

- `_check_travel_limits` enforces absolute `min_tick`/`max_tick` per servo. A
  joint already out of range is not trapped — moves that *reduce* the violation
  are allowed.
- `scripts/find_joint_limits.py` measures a joint's travel in small
  operator-confirmed steps, watching for the 4095→0 encoder wrap throughout.
- `matlab/init_arm.m` reads `data/joint_limits_rad.json`, so the IK solver stops
  producing solutions the arm cannot reach.
- `scripts/goto_tick.py` walks one joint to a target in sub-cap increments.

**Two bugs found in the limit tool itself.** It judged travel from
`move_and_verify`'s return value, which is stale mid-travel (the settle check
trips during the servo's acceleration ramp) — it recorded J2 as 11° of travel
when the real figure was ~51°. Fixed with a settle-then-read helper and a
two-consecutive-under-travels rule before declaring a stop. It also refused to
re-measure a joint that already had limits recorded, which prevented correcting
a bad measurement.

**Recorded:** J2 `[2883, 3468]` (51°), J3 `[200, 1096]` (79°). J1, J4, J5, J6
still unmeasured.

**J3's lower limit was originally measured at tick 6** — sitting on the encoder
seam, the same geometry that caused the J1 runaway in July. Pulled in to 200
(~18° of margin) because the arm has no need for that region; recorded in
`limit_basis` so nobody later reads +75° as the mechanical stop. The limit tool
now warns when a *measured* limit lands near the seam, which it previously only
did for the starting position.

### Ground collision is not a joint limit

J2's measured range is only ~51° of its real travel because the claw grounded out
at the elbow angle used when measuring. Fold the elbow differently and the same
joint angle is perfectly safe. A Cartesian floor is the honest form of that
constraint: `config.MIN_CLAW_HEIGHT_M` (5 mm), enforced in
`HardwareRobot.send_target_pose`. Deliberately below `PICK_Z_OFFSET` (10 mm) so
it cannot refuse the pick it exists to protect.

Both limits are annotated with `limit_basis` recording that they are
ground-derived and pose-specific.

## 3. Pivot — an actual pick attempt

With calibration unresolved, the operator chose to attempt a pick with a hand on
the power switch and learn from the result.

### Zero-motion dry run first

`scripts/test_pick_dry_run.py` walks the entire pick path and commands nothing.
A wrong hand-eye transform does not fail loudly — it yields a confident,
well-formed `PickTarget` and `run_once` would drive straight to it.

Result against the real arm, brick at a ruler-measured (280, 35):

| | |
|---|---|
| Pipeline said | (+641, +210) mm |
| Ruler said | (+280, +35) mm |
| Optical axis | 91° off the claw direction |
| `Pose` round-trip error | 2.22e-16 at pitch −23.4° — **clean** |

The gimbal-lock path that poisoned the hand-eye solve is *not* contaminating the
pick at this working pose.

### The board measured the camera directly

The ChArUco board was in frame, lying flat on the table, which gives an absolute
camera pose independent of FK and hand-eye:

| | camera height above table |
|---|---|
| Board (`solvePnP`) | **228.0 mm** |
| FK + hand-eye | 279.9 mm |

**52 mm out.** The stored transform is wrong in translation as well as rotation.
Notably this test uses no ruler and no assumption about the base origin, so it
condemns hand-eye regardless of where the base column actually is.

### Reach, not calibration, blocks the pick

Mapping the reachable envelope at the brick's height showed the arm reaches only
**~180 mm forward at table level** (220 mm at z=+80). At x=180 both J2 and J3 are
pinned at their recorded minimums — so the blocker is the *ground-derived joint
limits*, not the arm's geometry and not calibration. The brick was moved to
~145 mm, where nothing binds.

## 4. Incident — power cut dropped the arm, J3 lost torque

A commanded move ran at the servo's default full speed rather than the paced
stepping the operator expected, and power was cut mid-move. Cutting power drops
holding torque on **every** joint simultaneously; the arm fell face-first and
back-drove J3 hard enough to trip its overload protection.

**J3 then looked dead but was not.** It answered the bus at 7.4 V, 33 °C, with no
fault flags — and `torque=OFF` while every other joint read `ON`. Gravity had
moved it 54° (296 → 915 ticks) while limp. Torque Enable is the one register that
distinguishes an overload trip from a dead servo, and the driver never exposed it.

Four gaps, all fixed:

- **No speed control.** The driver only ever wrote Goal Position, so every move
  ran at full default speed and ended in a hard stop, putting peak torque far
  above what the pose needs statically. `set_speed` / `set_acceleration` write
  the SRAM registers at 0x2E / 0x29. Being SRAM they reset on every power cycle
  and must be re-applied per run.
- **No software stop.** Once a Goal Position is written the servo travels there
  whether or not anything is still talking to it — Ctrl-C did not stop the arm,
  so the power cut was the only option, and it drops the arm. `ServoBus.freeze`
  overwrites each goal with the joint's present position: motion halts, torque
  stays on, nothing falls. Wired to Ctrl-C in `goto_point.py` and available
  standalone as `scripts/freeze.py` from a second terminal.
- **No torque visibility or recovery.** `read_diagnostics` exposes voltage,
  temperature, load, faults and Torque Enable. `enable_torque` pins the goal to
  the **present** position before enabling, because the servo's stale goal would
  otherwise snap it back at full speed from a pose nobody chose.
  `scripts/servo_torque.py` drives both.
- **No motion pacing.** Operator requirement: all pick-path motion in hops of
  ±60 ticks with 0.5 s between. `ServoBus.move_joints_stepped` implements it,
  all joints advancing together each round — driving joints to their targets one
  at a time takes the arm through poses nobody planned.
  `HardwareRobot.send_target_pose` routes through it, so the real pick path and
  the bring-up scripts share one implementation.

`HardwareRobot` also now applies the motion profile at init; it never had one, so
real picks ran at full speed regardless of what the scripts did.

## 5. Probable root cause of the hand-eye failure — J5 `dir_sign`

The 49 saved samples still had usable joint angles (recovered by inverting the
stored FK). Re-solving them under all 32 possible `dir_sign` combinations, using
the Kabsch axis residual as the statistic:

```
as recorded  (+1 +1 +1 +1 +1)   39.0 deg   <- no rigid transform explains this
flip J5 only (+1 +1 +1 +1 -1)    3.4 deg   <- consistent
flip J2/3/4/5(+1 -1 -1 -1 -1)    3.5 deg   <- same solution in disguise (J2||J3||J4)
```

**J5's `dir_sign` is probably inverted.** It fits everything: the camera is
mounted on J5 (it visibly swings with it), so an inverted sign means FK believes
the camera rotated one way while it physically rotated the other — right
magnitude, wrong axis, exactly the anomaly. It also explains why Stage D passed:
vertical moves barely exercise J5.

**Unresolved conflict.** The operator twice observed J1 physically turning the
wrong way, but every J1-flipped combination scores ≥19°. Caveat on the data side:
the recovered angles came from inverting FK, and a 5-DOF arm has multiple joint
solutions for one tip pose, so the recovery may not have landed on the original
branch — in which case the J1 column means nothing. Caveat on the observation
side: J3 was limp for part of the second run, so the arm's shape did not match
any command. **Not settled. Needs one clean slow jog of J1 and J5 each**
(`scripts/jog_joint.py`), which is now safe at the capped speed.

## 6. A hand-eye-free path to the pick

`scripts/calibrate_board_to_base.py`. A brick's pixel can be intersected with the
**board plane** using only intrinsics and `solvePnP` — the camera's base-frame
pose never enters, which is what makes it immune to the hand-eye problem.
Verified working on the captured frame: the brick landed at square (9.9, 4.4) of
a 10×6 board, 253 mm from the camera, physically on the board as it must be.

What remains is where the board sits relative to the robot, and since it lies
flat that is only (x, y, yaw). Preferred input is `--touch`: jog the claw onto a
named ChArUco corner, whose board coordinates are known from the printed
geometry, and read FK. **Neither a ruler nor the operator's idea of the base
origin enters** — which matters, because a ruler measured from an assumed origin
that is off by some offset yields a transform in the operator's frame, and the
arm then misses every pick by exactly that offset, silently.

Bring-up scaffolding, not the merge path: it needs the board in view and sits
beside `PickPipeline`.

## 7. Pushed

Branch `hand-eye-diagnostics`, 15 commits, pushed to GitHub. `main` untouched at
`3b55ee7`. No calibration data committed — `.gitignore` covers intrinsics,
hand-eye, servo calibration, joint limits, samples, and the board→base tie, all
machine-specific.

---

## Outstanding

**Blocking a pick:**
- J3's torque needs re-enabling (`scripts/servo_torque.py --enable 3`), and
  watching for a repeat trip, which would mean damage rather than overload.
- The brick's **y** offset needs re-measuring; the 35 mm figure predates moving
  it closer.

**Blocking hand-eye:**
- J5 and J1 `dir_sign` unresolved (§5). One clean jog each settles it. If J5 is
  confirmed inverted, the 49 saved samples become consistent with no recapture.
- Camera intrinsics may want redoing now that focus is pinned — the 56-view July
  run had autofocus enabled.

**Not started:**
- J1, J4, J5, J6 travel limits unmeasured.
- J6 gripper open/closed tick range against `SERVO_GRIPPER_OPEN_RAD`/`CLOSE_RAD`
  (deliberately deferred by the operator).
- `PICK_Z_OFFSET` / `APPROACH_HEIGHT` still nominal.
- J2/J3's *mechanical* limits, as opposed to the current ground-derived ones.
  Re-measuring with the elbow folded so the claw cannot reach the ground would
  give back workspace out to ~280 mm, with the Cartesian floor guard handling
  ground collision — the honest model, since it depends on the whole arm's
  configuration rather than any one joint.

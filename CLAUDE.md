# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Computer vision pipeline (OpenCV, classical CV — no ML yet) for a **5-DOF** robotic arm that picks up a single red Lego brick from a table. The camera is **eye-in-hand** (mounted on the arm). The arm's own code does not exist yet; this repo is the vision side, built so the robot code merges in by implementing one interface. Development is on Windows / PowerShell.

The full path is implemented end-to-end against a **simulated** robot: detect brick → back-project its pixel to a table coordinate in the robot base frame → plan a top-down grasp → command the arm. What is *not* real yet: the calibration numbers (camera intrinsics + hand-eye transform are placeholders) and the robot backend (only `SimRobot` exists). See **Merge path** below — those are the two things to fill in.

See [README.md](README.md) for a quick project overview, setup, and the command list; this file goes deeper on architecture, the data flow, and the merge path onto real hardware. Session narratives live in `SESSION_LOG_2026-07-22.md` (bring-up) and `SESSION_LOG_2026-08-04.md` (hand-eye diagnosis, joint limits, the arm-drop incident).

**Current state in one line:** the vision pipeline is merge-ready and fully tested; **the arm is now built from a RULER SURVEY rather than the CAD import, which was the wrong shape by up to 68 mm** (see the section directly below), and all five `dir_sign` values are confirmed by physical jog; **`data/hand_eye.json` is still wrong and no saved capture can fix it** — every attempt so far rotated about too few axes to observe the camera's position (see "One diagnostic" below), so the open-loop pick path stays blocked. The CLOSED-LOOP path (`scripts/visual_servo.py`) needs no hand-eye and is the way forward; a pick is additionally limited by ground-derived joint limits capping forward reach at ~180 mm.

## COORDINATE FRAMES — the imported model is upside-down (read before touching kinematics)

**The single most important fact about this system, stated by the operator and verified
physically on 2026-07-22: the Simscape/`importrobot` model's base frame is FLIPPED relative to
the physical robot.** In the model's own coordinates the arm reaches *upward* (claw tip at
z = +63 mm above the base origin — as if picking objects off a ceiling). The physical arm does the
opposite: the claw hangs *down* toward the table the robot stands on. Every past error in this
area came from assuming the model frame is the physical frame.

**The exact relationship** (a 180° rotation about the shared X axis; self-inverse, so the same
formula converts both directions):

```
physical (x, y, z) = model (x, −y, −z)
model +X = physical forward (unchanged) · model +Y = physical RIGHT · model +Z = physical DOWN
```

Definitions, so language stays unambiguous:
- **"The table"** = the horizontal surface the robot stands on = the plane motor 1 (J1) sits on.
- **Physical frame** = X forward (the way the claw points at home), Z up (against gravity),
  Y left (right-handed). The base origin sits **~73 mm above the tabletop** (inside the base
  column, near shoulder height). `TABLE_Z_IN_BASE ≈ −0.073` (physical; estimate ±5 mm).
- At home, physically: wrist ~+4 mm above origin, claw tip ~63 mm **below** origin, ~10 mm above
  the table. **The claw tip is always below the wrist in any sane pose** — if "physical" numbers
  ever show the tip *above* the wrist, a frame conversion has been dropped somewhere.

**Where the conversion lives — one seam only:**
[`MatlabIKClient`](src/vision_pipeline/robot_interface/matlab_client.py) converts at the wire:
`request_ik` negates y,z of the target on the way in; `request_fk`/`request_fk_tip` left-multiply
returned transforms by `diag(1,−1,−1,1)` on the way out. Consequently **everything Python-side
speaks the physical frame** (HardwareRobot, pipeline, planning, all scripts, all config
geometry values), while **everything MATLAB-side speaks the model frame** (`ik_fk_server.m`, the
raw TCP JSON protocol, `test_ik_fk.m`, `IKtrials_v2.m` — including their target names: the
battery's "low" target is model-low, i.e. physically *high*). Never convert anywhere else;
never convert twice. Joint *angles* are scalars and are NOT affected by this Cartesian flip —
they pass through unchanged; only Cartesian poses/targets convert.

**Past mistakes this section exists to prevent (do not reintroduce):**
1. `TABLE_Z_IN_BASE = 0.0` — original placeholder, wrong by ~73 mm.
2. `= −0.084` — assumed the tip hangs `CLAW_LEN` straight below the wrist *in the model frame*.
3. `= +0.053` — used the model-frame tip z directly, concluding the tabletop is above the base
   origin. It is not; the frame is flipped.
4. Deriving joint `dir_sign` values by interpreting model-frame FK axes as physical directions
   ("viewed from above", "the left side") without applying the flip — this silently inverts
   every such conclusion. All dir_sign reasoning must be done in the physical frame (or, better,
   confirmed by a physical jog through `scripts/jog_joint.py`, whose predictions are physical
   now that the client converts).

**60-second re-verification recipe** (arm powered, MATLAB server up):
`python scripts/check_servo_health.py` → the B4 block must show the claw tip *below* the wrist,
and `tip z − (measured gap under the claw)` must land near **−74 mm** whatever pose the arm is
in. Both are pose-independent invariants; the absolute numbers are not (they move whenever the
arm does, which is what made three earlier `TABLE_Z_IN_BASE` values wrong). If the tip reads
*above* the wrist, or the tabletop lands positive, the conversion seam has been broken.

## The base yaw axis is NOT the model origin — it misses by 81 mm (2026-08-06)

[`scripts/audit_model_axes.py`](scripts/audit_model_axes.py) reads the imported model's kinematics back out of its own FK: rotating joint *k* moves the wrist by `M = A·Rot(θ)·A⁻¹`, a pure rotation about that joint's axis in the base frame, so `geometry.screw_axis(M)` recovers both the axis DIRECTION and WHERE IT IS. Two FK calls per joint, no model file parsing.

**The model is a structurally correct arm.** Yaw ⊥ three parallel pitches ⊥ wrist roll; J2↔J3 = 102.7 mm, J3↔J4 = 136.2 mm, J5→tip = 70.1 mm — **all three ruler-confirmed on the physical arm**, and J5 confirmed by eye as a roll (`init_arm.m` called it "wrist pitch"; that was a comment mislabel, now fixed). J2, J3 and J4 are parallel to **0.0°**, so the arm has only **two independent rotation axes**: the pitch chain and the J5 roll. That matters for hand-eye capture — "jog a different joint" means J5-vs-pitch, not J3-vs-J4.

**But the CAD origin is not the base column.** J1's axis passes **81 mm** from it, while at home the claw tip sits only **25 mm** from that axis. So `tip_xy / |tip_xy|` — "radial", the direction the arm reaches — comes out up to **108° from truly outward**, and `hypot(tip_x, tip_y)` reports 70 mm of reach where there is 25 mm.

That single mistake was live in three places:
- `jog_joint.describe_motion` announced "1.9 mm left" for pure pitch joints that **cannot move sideways**, which is what the operator caught by watching a J3 jog. It now decomposes about the JOINT'S OWN axis (`screw_axis` of the two predicted poses): a pitch reports out/up with a sideways term of exactly 0.0, and only a yaw reports left/right. `describe_rotation` likewise describes a roll by spin sense about the claw's own direction — base-frame naming called the J5 roll "tilting back/up", a pitch.
- `planning.visual_servo.radial_tangential` — inside the descent, so a reach correction pushed partly sideways and the sideways correction partly reached. **The two axes of that loop were fighting each other by construction.** `yaw_axis_xy` is now a REQUIRED argument, deliberately with no origin-defaulting overload, and `reach_from_axis` replaces every `hypot(tip)`. The "too close in for Cartesian control" guard was wrong in the dangerous direction — overstating reach, hence reporting better conditioning than the arm had.
- `MatlabIKClient.base_yaw_axis_xy()` is the one place it is measured (jog J1 4°, read the axis back), cached per session, so it cannot go stale if the model or frame convention changes.

**Also open: at all-zero joints the arm reaches along bearing −89.4°**, not +X, so every Cartesian x/y target is rotated ~90° from the arm's own forward. Stage D only ever validated the VERTICAL axis, which a rotation about Z leaves untouched — which is why nothing has caught it. **Hand-eye is immune** (`AX=XB` uses relative gripper poses, and a fixed base-frame change cancels exactly), so this is not the calibration failure. It does mean `goto_point.py --x` does not drive the direction it names. Unfixed: the visual-servo path does not use base X/Y (it works in radial/tangential, now correct), so it is not blocking a descend run.

### "Radial" was not a direction the arm could reach in (2026-08-07)

The yaw-axis fix above was necessary and not sufficient. Radial was still *inferred* — the horizontal direction from the base yaw axis out to the tool — and that is the direction the arm reaches **only when the tool is well away from that axis**. Reach comes from the **pitch chain**: J2/J3/J4 are parallel, so they move the tip in one fixed vertical plane, and J1 alone decides that plane's horizontal bearing. Nothing makes it point away from the yaw axis.

At the hover pose it does not come close. The tool sits **8.7 mm** from the yaw axis, so "radial" is the bearing of a near-zero vector:

```
tip (+71.4, +17.7) mm    yaw axis (+77.8, +23.6) mm
computed radial   -137.8 deg
pitch chain        +94.4 deg      <- 52.1 deg apart
```

Every radial nudge asked for a component the shoulder/elbow chain could supply only 61% of, and IK made up the rest with the two joints that could: base yaw and wrist roll. A 9 mm ask came back wanting **260 ticks (23°) of J5**. `scripts/visual_servo.py:reach_axis_xy` measures the direction instead — perturb a pitch joint, watch the tip, two FK calls — using whichever of J2/J3/J4 gives the largest horizontal response, since at a folded pose J3 is nearly pure vertical (0.47 mm of 4.14) while J4 gives 3.37 mm. **Pose-dependent, so never cached**, unlike the yaw axis.

| 9 mm nudge along | pan | roll | verdict |
|---|---|---|---|
| old radial | 0.35° / 15.56° | **22.85° / 20.04°** | shrink |
| measured reach | 0.09° / 0.53° | **0.00° / 0.53°** | accepted |

**Tangential was already right** and stays geometric: computed −47.8° against J1's actual tip motion at −46.8°. It is perpendicular to the radius by construction, which is exactly what base yaw does, and that holds however close in the tool sits.

#### The pan budget was capping the one axis base yaw is FOR (2026-08-07)

**Tangential was geometrically right and administratively throttled.** The pan guard rejects a solve that spends more than `SERVO_VISUAL_MAX_PAN_DEG` (1.5°) of base yaw, because for a *radial* nudge — reach, which lives in the shoulder/elbow plane — any yaw is redundancy the solver spent uninstructed, and the camera on the wrist pays for it. Applied to a *tangential* nudge that reasoning inverts: J2/J3/J4 are parallel pitches confined to one vertical plane and J5's lever arm is the shortest on the arm, so **sideways motion of the claw is what J1 is**, and the angle is fixed by geometry rather than chosen:

```
theta = d / r        d = the nudge, r = tip radius from the yaw axis
```

No solution uses less. A flat angle ceiling therefore does not limit waste there — **it limits the step size, to `MAX_PAN_DEG · r`,** while reporting the optimal solve as "over budget".

Measured mid-descent at r = 205 mm. The loop asked for 12 mm against a 55 px error — the right amount, as the part that did execute later confirmed at 4.7 px/mm:

| request | pan needed | |
|---|---|---|
| 12 mm | 3.35° | REFUSED, halve |
| 6 mm | 1.68° | REFUSED, halve |
| 3 mm | 0.84° | accepted — moved **14 px of the 55** |

The descent was injecting **~14 px of sideways error per step by itself**, visible in steps 1–5 before the sideways loop had engaged at all (+40, +29, +25, +10, −9 px). So the corrector was pinned at exactly break-even, the error never closed, and the `ProgressMonitor` stopped the run — correctly, for a cause it could not see. **The operator's report is the tell: J1 turned 0.79°, three times, invisibly, on a loop asking to turn it four times as far.** A correction that is capped rather than wrong-signed looks identical to a dead axis from outside.

Note also that the guard's own error message names 130 mm as the radius that fixes the conditioning problem, and the arm was at **205 mm** — well inside the regime the guard calls clean. It was not diagnosing conditioning; it was rate-limiting.

`CartesianActuator._pan_budget_deg` now gives tangential `d/r · SERVO_VISUAL_TANGENTIAL_PAN_SLACK` (1.4), floored at the flat ceiling so the fix can only loosen, capped at `SERVO_VISUAL_MAX_TANGENTIAL_PAN_DEG` (8°), and falling back to the flat ceiling below `SERVO_VISUAL_MIN_TANGENTIAL_RADIUS_M` where `d/r` explodes. **The slack is what still catches the case the guard was written for** — near the base axis the claw's 27 mm offset makes the tip's bearing hypersensitive to J1 and a solve wants many times `d/r`. Pinned by `test_a_tangential_nudge_may_spend_the_yaw_its_geometry_requires` (full 12 mm executes, above the flat ceiling) and `test_a_tangential_solve_that_wastes_yaw_is_still_shrunk` (3× the requirement executes under a quarter).

**Still open from that run: the descent's sideways coupling is unmodelled.** `DescentModel` fits the vertical axis (`dpx = a·dz + b·dr`) and the sideways axis has no model at all — it is corrected reactively, one step behind. ~14 px per step is not noise, it is a systematic term, and it is large because the aim point sits 158 px off the optical axis where camera pitch produces x motion. The corrector can now out-run it; modelling it would mean not having to.

#### Never delete a joint from an IK solution

The same run produced a worse mistake, now reverted. The `lock` field is ignored by the server (below), so `visual_servo.py` briefly enforced it by simply not commanding the held joints — on the theory that J5 is a wrist roll and therefore pure null space. Both halves were wrong. **The claw tip sits off the roll axis, so J5 translates it**; and an IK solution is a *coordinated* answer — the other joints are where they are BECAUSE the deleted one was going to move.

| nudge | held | actually executed |
|---|---|---|
| radial −9 mm | J1, J5 | **−0.11 mm — 1.2% of the request** |
| radial +9 mm | J1, J5 | +1.23 mm — 13.7% |
| tangential −9 mm | J5 | **+15.17 mm — the wrong way** |

That is a centring loop commanding 9 mm, moving 0.1 mm, seeing no pixel response and asking again with the same numbers — which is exactly how the run stalled at 99 px with J3 requesting the same −18 ticks twelve times running. **The tell was the request not changing**: the loop re-reads the arm's angles every iteration, so an unchanging ask means an unmoving arm.

A solution is now taken or refused **whole**, with `SERVO_VISUAL_MAX_ROLL_DEG` joining the pan guard and shrinking the request until a solve fits. Deletion is pinned shut by `test_an_ik_solution_is_commanded_whole_never_censored`.

## THE ARM IS NOW BUILT FROM A RULER SURVEY, NOT FROM CAD (2026-08-07, LIVE)

`matlab/init_arm.m` has `USE_SURVEY_GEOMETRY = true` and calls
[`build_arm_from_survey.m`](matlab/build_arm_from_survey.m) instead of
`importrobot`. Flip the flag to `false` for the old behaviour; the legacy import
sits untouched below it. Python's twin is
[`kinematics/arm_model.py`](src/vision_pipeline/kinematics/arm_model.py).

**Why: the CAD was the wrong shape by up to 68 mm** — see the next section. **How
much better: seven held-out tabletop touches spread 6.0 mm instead of 73.1**, and
MATLAB reproduces Python to 0.01 mm on every one of them
(`matlab/test_arm_from_survey.m`, four checks, offline).

**Nothing in Python changed.** A base transform inside `init_arm.m` puts the tree
in the frame the project already speaks, so all 26 files that call the server are
untouched. Verified end to end, including `MatlabIKClient`'s flip:
`audit_model_axes.py` section E now prints **90 / 146 / 152 / 71.3 / 0**, which is
the ruler exactly.

**Two things that DID change, both deliberate:**
- **X and Y now mean the arm's forward and left**, not the CAD's axes, which sat ~89.4° off the arm's own forward — a known unfixed wart nothing correct depended on. `goto_point.py --x` therefore drives a different direction than before (a correct one). The visual servo is indifferent: it builds targets as `tip + delta` and measures its reach direction.
- **Z is matched exactly** and is the axis that mattered, being the one `target_z` and the floor guard are expressed in.

**J1's rotation sense is UNCHANGED** — operator-confirmed. That was the one thing
the rebuild could have silently inverted, since J1's `dir_sign +1` was jogged
against the old tree and a wrong sign on the sideways axis turns the servo loop
into a runaway. It did not flip; no `dir_sign` needs revisiting.

**The acceptance test, after any change to the geometry or the frame:** run
`python scripts/audit_model_axes.py` and check section E against the ruler. It
exercises MATLAB, the wire, the client's frame conversion and the audit's own
arithmetic in one go. Section D's moment-arm warning should stay silent.

## RULER vs MODEL: the imported geometry is wrong by up to 68 mm (2026-08-07)

Operator measurements at the home pose, heights above the tabletop, against what
`importrobot` produces. **`scripts/audit_model_axes.py` section E prints the model
column**; the ruler column is the arm.

| | model | **ruler** | error |
|---|---|---|---|
| servo 2 shaft (shoulder) | 158.2 | **90** | **−68.2** |
| servo 3 shaft (elbow) | 106.4 | **146** | **+39.6** |
| servo 4 shaft (wrist pitch) | 108.0 | **152** | **+44.0** |
| wrist (servo 5) | 71.3 | **71.3** | **0** |
| claw tip | 5.5 | **0** | −5.5 |

**The shoulder number is the one that admits no argument.** That shaft is bolted to
the base column: no joint angle, no `dir_sign`, no `ticks_per_rad` and no backlash
can move it, and it is pose-independent (verified at home, hover and an arbitrary
pose — always 158.2 in the model). The model is simply wrong about where the
shoulder is.

**The model has the arm the wrong shape.** It puts the elbow 51.8 mm BELOW the
shoulder; the arm has it 56 mm ABOVE. The upper arm points ~30° down in the model
and ~33° up in reality. That is what produces the moment-arm inversion above, and
hence the x2.17 jog discrepancy.

**Also note the claw tip at 0 mm: at home the claw is ON the table**, not ~6 mm
proud as long assumed. `TABLE_Z_IN_BASE = -0.0732` is unaffected — that value came
from the touch mean and the ruler pair, not from home's gap.

### Why no fix has been applied yet

**The fix belongs in MATLAB, not in a Python correction layer, and IK decides that.**
A Python layer corrects FK trivially, but IK runs *inside* MATLAB against the wrong
geometry: using it would mean inverting a pose-dependent correction around a remote
solver on every call, and every consumer would have to know which side of the
correction it is on. The whole motion path goes through IK.

What is missing is data, not intent. Five VERTICAL numbers cannot determine a link
transform's 3-D translation; pinning z while leaving x and y at whatever the CAD
says — the same source that got z wrong by 68 mm — yields a model correct in one
axis at one pose and unknown elsewhere.

Reconstructing it in Python from the ruler alone was tried (scratch `ruler_fk.py`):
the arm reduces to a three-term planar chain because J2/J3/J4 are parallel, and the
claw tip at 0 mm pins it with no fitting. On the seven held-out touches it gives a
**36.5 mm** spread against MATLAB's 73.1 — better, not right. A 3-D scan of the
tick→angle scales on top of it bottoms out at 27.7 mm while running to the edge of
its range, so no tick calibration rescues either geometry.

**Suspect the touch data has a floor.** Those touches were taken by hand-pressing a
limp arm onto the table, which loads every joint against its gear backlash — several
degrees on these servos, and at a 150 mm moment arm 3° is 8 mm, so 20–40 mm across
three joints is plausible. The ruler heights carry none of that, which is why they
are the data to rebuild on.

**To finish it, measure the same five points' FORWARD distance from the base
column's centre at home.** Ten numbers total determine the planar chain, and then
the corrected transforms can be written into `init_arm.m` with the touches as an
independent check.

## THE MODEL'S J2 MOMENT ARM IS WRONG — confirmed by jog (2026-08-07)

**The fault is in the imported model, not in any JSON.** `data/servo_calibration.json`
is now fully confirmed: all five `dir_sign` by physical jog, J3's `ticks_per_rad`
at ×1.02. What is wrong is where the model puts the claw relative to J2's axis.

Three independent confirmations:

1. **A jog measured it.** `jog_joint.py --joint 2 --ticks -150` predicted 13.8 mm of claw travel; the operator measured **30.0 mm** (×2.17). At 12.6° that needs a moment arm of **137 mm**; the model says **63.1 mm**. Understated by 74 mm.
2. **The model contradicts the chain order.** J2 is upstream of J3, which is upstream of J4, so J2 must have the largest moment arm to the claw — it carries everything beyond it. The model inverts this: at the hover, J2 38.5 mm against J3 116.5 and J4 103.4. `audit_model_axes.py` section D reports and flags this.
3. **The operator falsified it with a ruler.** Jogged the same 12° on J2 and J3 from the hover: the model predicts 8 mm and 24 mm. **J2 moved the claw further**, which is the opposite of the prediction and the correct physical behaviour.

**This explains the whole 2026-08-07 cluster of symptoms**, all of which resisted
every calibration-layer fix:

- the seven-touch plane spreading **73 mm** with a residual that tracked J2's ANGLE;
- FK's absolute height being wrong (53 mm at one pose, 111 mm at another) while its DIFFERENTIALS stayed good — a wrong moment arm scales displacement with joint angle, so a small jog looks fine and a large excursion does not;
- the descent driving past the tabletop until J2 stalled 34 ticks from its limit.

**Do not chase this in `servo_calibration.json` again.** It was scanned
exhaustively and refuted: `ticks_per_rad` scale and `home_tick` offset on J2/J3/J4
singly and as 2-D grids (all ran to the edge of their range), all 32 `dir_sign`
combinations, and gravitational sag (refuted — within one posture cluster the
moment arm swings 81.7 → 116.5 mm while FK z moves only 56.7 → 53.2). The fix is
in `Robomainassemjoints.slx` / `Robomainassem_DataFile.m`, or in how `init_arm.m`
maps servo 2 onto a model body.

**A METHOD NOTE WORTH MORE THAN THE FINDING.** Four statistical arguments agreed
with each other that J2/J3/J4 were sign-flipped — including an implied table
height landing within 0.8 mm of an independent ruler reading the fit never saw —
and all four were wrong. One jog refuted them in thirty seconds. The operator got
there from first principles instead: *J2 is earliest in the chain, so it must move
the claw more than J3 and far more than J4.* The numbers proving it had already
been generated (4.14 mm for J3 against 1.34 for J2 per 2°) and read past. Prefer a
structural check with a falsifiable ruler prediction over any fit to one data set;
this is the same lesson as the five hand-eye solvers agreeing at 80 mm against a
24 mm ruler.

## The touch plane said J2, J3 and J4 are sign-flipped — REFUTED (2026-08-07)

Seven touches of the tabletop across three distinct postures
([`scripts/measure_table_plane.py`](scripts/measure_table_plane.py)) are all one
flat plane, so FK must return one z for all of them. Under the stored signs it
spreads **73.1 mm**. Re-solved under all 32 `dir_sign` combinations:

```
  spread   dir_sign              implied table
     7.6   (1, +1, +1, +1, -1)      -73.2 mm     <-- 10x better
    73.1   (1, -1, -1, -1, -1)                   <-- stored, 21st of 32
```

**Four things agree, and only one of them is the fit.**

1. spread collapses 73.1 -> **7.6 mm**;
2. the implied table lands at **−73.2 mm** against the 2026-07-22 ruler pair's **−74.0 / −75.7 mm** — a measurement the fit never saw;
3. reach becomes physically sensible. Base −y is the arm's forward (the frame is rotated −89.4°, see above): flipped puts the touches **83–167 mm** in front of the base, stored puts them **8–27 mm**, i.e. inside the base column;
4. home's claw then sits **5.5 mm above** that table, matching the long-standing "~6 mm above the table".

It also corroborates the hand-eye `--search`, which ranked J3/J4-flipped best (board spread 10.0 vs 15.0 mm, TSAI-vs-PARK 1.2 mm/0.2° vs 2.5 mm/5.8°) and was recorded as a hypothesis. J1 is indifferent here — it cannot change tip height — and J5 is unresolved either way (7.6 vs 7.7 mm).

### …and the jog REFUTED it. The signs are right (2026-08-07)

`jog_joint.py --joint 3 --ticks 150` predicted, under the stored signs, that the
claw would move **16.1 mm in and 30.8 mm DOWN**. The operator reported it matched,
and volunteered the magnitude: **37.0 mm travelled against 36.1 mm predicted
(×1.02)**. So J3's `dir_sign` AND its `ticks_per_rad` are confirmed by direct
observation, and the flip hypothesis is dead — J3 and J4 are both jog-confirmed,
and flipping **J2 alone** makes things worse, not better (46.1 mm spread, and a
table at +116.7 mm, above the base origin, which is nonsense).

**This is why the rule exists.** Four mutually consistent statistical arguments,
one of them cross-validated against an independent ruler reading, all pointed the
wrong way. A single physical jog settled it in thirty seconds. Statistical
agreement among analyses of the same data set is not evidence — the same lesson
the hand-eye solvers taught (five methods agreeing at 80 mm against a 24 mm ruler).

**Gravitational sag was also refuted.** It predicts error tracking the moment arm;
within one posture cluster the arm swings 81.7 → 116.5 mm while FK z moves only
56.7 → 53.2. The residual tracks J2's ANGLE, not load.

**What is left, and it is the one untested parameter on the arm: J2's `dir_sign`
has never been jogged.** From `J1 2007 J2 3213 J3 2698 J4 1615 J5 2745`, with 613
ticks of headroom below:

| jog | `J2 = -1` (stored) predicts | `J2 = +1` predicts |
|---|---|---|
| **J2 −150** | **7.4 mm UP** | **13.6 mm DOWN** |
| J2 +150 | 4.3 mm down | 12.0 mm up |

`python scripts/jog_joint.py --joint 2 --ticks -150 --clearance-mm <measured>`.
Opposite directions again, so it cannot be misread.

**`TABLE_Z_IN_BASE = -0.0732` stands regardless**: the flipped fit gave −73.2, the
stored signs give −73.7 via home's 6 mm gap, and the ruler gave −74.0. Four routes
inside 2.5 mm.

## FK's ABSOLUTE height is not trustworthy; its DIFFERENTIAL height is (2026-08-07)

The operator touched the tabletop from several postures and read the ticks. Every
touch is the same flat plane, so FK must return one z for all of them. It does
not:

| touch | J2 | J3 | FK tip z |
|---|---|---|---|
| home (all joints zero) | 0° | 0° | −67.7 mm |
| C | −76.4° | +53.2° | **+4.6 mm** |
| A (board base) | −90.8° | +75.4° | **+37.3 mm** |

A and C are **37.7 mm apart horizontally** and **32.6 mm apart in FK z**. Including
home the spread is 105 mm. The postures differ by 107° of total joint travel while
the tip barely moves — the regime where kinematic error is most amplified.

**The fault is not in `servo_calibration.json`.** Every hypothesis at that layer was
scanned and refuted (scratch `fit_calibration.py`), each searched for a value that
puts BOTH touches on the plane home defines:

- `ticks_per_rad` scale on J2, J3, J4 individually, or all three together — best residual pair +63/+48, and the least-bad common scale (0.44) would mean 360° spans 1800 ticks on a 12-bit encoder;
- `home_tick` offset on J2, J3, J4 over ±600 ticks — best +55/+18;
- a free **two-parameter** J2×J3 scale grid — **zero** pairs fit both touches within 5 mm;
- all **32 `dir_sign`** combinations — best spread 42 mm, stored ranks 8th at 60.9 mm.

So the residual lives in the imported model's geometry or in an assumption about
it, not in the tick↔radian layer. What is NOT contradicted: link lengths (ruler),
`dir_sign` (physical jog), and **differential** height — Stage D's 45 mm commanded
lift moved the FK tip 40.7 mm against a 39 mm ruler reading.

**Consequence, and it caused the 2026-08-07 stall.** A descent commanded to an
absolute `target_z` is aiming at a number FK cannot deliver: the run drove down
past the surface until J2 reached 34 ticks of its limit and stalled, and the
stalled joint's current is what produced the checksum storm and the bus failure.
Prefer a descent expressed as "N mm from here" over one expressed as "to z = −57.7".

**One plane can refute but cannot identify.** Zhuang, Motaghedi & Roth, *Robot
Calibration with Planar Constraints* (ICRA 1999): a single-plane constraint leaves
the identification matrix rank deficient; **three mutually non-parallel planes** are
the minimum, and then only if the identification Jacobian is nonsingular and the
points on each plane are not collinear. So the scans above are eliminations, not a
fit. [`scripts/measure_table_plane.py`](scripts/measure_table_plane.py) collects the
touches and reports the spread; closing it needs two more surfaces (a book on edge,
a box side).

**The camera route is shut, for now.** With a correct hand-eye a board flat on the
table gives the plane for free — `solvePnP` for the board in camera coordinates,
FK @ hand-eye for the camera in base coordinates — and robot-world hand-eye
(`AX = ZB`, Zhuang/Roth/Sudhakar 1994; `cv2.calibrateRobotWorldHandEye`) solves
`Z` = base→world directly, whose translation IS the table height when the board
lies flat. Both depend on `data/hand_eye.json`, which is known wrong; `AX=ZB` on
the current samples returns 257 mm against a 24 mm ruler measurement.

### Sources

- H. Zhuang, S. Motaghedi & Z. Roth, [*Robot Calibration with Planar Constraints*](https://ieeexplore.ieee.org/document/770073/), ICRA 1999 — one plane is insufficient; three mutually non-parallel planes, non-collinear points, nonsingular identification Jacobian.
- H. Zhuang, Z. Roth & R. Sudhakar, [*Simultaneous Robot–World and Hand–Eye Calibration*](https://ieeexplore.ieee.org/document/704233/), IEEE T-RA 10(4), 1994 — the `AX = ZB` formulation whose `Z` is base→world.
- A. Li et al., [*Solving the Robot-World Hand-Eye(s) Calibration Problem with Iterative Methods*](https://arxiv.org/abs/1907.12425), Machine Vision and Applications 2017 — iterative solvers for the same, more robust than the closed forms.
- [*A novel robot calibration method with plane constraint*](https://arxiv.org/pdf/2208.02652), arXiv:2208.02652, and [SCALAR](https://arxiv.org/pdf/1803.00747), arXiv:1803.00747 — Levenberg–Marquardt identification under planar constraints, and how many planes each sensing modality needs.
- [MathWorks, *Estimate Pose of Moving Camera Mounted on a Robot*](https://www.mathworks.com/help/vision/ug/estimate-pose-of-moving-camera-mounted-on-a-robot.html) — the practical eye-in-hand workflow for taking a board on a table into the base frame.

## home_tick was wrong by up to 268 ticks, and nothing could see it (2026-08-05)

**`matlab/init_arm.m` defines home as all five joint angles ZERO** (`homeAngles = zeros(1,6)`, `HomePosition` reset to 0 per motor joint). FK there returns claw tip `(+70.0, −0.1, −67.7)` mm — 70 mm in front of the base, dead centre in y, hanging below the wrist. That is the arm's reference frame; `home_tick` in `data/servo_calibration.json` is the tick reading at that physical pose, and nothing else.

The stored values were wrong: **J1 −28, J2 −268 (−23.6°), J3 +179, J4 −126, J5 +250 ticks.** Corrected by parking the arm at the pose and reading `hold_pose.py`.

**Why no check caught it.** A `home_tick` offset cancels out of any *differential* measurement — Stage D's 45 mm lift (FK 40.7 vs ruler 39) is differential, so it validated `ticks_per_rad` and the link lengths while saying nothing about the origin. It does not affect `dir_sign` (a direction) or `ticks_per_rad` (a scale). B3's "home agreement" only compares ticks to the stored number, so a wrong stored number agrees with itself. Every automated check in the repo was blind to it.

**What did catch it, and the test to reuse:** at the operator's home pose the old calibration put the claw tip at z = **−75.5 mm**, against a table plane independently measured at −74.0 mm. *The claw cannot be through the table.* Under the corrected values it sits at −67.7 mm, i.e. **+6.3 mm above the table** — and lands dead centre in y, which a home pose must.

**Consequence:** every absolute Cartesian result taken before 2026-08-05 used a base frame offset by these amounts, **including the 53 hand-eye samples**. But those samples are *recoverable* — see below.

### The hand-eye samples were never bad data, only mislabelled (2026-08-05)

[`scripts/reinterpret_hand_eye.py`](scripts/reinterpret_hand_eye.py). Each sample is a pair, and **only one half was ever wrong**: `T_cam_board` comes from `solvePnP` on the board — camera and board only, no arm, no FK, no joint calibration. `T_base_gripper` came from FK of joint angles computed with a calibration that has since changed twice (`home_tick`, and J5's `dir_sign`). So the arm really was where it was; the code mislabelled which joint angles that pose corresponded to.

**The stored matrices cannot be patched by multiplying them by anything** — a joint-angle change maps to a pose correction that depends on the pose. The angles have to come back first: `T_stored → invert FK → θ_rec → (scale, offset) → θ_true → FK → T_fixed`. The inversion is well posed here, unlike the usual warning about 5-DOF position IK, because these store the full 4×4 **wrist** pose: a 6-DOF pose on the reachable manifold pins the configuration to a discrete set. All 53 recovered, worst reproduction error **0.0008 mm**, each seeded from the previous so the branch stays continuous.

**A calibration change is not always an offset.** `home_tick` *shifts* an angle; `dir_sign` *reflects* it:

```
θ_true = (d_now/d_cap)·θ_rec  +  d_now·(h_cap − h_now)/ticks_per_rad
```

Assuming the scale was always 1 made the first run of this fail (145.7 → 151.2 mm, no improvement) and briefly looked like proof the samples were junk. J5 was `dir_sign +1` when the samples were recorded and `−1` eighty minutes later, so its term is a reflection.

**Result — the referee is board spread**, which needs no arm and no ruler: the board never moved, so `T_base_gripper @ T_gripper_camera @ T_cam_board` must land on one base-frame point for every sample.

| | board spread | TSAI vs PARK |
|---|---|---|
| as recorded | 145.7 mm | 150 mm / 98° |
| corrected, stored signs | **15.0 mm** | 2.5 mm / 5.8° |
| corrected, J3+J4 also flipped | **10.0 mm** | 1.2 mm / 0.2° |

A tenfold collapse, and two independent solvers going from 98° apart to sub-degree. **This is what makes `hand_eye.json` recoverable without a recapture.**

**Do not use this to settle a `dir_sign`.** The rule stands: only a physical jog or a ruler can. `--search` ranking J3/J4-flipped best is a *hypothesis*, not a finding.

#### Solvers agreeing with each other is not evidence (2026-08-05)

The camera offset was measured with a ruler: **24 mm** from the wrist. Every solver OpenCV offers was then run on the corrected samples:

| | stored signs | J3/J4 flipped |
|---|---|---|
| TSAI / PARK / HORAUD (separable) | 136–138 mm | 80 mm |
| DANIILIDIS (dual quaternion, simultaneous) | 138 mm | 83 mm |
| ANDREFF (simultaneous) | **did not converge** | 72 mm |
| robot-world `AX=ZB` (different formulation) | 257 mm | 81 mm |

Under the flipped hypothesis five independent methods — including a *different problem formulation* — agree to within 11 mm. **And they are all wrong**: the ruler says 24 mm. Mutual agreement between solvers fed the same badly-conditioned data proves only that they share the same weakness. (The stored-signs column is separately damning: ANDREFF refusing to converge and `AX=ZB` landing 120 mm away means that hypothesis does not describe a rigid transform at all.)

**The root cause is the capture, not the solve.** Median rotation between poses was **15.5°**, with 2–7 mm of gripper translation. The hand-eye literature and [MVTec's HALCON docs](https://www.mvtec.com/doc/halcon/12/en/hand_eye_calibration.html) call for **≥30°, ideally 60°**, over ≥8 poses with ≥2 non-parallel rotation axes. Our axis spread was fine (90°); the rotation *magnitude* was half the minimum. Camera translation is recovered from how far the camera **swings** about the unknown offset, so small rotations leave it barely determined while every residual still looks healthy. Note also that TSAI/PARK/HORAUD are all *separable* (rotation first, then translation), so rotation error propagates straight into translation — `DANIILIDIS` and `ANDREFF` solve both at once and are reported as more noise-robust.

What is now in the code:
- `hand_eye.BoardResult.solutions` runs all five methods; `method_spread_mm` is their widest disagreement.
- `hand_eye.capture_health()` judges the **capture** — pose count, rotation magnitude against `config.CALIB_HAND_EYE_MIN_ROTATION_DEG`, axis spread — and is the check that would have caught this on the day.
- `hand_eye.offset_disagrees_with_ruler()` gates a solve against `config.HAND_EYE_EXPECTED_OFFSET_MM` (24 mm). **This is the only check in the whole pipeline that is independent of the data being solved.**

#### One diagnostic, and the two invariants no solver can flatter (2026-08-06)

[`scripts/hand_eye_report.py`](scripts/hand_eye_report.py) replaces `check_hand_eye.py` and a throwaway pair-residual script. Read-only unless `--fix`, so it is safe to run mid-capture. **Its six sections are in the order to trust them in**, and the ordering is the point: sections 2–4 need no solve at all, and a solver fed inconsistent data returns a confident wrong answer rather than an error.

**Screw congruence (Chen 1991) is the check the repo was missing.** `AX = XB` makes A and B conjugate, and conjugation preserves *both* screw invariants, so for every pose pair — whatever X is:

```
angle(A) == angle(B)      how far it turned
pitch(A) == pitch(B)      how far it slid ALONG that same axis
```

Only the angle half was ever tested. Adding pitch (`geometry.screw_pitch`) immediately caught four pairs the angle check passed cleanly, one of them at **0.06° angle error with 115 mm of pitch error**. Angle is blind to any fault that turns the right amount about the right axis and slides the wrong way — a misdetected board, a stale frame.

**The two halves indict different subsystems, and that is what localises a fault.** Angle depends only on rotations, so every board in one capture must agree on it. Pitch also depends on translations, which are measured per board. Within one 2026-08-06 capture:

| board | n | angle median | pitch rms | pitch correlation | |
|---|---|---|---|---|---|
| 1 | 9 | 1.12° | **6.1 mm** | **+0.995** | clean |
| 2 | 22 | 1.08° | 63.8 mm | +0.631 | inconsistent |
| 3 | 7 | 0.97° | 147.3 mm | **−0.663** | **mirrored** |

Same arm, same FK, angles agreeing to ~1° as they must — while pitch varies 24× across boards. **A fault that varies board to board while the arm is held constant cannot be in the arm.** A *negative* correlation means the camera's translations run opposite to the arm's, which no rigid transform can do but `solvePnP`'s two-fold planar ambiguity can: it reflects a flat board's normal, preserving rotation magnitude and mirroring translation. `CongruenceQuality.mirrored` reports it.

**`select_best` ranked boards by SAMPLE COUNT, so every tool had been solving the worst board available.** It now takes the accumulator and ranks by consistency first, count second — 22 mutually contradictory samples are worth less than 9 consistent ones, and no solver will tell you which you have. Callers must pass the accumulator or it silently falls back to counting.

**Outliers are named by consensus over ALL pairs, not neighbours** (`congruence_disagreement` / `congruence_outliers`). Congruence holds between *any* two poses, so capture order carries no meaning; worse, an isolated bad consecutive pair implicates both endpoints equally. Scored against every other sample a real culprit disagrees with nearly all of them (`> config.CALIB_HAND_EYE_MAX_DISAGREEMENT`, 0.5) while its innocent neighbour disagrees only with it. This is the model-free reduction of what [ethz-asl/hand_eye_calibration](https://deepwiki.com/ethz-asl/hand_eye_calibration/4.2-ransac-methods) does with RANSAC.

**Observability, from the robot-calibration literature.** `hand_eye.observability()` returns O1–O4 from the singular values of the stacked `(R_a − I)`, the matrix whose conditioning governs translation. Read **O3** (smallest singular value, E-optimality — [Sun & Hollerbach, ICRA 2008](https://cse.usf.edu/~yusun/lab_pub/icra0801.pdf)) rather than the condition number: `‖(R−I)v‖ = 2 sin(θ/2)·|v⊥|` is small when rotations are **small** *or* when they **share an axis**, so O3 catches both failures in one number, while a condition number is scale-invariant and rates a capture of uniformly tiny rotations a perfect 1.0. `CALIB_HAND_EYE_MIN_O3 = 1.5` is what `CALIB_HAND_EYE_MIN_SAMPLES` worth of good poses produces, so the two gates agree rather than contradict.

**The archives hold nothing recoverable.** `preDropSample11` is fully contained in the current file. The 72-sample `112750` archive shares no sample with it and carries 8–18° angle medians on four of six boards — the superseded-calibration signature, exactly what `reinterpret_hand_eye.py` exists for; its two clean boards have 6–7 samples each.

##### Sources

The checks above are not invented here; each implements something standard that this repo had been missing.

- **Screw congruence** — H. Chen, *A screw motion approach to uniqueness analysis of head-eye geometry*, CVPR 1991. A and B are conjugate, hence equal rotation angle and equal pitch. Restated with the dual-quaternion reading (the two invariants are the scalar parts) in K. Daniilidis, [*Hand-Eye Calibration Using Dual Quaternions*, IJRR 18(3), 1999](https://www.cis.upenn.edu/~kostas/mypub.dir/ijrr99.pdf), and in M. Ulrich & C. Steger, [*Hand-Eye Calibration of SCARA Robots Using Dual Quaternions*, OGRW 2014](https://mv.in.tum.de/_media/members/steger/publications/2014/ogrw-2014-ulrich-et-al.pdf) — the latter is also where the degenerate-motion analysis comes from.
- **Observability indices O1–O4** — Y. Sun & J. Hollerbach, [*Observability Index Selection for Robot Calibration*, ICRA 2008](https://cse.usf.edu/~yusun/lab_pub/icra0801.pdf); compared further in A. Joubair et al., *Comparison of the efficiency of five observability indices for robot calibration*, Mechanism and Machine Theory 70, 2013. O3 (minimum singular value, E-optimality) is their pick as the best single predictor of pose uncertainty, which is why it is the gate here rather than the condition number.
- **Consensus outlier rejection** — [ethz-asl/hand_eye_calibration](https://deepwiki.com/ethz-asl/hand_eye_calibration/4.2-ransac-methods), whose RANSAC variants classify inliers either by an RMSE threshold after a minimal-sample solve or by *dual-quaternion scalar-part equality* with no model fitted at all. `congruence_disagreement` is the second of those, generalised from a threshold to a vote over every pair.
- **Capture geometry minima** — [MVTec HALCON hand-eye calibration docs](https://www.mvtec.com/doc/halcon/12/en/hand_eye_calibration.html): ≥8 poses, ≥30° rotation between them (60° better), ≥2 non-parallel rotation axes. These set `CALIB_HAND_EYE_MIN_SAMPLES` and `CALIB_HAND_EYE_MIN_ROTATION_DEG`.
- **Active pose selection**, not implemented but the obvious next step if capture keeps being the bottleneck — [*Next-Best-View Selection for Robot Eye-in-Hand Calibration*, arXiv:2303.06766](https://arxiv.org/pdf/2303.06766) picks the next pose by predicted information gain, which is the same Fisher-information argument the observability indices approximate offline.

### Named poses, and starting every run from one

`config.SERVO_HOME_TICKS` / `SERVO_HOVER_TICKS` + [`robot_interface/poses.py`](src/vision_pipeline/robot_interface/poses.py) (`goto`, `at_pose`, `describe_move`), driven by [`scripts/goto_pose.py`](scripts/goto_pose.py). Raw ticks, never IK — a recovery pose has to work when the clever paths do not, and IK depends on a hand-eye transform that is known wrong.

**The automatic hover move is offered again, right after it runs** (`go_to_hover`, 2026-08-07). `poses.goto` walks every joint in sub-cap hops and re-checks the travel limits *per hop*, so a joint that is out of range — back-driven by a power cut, left there by a stalled descent — is refused part-way while the others arrive, and the run then continues from a pose that **looks like the hover in the log because the move was commanded** and is not one. It now prints the per-joint residual, names any joint outside `SERVO_VISUAL_HOVER_TOLERANCE_TICKS` (15) with the limits that refused it, and asks "Drive to hover again?". A second attempt starts from where the first got to, so it makes progress the first could not. It is also the natural moment to reposition the brick, since the go-ahead that follows asks the operator to approve the view the hover just produced. Bounded by the operator answering, not a count; skipped under `--no-wait`.

**HOVER is where a run should start.** The camera is eye-in-hand, so a brick that is not in view cannot be detected, probed, or servoed to; beginning from an arbitrary pose means hand-positioning the arm until the brick appears, and the probe then measures its gains against whatever geometry that happened to be. `visual_servo.py` drives there automatically (`--no-hover` opts out).

**The go-ahead comes AFTER the hover** (fixed 2026-08-06). It used to be asked first, so the operator approved a view that the very next move threw away — the arm started wherever the last run left it, the brick had to be hand-framed from that posture, and then the hover swung the camera somewhere else. The reach-conditioning warning moved with it, for the same reason: it was measuring a pose the run never visits. `--dry-run` keeps the old order, since it never touches the arm.

Re-measured 2026-08-06: `{1: 2057, 2: 3145, 3: 2205, 4: 1841, 5: 2688}`, tip ~195 mm above the table. **Note it sits only ~7 mm from the base column**, well inside `SERVO_VISUAL_MIN_RADIUS_M` (120 mm). Centring is unaffected — J1 pans the view perfectly well there even though it barely translates the claw — but the GRASP is, because the brick has to end up somewhere the claw can actually reach. J6 is deliberately excluded from the pose so driving to it never opens or closes the gripper.

`tests/test_poses.py` pins `SERVO_HOME_TICKS` against each joint's `home_tick` and asserts the home pose converts to exactly 0.0 rad — the two are copies of one measurement, and drift between them silently offsets every angle the arm reports.

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

# HARDWARE — read-only, safe any time (run these first after a power cycle)
python scripts/check_servo_health.py          # bus, reads, home agreement, FK cross-check
python scripts/servo_torque.py                # which joints are actually HOLDING
python scripts/test_pick_dry_run.py --save data/dryrun.png   # whole pick path, commands nothing
python scripts/audit_model_axes.py            # the model's own joint axes + a ruler sheet

# CALIBRATION ANALYSIS — no arm, no camera, safe to run DURING a capture
python scripts/hand_eye_report.py             # inventory, per-board quality, outliers, solve, verdict
python scripts/hand_eye_report.py --fix       # drop the congruence outliers it names

# HARDWARE — these MOVE the arm. Ctrl-C freezes it; power cut is the last resort.
python scripts/goto_point.py --x 145 --y 35 --z -64 --hover-only   # ruler-driven, no vision
python scripts/jog_joint.py --joint 5 --ticks 150                  # one joint, dir_sign check
python scripts/goto_tick.py --joint 2 --ticks 2883                 # one joint to a tick
python scripts/find_joint_limits.py --joint 3                      # measure travel
python scripts/close_claw.py                  # J6 only, one approved jog at a time
python scripts/close_claw.py --open           # let go

# HARDWARE — recovery
python scripts/freeze.py                      # STOP the arm without dropping it
python scripts/servo_torque.py --enable 3     # re-enable a limp joint where it sits
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
Interactive, hardware-dependent tools, not part of pytest. They use a **ChArUco** board (chessboard + a unique ArUco marker per square), not a plain chessboard — every corner is individually identified, so detection survives partial/angled views, which matters because the eye-in-hand camera's view of the board swings around as the arm moves (OpenCV 5.0 `cv2.aruco.CharucoDetector` + `board.matchImagePoints`). The board geometry lives in `config.CALIB_ARUCO_DICT` / `CALIB_CHARUCO_SQUARES_X/Y` / `CALIB_SQUARE_SIZE_M` / `CALIB_MARKER_SIZE_M` / `CALIB_BOARD_COUNT`, built through [`calibration/charuco.py`](src/vision_pipeline/calibration/charuco.py) — the **single source of truth**: generator and detectors both construct boards there, so the printed board and the detector's expectation can't drift. **Multiple DISTINCT boards** (`CALIB_BOARD_COUNT`, default 6): board *i* takes dictionary IDs `i*30..i*30+29`, so several tiled in one frame are each resolved independently — printing one board N times instead gives duplicate IDs the detector can't disambiguate, which is a *silent* miscalibration, not a visible failure. Two square sizes exist on purpose: `CALIB_SQUARE_SIZE_M` (ruler-**measured**, used for detection/pose) vs `CALIB_SQUARE_SIZE_NOMINAL_M` (what the generator asks the printer for) — rendering at the measured size would apply the printer's scale factor twice. **Never substitute a board from elsewhere** (a random online image): a mismatched dictionary/geometry doesn't just fail to detect, it can silently feed wrong 3D coordinates and yield a confidently-wrong calibration.
- [`scripts/generate_charuco_board.py`](scripts/generate_charuco_board.py) — renders `CALIB_BOARD_COUNT` distinct boards to `data/charuco_board_1.png .. _N.png` at a known DPI (print at 100%, measure a square, correct `CALIB_SQUARE_SIZE_M` if the printer rescaled). Board 1's IDs match the old single board, so an existing printout of it stays valid. Run this **first**; regenerate whenever the config geometry changes.
- [`scripts/calibrate_camera_intrinsics.py`](scripts/calibrate_camera_intrinsics.py) — `cv2.calibrateCamera` (fed by `matchImagePoints`), camera only, writes `data/camera_intrinsics.json` via `save_intrinsics`. Runs every board's detector per frame; each detected board banks an independent view, so laying several in view at once collects views faster. Do this before hand-eye.
- [`calibration/hand_eye.py`](src/vision_pipeline/calibration/hand_eye.py) — the shared sampling + solve logic behind BOTH the terminal script below and the web dashboard's calibration panel, so the math exists exactly once. `HandEyeAccumulator` buckets `(T_base_gripper, T_cam_board)` samples **per board** (the workspace tiles `CALIB_BOARD_COUNT` boards, so one frame can see several) and solves each independently with `cv2.calibrateHandEye`; two boards converging on the same camera offset is a stronger trust signal than TSAI-vs-PARK alone, since TSAI/PARK share their input data and can't catch a bad sample set. `detect_board_poses` wraps `charuco.detect` + `solvePnP` per board. Callers **must** pass the gripper pose as a raw FK 4x4 (e.g. from `MatlabIKClient.request_fk`), never routed through `Pose` — `geometry.transform_to_pose` forces `roll=0` near pitch=±90°, and the top-down tool orientation sits close to that gimbal-lock singularity, which would poison the solve with rotation error. Covered by fully-synthetic [test_hand_eye.py](tests/test_hand_eye.py) (plants a known transform, confirms the solve recovers it).
- [`scripts/calibrate_hand_eye.py`](scripts/calibrate_hand_eye.py) — terminal driver for the above. **Drives the real arm**, but by JOINT-space jogs (same raw-tick primitive as `jog_joint.py`), not a Cartesian sweep: this arm's small-Cartesian-move-near-the-workspace behavior (see "COORDINATE FRAMES" / `back` probe note below) means a Cartesian sweep mostly gets refused by `SERVO_MAX_MOVE_DELTA_TICKS`, and jogging J4/J5 directly gives the wrist-rotation diversity `calibrateHandEye` actually needs. Writes `data/hand_eye.json` via `save_hand_eye`, using whichever board has the most samples (ties broken by lower TSAI-vs-PARK disagreement). Needs intrinsics first; boards must stay stationary for the whole session.
- The same capture flow is also available from the web dashboard — see "Arm observation / jog web UI" below (`run_arm_ui.py --hardware --calibrate`) — which drives it by clicking Record Sample after jogging instead of keyboard shortcuts in an OpenCV window.
- [`scripts/validate_pixel_to_world.py`](scripts/validate_pixel_to_world.py) — end-to-end check that the whole chain (intrinsics + hand-eye + `TABLE_Z_IN_BASE`) actually works, in millimetres. Picks one ChArUco corner (true position known from the board's own geometry), observes it from ≥2 arm poses, and cross-checks three independent answers: single-view back-projection (assumes the table plane), two-view triangulation (`triangulate_pixels`, no plane assumption), and cross-pose board-spread (also re-measures the table height as a byproduct). `--goto-corner` commands the claw tip to the recovered point via `HardwareRobot` for a final ruler-against-reality check.

### Robot seam
- [`RobotInterface`](src/vision_pipeline/robot_interface/base.py) — the entire contract to the arm: `get_end_effector_pose` (FK), `send_target_pose`, `set_gripper`. `Pose` (base-frame, meters + degrees) is the shared type.
- [`SimRobot`](src/vision_pipeline/robot_interface/sim.py) — in-memory backend: reports a fixed gripper pose and records every command. This is what makes the pipeline testable/demoable with no hardware.
- [`PickTarget` + `plan_pick_sequence`](src/vision_pipeline/planning/pick.py) — `PickTarget` is the vision→robot handoff (x,y,z,yaw; roll/pitch are *not* here because a 5-DOF top-down pick keeps the tool pointing down — those fixed angles come from config). `plan_pick_sequence` turns one target into the backend-agnostic hover→descend→close→lift step list, so every backend runs the same grasp and only implements the primitive moves.

- [`Camera`](src/vision_pipeline/capture/camera.py) — thin `cv2.VideoCapture` wrapper (context manager + frame iterator) so detection depends on this interface, not OpenCV's camera API directly.

### Arm observation / jog web UI (bring-up tool)

[`scripts/run_arm_ui.py`](scripts/run_arm_ui.py) launches a Flask dashboard (`src/vision_pipeline/webui/`) showing the live camera feed and letting you jog joints J1–J6 by hand — built for hardware bring-up, since `ServoBus` (above) hasn't been exercised on physical hardware yet. It sits *beside* `PickPipeline`, not inside it — nothing here is on the pick pipeline's path.

- [`JointController`](src/vision_pipeline/robot_interface/joint_controller.py) is the joint-space seam this fills in: `RobotInterface`/`HardwareRobot` only expose Cartesian `Pose` (xyz, solved through IK), and `SimRobot` has no joint model at all — individual-joint control previously existed only at the raw-tick `ServoBus` level. `MockJointController` (in-memory, seeded at each joint's `home_tick`) and `ServoJointController` (thin wrapper over a real `ServoBus`) both implement `read_joint`/`jog`/`set_gripper`/`degrees_per_tick`; `scripts/run_arm_ui.py --hardware` selects which one backs the dashboard.
- [`servo_calibration.py`](src/vision_pipeline/robot_interface/servo_calibration.py) factors the JSON-or-`config.SERVO_CALIBRATION_FALLBACK` loading and tick↔radian conversion out of `ServoBus` into serial-free functions, so `MockJointController` converts ticks↔degrees using the exact same calibration `ServoBus` would, without opening a serial port. `ServoBus` keeps its own copy (its byte-level tests pin that behavior) — this is a standalone twin, not a replacement.
- [`webui/app.py`](src/vision_pipeline/webui/app.py) (`create_app`) wires one `JointController` and a background-threaded [`CameraStreamer`](src/vision_pipeline/webui/camera_stream.py) (owns the one `Camera`, degrades to a placeholder frame if no webcam) into Flask routes: `/` (dashboard), `/video_feed` (MJPEG), `/api/joints`, `/api/joints/<id>/jog`, `/api/gripper`. Every controller call goes through one lock — `ServoBus` isn't thread-safe, and Flask's dev server is multithreaded. `CameraStreamer`'s overlay is pluggable (a frame→frame callable) rather than hardcoded to brick detection, which is what lets calibration mode below draw ChArUco corners on the same feed instead.
- The dashboard shows each joint's position in both degrees and raw ticks, with jog buttons whose step size is set by a per-joint tick slider; the slider also previews its size in degrees (`JointController.degrees_per_tick`) so a tick count is legible without doing the conversion by hand.
- `run_arm_ui.py --hardware --calibrate` additionally connects a `MatlabIKClient` and enables a hand-eye calibration panel: jog to a pose where a board is visible, click Record Sample (grabs a frame, runs `calibration/hand_eye.detect_board_poses`, reads FK, adds to a `HandEyeAccumulator`), repeat across varied poses, then Solve & Save writes `data/hand_eye.json`. Backed by `/api/calib/detect|sample|samples|solve`, which 501 (not 404) if `create_app` wasn't given an `ik_client` — the routes exist, calibration mode just isn't on. This is the same `calibration/hand_eye.py` logic `scripts/calibrate_hand_eye.py` uses, so the two capture paths can't drift apart.

## Merge path (what's left to make this drive a real arm)

1. **Calibrate the camera intrinsics** (chessboard + `cv2.calibrateCamera`), write `data/camera_intrinsics.json`. Until then `config.CAMERA_*` placeholders are used and world coordinates are only roughly right.
2. **Hand-eye calibrate** (`cv2.calibrateHandEye`) once the arm exists, write the 4x4 gripper→camera transform to `data/hand_eye.json`. Placeholder is identity (camera == gripper), which is wrong for any real mount.
3. **Measure table geometry** into config: `TABLE_Z_IN_BASE`, `PICK_Z_OFFSET`, `APPROACH_HEIGHT`, and confirm `PICK_ROLL_DEG`/`PICK_PITCH_DEG` match the arm's "tool pointing down" convention. **`TABLE_Z_IN_BASE` is done** (−0.074, corroborated to 1.7 mm — see "Stage D" below); the rest are still nominal.

   **5-DOF orientation caveat:** this specific arm is genuinely 5-DOF and its IK (`matlab/`, below) is **position-only** (`weights = [0 0 0 1 1 1]`) — there is no spare joint for wrist yaw/roll/pitch. `PickTarget.yaw_deg` is still *computed* by vision, but `HardwareRobot` **silently drops** all orientation (`roll/pitch/yaw`) and commands `(x, y, z)` only; it logs the dropped angles at debug level. `PICK_ROLL_DEG`/`PICK_PITCH_DEG` describe the arm's *fixed* mechanical top-down pose, they are not commanded. Accepted because the grasp is top-down on a near-symmetric brick; revisit only if orientation-dependent grasping becomes a requirement.
4. **Write a real `RobotInterface` backend** (sibling to `SimRobot`) against your ROS/serial/TCP arm, implementing the three methods. Swap it in wherever `SimRobot()` is constructed (`scripts/run_pick_demo.py`, or your own entry point). Nothing else changes. **One such backend now exists** — see "Real hardware backend" below.

The agreement point with the robot code is the `Pose`/`PickTarget` convention: base-frame origin, meters, degrees, RPY as defined in `geometry.py`. Align units/axes with the kinematics code there.

### Real hardware backend (MATLAB IK/FK + Feetech servos)

[`HardwareRobot`](src/vision_pipeline/robot_interface/hardware.py) is a concrete `RobotInterface` backend for the physical arm, composed of two seams that are tested independently before being combined:

- **MATLAB IK/FK bridge** — the validated position-only IK solver (`matlab/IKtrials_v2.m`, kept verbatim as reference) is hosted as a persistent TCP JSON server, [`matlab/ik_fk_server.m`](matlab/ik_fk_server.m), on `localhost:9999`. Shared setup (`importrobot` of `Robomainassemjoints.slx`, servo→joint mapping, idler freeze, `ClawTip` end effector, IK solver) lives in the `matlab/init_arm.m` **script** — deliberately a script, not a function, because `importrobot`'s compile step reads base-workspace variables (`smiData` from `Robomainassem_DataFile.m`, and the `j1..j6` "From Workspace" placeholder signals) that must exist *before* the import. FK returns the **wrist (Body08)**, not the claw tip, so it composes with the separate hand-eye calibration. [`MatlabIKClient`](src/vision_pipeline/robot_interface/matlab_client.py) is the stdlib-socket Python client; `IKUnreachableError` surfaces an out-of-workspace target. Verify with `matlab/test_ik_fk.m` (offline, MATLAB-side) and `scripts/test_matlab_bridge.py` (against a running server).
- **Servo bus** — [`ServoBus`](src/vision_pipeline/robot_interface/servo_driver.py) speaks the Feetech STS3215 (Dynamixel-1.0-compatible) serial protocol over a Waveshare adapter, with **write-then-read-back verification** on every move. Per-servo calibration (`home_tick`, `ticks_per_rad`, `dir_sign`) loads from `data/servo_calibration.json` with a `config.SERVO_CALIBRATION_FALLBACK` placeholder (same pattern as camera intrinsics). J1–J5 are IK-driven; **J6 is the gripper** (open/close), mapped to `set_gripper`. Also exposes `ping`/`scan_ids`/`write_servo_id` for one-time bus setup — Feetech servos ship at a shared factory ID, so each must be assigned a unique ID (1..6 for J1..J6) before wiring the chain; [`scripts/set_servo_id.py`](scripts/set_servo_id.py) drives that one servo at a time. `write_servo_id` does the Feetech EEPROM dance (unlock→write ID→re-lock, re-lock addressed to the *new* ID). The move/read/ID-change protocol is unit-tested against a fake-serial servo emulator ([test_servo_driver.py](tests/test_servo_driver.py), exact-byte assertions), and **has now been exercised on physical hardware** — see "Bring-up status" below.

`HardwareRobot` owns the joint-angle state in Python (seeded from a servo read-back at init, refreshed from the **verified** read-back after every move) and treats MATLAB as a pure stateless math service: `get_end_effector_pose` = live FK of the last verified angles; `send_target_pose` = IK on `(x,y,z)` → `move_and_verify` J1–J5.

#### Bring-up status (2026-07-22)

The arm-side chain is **live and validated on the vertical axis**; the camera-side calibration is still placeholders. Bring-up ran in lettered stages, each a script kept in `scripts/`:

- **Stage B — read-only health** ([`check_servo_health.py`](scripts/check_servo_health.py)): bus enumeration, read stability, home agreement, and a MATLAB FK cross-check against the physical arm. Commands no motion; safe to run any time and the first thing to run after a power cycle.
- **Stage C — `dir_sign` per joint** ([`jog_joint.py`](scripts/jog_joint.py)): jogs ONE joint by a raw tick delta and compares the physical result against an FK prediction. **Four of five confirmed by physical jog: J1, J3, J4, J5. J2's `dir_sign_basis` reads "provenance not recorded" and it is the one that has never been tested** — this file claimed all five until 2026-08-07, when `check_servo_health.py` B5 was actually read. Raw ticks, not IK, deliberately: going through IK would fold the unknown sign into the solve. It predicts **claw-tip** motion, not wrist — J4's first attempt was void because the operator watched the claw while the script predicted the wrist, and for wrist joints those describe different axes.
- **Stage D — IK round-trip** ([`validate_ik_roundtrip.py`](scripts/validate_ik_roundtrip.py)): the first stage where IK commands the arm. Lifts first, then probes ±15 mm on each axis from the raised pose, then descends to the real pick height. **Vertical axis passed**; lateral probes outstanding.

**What Stage D established, and the trap in reading it.** The script prints an FK-vs-IK error, and *that number cannot validate `dir_sign` or `ticks_per_rad`* — `rad_to_ticks` and `ticks_to_rad` apply the same calibration on the way out and back, so it cancels and reports ~0 even with a sign inverted. **Only a ruler against the physical arm closes that loop.** Doing it: a 45 mm commanded lift moved the FK tip 40.7 mm (the servos settled 15–47 ticks short of goal — ~4 mm of real open-loop positioning error, which FK tracked correctly because it reads back actual positions). The operator measured the gap under the claw before and after: 5 mm → 44 mm, a 39 mm physical rise. **FK 40.7 vs ruler 39.** The two implied tabletop heights (−75.7, −74.0), taken 40 mm apart vertically, agree to 1.7 mm — a wrong vertical scale would have made them diverge. So the vertical kinematic chain (`ticks_per_rad` × link lengths, through the frame conversion) is good to ~2 mm over a 40 mm move, and J3/J4's `dir_sign` is confirmed by the arm having gone *up*.

**Still open on the arm side:** the lateral probes (`forward`/`left`/`right` — where a J1/J5 sign or scale error would surface, since vertical barely exercises them). **J6 is now measured** — see "The gripper" below. The `back` probe is *expected* to be refused by the tick cap: the arm works close in (tip x ≈ 75 mm) where the tip is near the J1 axis, so small Cartesian moves demand large J1/J5 swings. That refusal is the safety system working, not a driver bug.

#### The gripper: measured at last, and the placeholder was inverted (2026-08-07)

J6 is the one joint the pick path never exercised on hardware, so its config placeholder survived every check by never being executed. Operator-measured, read off `hold_pose.py`:

| J6 tick | |
|---|---|
| **3219** | fully OPEN |
| **3003** | closed ON THE BRICK — the working grip |
| **2732** | fully closed, jaws touching. *"It should never be this much."* |

**Ticks DECREASE as the claw closes**, and J6's `home_tick` is **2741** — nine ticks off the jaws being shut. So `SERVO_GRIPPER_OPEN_RAD = 0.0` ("home position = fully open") meant *open the claw* commanded it to clamp, and `CLOSE_RAD = +0.2` (2806) opened it slightly from there. Both wrong, and inverted relative to each other. Now 1.4665 / 0.8038 rad, with `tests/test_gripper.py` pinning the conversion against `data/servo_calibration.json` so the tick measurement and the radian constant cannot drift apart — the same guard `test_poses.py` puts on `home_tick`.

**2732 is a damage limit, not a target.** It is J6's `min_tick` (its `limit_basis` is the first on this arm where *both* ends are real ends of travel rather than ground-collision stops), so `ServoBus` refuses it independently of any script. A close driven into that stop with a brick in the jaws stalls the servo against the brick, and a stalled servo draws heavy current — that is what put the checksum storm on the bus during the 2026-08-07 descent.

[`robot_interface/gripper.py`](src/vision_pipeline/robot_interface/gripper.py) holds the jog loop; [`scripts/close_claw.py`](scripts/close_claw.py) is a thin driver over it, and **`visual_servo.py` offers the same thing at the end of a descent** through the same code. J6 only, **one operator-approved jog at a time** (`SERVO_GRIPPER_JOG_TICKS` = 40, so open→grip is ~6 approvals). The standalone script still has to exist because the recovery case — "that descent ended badly, open the claw" — cannot go through a descent.

**A stall while closing is SUCCESS**, and it is the one way this differs from every other motion primitive here: in `goto_tick.py` a joint that stops short is obstructed and the script backs off, whereas here the obstruction is the brick. `GripResult.holding` is true **only** for `GRIPPED`. `REACHED` means the claw arrived at the commanded position having met nothing — the jaws shut on air — so a caller reading "no exception" as "got it" has the answer backwards. Exactly one command is spent discovering the stall; the loop cannot know before a jog produces no motion, and a second push into a stalled servo is what draws the current.

**The stall is where the jaws TOUCH, not where they hold**, so the grip offer does not end there. A Lego brick is smooth plastic and first contact will drop it; closing further is how a Feetech servo is asked to grip harder, since it turns goal-position error into torque. `squeeze_and_lift` therefore offers **Enter** (squeeze `SERVO_GRIPPER_SQUEEZE_TICKS` = 20 more, repeat as often as you like), **k** (accept and drive to hover), **n** (leave it). Enter repeating rather than one bigger number because the right grip is something the operator can feel and no sensor here can measure.

`gripper.squeeze_once` is deliberately the opposite reading of the same observation: in `close_in_jogs` no motion means stop, here **no motion is the point**. What bounds it is not travel but `commanded_past` — accumulated goal error past first contact, which is what becomes torque. `SERVO_GRIPPER_MAX_SQUEEZE_TICKS` (120) is advisory and says so; the full-close stop is what actually refuses.

**`k` lifts with the brick held, and that is only safe because HOVER omits J6** — the joint holding the brick is not in the pose, so it is not commanded, so it keeps holding. Pinned twice (`test_the_hover_pose_does_not_touch_the_gripper`, `test_the_lift_never_commands_the_gripper`).

**The offer is gated on `descend()` returning True.** A descent refused by the floor guard, stalled, or stopped by the progress monitor has left the claw somewhere nobody chose, and offering to close there invites a grab at the table. It asks before starting (`--no-grasp` suppresses the question), then asks again per jog — the gripper is the only joint whose job is to stall, and where it stalls depends on where the brick really is, which is precisely what the vision chain is still worst at. The operator is the sensor.

#### J5's `dir_sign` is probably inverted — the likely cause of the hand-eye failure (2026-08-04, UNRESOLVED)

Hand-eye calibration failed repeatedly with one signature: **FK and the camera agreed on rotation *magnitude* (0.3° median) but disagreed on rotation *axis* (Kabsch residual 25–40°)**. For a rigidly-mounted camera both must hold, and no rigid transform explains data where only one does. Board square size, resolution, planar pose ambiguity, outliers, stale frames, MATLAB joint axes, `ticks_per_rad`, joint zero offsets and autofocus were each eliminated **by measurement** (see `SESSION_LOG_2026-08-04.md` §1).

Re-solving the 49 saved samples under all 32 `dir_sign` combinations:

```
as recorded  (+1 +1 +1 +1 +1)   39.0 deg   <- no rigid transform explains this
flip J5 only (+1 +1 +1 +1 -1)    3.4 deg   <- consistent
flip J2/3/4/5(+1 -1 -1 -1 -1)    3.5 deg   <- same solution in disguise (J2||J3||J4)
```

It fits: **the camera is mounted on J5** and visibly swings with it, so an inverted sign means FK believes the camera rotated one way while it physically rotated the other — right magnitude, wrong axis. It also explains why Stage D passed, since vertical moves barely exercise J5.

**Do not flip it on this evidence alone.** Two unresolved conflicts: the operator twice observed *J1* turning the wrong way, yet every J1-flipped combination scores ≥19°; and the recovered angles came from inverting stored FK, where a 5-DOF arm's multiple joint solutions per tip pose mean the recovery may not be the original branch. Settle it with one clean `scripts/jog_joint.py` per joint — small, isolated, and now safe at the capped speed. **If J5 is confirmed inverted, the 49 saved samples become consistent with no recapture.**

**Corollary for any future `dir_sign` work:** an FK-vs-IK error number cannot validate a sign (`rad_to_ticks`/`ticks_to_rad` cancel it), and neither can a hand-eye solve's own residual. Only a physical jog or a ruler closes that loop.

#### The stored hand-eye transform is WRONG — do not trust `data/hand_eye.json` (2026-08-04)

Measured against the board, which lies flat on the table and so gives an absolute camera pose independent of FK and hand-eye:

| | camera height above table |
|---|---|
| Board (`solvePnP`) | **228.0 mm** |
| FK + hand-eye | 279.9 mm |

52 mm out, and the optical axis points 91° off the claw direction. A brick back-projects ~400 mm from where a ruler puts it. **This test uses no ruler and no assumption about the base origin**, so it condemns the transform regardless of where the base column actually is. Run [`scripts/test_pick_dry_run.py`](scripts/test_pick_dry_run.py) to reproduce — it walks the whole pick path and commands nothing. Wrong calibration does not fail loudly; it yields a confident, well-formed `PickTarget` that `run_once` would drive straight to.

[`scripts/calibrate_board_to_base.py`](scripts/calibrate_board_to_base.py) is the route around it: a brick's pixel intersected with the **board plane** needs only intrinsics and `solvePnP`, so the camera's base-frame pose never enters. Prefer its `--touch` mode (claw onto a known ChArUco corner, read FK) over `--add` (ruler) — a ruler measured from an assumed base origin that is off by some offset produces a transform in the *operator's* frame, and the arm then misses every pick by exactly that offset, silently and consistently. Bring-up scaffolding, not the merge path.

#### Joint travel limits, and why ground collision is not one (2026-08-04)

`ServoBus` capped how FAR one command travelled but had no idea WHERE a joint could go, so small legal steps walked J3 and then J4 into hard stops. `_check_travel_limits` now enforces absolute `min_tick`/`max_tick`; a joint already out of range is not trapped, since moves that *reduce* the violation are allowed. Measure with [`scripts/find_joint_limits.py`](scripts/find_joint_limits.py) (operator-confirmed steps, watches for the 4095→0 wrap); `matlab/init_arm.m` reads the resulting `data/joint_limits_rad.json` so IK stops proposing unreachable solutions. [`scripts/goto_tick.py`](scripts/goto_tick.py) walks one joint to a target in sub-cap increments.

**Recorded: J1 `[1670, 2370]`, J2 `[2600, 3750]`, J3 `[500, 3350]`. J4/J5/J6 unmeasured.** Read `limit_basis` before trusting any of them — **not one of these six numbers is a mechanical stop.** J2's and J3's `max` are ground-collision stops measured at one elbow angle; J3's `min` is pure encoder-seam protection; J1's pair is a deliberate **working envelope** added after two runaways, not a measurement. J3's `max` was raised from 3161 to 3350 because 3161 **excluded the home pose itself** (3298) — the arm was not legally allowed to return to where it starts.

J3's numbers moved twice on 2026-08-05 and neither end means what the field name suggests — read `limit_basis` in the JSON before trusting either. Its encoder was re-centred (every stored tick shifted **+2065**; any J3 tick quoted in an older note or log is in the pre-re-centre numbering and must have +2065 applied to compare). `max_tick` 3161 is still the ground-collision stop. **`min_tick` 500 is not a measurement at all** — it is pure seam protection, deliberately widened from 2128 at operator request once the re-centre left the old value guarding nothing. It permits J3 out to **+230°, some 138° past anything ever measured**; the joint's mechanical stop in that direction has never been found, and the max-delta cap plus the operator watching are the only guards there. `find_joint_limits.py --joint 3` is safe to run for the first time now that the seam is far away, and closing that gap is the outstanding item.

**Both recorded limits are ground-derived, not mechanical** — the joint stopped because the *claw* reached the table at the elbow angle used when measuring. Fold the elbow differently and the same joint angle is safe. They carry a `limit_basis` field saying so. The honest form of that constraint is Cartesian: `config.MIN_CLAW_HEIGHT_M` (5 mm), enforced in `HardwareRobot.send_target_pose`, deliberately below `PICK_Z_OFFSET` (10 mm) so it cannot refuse the pick it exists to protect.

**Consequence, and it bites:** at table height the arm reaches only **~180 mm forward** (220 mm at z = +80), with J2 and J3 both pinned at their minimums. That is the joint limits, not the arm's geometry. Re-measuring the *mechanical* stops with the elbow folded so the claw cannot ground out would give back workspace to ~280 mm.

##### The 0/4095 encoder seam, and how J3 fell through it (2026-08-05)

A limit that lands on the seam is unusable — J3's travel was first measured to tick 6, the same geometry as the July J1 runaway — so `min_tick` was pulled in to 200 (~18° of margin). On 2026-08-05 that margin was cut to 63 (5°) at operator request, because J3 kept sagging below 200 and the limit was refusing poses the arm genuinely works in. **That was half a fix.** Shrinking the margin is only safe once the seam has been moved; it had not been. J3 subsequently crossed the seam during a visual-servo run and came back reading **4079**.

**A wrapped joint is not "out of range", and the two need opposite responses.** In raw ticks 4079 looks like ~3000 past a maximum of 1096; physically it is 80 ticks *below* a minimum of 63. Everything downstream misread it: `_check_travel_limits`'s "moves back toward range" allowance ran backwards, so the one direction that recovers the joint was the one refused, under a message telling the operator to re-measure a range that was entirely correct. And no goal position fixes it — the servo drives goals linearly and would take the long way round the circle, through every hard stop in between.

- `servo_calibration.unwrap_tick(tick, reference)` / `wrapped_past_seam(tick, lo, hi)` are the shared primitives. Unwrapping produces a *continuous* joint coordinate that may fall outside 0..4095; anything written to a register must come back with `% TICK_SPAN`.
- `ServoBus._check_travel_limits` now names the wrap and points at the fix. It refuses motion in **both** tick directions (neither is "toward range" across a seam) but still allows **zero-motion holds**, because that is what `freeze`/`hold_pose` write when someone is catching a falling arm — refusing it would rebuild the trap documented one line above it in that function.
- `scripts/recentre_joint.py --here` is the way out, and it needs **no motion**: the Feetech one-key midpoint redefines the joint's present pose as tick 2048, so the seam moves instead of the arm. It applies `--here` automatically on detecting a wrap, since such a joint cannot be driven to the middle of its travel first. The offset must be measured against the *unwrapped* present position — using the raw reading shifts every stored limit by a whole encoder turn, silently.
- Re-centring moves the seam, **not the joint**: J3 is 80 ticks under its minimum before and after. What changes is that an ordinary `goto_tick` can then walk it back.

`find_joint_limits.py` warns when a *measured* limit lands near the seam, not just the starting position. The durable lesson: **re-centre first, then measure, then set margins.** A thin margin next to a seam is a countdown, not a setting.

#### The IK solver is redundant for this task, and that redundancy leaks (2026-08-05)

Five actuated joints against a **3-DOF position target** (`weights = [0 0 0 1 1 1]`) leaves a **2-dimensional null space**. A position-only solve has no preference within it: every point is an equally correct answer, so the solver returns whichever its iteration lands on. Seeding from the current angles biases that; it does not constrain it. Since the camera rides on the wrist, redundancy spent on base yaw pans the whole image — feeding straight back into the visual loop trying to correct it.

Two defects found by reading `matlab/ik_fk_server.m`, both fixed:

1. **The random-restart discarded the seeded posture.** When the seeded attempt missed by >2 mm it reseeded with `randomConfiguration` and then accepted whichever candidate had the *lowest position error*, regardless of posture — so a 0.1 mm solution half a workspace away beat a 3 mm one right where the arm stood, well inside the 10 mm `IK_TOL` governing acceptance anyway. Now: among candidates meeting tolerance, take the one **closest in joint space** to the seed. Accuracy beyond `IK_TOL` buys nothing this arm can execute (open-loop error is several mm); posture change is paid for in real motion and a swung camera. The response gained `move_rad` so callers can see posture cost, which a position residual cannot show — the runaway solve reported **0.0 mm**.
2. **No way to hold a joint.** The `ik` request takes an optional `lock` list of joint numbers, pinned at their seed angle via `PositionLimits` (the only mechanism `rigidBodyTree` offers), restored on every exit path by `onCleanup` — `robot` is a global handle object, so a leaked pin would silently freeze that joint for the rest of the server's life.

**Measured, live, from one pose:** descents of 5/10/20 mm each needed **1.2 ticks of J1**; a 40 mm descent came back wanting **213 ticks of J1 and 1275 of J5**. So base yaw is *not* geometrically required by a descent — that was redundancy being spent uninstructed. `visual_servo.py` passes `lock=[1]` on every descent solve (`--no-lock-base` to disable).

##### The `lock` field DOES NOT WORK. Enforce it caller-side (2026-08-06)

`PositionLimits` pinning has never held. With `lock=[5]`, `lock=[1]` and `lock=[1,5]` the server returns a **byte-identical** solution — the field is parsed (`lock_drift_rad` reports the right joints) and then ignored by the solve. Three fixes were tried and all failed: a degenerate `[v, v]` interval; a narrow `[v−ε, v+ε]` band with `HomePosition` saved and restored; and `release(ik)` to defeat the System object's cached `RigidBodyTree`. The last of these *should* have worked — `inverseKinematics` locks on first call, which is exactly why the limits from `data/joint_limits_rad.json` DO take effect (`init_arm.m` applies them before the solver is built) — but it did not. `generalizedInverseKinematics` with a real joint-position constraint is the untried option if this is ever worth revisiting.

**So `lock` is advisory and the caller must enforce it.** `visual_servo.CartesianActuator.apply` simply does not command a held joint, whatever the solver returned. The nudge then misses its requested point, which is fine *here* and would not be open-loop: the loop measures the pixel response of whatever the arm actually did. An unrequested joint motion is not a small error to tolerate, it is a disturbance to the image the loop reads.

Two consequences worth keeping straight:
- **A held joint can leave nothing to move with.** If the solver's whole answer was the joint being refused, dropping it makes the nudge a no-op that still reports its requested size — and the probe divides a pixel shift by a move that never happened, manufacturing a gain from detection noise. `apply` aborts on exactly that (held ∧ dropped ∧ nothing left), not on a solve that was already a no-op for its own reasons.
- **The drift log is `debug`, not `warning`.** The pin never holds, so at warning level it fires on every nudge of every descent — dozens of identical lines about a condition handled two statements later, which is where a real fault goes to hide.

[`scripts/check_ik_lock.py`](scripts/check_ik_lock.py) tests **both layers separately** and only blocks on the load-bearing one. Section 1 failing is the documented status quo; what blocks a run is a stale server (caught by the `server_build` marker), a held joint still moving after the drop, or an axis with no motion left. Measured 2026-08-07 from the hover: a 3 mm nudge wanted **120 ticks (10.5°) of J5 roll**, all of it dropped.

**Any change to `matlab/` requires restarting the server (`>> ik_fk_server`) to take effect.**

#### Descending and re-aiming are ONE degree of freedom (2026-08-05)

Lowering the claw and correcting the brick's **vertical** position in the image are not independent. Both ride the shoulder/elbow chain, and the camera is on the wrist, so descending swings the view — **~7 px per mm**, measured. A 20 mm step throws the brick ~140 px up the frame, against a 55 px acceptance box.

Running them as two loops (IK for the descent, a joint jog to re-centre) makes the arm fight itself: the descent throws the brick up, the re-centring jog drags it back *and raises the tip by more than the step gained*, and the next step spends itself undoing that. Observed: four steps, **4 mm of net descent**, then the brick left the frame.

[`DescentModel`](src/vision_pipeline/planning/visual_servo.py) fixes it by modelling both terms — `dpx = a·dz + b·dr` — and issuing **one IK solve per step** whose target descends *and* reaches out by the amount that cancels the swing the descent is about to cause. IK spreads that across J2/J3/J4, which is the whole reason an IK solver is in this loop. Both coefficients are fitted from the descent's own motion (step 1 straight down, step 2 with a radial offset; both descend in full, so neither is a wasted probe), refitted every step because the coupling weakens as the arm unfolds, and rank-checked so two pure descents can't fabricate a `b`.

**The step size is derived, not fixed.** Reach is capped per step (`SERVO_VISUAL_DESCEND_MAX_REACH_MM`), so when a descent would inject more error than the reach can absorb — 20 mm needs 56 mm of reach at the measured coefficients, more than double the cap — `plan_step` shrinks it, to zero if necessary. `dz = 0` is a legitimate outcome: a re-aim at constant height, the one correction that cannot undo a descent. Progress resumes by itself. It never climbs to improve the aim and never descends further than asked.

Sideways error still goes through a J1 jog on purpose: J1 is base yaw, it cannot change tip height, so it cannot undo a descent step and has no business in the combined solve.

**That jog ran J1 away on 2026-08-05 and the operator cut power.** Its gain came from a probe taken *before* the descent, at a different posture; once wrong-signed, each correction enlarged the error and the next was bigger. `centre()` has carried a `ProgressMonitor` against exactly this since it was written — the descent branch was added without one, and J1 has **no measured travel limits**, so `ServoBus` could not refuse it either. Nothing in the system was watching. Three independent bounds now, each separately tested by turning the runaway on and removing the others:

1. the first correction that makes the error worse stops the descent (fires after ~2 moves);
2. a `ProgressMonitor` across the whole descent;
3. `SERVO_VISUAL_SIDEWAYS_BUDGET_TICKS` — a hard cumulative ceiling that holds even when the pixel readings themselves are wrong, which is the case where (1) and (2) are blind. `--no-sideways` disables the jog entirely.

**The general lesson: any autonomous correction loop needs a progress check, and a joint with no measured travel limits has no backstop below it.** J1's limits are still unmeasured — the July runaway and this one are the same gap.

**Operator safety convention:** stand by the power cut whenever a commanded move turns any joint more than `config.SERVO_WATCH_POWER_MOVE_DEG` (45°). Both `jog_joint.py` and `validate_ik_roundtrip.py` preview per-joint degree deltas and print a banner when a move crosses it.

**Two stops exist, and the power cut is the WORSE one — reach for it second.** Once a Goal Position is written the servo travels there regardless of whether anything is still talking to it, so killing the script does not stop the arm.
- **Freeze (`ServoBus.freeze`, Ctrl-C in `goto_point.py`, or `scripts/freeze.py` from a second terminal)** overwrites each joint's goal with its present position. Motion halts, torque stays on, nothing falls. Needs a working serial link.
- **The power cut** drops holding torque on *every* joint simultaneously and the arm falls under its own weight; there is no way to power down one servo independently. Use it when freeze doesn't visibly stop the arm within a second, or when the bus is unresponsive. On 2026-08-04 a mid-move power cut dropped the arm face-first and back-drove J3 hard enough to trip its overload protection — the servo came back answering the bus normally, at healthy voltage with no fault flags, but with torque disabled and the joint sagged 54°. `scripts/servo_torque.py` reports the Torque Enable register (the only thing that distinguishes this from a dead servo) and re-enables a limp joint *at its present position*, so it holds rather than snapping back to a stale goal.

**Servo speed is capped in software, not by the servos.** `config.SERVO_MOVE_SPEED_TICKS_S` / `SERVO_MOVE_ACCEL` are written via `ServoBus.set_motion_profile`; without them every move runs at the servo's full default speed, which ends each step in a hard stop and puts peak torque far above what the pose needs statically. They live in the servo's SRAM, so they reset on every power cycle and must be re-applied per run — `goto_point.py` does this at startup and prints the limit.

**Speed and pacing are ONE setting.** `SERVO_MOVE_SPEED_TICKS_S` caps how fast a single hop runs; `PICK_STEP_PAUSE_S` is the rest between hops. On 2026-08-07 the pause was halved (0.5 → 0.25 s) at the operator's request — a long paced move was mostly dead time — and the speed dropped with it (200 → 160), so a 60-tick hop goes from 0.30 + 0.50 s to 0.375 + 0.25 s: **×1.28 overall, with the stops cut in half.** Shortening the pause alone would have been ×1.6. The pause is not idle time, it is the window in which a wrong move gets caught by hand, and slowing the hop buys part of that window back in a better form — during a pause the arm is already wherever the bad command put it, whereas during travel it is still on its way and a freeze still helps.

## Tuning is centralized in config.py

All thresholds live in [`config.py`](src/vision_pipeline/config.py) — detection tunables (HSV bounds, `MIN_CONTOUR_AREA`, `MORPH_KERNEL_SIZE`, Hough parameters, `STUD_ROI_REFERENCE_PX`/`STUD_ROI_MAX_UPSCALE`/`STUD_HOUGH_PARAM2_UPSCALED`, `STUD_REGION_CLOSE_*`, `SPECULAR_*`, `SHAPE_*`, `STUD_WEIGHT`/`SHAPE_WEIGHT`/`DETECTION_CONFIDENCE_THRESHOLD`) **and** the geometry/calibration tunables (`CAMERA_*` intrinsics, `HAND_EYE_PATH`, `TABLE_Z_IN_BASE`, `PICK_Z_OFFSET`, `APPROACH_HEIGHT`, `PICK_ROLL_DEG`/`PICK_PITCH_DEG`). Detection and calibration classes read these as **constructor defaults**, so tests/callers override per-instance without touching config. When behaviour is wrong, retune here rather than editing logic. Key relationships baked into current values:
- High saturation floor (150) deliberately rejects skin tones (low-saturation orange-red) so the detector doesn't fire on a hand. Raise it if background skin/wood leaks in; lower the value floor (90) first if a real brick is missed in dim light.
- `MIN_CONTOUR_AREA` (350) is deliberately well above single-digit-pixel JPEG/lighting noise speckles (seen as low as ~30-300px² in real photos) — those tiny blobs are pixel-grid-quantized near-rectangles almost by accident and can otherwise slip through on shape score alone. Don't lower this much without also re-checking the noise-speck real-photo regression case.
- `STUD_WEIGHT`/`SHAPE_WEIGHT`/`DETECTION_CONFIDENCE_THRESHOLD` trade false positives vs. missed bricks the same way `MIN_STUDS` used to alone: raise the threshold or `STUD_WEIGHT` if smooth red objects get misclassified; lower the threshold, or raise `STUD_ROI_REFERENCE_PX`/`STUD_REGION_CLOSE_FRAC` if a real brick far from the camera (or at a bad, glare-prone angle) is being missed. Be cautious raising `SHAPE_WEIGHT` or loosening `SHAPE_RECT_SCORE_LOW`/`STUD_HOUGH_PARAM2_UPSCALED` much further — real photos show a hand's silhouette and a genuinely distant brick's silhouette land in overlapping ranges on fill ratio and on Hough's own permissiveness, so those two knobs don't have a clean global setting that helps one without also helping the other; tune against `tests/sample_images/`, not intuition.

## Tests

Everything runs without a camera or robot. The suite has five parts:
- **Detection** ([test_color_detector.py](tests/test_color_detector.py), [test_lego_detector.py](tests/test_lego_detector.py), [test_shape_detector.py](tests/test_shape_detector.py), [test_stud_detector.py](tests/test_stud_detector.py)) — synthetic frames (red square; red rectangle with/without drawn "studs"; a smooth red oval) pin the core rule *red + (studs or brick-shaped) = brick, red alone (round, no studs) = not a brick* — including a dedicated case proving a rectangular region with zero resolvable studs is still accepted on shape confidence alone (the far-away/glare-corrupted case). Plus real-photo regression tests parametrized over `tests/sample_images/`, keyed by the `SAMPLE_TRUTH` dict (`True` = must detect, `False` = must reject; missing files skip, not fail). Add a photo → add it to `SAMPLE_TRUTH`. The `False` entries (red cup, hand) guard the false positives the stud+shape stage exists to fix.
- **Geometry** ([test_geometry.py](tests/test_geometry.py)) — exact-answer math tests for transforms and ray/plane intersection; they lock the RPY convention.
- **Calibration** ([test_pixel_to_world.py](tests/test_pixel_to_world.py)) — the load-bearing one: a **projection round-trip**. It projects a known table point through the camera to a pixel, then back-projects through `PixelToWorldCalibrator` and asserts recovery. If projection and back-projection agree, the whole chain is self-consistent. [test_hand_eye.py](tests/test_hand_eye.py) does the same for hand-eye: plants a known `T_gripper_camera`, synthesizes the samples a real session would record (varied rotation + translation, since `calibrateHandEye` needs rotation diversity to be well-conditioned), and asserts the per-board solve recovers it — this is what would have caught the `Pose`-round-trip gimbal-lock bug fixed in `calibrate_hand_eye.py`, since it feeds samples as raw FK 4x4s the way callers now must.
- **Pipeline** ([test_pipeline.py](tests/test_pipeline.py)) — end-to-end against `SimRobot`: a synthetic brick frame yields a `PickTarget` and the exact hover→descend→close→lift command sequence (4 poses, gripper `[open, close]`). This is the "vision side is merge-ready" regression guard.
- **Joint control & web UI** ([test_joint_controller.py](tests/test_joint_controller.py), [test_webui.py](tests/test_webui.py)) — `MockJointController` jog/clamp/degree-conversion correctness, and the Flask dashboard's REST endpoints via `app.test_client()` wired to the mock (no camera or hardware needed). `ServoJointController` is checked against a small duck-typed fake bus, since `ServoBus`'s own wire protocol is already exhaustively covered by [test_servo_driver.py](tests/test_servo_driver.py). The calibration panel's routes are covered the same way: a `_FakeIKClient` stands in for `MatlabIKClient`, and `detect_board_poses` is monkeypatched to a scripted sequence so the sample→solve round trip runs through the real Flask routes without a camera; `config.HAND_EYE_PATH` is monkeypatched to a `tmp_path` so the test never touches the real `data/hand_eye.json`.

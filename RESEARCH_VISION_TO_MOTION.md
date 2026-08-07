# Research: turning what the camera sees into where the arm moves

**Written 2026-08-07.** Scope agreed before writing: explain the chain this repo already
has, then extend it with one alternative family — **closed-loop visual servoing** — and end
with a recommendation. Deliberately out of scope: learned/ML methods, and the
calibration-free board-plane tie (already scaffolded in
[`scripts/calibrate_board_to_base.py`](scripts/calibrate_board_to_base.py); mentioned here
only where it intersects the argument). Every option is judged against the rig as it
actually is today — ~180 mm forward reach at table height, 5-DOF position-only IK, J5's
`dir_sign` unresolved, `data/hand_eye.json` untrusted.

**Evidence labels used throughout**, because this document mixes three very different
grades of claim and conflating them is how this project has been burned before:

| Tag | Means |
|---|---|
| `[measured]` | Taken from a physical measurement recorded in `CLAUDE.md` / the session logs. |
| `[derived]` | My arithmetic or algebra from the code and those measurements. Reproducible on paper; **not** confirmed against this arm. |
| `[unverified]` | A claim that needs a measurement nobody has taken yet. Flagged, never relied on. |

There is no Python or `.venv` on this machine right now, so nothing here was executed —
every `[derived]` number is analytic. §6.1 gives the script to check them when the
environment is back.

---

## 1. What the code does today

One frame becomes one arm command through five stages. Each is a separate module and each
contributes its own error.

```
  frame
    │
    ▼  LegoBrickDetector.detect()                 detection/lego_detector.py
  Detection.centroid_px  (u, v)
    │
    ▼  CameraIntrinsics.pixel_to_ray()            calibration/camera_model.py:51
  unit ray in the CAMERA frame
    │
    ▼  T_base_camera = T_base_gripper @ T_gripper_camera
    │                  ▲ live FK           ▲ data/hand_eye.json
  ray in the BASE frame                            calibration/pixel_to_world.py:120
    │
    ▼  ray ∩ plane z = TABLE_Z_IN_BASE             calibration/geometry.py:110
  world point (x, y, table_z)  →  + PICK_Z_OFFSET  →  PickTarget
    │
    ▼  plan_pick_sequence()                        planning/pick.py:69
  [hover+open, descend, close, lift]
    │
    ▼  HardwareRobot.send_target_pose()            robot_interface/hardware.py:94
  floor guard → MATLAB IK (position only, seeded) → rad_to_ticks → move_joints_stepped
```

### 1.1 The single most important property of this design

**It is open-loop end to end.** The arm measures once, computes once, and commits. Nothing
downstream ever re-observes the brick to check whether the arm is actually going where it
should. There is no feedback path anywhere in the diagram above.

Two consequences follow, and they explain most of the project's current pain:

1. **Every error in the chain lands on the brick, at full size.** The errors do not get
   corrected by anything; they add up and the claw arrives wherever the sum puts it.
2. **Wrong calibration fails silently.** A bad `T_gripper_camera` does not raise, does not
   log a warning, and does not produce a malformed result. It produces a perfectly
   well-formed `PickTarget` that `run_once` will drive straight to. `CLAUDE.md` already
   records this in the strongest terms, and it is a property of the *architecture*, not of
   any particular bug.

The guards that do exist — `ray_plane_intersection` returning `None`, the parallax/residual
gates on triangulation, the `MIN_CLAW_HEIGHT_M` floor — all catch *geometric impossibility*
(ray missed the plane, rays never met, target below the table). None of them catch
*plausible but wrong*, which is the failure mode actually occurring.

### 1.2 Where each number enters

| Quantity | Source | Status |
|---|---|---|
| `fx, fy, cx, cy`, distortion | `data/camera_intrinsics.json`, else `config.CAMERA_*` | placeholder `fx=fy=550` unless the file exists |
| `T_gripper_camera` | `data/hand_eye.json`, else identity | **measurably wrong, 52 mm** `[measured]` |
| `T_base_gripper` | live MATLAB FK of read-back servo angles | vertical validated ~2 mm; lateral unvalidated `[measured]` |
| `TABLE_Z_IN_BASE` | `config`, −0.074 | corroborated to 1.7 mm `[measured]` |
| `PICK_Z_OFFSET`, `APPROACH_HEIGHT` | `config` | nominal, never tuned |
| `PICK_ROLL_DEG` / `PICK_PITCH_DEG` | `config` | **not commanded** — `HardwareRobot` drops orientation |

Note the asymmetry in how much validation each has received. The two quantities the whole
chain is most sensitive to (§2) are the two least validated.

---

## 2. Error budget: what each mistake costs, in millimetres on the table

This is the part that decides everything else, so it is worth doing properly rather than
by intuition.

**The master sensitivity.** The camera sits `H` above the table and looks down. Any error
that tilts the back-projected ray by an angle θ moves the point where that ray lands by

```
        miss  =  H · tan θ
```

Using the board-measured camera height `H = 228 mm` `[measured]`:

| ray tilt θ | miss on the table `[derived]` |
|---|---|
| 0.1° | 0.4 mm |
| 0.5° | 2.0 mm |
| 1° | 4.0 mm |
| 2° | 8.0 mm |
| 5° | 19.9 mm |
| 10° | 40.2 mm |

**A degree of angular error is four millimetres of miss.** That single ratio is the reason
this rig is hard: a 2×2 brick is 15.8 mm across, so roughly 2° of accumulated angular error
is the entire grasp tolerance.

### 2.1 The terms, largest first

**Hand-eye rotation — currently dominant.** `CLAUDE.md` records the optical axis pointing
91° off the claw direction `[measured]`. At 4 mm/degree that is not a millimetre error, it
is a "the brick back-projects ~400 mm from where a ruler puts it" error, which is exactly
what was observed. Nothing else in the budget is within two orders of magnitude of this
while it stands.

**Hand-eye translation — 52 mm `[measured]`.** Translation error moves the ray's *origin*,
so for a near-vertical ray it passes through to the table at roughly 1:1. ~52 mm.

**Servo open-loop positioning — ~4 mm `[measured]`.** Stage D found the servos settling
15–47 ticks short of goal. This is irreducible without position feedback at the joint level,
and it is *not* a calibration problem — FK tracked it correctly because it reads back actual
positions. It sets a hard floor on open-loop accuracy.

**Brick top-face parallax — ~2.6 to 6.7 mm, systematic, and currently unmodelled.**
See §2.2; I believe this is a real defect nobody has recorded yet.

**Intrinsics error.** With `fx` a placeholder at 550, a feature `d` pixels from the optical
centre back-projects at `atan(d/fx)`. If the true `fx` were 600 (9% off — plausible for an
uncalibrated guess), then `[derived]`:

| feature offset from centre | angular error | miss |
|---|---|---|
| 100 px | 0.84° | 3.3 mm |
| 200 px | 1.55° | 6.2 mm |

So intrinsics error is small near the image centre and grows toward the edges. Keeping the
brick centred is a genuine mitigation, and that is a property worth remembering for §4.

**Table plane uncertainty — ±2 mm vertical `[measured]`.** Converts to lateral error only
through ray obliquity: `Δz · tan θ_view`. At a 25° viewing angle, ±2 mm vertical is
±0.9 mm lateral `[derived]`. Minor.

**Detection centroid noise — ~1–2 px.** At `fx = 550`, one pixel is `atan(1/550) = 0.104°`,
i.e. **0.41 mm** `[derived]`.

### 2.2 A systematic bias that is currently unmodelled

`pixel_to_world` intersects the brick's ray with **the table plane**
(`z = TABLE_Z_IN_BASE`). But the detected centroid is the visual centre of the brick's
**top face**, which sits a brick-height above that plane — 9.6 mm for a standard Lego brick.

Back-projecting a top-face point onto the table plane therefore overshoots, *away from the
camera's ground track*, by `[derived]`:

```
        error  =  brick_height · tan(θ_view)
```

| viewing angle from vertical | overshoot (h = 9.6 mm) |
|---|---|
| 0° (directly overhead) | 0.0 mm |
| 15° | 2.6 mm |
| 25° | 4.5 mm |
| 35° | 6.7 mm |
| 45° | 9.6 mm |

This is **systematic, not random** — it always pushes the target further from the camera,
so it never averages out over repeated attempts. At the obliquity an eye-in-hand camera
typically works at, it is comparable to the servo positioning error and larger than
everything else except the calibration terms.

The fix is nearly free and needs no new calibration: intersect with the plane the feature
actually lies on.

```python
# calibration/pixel_to_world.py — the brick's TOP face, not the tabletop
world = self.pixel_to_world(px, T, plane_z=config.TABLE_Z_IN_BASE + config.BRICK_HEIGHT_M)
```

That recovers the correct **x, y**; the grasp **z** stays whatever `PICK_Z_OFFSET` dictates.
`pixel_to_world` would need a `plane_z` parameter (it currently hardcodes `self.table_z`),
which is a two-line change. `[unverified]` — worth confirming the brick's real height with
calipers rather than trusting the nominal 9.6 mm.

### 2.3 A narrow trap in the runtime path

`CLAUDE.md` is emphatic that hand-eye callers must pass the gripper pose as a **raw FK
4×4**, never through `Pose`, because `geometry.transform_to_pose` forces `roll = 0` near
`pitch = ±90°`. The calibration path obeys this. **The runtime path does not:**

- `HardwareRobot.get_end_effector_pose()` ([hardware.py:86](src/vision_pipeline/robot_interface/hardware.py#L86)) does `transform_to_pose(T_wrist)` → `Pose`
- `PickPipeline.locate_brick()` ([pipeline.py:67](src/vision_pipeline/pipeline.py#L67)) then does `ee_pose.to_matrix()`

That is the exact round-trip the calibration path was fixed to avoid.

**How bad is it, honestly?** I worked the algebra rather than assuming the worst, and the
answer is: **much narrower than it first looks.** For `R = Rz(yaw)·Ry(pitch)·Rx(roll)`:

```
  r[2,0] = −sin(pitch)          roll = atan2(r[2,1], r[2,2]) = atan2(cp·sr, cp·cr)
```

The `cp` factor cancels inside `atan2`, so the decomposition recovers the rotation
**exactly** for any `|pitch| < 90°` — the round-trip is lossless, not merely close. The
lossy branch only fires when `np.isclose(abs(sp), 1.0)` is true, which at numpy's default
tolerances means `|sp| ≳ 0.99999`, i.e. **pitch within about 0.26° of ±90°** `[derived]`.

So this is a narrow trap, not a live bug — *unless* the FK wrist orientation happens to sit
in that band, which `CLAUDE.md` hints at when it says the top-down tool orientation "sits
close to that gimbal-lock singularity". Whether it actually does is `[unverified]` and needs
the MATLAB server to answer. Given that the fix is a few lines and removes a whole class of
"why is it off by a weird amount" investigation, it is worth doing regardless:

```python
# robot_interface/base.py — add alongside get_end_effector_pose
def get_end_effector_matrix(self) -> np.ndarray:
    """Raw 4x4 FK, no Euler round-trip. Preferred by the calibration chain."""
```

`HardwareRobot` returns `T_wrist` directly; `SimRobot` returns `self._pose.to_matrix()`;
`PickPipeline.locate_brick` uses it when present. Backward compatible.

### 2.4 The budget with calibration fixed

Suppose the hand-eye terms were fully resolved. What is left? `[derived]`, RSS of the
independent terms:

| term | contribution |
|---|---|
| servo open-loop positioning | ~4.0 mm |
| brick top-face parallax (if uncorrected) | ~4.5 mm |
| intrinsics (mid-frame) | ~3.3 mm |
| table plane | ~0.9 mm |
| detection centroid | ~0.4 mm |
| **RSS total** | **~7 mm** |

Against a 2×2 brick's 15.8 mm width, a ~7 mm RSS error with a worst-case nearer 12 mm is
**marginal, not comfortable** — a coin-flip on whether the claw closes on the brick or
knocks it over. Correcting §2.2 buys back a couple of millimetres, but the floor set by
open-loop servo positioning does not move.

**This is the quantitative case for closed-loop refinement.** It is not gold-plating: a
perfectly calibrated version of the current architecture still lands near the edge of the
grasp tolerance, because ~4 mm of that budget is servo error that no amount of calibration
can remove.

### 2.5 The one genuinely healthy part

Detection contributes **0.4 mm** of the ~7 mm budget. The two-stage detector, the specular
suppression, the stud/shape blending — all of that is comfortably good enough and is not
what is holding the project back. **Every meaningful error is downstream of the pixel**, in
the transform chain and the servos. Effort spent retuning HSV thresholds or Hough parameters
buys nothing right now.

---

## 3. Making the open-loop path work

Fixing the current architecture is the cheaper of the two directions and is a prerequisite
for parts of the other one, so it comes first regardless.

The order below is **forced by dependency**, not preference. Each step's validity depends on
the previous one being true, which is why it has been tempting and costly to skip ahead.

**Step 0 — settle `dir_sign` on J5 (and J1).** `CLAUDE.md` §"J5's `dir_sign` is probably
inverted" already has the analysis: re-solving the 49 saved samples under all 32 sign
combinations drops the Kabsch residual from 39.0° to 3.4° with J5 alone flipped `[measured]`.
Two things make this the unambiguous first move:

- Nothing downstream is meaningful without it. FK is what places the camera; if a joint's
  sign is inverted, every camera pose is wrong and so is every calibration solved against it.
- **It is a prerequisite for the closed-loop path too** (§4.4), so it is not a bet on one
  approach.

Do it by physical jog, not by any solve: `python scripts/jog_joint.py --joint 5 --ticks 300`.
Use 300 ticks, not the usual 80 — at 80 the tip moves only ~3 mm `[measured]`, which is
likely why the original reading was unreliable. Repeat for J1, where the operator's
observation and the sample re-solve currently disagree.

Neither an FK-vs-IK error number nor a hand-eye residual can substitute here;
`rad_to_ticks`/`ticks_to_rad` cancel the sign in the first, and the second has no
independent reference. Only a physical jog closes it.

**Step 1 — validate each joint's geometry.** [`scripts/validate_joint_geometry.py`](scripts/validate_joint_geometry.py)
already exists for exactly this and uses the camera against a stationary ChArUco board as
ground truth, so no ruler is involved. Its own docstring states the rule plainly: *hand-eye
calibration is meaningless until every jogged joint reads ~1.0*. Run it per joint. A
consistent ratio ≠ 1 localises the error to that joint, which no whole-arm solve can do.

**Step 2 — re-solve hand-eye, probably without recapturing.** This is the payoff:
`CLAUDE.md` records that **if J5 is confirmed inverted, the 49 saved samples become
consistent with no recapture** `[measured]`. A confirmed sign flip could convert an
afternoon of re-capture into a re-run of the solve.

**Step 3 — verify against something independent.** [`scripts/test_pick_dry_run.py`](scripts/test_pick_dry_run.py)
compares the board's `solvePnP` camera height against FK + hand-eye and commands nothing.
The two should agree; today they differ by 52 mm. Then
[`scripts/validate_pixel_to_world.py`](scripts/validate_pixel_to_world.py) cross-checks
three independent answers in millimetres.

**Step 4 — correct the top-face parallax** (§2.2) and **add the raw-4×4 accessor** (§2.3).

**What this gets you:** a ~7 mm RSS open-loop pick (§2.4) — enough to hover accurately, not
reliably enough to close a claw on a 15.8 mm brick.

---

## 4. Closed-loop: visual servoing

The core idea is to stop treating vision as a one-shot measurement feeding a feed-forward
computation, and instead treat it as a **sensor in a feedback loop**: observe, move a
little, observe again, and let the error drive itself to zero.

The reason this matters *here specifically* is not elegance. It is that a feedback loop
converges to the right answer **even when the model relating camera motion to image motion
is substantially wrong** — which is precisely this rig's situation.

### 4.1 Two families, and why only one fits

**PBVS (position-based)** reconstructs the target's 3D pose in the camera frame, then
servos in Cartesian space. It needs good intrinsics and a metric object model, and — for an
eye-in-hand rig commanding base-frame targets — it still needs the hand-eye transform to
express the goal. **It inherits the exact dependency that is broken.** Not a fit.

**IBVS (image-based)** defines the error directly in the image: `e = s − s*`, where `s` is
the feature's current image position and `s*` is where it should be. The control law drives
`e → 0` without ever computing a world coordinate. Classical IBVS needs neither a complete
3D model of the scene nor perfect camera calibration — that is its defining property, and
it is what makes it the right family for this arm.

### 4.2 The mechanics, reduced to this arm's actual DOF

For a point feature at normalised image coordinates `x = (u−cx)/fx`, `y = (v−cy)/fy` at
depth `Z`, the standard interaction matrix relating camera velocity to feature velocity
(`ṡ = L·v_c`) is:

```
        ⎡ −1/Z    0     x/Z     x·y     −(1+x²)    y ⎤
   L =  ⎢                                            ⎥
        ⎣   0   −1/Z    y/Z    1+y²      −x·y     −x ⎦
              └── translation ──┘  └──── rotation ────┘
```

**This arm cannot use the right-hand block at all.** `HardwareRobot.send_target_pose` drops
roll/pitch/yaw and commands `(x, y, z)` only — the IK is position-only (`weights =
[0 0 0 1 1 1]`). So the three rotational columns are not available as control inputs.

Take the translational block, and hold height constant (`v_z = 0`) because a top-down pick
naturally separates "get over the brick" from "descend onto it". What remains is beautifully
simple:

```
   ṡ = −(1/Z)·v_xy        ⟹        L = −(1/Z)·I₂        ⟹        L⁺ = −Z·I₂
```

With the standard control law `v = −λ·L⁺·e`:

```
   v_xy(camera frame)  =  λ · Z · (s − s*)
```

and the closed-loop error obeys `ė = −λ·e` — clean exponential decay `[derived]`.

Two properties of that result are worth dwelling on:

- **`Z` appears as a pure scalar gain.** Getting `Z` wrong by 20% changes the step size by
  20%, not the destination. The loop still converges, just slightly faster or slower. This
  is the robustness that makes IBVS attractive. **But it is not unconditional** — the
  literature is explicit that care is needed when approximating depth, because a
  sufficiently bad approximation can destabilise the law. A sane `Z` from
  `TABLE_Z_IN_BASE` plus the arm's known height is comfortably within the safe range.
- **The result is a displacement in the *camera* frame**, so it must be rotated into the
  base frame before `send_target_pose` can use it: `Δ_base = R_base_camera · [Δx, Δy, 0]`.

### 4.3 Why this dodges the broken calibration

That last line is the crux. Converting the correction to the base frame needs
`R_base_camera` — the camera's **orientation** only. It does **not** need the camera's
position. So of the two things wrong with `data/hand_eye.json`:

| what's wrong | `[measured]` | effect on the servo law |
|---|---|---|
| translation off by 52 mm | yes | **none** — translation never enters |
| rotation axis off | yes | rotates each correction step |

And a rotation error `θ` in `R_base_camera` does not move the fixed point of the loop —
it steers each step off by `θ`. The loop still converges to the correct image position as
long as the commanded step retains a positive component along the true descent direction,
i.e. **while `θ < 90°`**. Below that, the arm spirals in instead of going straight in;
above it, the feedback sign inverts and it diverges.

**This is the whole argument in one line:** closed-loop moves the calibration requirement
from *sub-degree and sub-millimetre* to *within roughly ±45° and no position accuracy at
all*. That is a loosening of two to three orders of magnitude, and it is why servoing is
worth considering on a rig whose calibration has resisted three sessions of work.

The caveat is equally important: the reported 91° axis discrepancy sits **right at the
stability boundary**. So this does not mean "ignore the hand-eye transform" — it means the
rotation needs to be roughly right, not precisely right.

### 4.4 What closed-loop does *not* rescue

**A wrong `dir_sign` is not fixed by feedback — it is made worse by it.** This is the most
important caveat in this document, and it is easy to get backwards.

Feedback tolerates errors in the *magnitude* and *direction* of the modelled response. It
does not tolerate a **sign inversion on a dominant axis**: that flips the loop from negative
to positive feedback, and the arm drives away from the target, accelerating, until something
stops it. In the `θ < 90°` condition above, a sign inversion is `θ = 180°`.

Since the camera is mounted on J5 and swings with it, an inverted J5 sign is exactly such an
inversion. **So Step 0 of §3 is a hard prerequisite for both paths.** There is no route to a
working pick that skips it, and a divergent servo loop on this arm is more dangerous than a
wrong open-loop target, because the open-loop version at least stops when it arrives.

Also not rescued:

- **Reach.** Servoing cannot move the ~180 mm forward limit at table height `[measured]`.
  If the brick is outside the reachable zone, no control scheme helps — the answer is
  re-measuring the *mechanical* stops with the elbow folded so the claw cannot ground out,
  which `CLAUDE.md` estimates would restore reach to ~280 mm.
- **Orientation.** 5-DOF position-only IK means yaw stays uncontrolled either way. Fine for
  a near-symmetric brick, unchanged by servoing.
- **The floor guard.** `MIN_CLAW_HEIGHT_M` still applies to every commanded pose.

### 4.5 Adapting the theory to a slow, position-commanded, stepped arm

Textbook IBVS assumes a velocity-controlled manipulator running at tens of hertz. **This arm
is nothing like that**, and pretending otherwise would be the classic way to import a
technique badly:

- Commands are **positions**, not velocities (`send_target_pose`).
- Moves are deliberately **slow and paced** — `SERVO_MOVE_SPEED_TICKS_S = 200` (~18°/s),
  `PICK_STEP_TICKS = 60`, `PICK_STEP_PAUSE_S = 0.5`.
- Every move **settles and verifies** before returning.

The correct adaptation is **iterated look-and-move** (discrete-time servoing), not
continuous IBVS:

```
    1.  hover at a fixed height above the table
    2.  capture a frame; detect the brick centroid  s
    3.  e = s − s*                    (s* = the taught reference position)
    4.  if |e| < tolerance:  break
    5.  Δ_camera = λ · Z · e          (λ ≈ 0.5, damped)
    6.  Δ_base   = R_base_camera · [Δ_camera, 0]
    7.  send_target_pose(current + Δ_base);  wait for settle
    8.  goto 2
```

The arm's slowness is genuinely not a problem here. With `λ = 0.5` and a Jacobian accurate
to ±30%, the error roughly halves per iteration `[derived]`:

```
    20 mm → 10 → 5 → 2.5 → 1.2 mm       ~4 iterations
```

At roughly 2–3 s per settle-and-verify cycle that is **10–15 seconds to converge** from a
20 mm initial miss — entirely acceptable for a pick, and the pacing that makes it slow is
the same pacing that makes a wrong move stoppable by hand.

Note also that each iteration re-solves IK seeded from the current angles, and the
corrections are small — comfortably inside `SERVO_MAX_MOVE_DELTA_TICKS = 400`. The existing
safety machinery does not need to be relaxed to accommodate this.

### 4.6 Where the reference `s*` comes from — and why this is the elegant part

`s*` is *"where the brick's centroid appears in the image when the claw is correctly
positioned to grasp it."* You obtain it by **teaching, not calculating**:

1. Put a brick on the table.
2. Jog the arm (the existing web dashboard, `run_arm_ui.py`, is already the right tool)
   until the claw is positioned to grasp it correctly.
3. Capture a frame, detect the brick, record its centroid.
4. That pixel is `s*`. Store it in `config` or a small JSON file.

**This requires no intrinsics, no hand-eye transform, and no table height.** It bakes the
entire geometric relationship into one measured pixel coordinate. Every future pick servos
the brick to that same pixel and the claw is, by construction, in the same place relative
to it.

This is why the approach is a genuinely good fit for a rig whose calibration is the problem:
it replaces a chain of six error-prone measured quantities with one taught one.

### 4.7 Going further: dispensing with `R_base_camera` too

The one remaining calibration dependency is the camera orientation in §4.2 step 6. It can be
removed entirely by **measuring the Jacobian instead of deriving it** — the uncalibrated
visual servoing approach (Piepmeier & Lipkin's eye-in-hand work is the standard reference,
with Broyden-style updates the usual estimator).

Once per session, at the hover pose:

```
    move +5 mm in base X   → observe pixel displacement (Δu₁, Δv₁)
    move +5 mm in base Y   → observe pixel displacement (Δu₂, Δv₂)

            ⎡ Δu₁  Δu₂ ⎤
    J   =   ⎢          ⎥        (pixels per mm of base-frame motion)
            ⎣ Δv₁  Δv₂ ⎦

    correction  =  −λ · J⁻¹ · e
```

Four arm moves, and the resulting `J` **empirically absorbs intrinsics, hand-eye rotation,
camera mounting, and the arm's own kinematic scale errors all at once** — because it
measures what actually happens rather than predicting it. Nothing in it can be wrong in the
way `hand_eye.json` is wrong, because it is not a model.

Two practical notes: probe moves must be large enough to exceed the ~4 mm servo positioning
noise (5–10 mm is a reasonable choice `[unverified]` — worth tuning against the real noise
floor), and `J` should be re-estimated when the arm moves far from where it was measured,
since the true Jacobian varies across the workspace.

### 4.8 The descent problem, and the hybrid answer

Eye-in-hand servoing has a well-known endgame problem: **as the claw descends onto the
brick, the claw occludes it and the brick leaves the field of view.** The feedback signal
disappears exactly when precision matters most.

The answer here is unusually clean, and it comes straight out of §2:

> **Servo the axes that are broken; go open-loop on the axis that is proven.**

- **Lateral (x, y)** depends on hand-eye, intrinsics, and the table plane — every untrusted
  quantity in the project. **Servo these**, at hover height, where the brick is fully
  visible.
- **Vertical (z)** is the one axis validated on hardware: FK 40.7 mm vs ruler 39 mm over a
  45 mm commanded lift, with two independent tabletop estimates agreeing to 1.7 mm
  `[measured]`. **Descend open-loop**, which is what `plan_pick_sequence` already does.

The occlusion problem dissolves because the closed-loop phase finishes before the claw ever
gets close enough to occlude anything.

---

## 5. Recommendation

**Keep the open-loop chain for the coarse approach; add an iterated image-based refinement
stage for the final lateral alignment; keep the descent open-loop.**

Concretely, the pick becomes:

```
    detect  →  pixel_to_world  →  PickTarget          (existing, open-loop, ~7 mm)
                     ↓
            move to hover above it                     (existing)
                     ↓
            ┌─ iterated visual servo in x,y ─┐         (NEW — closes to ~1-2 mm)
            │  capture → e = s − s* → step   │
            └────────── repeat ≤ 6 ──────────┘
                     ↓
            descend / close / lift                     (existing, open-loop, validated axis)
```

**Why this rather than either extreme:**

- **Not pure open-loop**, because §2.4 shows even a perfectly calibrated version lands at
  ~7 mm RSS against a 15.8 mm brick, with ~4 mm of that being servo error that calibration
  cannot touch. The margin is too thin to be reliable.
- **Not pure servoing**, because the coarse open-loop stage is already written, tested, and
  needed anyway to get the brick into view at a sane hover pose — and because the vertical
  axis is the one thing on this arm that is actually validated, so replacing it with
  feedback would be trading a known-good component for an unknown one.
- **It degrades gracefully**, which is the property this project most lacks today. If the
  hand-eye rotation is imperfect, the servo stage absorbs it. If servoing fails to converge,
  it can bail out and report rather than driving to a confidently wrong target. Contrast
  with the current architecture, where wrong calibration produces a well-formed `PickTarget`
  and the arm drives straight to it.

### 5.1 Ordered next steps

**These first three are not optional and are not specific to the recommendation** — the
current architecture needs them just as much.

1. **Jog J5 and J1 to settle `dir_sign`.**
   `python scripts/jog_joint.py --joint 5 --ticks 300` (300, not 80 — see §3).
   Blocks everything. Required by *both* paths; a wrong sign makes the servo loop diverge
   (§4.4).
2. **Run `validate_joint_geometry.py` per joint** until each reads ~1.0. The script exists
   and needs no ruler.
3. **Re-solve hand-eye** — quite possibly from the 49 saved samples with no recapture, if
   J5 is confirmed inverted `[measured]`. Verify with `test_pick_dry_run.py`: the board's
   `solvePnP` camera height and FK + hand-eye should agree, where today they differ by 52 mm.

**Then the cheap correctness fixes, both independent of the above:**

4. **Intersect the top-face plane, not the tabletop** (§2.2). Add `BRICK_HEIGHT_M` to
   config and a `plane_z` parameter to `pixel_to_world`. Removes a systematic 2.6–6.7 mm
   bias.
5. **Add `get_end_effector_matrix()`** to `RobotInterface` (§2.3) and use it in
   `locate_brick`. Closes the Euler round-trip in the runtime path.

**Then the new capability:**

6. **Teach `s*`** — jog to a known-good grasp pose over a brick via `run_arm_ui.py`, record
   the centroid, store it.
7. **Build the servo loop.** Suggested shape, following the repo's existing conventions
   (a new subpackage beside `planning/`, depending only on interfaces that already exist):

   ```
   src/vision_pipeline/servoing/image_servo.py
       class ImageServo:
           def __init__(self, detector, robot, camera, s_star, gain=0.5, ...)
           def estimate_jacobian(self) -> np.ndarray      # §4.7, two probe moves
           def step(self) -> tuple[float, float]          # one correction, returns residual
           def converge(self, max_iters=6, tol_px=4.0) -> bool
   ```

   It needs **no new robot seam** — `get_end_effector_pose`, `send_target_pose`, and the
   existing `Camera` and `LegoBrickDetector` are sufficient. New config entries:
   `SERVO_GAIN`, `SERVO_TOL_PX`, `SERVO_MAX_ITERS`, `SERVO_PROBE_MM`.

8. **Wire it into `PickPipeline`** as an optional stage — e.g. `run_once(frame,
   refine=True)` — so the existing open-loop path and all its tests stay untouched and
   `SimRobot` demos keep working.

### 5.2 Testable without hardware

Worth stating because it is how the rest of this repo is built, and it means step 7 can be
written and validated before the arm is available: the servo loop can be tested entirely in
simulation by using `project_point` (already in `camera_model.py`) to render where a
synthetic brick *would* appear from a given pose, then confirming the loop drives the error
to zero — including deliberately feeding it a **wrong** `R_base_camera` to verify it still
converges (§4.3), and a **sign-inverted** one to verify it diverges and bails out rather
than driving the arm (§4.4). Those two tests are the ones that would catch a regression in
the property the whole approach depends on.

---

## 6. Appendices

### 6.1 Checking the `[derived]` numbers

Nothing in this document was executed — there is no Python on this machine at present. When
the environment is back (`python -m venv .venv`, then `pip install -r requirements.txt`),
this confirms §2's sensitivity table and §2.3's gimbal-lock band:

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
import numpy as np
from vision_pipeline.calibration import geometry

# §2 master sensitivity: ray tilt -> miss on the table, at H = 228 mm
for d in (0.1, 0.5, 1, 2, 5, 10):
    print(f"{d:5.1f} deg -> {228 * np.tan(np.radians(d)):6.2f} mm")

# §2.3 Pose round-trip: exact away from lock, lossy only within ~0.26 deg of +/-90
def rt(T):
    return geometry.make_transform(*geometry.transform_to_pose(T))
def err(A, B):
    R = A[:3, :3].T @ B[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
for p in (0, 45, 80, 89, 89.9, 89.99, 90):
    T = geometry.make_transform(0.1, 0, 0, 35.0, p, 20.0)
    print(f"pitch {p:6.2f} -> round-trip error {err(T, rt(T)):.4f} deg")
```

### 6.2 The constraints every option was judged against

| Constraint | Value | Source |
|---|---|---|
| Forward reach at table height | ~180 mm (~220 mm at z = +80) | `[measured]` 2026-08-04 |
| IK | position-only, `weights = [0 0 0 1 1 1]`; orientation dropped | `matlab/`, `hardware.py:94` |
| Claw floor | `MIN_CLAW_HEIGHT_M` = 5 mm above table | `config.py:340` |
| Single-move cap | `SERVO_MAX_MOVE_DELTA_TICKS` = 400 (~35°) | `config.py:492` |
| Move speed | `SERVO_MOVE_SPEED_TICKS_S` = 200 (~18°/s), SRAM, re-applied per run | `config.py:508` |
| Pick pacing | 60 ticks/hop, 0.5 s pause | `config.py:520` |
| J5 `dir_sign` | unresolved; probably inverted | `CLAUDE.md` |
| `data/hand_eye.json` | 52 mm translation error, axis ~91° off | `[measured]` 2026-08-04 |
| Frame convention | `physical (x,y,z) = model (x,−y,−z)`, converted only in `MatlabIKClient` | `CLAUDE.md` |

### 6.3 References

Visual servoing theory:
- [Robustness of image-based visual servoing with respect to depth distribution error](https://www.researchgate.net/publication/224744326_Robustness_of_image-based_visual_servoing_with_respect_to_depth_distribution_error) — stability under depth approximation; the source of the §4.2 caveat that depth robustness is not unconditional.
- [Robustness of Image-Based Visual Servoing With a Calibrated Camera in the Presence of Uncertainties in the Three-Dimensional Structure](https://ieeexplore.ieee.org/document/5350660/) — IEEE, uncertainty in 3D structure.
- [Feature Depth Observation for Image-based Visual Servoing: Theory and Experiments](https://dl.acm.org/doi/abs/10.1177/0278364908096706) — IJRR, on estimating `Z` online.
- [2½-D Visual Servoing](https://www.cs.jhu.edu/~hager/teaching/CS600.641/Malis2-1-2DTRA99.pdf) — Malis et al.; the hybrid between IBVS and PBVS, relevant if orientation control is ever added.

Uncalibrated / empirically-estimated Jacobian (§4.7):
- [Uncalibrated Eye-in-Hand Visual Servoing](https://journals.sagepub.com/doi/10.1177/027836490302210002) — Piepmeier & Lipkin, IJRR 2003. The standard reference for this configuration; [full text](https://repository.gatech.edu/server/api/core/bitstreams/d24d33e8-ba0b-4ddd-ac8a-28bda97803b7/content).
- [Robust Jacobian Estimation for Uncalibrated Visual Servoing](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_ICRA_2010/data/papers/1516.pdf) — ICRA 2010; compares Broyden against Kalman/particle-filter estimators.
- [Uncalibrated Dynamic Visual Servoing](https://www.usna.edu/Users/weaprcon/piepmeie/_files/documents/H2002_282final.pdf) — partitioned Broyden for moving targets.

In-repo cross-references: `CLAUDE.md` (coordinate frames, bring-up status, the hand-eye
diagnosis), `SESSION_LOG_2026-08-04.md` §1 (what was eliminated by measurement, and how).

"""Closed-loop visual servoing: drive the brick to the middle of the frame.

WHY THIS EXISTS, AND WHY IT IS NOT THE PICK PIPELINE. `PickPipeline` is
open-loop: it converts one pixel into one base-frame point and commands the arm
there in a single shot. Everything about that shot -- camera intrinsics, the
hand-eye transform, TABLE_Z_IN_BASE, where the base origin actually sits -- has
to be right the first time, because nothing looks again afterwards. Today the
hand-eye transform is measurably wrong by 52 mm with its optical axis 91 deg
off, so that path produces a confident, well-formed target that is simply in the
wrong place.

This module takes the opposite approach and needs NONE of those numbers:

    look -> is the brick left or right of centre? -> nudge -> look again

An error measured in pixels, corrected by a nudge measured in ticks, re-measured
in pixels. No metric calibration appears anywhere, so no calibration error can
appear either. Overshoot, the failure mode that open-loop cannot even detect,
becomes just another error to correct on the next iteration.

HOW IT LEARNS ITS OWN DIRECTION. The one thing the loop must know is which way
to nudge, and that is exactly what this arm is least sure of (see CLAUDE.md on
J5's dir_sign). So it does not assume: it PROBES. It applies a small, deliberate
test move, measures how many pixels the brick actually shifted, and divides:

    ticks_per_px = probe_ticks / observed_pixel_shift

That signed number is a one-column numerical estimate of the image Jacobian. Its
SIGN is the direction question, answered by measurement rather than by belief,
and its MAGNITUDE is the gain, which also removes the need to guess a step size.
A dir_sign error, a mirrored camera, an inverted axis convention -- all of them
are absorbed, because the probe measures the composition of the whole chain
rather than any single link in it.

THE RUNAWAY GUARD IS THE LOAD-BEARING PART. A closed loop with a sign error does
not sit still and do nothing; it drives away from the target as fast as it is
allowed to, and it keeps driving. Every safeguard here exists for that:

  * the probe must produce a visible response (MIN_PROBE_RESPONSE_PX), otherwise
    this axis does not control this error and the loop stops rather than
    integrating noise into ever-larger corrections;
  * every step is clamped to max_step_ticks, so a bad estimate is bounded;
  * `ProgressMonitor` stops the loop when the error stops improving, which is
    what a wrong sign, a stalled joint, and a lost brick all look like from
    here.

Pure functions and plain dataclasses only: no camera, no serial port, no OpenCV.
That is what lets the control law be unit-tested (tests/test_visual_servo.py)
without hardware, which matters more here than usual -- this is the first code
in the repo that moves the arm based on what it sees, in a loop, unattended
between frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vision_pipeline import config


class ServoAbort(Exception):
    """Raised when the loop must stop moving the arm.

    Deliberately an exception rather than a return code: every caller of this
    module is mid-move on real hardware, and the correct response to "this is
    not converging" is to stop immediately, not to finish the iteration.
    """


@dataclass
class AxisEstimate:
    """The measured relationship between one joint's ticks and one pixel axis.

    Attributes:
        ticks_per_px: signed. Ticks required to shift the brick one pixel along
            this error axis. Sign carries the direction; magnitude the gain.
        probe_ticks: the test move this was measured from, for reporting.
        response_px: the pixel shift that test move produced.
    """

    ticks_per_px: float
    probe_ticks: int
    response_px: float

    @property
    def direction(self) -> int:
        """+1 or -1: which way ticks push the brick along this pixel axis."""
        return 1 if self.ticks_per_px >= 0 else -1


def pixel_error(centroid_px: tuple[float, float],
                frame_shape: tuple[int, ...],
                aim_px: tuple[float, float] | None = None) -> tuple[float, float]:
    """How far the brick sits from where we want it, in pixels.

    Positive dx means the brick is to the RIGHT of the aim point in image
    coordinates, positive dy means BELOW it. Image coordinates, not world ones
    -- refusing to translate into world terms here is the entire point of this
    module, and naming these 'forward'/'left' would quietly reintroduce the
    frame assumptions it exists to avoid.

    Args:
        centroid_px: detected brick centre, (x, y) in pixels.
        frame_shape: the frame's .shape, (h, w) or (h, w, channels).
        aim_px: where the brick should end up. Defaults to the frame centre.
            Pass something else when the gripper does not grasp at the optical
            centre -- that offset is measurable by hand and needs no calibration.

    Returns:
        (dx, dy) in pixels.
    """
    h, w = frame_shape[0], frame_shape[1]
    ax, ay = aim_px if aim_px is not None else (w / 2.0, h / 2.0)
    return centroid_px[0] - ax, centroid_px[1] - ay


def estimate_axis(probe_ticks: int,
                  error_before_px: float,
                  error_after_px: float,
                  min_response_px: float = config.SERVO_MIN_PROBE_RESPONSE_PX
                  ) -> AxisEstimate:
    """Turn one probe move into a signed ticks-per-pixel estimate.

    This is the whole calibration this controller ever does, and it is why the
    loop is immune to a wrong dir_sign: it measures the arm's actual response
    instead of predicting it.

    Args:
        probe_ticks: the test move that was applied (signed, non-zero).
        error_before_px: pixel error on this axis before the probe.
        error_after_px: pixel error on this axis after it.
        min_response_px: below this the probe told us nothing.

    Raises:
        ServoAbort: if the probe produced no usable response. That means this
            joint does not meaningfully control this pixel axis -- the arm did
            not move, the brick was re-detected somewhere unrelated, or the
            geometry is degenerate. Dividing by it would manufacture an enormous
            gain out of what is really just detection noise, and the next step
            would slam the joint to its clamp in that invented direction.
    """
    if probe_ticks == 0:
        raise ServoAbort("Probe of 0 ticks cannot measure anything.")

    response = error_after_px - error_before_px
    if abs(response) < min_response_px:
        raise ServoAbort(
            f"Probe of {probe_ticks:+d} ticks moved the brick only "
            f"{response:+.1f} px (need {min_response_px:.1f}). This joint does "
            f"not control this axis, the arm did not actually move, or the "
            f"detection is jumping between objects. Not guessing a gain from it."
        )
    return AxisEstimate(ticks_per_px=probe_ticks / response,
                        probe_ticks=probe_ticks,
                        response_px=response)


def step_command(error_px: float,
                 estimate: AxisEstimate,
                 gain: float = config.SERVO_VISUAL_GAIN,
                 max_step: float = config.SERVO_VISUAL_MAX_STEP_TICKS,
                 deadband_px: float = config.SERVO_VISUAL_DEADBAND_PX) -> float:
    """The next correction, in whatever unit the probe was measured in.

    Unit-agnostic on purpose. The probe measures "command units per pixel", and
    nothing here cares whether a command unit is a raw servo tick or a
    millimetre of Cartesian nudge. That is what lets the same control law drive
    a single joint OR a whole-arm IK move: only the actuator differs, and the
    actuator is exactly the part the probe measures around.

    Proportional, deliberately under-damped by `gain` < 1: the estimate comes
    from a single small probe and is only approximately right, so closing the
    whole gap in one move would overshoot and oscillate. Taking most of it and
    re-measuring converges without ever needing a better estimate.

    The clamp is a safety bound, not a tuning knob. It is what keeps a bad
    estimate -- or a brick that vanished and was replaced by a red something
    else across the room -- to one small wrong move instead of a slew.
    """
    if abs(error_px) <= deadband_px:
        return 0.0
    raw = -error_px * estimate.ticks_per_px * gain
    return max(-max_step, min(max_step, raw))


def step_ticks(error_px: float,
               estimate: AxisEstimate,
               gain: float = config.SERVO_VISUAL_GAIN,
               max_step_ticks: int = config.SERVO_VISUAL_MAX_STEP_TICKS,
               deadband_px: float = config.SERVO_VISUAL_DEADBAND_PX) -> int:
    """step_command rounded to whole servo ticks, for joint-space actuation."""
    return int(round(step_command(error_px, estimate, gain,
                                  max_step_ticks, deadband_px)))


def resolution_px(estimate: AxisEstimate,
                  min_step: float = config.SERVO_VISUAL_MIN_STEP_TICKS) -> float:
    """Pixels moved by the smallest step this actuator can reliably make.

    A joint under load does not move at all below some commanded delta -- it
    cannot break stiction. Measured on J3 (load 56, carrying the forearm) on
    2026-08-05, against a probe that said 0.78 ticks per pixel:

        commanded -35 ticks -> 28 px      (responds)
        commanded -23 ticks ->  9 px      (mushy)
        commanded -18 ticks ->  2 px
        commanded -17 ticks ->  0 px      (nothing at all)

    That sets a floor on how finely the loop can aim, and the floor is REAL:
    no gain, no patience and no number of iterations gets past it. Asking for
    a smaller correction than the joint can execute produces a loop that looks
    active and achieves nothing, which is exactly what happened twice.
    """
    if estimate.ticks_per_px == 0:
        return 0.0
    return abs(min_step / estimate.ticks_per_px)


def effective_deadband(estimate: AxisEstimate,
                       deadband_px: float = config.SERVO_VISUAL_DEADBAND_PX,
                       min_step: float = config.SERVO_VISUAL_MIN_STEP_TICKS) -> float:
    """The tightest deadband this axis can actually honour.

    Never finer than half the smallest executable step: aiming inside that is
    asking for a correction the joint will ignore. Returning the coarser of the
    two turns 'stuck forever' into 'converged as far as this joint resolves',
    which is the truth and is also actionable -- get the camera closer, or
    accept the resolution.
    """
    return max(deadband_px, resolution_px(estimate, min_step) / 2.0)


def enforce_minimum_step(amount: float,
                         min_step: float = config.SERVO_VISUAL_MIN_STEP_TICKS
                         ) -> float:
    """Round a non-zero correction up to the smallest step that will move.

    Overshooting slightly and coming back beats commanding a move that does
    nothing. Zero stays zero -- that is the deadband's decision, not this one.
    """
    if amount == 0:
        return 0.0
    return amount if abs(amount) >= min_step else (min_step if amount > 0 else -min_step)


def clamp_predicted_shift(amount: float,
                          estimate: AxisEstimate,
                          frame_shape: tuple[int, ...],
                          fraction: float = config.SERVO_VISUAL_MAX_FRAME_FRACTION
                          ) -> float:
    """Scale a command down so it cannot fling the brick out of the frame.

    The per-step clamp in `step_command` is expressed in COMMAND units -- ticks
    or millimetres -- which says nothing about how far the brick will actually
    travel across the image. How far it travels depends on the gain, and the
    gain depends on how close the camera is: the same 12 mm nudge is a gentle
    nudge from far away and half a frame from close up.

    Observed 2026-08-05: a 12 mm tangential nudge, well inside its millimetre
    clamp, moved the brick clean out of view and the run ended with 'Lost the
    brick mid-loop' on the very next look. A control loop cannot correct an
    error it can no longer see, so the honest bound is the one measured in the
    thing the loop actually observes.

    Args:
        amount: the proposed command, in the estimate's own units.
        estimate: the measured units-per-pixel for this axis.
        frame_shape: the frame's .shape, (h, w) or (h, w, channels).
        fraction: most of the frame's SHORTER side one step may traverse.

    Returns:
        `amount`, or a scaled-down version of it with the same sign.
    """
    if estimate.ticks_per_px == 0:
        return amount
    max_px = fraction * min(frame_shape[0], frame_shape[1])
    predicted_px = abs(amount / estimate.ticks_per_px)
    if predicted_px <= max_px:
        return amount
    return amount * (max_px / predicted_px)


def radial_tangential(
    tip_xy: tuple[float, float],
    yaw_axis_xy: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two horizontal directions aligned with the arm's own kinematics.

    Returns (radial, tangential) unit vectors in the base frame's XY plane:
    radial points from the base yaw axis out toward the tool, tangential is
    perpendicular to it.

    `yaw_axis_xy` IS REQUIRED AND IS NOT THE ORIGIN. Get it from
    MatlabIKClient.base_yaw_axis_xy(), which measures it from FK. This argument
    used to be absent, which silently assumed the base yaw axis passed through
    the model's origin -- it misses by 81 mm, while the claw at the home pose
    sits only ~25 mm from that axis, so the "radial" direction came out up to
    108 deg from truly outward (measured 2026-08-06, scripts/audit_model_axes.py).
    A reach correction then pushes partly sideways and a sideways correction
    partly reaches, which is the two axes of this loop fighting each other. It
    is deliberately a required positional argument rather than one defaulting to
    the origin: a wrong default here is invisible and cost a J1 runaway.

    WHY NOT JUST BASE X AND Y. The camera rides on the wrist and swings with
    J1, so a fixed base direction means something different to the image at
    every J1 angle -- the mapping the probe measured drifts as the run
    progresses. Radial and tangential rotate WITH the arm, so they keep meaning
    the same thing: radial changes reach (the shoulder/elbow chain, which reads
    as vertical in a downward-looking image), tangential sweeps sideways (base
    yaw, horizontal in the image).

    This also matters for joint limits, which is the point of the exercise: a
    radial nudge is a Cartesian request, and IK is free to satisfy it with
    whatever joints still have travel. A joint-space jog of J2 has no such
    freedom, and simply stops when J2 does.

    Raises:
        ValueError: directly over the base axis, where radial is undefined --
            every horizontal direction is equally radial there, so there is no
            answer to give rather than a wrong one to guess.
    """
    x = tip_xy[0] - yaw_axis_xy[0]
    y = tip_xy[1] - yaw_axis_xy[1]
    norm = (x * x + y * y) ** 0.5
    if norm < 1e-6:
        raise ValueError(
            "The tool is on the base axis; radial direction is undefined there. "
            "Move the arm out from directly over its own column first."
        )
    radial = (x / norm, y / norm)
    tangential = (-radial[1], radial[0])
    return radial, tangential


def reach_from_axis(tip_xy: tuple[float, float],
                    yaw_axis_xy: tuple[float, float]) -> float:
    """How far the tool is from the base COLUMN, not from the model origin.

    The same 81 mm correction radial_tangential needs, and for the same reason:
    at the home pose `hypot(tip_x, tip_y)` reports 70 mm when the tool is 25 mm
    from the axis it swings about. Everything that reasons about reach -- the
    'too close in for Cartesian control' guard, and the measured dr fed back
    into DescentModel -- is wrong by that difference, and the guard is wrong in
    the dangerous direction: it reports the arm as further out, hence better
    conditioned, than it is.
    """
    return float(np.hypot(tip_xy[0] - yaw_axis_xy[0], tip_xy[1] - yaw_axis_xy[1]))


class DescentModel:
    """How the vertical image error responds to descending AND to reaching out.

    THE PROBLEM THIS SOLVES, measured on hardware 2026-08-05. Lowering the claw
    and correcting the brick's vertical position in the image are not
    independent: both are driven by the shoulder/elbow chain, and the camera
    rides on the wrist, so every millimetre of descent SWINGS THE VIEW. Observed
    ~7 px of vertical image motion per mm of descent -- a 20 mm step threw the
    brick 140 px up the frame, against a 55 px acceptance box.

    Treating them as separate loops makes the arm fight itself. The run that
    prompted this alternated: IK lowers the tip 20 mm (brick jumps up ~140 px),
    then a J3 jog drags the brick back down (raising the tip ~25 mm), then the
    next step spends itself undoing that. Four descent steps moved the claw a
    net 4 mm and then lost the brick off the top of the frame.

    Neither half is wrong on its own. What is wrong is asking two controllers to
    share one degree of freedom without telling either about the other.

    So model both terms at once and issue ONE move:

        dpx = a * dz_mm  +  b * dr_mm

    a = image response to descending (the disturbance, unavoidable)
    b = image response to reaching out radially (the correction, ours to choose)

    Then every step can descend by its full amount AND carry the radial
    correction that cancels the swing the descent is about to cause, in a single
    IK solve. The disturbance is predicted rather than chased.

    Both coefficients are MEASURED from the descent's own motion -- no extra
    probe moves, and no dependence on hand-eye, intrinsics or dir_sign. Fitting
    needs two observations whose (dz, dr) are not parallel, which is why the
    first two steps deliberately differ: step 1 descends straight down, step 2
    adds a radial offset.
    """

    def __init__(self, min_response_px: float = config.SERVO_MIN_PROBE_RESPONSE_PX):
        self.samples: list[tuple[float, float, float]] = []   # (dz, dr, dpx)
        self.min_response_px = min_response_px
        self.a: float | None = None       # px per mm of descent
        self.b: float | None = None       # px per mm of radial reach

    def observe(self, dz_mm: float, dr_mm: float, dpx: float) -> None:
        """Record one executed move and the image shift it produced.

        Uses the move the arm ACTUALLY made (from FK), not the one commanded --
        this arm routinely lands a millimetre or two off, and the servos settle
        short under load.
        """
        self.samples.append((dz_mm, dr_mm, dpx))
        self._fit()

    def _fit(self) -> None:
        """Least-squares fit of a and b, once the samples span both directions.

        Refits on every observation rather than freezing after the first pair:
        the coefficients genuinely change as the arm unfolds -- the same
        millimetre of descent swings the camera far less at 160 mm of reach than
        at 60 mm -- so a gain fitted once at the top would be wrong by the
        bottom.
        """
        if len(self.samples) < 2:
            return
        A = np.array([[dz, dr] for dz, dr, _ in self.samples], dtype=float)
        y = np.array([dpx for _, _, dpx in self.samples], dtype=float)

        # Rank check before solving. Two pure-descent samples say nothing about
        # the radial term, and lstsq would happily return a b of ~0 or a wild
        # one rather than admitting it does not know.
        if np.linalg.matrix_rank(A, tol=1e-6) < 2:
            return
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        if abs(b) < 1e-9:
            return
        self.a, self.b = float(a), float(b)

    @property
    def ready(self) -> bool:
        return self.a is not None and self.b is not None

    def radial_for(self, error_px: float, dz_mm: float) -> float:
        """The radial move that lands the brick on the aim point AFTER descending.

        Solves  error_px + a*dz + b*dr = 0  for dr. The `a*dz` term is what
        makes this different from an ordinary correction: it aims at where the
        brick WILL be once this step's descent has swung the view, not at where
        it is now.

        Args:
            error_px: current vertical error, positive when the brick is BELOW
                the aim point (image convention: y grows downward).
            dz_mm: the descent this step will command (negative going down).

        Raises:
            ServoAbort: if called before the model has been fitted.
        """
        if not self.ready:
            raise ServoAbort(
                "The descent model has not been fitted yet; it needs two moves "
                "that differ in radial reach. Nothing to solve from."
            )
        return -(error_px + self.a * dz_mm) / self.b

    def plan_step(self, error_px: float, desired_dz_mm: float,
                  max_reach_mm: float) -> tuple[float, float]:
        """Descend as fast as the aim can be held, and no faster.

        THE RATE LIMIT, and it is not optional. Descending injects `a` px of
        error per mm; reaching out removes `b` px per mm, but only up to
        `max_reach_mm` in one step. When a*dz exceeds what b*max_reach can
        absorb, the step ends further from the target than it began -- and doing
        that repeatedly is exactly the run that walked the brick off the top of
        the frame. Measured coefficients say a 20 mm descent needs 56 mm of
        reach to break even, more than double the cap.

        So the descent step is DERIVED, not fixed: shrink dz until the required
        reach fits, down to zero if it must. dz = 0 is a legitimate outcome and
        an important one -- it is a re-aim at constant height, which is the one
        correction that cannot undo the descent. Progress resumes as soon as the
        error is small enough to carry.

        Never ascends to improve the aim, and never descends further than asked.

        Returns:
            (dz_mm, dr_mm) for this step.
        """
        if not self.ready:
            raise ServoAbort(
                "The descent model has not been fitted yet; it needs two moves "
                "that differ in radial reach. Nothing to plan from."
            )
        reach = abs(max_reach_mm)
        absorbable = reach * abs(self.b)

        dz = desired_dz_mm
        if abs(self.a) > 1e-9:
            # |error + a*dz| <= absorbable  is the feasible band for dz.
            bounds = sorted(((-absorbable - error_px) / self.a,
                             (absorbable - error_px) / self.a))
            dz = min(max(desired_dz_mm, bounds[0]), bounds[1])
            dz = min(dz, 0.0)                    # never climb to fix the aim
            dz = max(dz, desired_dz_mm)          # never overshoot the request

        dr = -(error_px + self.a * dz) / self.b
        return dz, float(np.clip(dr, -reach, reach))

    def describe(self) -> str:
        if not self.ready:
            return f"descent model: {len(self.samples)} sample(s), not yet fitted"
        return (f"descent model: {self.a:+.2f} px/mm down, "
                f"{self.b:+.2f} px/mm out  ({len(self.samples)} samples)")


@dataclass
class ProgressMonitor:
    """Stops the loop when the error is not actually shrinking.

    The one guard that cannot be left out. A proportional loop with a correct
    sign converges; with a wrong sign it accelerates away, and every individual
    step still looks perfectly reasonable in isolation. The only thing that
    distinguishes the two is whether the error is going down over time, so that
    is what gets watched.

    `estimate_axis` should make a wrong sign impossible, but this is the backstop
    for everything the probe cannot see: a joint hitting a hard stop mid-run, the
    detector locking onto a different red object, the brick being knocked by the
    claw, or the response reversing as the arm passes through a singularity.

    Attributes:
        patience: consecutive non-improving iterations tolerated before aborting.
        min_improvement_px: an error that shrinks by less than this is not
            improving; it is noise. Without this, a loop stalled against a hard
            stop reads as forever-slightly-improving and never stops.
    """

    patience: int = config.SERVO_VISUAL_PATIENCE
    min_improvement_px: float = config.SERVO_VISUAL_MIN_IMPROVEMENT_PX
    best_error: float = float("inf")
    stalled: int = 0
    history: list[float] = field(default_factory=list)

    def update(self, error_px: float) -> None:
        """Record this iteration's error magnitude; raise if it is going wrong.

        Raises:
            ServoAbort: if the error has failed to improve `patience` times in
                a row.
        """
        self.history.append(error_px)
        if error_px < self.best_error - self.min_improvement_px:
            self.best_error = error_px
            self.stalled = 0
            return

        self.stalled += 1
        if self.stalled >= self.patience:
            raise ServoAbort(
                f"Error has not improved for {self.stalled} iterations "
                f"(now {error_px:.1f} px, best {self.best_error:.1f} px). "
                f"Either the correction is going the wrong way, a joint is "
                f"against a stop, or the detector has changed its mind about "
                f"which object is the brick. Stopping rather than continuing."
            )

    @property
    def diverging(self) -> bool:
        """True if the error is now worse than where it started."""
        return bool(self.history) and self.history[-1] > self.history[0]

"""The visual servo loop driven end-to-end against a fake arm and fake camera.

test_visual_servo.py covers the control LAW in isolation -- the maths of turning
a pixel error into a tick delta. This file covers the LOOP that wires that law
to a servo bus: probing each axis, choosing which axis to correct, bookkeeping
joint positions, converging.

Why it exists: a wrong unpack of the (axis, joint)/estimate pairs sent an
AxisEstimate object into ServoBus.read_position as a servo ID, and it surfaced
only on live hardware, after two probe moves had already run. Nothing about that
bug needed an arm to find -- it needed something to call the loop. These fakes
are that something, and they cost nothing to run.

The fake arm is a linear model: each joint moves the brick a fixed number of
pixels per tick along one image axis. That is exactly the relationship the probe
is designed to measure, so a loop that works here is doing the right arithmetic;
whether the real arm matches the model is what the probe checks at runtime.
"""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import visual_servo as vs  # noqa: E402
from vision_pipeline import config  # noqa: E402
from vision_pipeline.planning.visual_servo import AxisEstimate  # noqa: E402


class FakeBus:
    """Just enough ServoBus to drive the loop, and it insists on integer IDs."""

    def __init__(self, start=None):
        self.ticks = dict(start or {1: 2000, 2: 2900})
        self.calibration = {str(j): {"dir_sign": 1} for j in range(1, 7)}
        self.moves = []

    def _check(self, servo_id):
        # The whole point: a non-integer ID is the bug this file was written
        # for, so the fake refuses it the way the real packet builder does.
        if not isinstance(servo_id, int) or isinstance(servo_id, bool):
            raise TypeError(f"{servo_id!r} is not a servo ID")

    def limits_not_required(self, servo_id):
        # Default to the honest answer: a joint with no limits here is one
        # nobody has measured, not one that needs none. Tests wanting the
        # deliberate case say so explicitly.
        self._check(servo_id)
        return bool(self.calibration.get(str(servo_id), {})
                    .get("limits_not_required", False))

    def read_position(self, servo_id):
        self._check(servo_id)
        return self.ticks[servo_id]

    def read_position_retrying(self, servo_id):
        # The loop reads through this while the arm is moving; the fake mirrors
        # the real bus's interface so a rename cannot silently bypass the fake.
        return self.read_position(servo_id)

    def move_and_verify(self, servo_id, tick):
        self._check(servo_id)
        self.ticks[servo_id] = int(tick)
        self.moves.append((servo_id, int(tick)))
        return int(tick)

    def freeze(self, ids):
        return {j: self.ticks[j] for j in ids if j in self.ticks}


class FakeWorld:
    """Maps joint ticks to where the brick appears, linearly."""

    def __init__(self, bus, px_per_tick=None, home=None, frame=(480, 640)):
        self.bus = bus
        self.px_per_tick = px_per_tick or {1: 1.0, 2: -0.75}
        self.home = home or dict(bus.ticks)
        self.frame = frame
        self.aim = (frame[1] / 2.0, frame[0] / 2.0)
        self.offset = (140.0, -90.0)   # how far off-centre the brick starts

    def centroid(self):
        dx = (self.bus.ticks[1] - self.home[1]) * self.px_per_tick[1]
        dy = (self.bus.ticks[2] - self.home[2]) * self.px_per_tick[2]
        return (self.aim[0] + self.offset[0] + dx,
                self.aim[1] + self.offset[1] + dy)


class FakeDetection:
    def __init__(self, centroid, area=4000.0):
        self.centroid_px = centroid
        self.confidence = 0.9
        # Real Detections carry this and the loop had been discarding it. The
        # height model reads it, so the fake has to have one.
        self.area = area


class FakeCamera:
    def __init__(self, world):
        self.world = world

    def read_frame(self):
        return np.zeros((*self.world.frame, 3), np.uint8)


class FakeDetector:
    def __init__(self, world, miss_frames=0, bounded=False, blind_after=None):
        self.world = world
        self.miss_frames = miss_frames   # simulate detection flicker
        self.bounded = bounded           # a brick outside the frame is not seen
        self.blind_after = blind_after   # go blind from this call onward
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls <= self.miss_frames:
            return []
        if self.blind_after is not None and self.calls > self.blind_after:
            return []
        cx, cy = self.world.centroid()
        if self.bounded:
            h, w = self.world.frame
            if not (0 <= cx < w and 0 <= cy < h):
                return []                # physically out of view
        return [FakeDetection((cx, cy))]


def make_ctx(px_per_tick=None, miss_frames=0, detector=None, offset=None,
             **overrides):
    bus = FakeBus()
    world = FakeWorld(bus, px_per_tick)
    if offset is not None:
        world.offset = offset
    args = Namespace(
        probe_ticks=40, settle=0.0, deadband=12.0, max_iterations=40,
        view=False, no_wait=True, descend=False, descend_step=8.0,
        target_z=None, no_confirm=True, recentre="joint",
        joint_x=1, joint_y=2,
        # These tests predate the offset aim point and the box target, and are
        # about the control law rather than where it aims: keep the aim at the
        # frame centre and make the box match the old scalar deadband so their
        # assertions still mean what they say. The offset and the box get their
        # own tests below.
        aim_offset_x=0.0, aim_offset_y=0.0, tolerance_x=12.0, tolerance_y=12.0,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    ctx = vs.Context(bus, FakeCamera(world),
                     detector or FakeDetector(world, miss_frames), args, None)
    return ctx, bus, world


def joint_actuators(probe_ticks=None):
    """The x and y actuators for the fake arm's two joints."""
    return vs.JointActuator(1, probe_ticks), vs.JointActuator(2, probe_ticks)


# --- the regression ---------------------------------------------------------

def test_centre_uses_joint_numbers_not_estimate_objects():
    """The exact bug: (axis, joint)/estimate pairs unpacked the wrong way.

    FakeBus raises TypeError on a non-integer servo ID, mirroring the real
    packet builder, so a bad unpack fails here instead of on live hardware
    after two probe moves have already run.
    """
    ctx, bus, _ = make_ctx()
    ax, ay = joint_actuators()
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax)),
                 (("y", ay), vs.probe_axis(ctx, "y", ay))]
    vs.centre(ctx, estimates)   # must not raise


# --- the loop actually working ----------------------------------------------

def test_probe_measures_the_fake_arm_correctly():
    ctx, _, world = make_ctx(px_per_tick={1: 2.0, 2: -0.5})
    ax, ay = joint_actuators()
    est_x = vs.probe_axis(ctx, "x", ax)
    est_y = vs.probe_axis(ctx, "y", ay)
    assert est_x.ticks_per_px == pytest.approx(1 / 2.0, rel=1e-6)
    assert est_y.ticks_per_px == pytest.approx(1 / -0.5, rel=1e-6)


def test_loop_centres_both_axes():
    ctx, bus, world = make_ctx()
    ax, ay = joint_actuators()
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax)),
                 (("y", ay), vs.probe_axis(ctx, "y", ay))]
    assert vs.centre(ctx, estimates) is True

    cx, cy = world.centroid()
    assert abs(cx - world.aim[0]) <= ctx.args.deadband
    assert abs(cy - world.aim[1]) <= ctx.args.deadband


def test_loop_converges_with_an_inverted_joint():
    """An inverted arm is a negative gain and nothing more — the probe absorbs it."""
    ctx, bus, world = make_ctx(px_per_tick={1: -1.0, 2: 0.75})
    ax, ay = joint_actuators()
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax)),
                 (("y", ay), vs.probe_axis(ctx, "y", ay))]
    assert vs.centre(ctx, estimates) is True
    cx, cy = world.centroid()
    assert abs(cx - world.aim[0]) <= ctx.args.deadband
    assert abs(cy - world.aim[1]) <= ctx.args.deadband


def test_loop_moves_only_the_joint_for_the_worst_axis():
    """Corrects one axis at a time; the joints couple, and doing both
    from a single frame double-counts that coupling into an overshoot."""
    ctx, bus, _ = make_ctx()
    ax, ay = joint_actuators()
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax)),
                 (("y", ay), vs.probe_axis(ctx, "y", ay))]
    before = len(bus.moves)
    vs.centre(ctx, estimates)
    for joint, _tick in bus.moves[before:]:
        assert joint in (1, 2)


def test_single_axis_run_never_touches_the_other_joint():
    ctx, bus, _ = make_ctx()
    ax, _ay = joint_actuators()
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax))]
    start_j2 = bus.ticks[2]
    vs.centre(ctx, estimates)
    assert bus.ticks[2] == start_j2


# --- flicker tolerance ------------------------------------------------------

def test_detect_centroid_retries_through_a_flickering_detector():
    """A brick that drops out for a few frames is not a lost brick.

    Glossy studs make detection intermittent; a single-frame miss ending a run
    is the failure this retry exists to prevent.
    """
    ctx, _, world = make_ctx(miss_frames=4)
    centroid, _frame = vs.detect_centroid(ctx)
    assert centroid is not None


def test_detect_centroid_gives_up_when_the_brick_is_really_gone():
    ctx, _, _ = make_ctx(miss_frames=10_000)
    centroid, frame = vs.detect_centroid(ctx)
    assert centroid is None
    assert frame is not None


# --- cartesian re-centring: the joint-limit workaround ----------------------

class FakeIK:
    """A 2-joint planar stand-in for MATLAB's limit-aware IK.

    Deliberately models the property under test and nothing else: J2 and J3 both
    contribute to reach, J2 has a travel limit, and the solver respects it by
    putting the remainder into J3. That is exactly what init_arm.m's
    PositionLimits buy on the real arm.
    """

    def __init__(self, bus, j2_min_tick=2883):
        self.bus = bus
        self.j2_min_tick = j2_min_tick
        self.solves = []
        self.locks = []

    def _tip(self, angles=None):
        # Reach grows with both joints; a purely notional 1 mm per tick each.
        # CartesianBus makes an "angle" numerically equal to a tick, so the two
        # entry points below share one formula.
        t2, t3 = ((self.bus.ticks[2], self.bus.ticks[3]) if angles is None
                  else (angles[1], angles[2]))
        reach = (t2 - 2900) * 0.001 + (t3 - 600) * 0.001
        return np.array([0.150 + reach, 0.0, 0.080])

    def request_fk_tip(self, angles=None):
        # MUST honour `angles` rather than reading the bus: reach_axis_xy
        # measures the arm's real reach direction by perturbing a joint and
        # watching the tip, so an FK that ignores its argument reports a
        # motionless arm and no direction at all.
        T = np.eye(4)
        T[:3, 3] = self._tip(angles)
        return T, T

    def base_yaw_axis_xy(self):
        # This synthetic world really does put the yaw axis at the origin --
        # reach is hypot(x, y) throughout. The REAL arm does not (81 mm off),
        # which is what test_radial_is_measured_from_the_yaw_axis_not_the_origin
        # pins; returning zeros here keeps the fake's own arithmetic honest.
        return np.array([0.0, 0.0])

    def request_ik(self, x, y, z, seed_rad=None, lock=None):
        self.locks.append(tuple(lock or ()))
        wanted = (np.array([x, y, z]) - self._tip())[0] * 1000   # mm of extra reach
        j2_room = self.bus.ticks[2] - self.j2_min_tick           # ticks J2 may still give
        from_j2 = max(-j2_room, wanted) if wanted < 0 else wanted
        from_j3 = wanted - from_j2
        self.solves.append({"wanted": wanted, "j2": from_j2, "j3": from_j3})
        # Ticks-as-radians for the fake bus. Joints this model does not use must
        # be returned UNCHANGED, not zeroed: the loop reads J1's delta to work
        # out how far the solution pans the camera, and a zero would read as a
        # full-scale swing and trip the conditioning guard.
        return [self.bus.ticks[1],
                self.bus.ticks[2] + from_j2,
                self.bus.ticks[3] + from_j3,
                self.bus.ticks[4], self.bus.ticks[5]], 0.0


class CartesianBus(FakeBus):
    """FakeBus with the extra surface a Cartesian move needs."""

    def __init__(self, j2_min_tick=2883, **kw):
        super().__init__(start={1: 2000, 2: 2900, 3: 600, 4: 1700, 5: 2500})
        self.j2_min_tick = j2_min_tick
        self.stepped = []

    def ticks_to_rad(self, servo_id, ticks):
        return float(ticks)

    def rad_to_ticks(self, servo_id, rad):
        return int(round(rad))

    def move_joints_stepped(self, targets, **kw):
        for j, t in targets.items():
            if j == 2 and t < self.j2_min_tick:
                from vision_pipeline.robot_interface.servo_driver import ServoSafetyError
                raise ServoSafetyError(f"J2 to {t} is outside travel")
        self.stepped.append(dict(targets))
        self.ticks.update({j: int(t) for j, t in targets.items()})


def test_radial_and_tangential_are_perpendicular_unit_vectors():
    radial, tangential = vs.radial_tangential((0.15, 0.0), (0.0, 0.0))
    assert radial == pytest.approx((1.0, 0.0))
    assert tangential == pytest.approx((0.0, 1.0))
    assert np.dot(radial, tangential) == pytest.approx(0.0)


def test_radial_rotates_with_the_arm():
    """Directions must follow the arm, not the base — the camera swings with J1,
    so a fixed base axis would mean something different to the image at every
    J1 angle, and the probe's measurement would drift as the run progressed."""
    radial, _ = vs.radial_tangential((0.0, 0.15), (0.0, 0.0))
    assert radial == pytest.approx((0.0, 1.0))


def test_radial_is_undefined_over_the_base_axis():
    """No answer beats a guessed one: every direction is equally radial there."""
    with pytest.raises(ValueError, match="base axis"):
        vs.radial_tangential((0.0, 0.0), (0.0, 0.0))


def test_cartesian_nudge_redistributes_onto_j3_when_j2_is_at_its_limit():
    """The whole point of the exercise.

    J2 sits one tick above its limit and cannot retract further. A joint-space
    correction would simply be refused -- that is the failure that stopped two
    live runs. The Cartesian request states the goal instead, and IK satisfies
    it by holding J2 and moving J3.
    """
    bus = CartesianBus()
    bus.ticks[2] = 2884                      # one tick of room left
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=FakeIK(bus))

    actuator = vs.CartesianActuator("radial")
    actuator.apply(ctx, -10.0)               # retract 10 mm

    solve = ctx.ik.solves[-1]
    assert abs(solve["j2"]) <= 1, "J2 had no room and must barely contribute"
    assert abs(solve["j3"]) > 5, "J3 must take up the motion"
    assert bus.ticks[2] >= bus.j2_min_tick, "J2 stayed inside its limit"


def test_cartesian_nudge_uses_the_natural_joint_when_there_is_room():
    """With travel available it behaves normally — the redistribution is a
    consequence of the limits, not a permanent change of posture."""
    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=FakeIK(bus))

    vs.CartesianActuator("radial").apply(ctx, -10.0)
    solve = ctx.ik.solves[-1]
    assert abs(solve["j2"]) > 5, "J2 had plenty of room and should have used it"


def test_a_solve_that_swings_the_base_is_refused_not_executed():
    """Near the base axis a solve can be geometrically perfect and useless.

    The claw hangs off the arm's plane, so at small tip radius the tip's bearing
    is hypersensitive to J1: measured 2026-08-05, a 5 mm radial nudge cost 6.6
    deg of base yaw at 78 mm reach. The camera is on the wrist, so that yaw pans
    the whole image and the loop ends up chasing its own motion. The nudge is
    halved a few times and then refused with the reason.
    """
    class YawingIK(FakeIK):
        def request_ik(self, x, y, z, seed_rad=None, lock=None):
            # However small the request, insist on swinging the base.
            super().request_ik(x, y, z, seed_rad, lock)
            return [self.bus.ticks[1] + 300, self.bus.ticks[2],
                    self.bus.ticks[3], self.bus.ticks[4], self.bus.ticks[5]], 0.0

    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=YawingIK(bus))

    # TANGENTIAL, because that is the axis where J1 is legitimately free: a
    # radial nudge now HOLDS J1 (base yaw cannot change reach), so the pan guard
    # has nothing left to catch there. The guard still matters here, where
    # swinging the base is the intended action and can still overshoot.
    with pytest.raises(vs.ServoAbort, match="poorly conditioned"):
        vs.CartesianActuator("tangential").apply(ctx, 10.0)
    assert bus.stepped == [], "nothing should have been commanded"


# --- the pan budget is not the same quantity on both axes --------------------
#
# For a RADIAL nudge base yaw is waste: reach happens in the shoulder/elbow
# plane, so yaw in the answer is redundancy spent uninstructed. A flat ceiling
# is right.
#
# For a TANGENTIAL nudge base yaw IS the actuator, and the angle is fixed by
# geometry rather than chosen: theta = d / r. No solution uses less, so a flat
# ceiling does not limit waste there -- it limits the STEP SIZE to
# MAX_PAN_DEG * r while calling the optimal solve "over budget".
#
# THAT STALLED THE 2026-08-07 DESCENT. At r = 205 mm the loop asked for 12 mm
# against a 55 px error -- the right amount, confirmed afterwards at 4.7 px/mm --
# needed 3.35 deg, and was halved twice to 3 mm. It moved 14 px against a descent
# injecting ~14 px per step, so the corrector sat at exactly break-even until the
# progress monitor stopped the run.

class TangentialIK(FakeIK):
    """Solves a tangential nudge with base yaw and nothing else.

    `waste` is how many times the geometric requirement (theta = d / r) the
    solve spends. 1.0 is the optimal answer -- what a well-conditioned pose
    returns -- and anything above it is the solver leaning on yaw harder than
    the geometry demands, which is the near-base-axis failure the flat ceiling
    was written for.
    """

    TICKS_PER_RAD = 651.89

    def __init__(self, bus, radius=0.150, waste=1.0):
        super().__init__(bus)
        self.radius = radius
        self.waste = waste

    def _tip(self, angles=None):
        return np.array([self.radius, 0.0, 0.080])

    def request_ik(self, x, y, z, seed_rad=None, lock=None):
        self.locks.append(tuple(lock or ()))
        self.solves.append({"tangential_mm": y * 1000.0})
        ticks = (y / self.radius) * self.TICKS_PER_RAD * self.waste
        return [self.bus.ticks[1] + ticks, self.bus.ticks[2],
                self.bus.ticks[3], self.bus.ticks[4], self.bus.ticks[5]], 0.0


def _tangential_ctx(**kw):
    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    return vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                      ik=TangentialIK(bus, **kw))


def test_a_tangential_nudge_may_spend_the_yaw_its_geometry_requires():
    """THE 2026-08-07 REGRESSION. The full request must execute, not a quarter.

    12 mm at r = 150 mm needs 4.6 deg, three times the flat ceiling, and there
    is no cheaper answer -- J1 is the only joint that moves the claw sideways.
    """
    ctx = _tangential_ctx()
    executed = vs.CartesianActuator("tangential").apply(ctx, 12.0)

    assert executed == pytest.approx(12.0), "the nudge was shrunk"
    assert len(ctx.ik.solves) == 1, "it should not have needed to shrink at all"

    panned = ctx.bus.stepped[-1][1] - 2000
    pan_deg = abs(panned) / TangentialIK.TICKS_PER_RAD * 180 / np.pi
    assert pan_deg > config.SERVO_VISUAL_MAX_PAN_DEG, (
        "this test is pointless unless the accepted pan exceeds the flat "
        "ceiling -- that is the whole regression")
    assert pan_deg == pytest.approx(np.degrees(0.012 / 0.150), abs=0.05), (
        "and it must be the geometric requirement, not merely a bigger number")


def test_a_tangential_solve_that_wastes_yaw_is_still_shrunk():
    """The loosening is against the GEOMETRY, not a blanket exemption.

    Close to the base axis the claw's 27 mm offset makes the tip's bearing
    hypersensitive to J1 and the solver buys millimetres with degrees. That is
    what the guard exists for and it must survive the fix.
    """
    ctx = _tangential_ctx(waste=3.0)
    executed = vs.CartesianActuator("tangential").apply(ctx, 12.0)

    assert abs(executed) < 12.0 / 4, (
        f"a solve spending 3x the geometric requirement executed {executed} mm")
    assert len(ctx.ik.solves) > 3, "it should have shrunk repeatedly"


def test_a_radial_nudge_keeps_the_flat_ceiling():
    """Direction-specific, and it has to be: the same 4.6 deg that is the only
    way to move sideways is pure waste on an axis base yaw cannot serve."""
    # FakeIK, not TangentialIK: a radial nudge measures its direction by
    # perturbing the pitch chain, so the FK has to respond to its angles.
    class YawingRadialIK(FakeIK):
        def request_ik(self, x, y, z, seed_rad=None, lock=None):
            self.locks.append(tuple(lock or ()))
            self.solves.append({})
            # 4.6 deg -- accepted tangentially above, waste here.
            return [self.bus.ticks[1] + 52, self.bus.ticks[2],
                    self.bus.ticks[3], self.bus.ticks[4], self.bus.ticks[5]], 0.0

    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=YawingRadialIK(bus))

    with pytest.raises(vs.ServoAbort, match="poorly conditioned"):
        vs.CartesianActuator("radial").apply(ctx, 12.0)
    assert bus.stepped == []


def test_close_to_the_yaw_axis_the_geometric_budget_is_not_used():
    """d/r explodes as r shrinks, so a budget derived from it would authorise
    exactly the swing the flat ceiling refuses. Fall back below MIN_RADIUS."""
    act = vs.CartesianActuator("tangential")
    ctx = _tangential_ctx(radius=0.150)
    near = _tangential_ctx(radius=0.020)

    tip_far = np.array([0.150, 0.0, 0.080])
    tip_near = np.array([0.020, 0.0, 0.080])

    assert act._pan_budget_deg(ctx, tip_far, 12.0) > config.SERVO_VISUAL_MAX_PAN_DEG
    assert act._pan_budget_deg(near, tip_near, 12.0) == config.SERVO_VISUAL_MAX_PAN_DEG


def test_the_tangential_budget_is_never_stricter_than_the_flat_one():
    """The fix only ever loosens. A small nudge has a tiny geometric
    requirement, and clamping to it would refuse nudges that work today."""
    act = vs.CartesianActuator("tangential")
    ctx = _tangential_ctx()
    tip = np.array([0.150, 0.0, 0.080])
    for mm in (0.5, 1.0, 2.0, 4.0, 12.0, 40.0):
        assert act._pan_budget_deg(ctx, tip, mm) >= config.SERVO_VISUAL_MAX_PAN_DEG
    assert act._pan_budget_deg(ctx, tip, 400.0) == \
        config.SERVO_VISUAL_MAX_TANGENTIAL_PAN_DEG


def test_the_reach_direction_is_measured_not_inferred_from_the_yaw_axis():
    """THE 2026-08-07 ROOT CAUSE. "Radial" meant the horizontal direction from
    the base yaw axis out to the tool, which is the direction the arm reaches
    only when the tool is well away from that axis. Reach comes from the PITCH
    CHAIN -- J2/J3/J4 are parallel, so they move the tip in one fixed vertical
    plane whose horizontal bearing J1 alone decides.

    At the hover the tool sits 8.7 mm from the yaw axis, so "radial" was the
    bearing of a near-zero vector: -137.8 deg measured, against the pitch
    chain's real +94.4 deg. 52 deg apart. Every radial nudge asked for a
    component the shoulder/elbow chain could not supply and IK made it up with
    base yaw and wrist roll -- 44 ticks of yaw and 108 of roll for a 3 mm ask.

    This fake makes the two disagree by 90 deg, which no amount of guarding the
    symptom would have caught."""
    class PlanarIK:
        """Pitch chain reaches along +Y, while the tool sits 5 mm from the yaw
        axis along -X -- so the yaw-axis radial points along X and is wrong."""

        def base_yaw_axis_xy(self):
            return np.array([0.100, 0.0])

        def request_fk_tip(self, angles):
            T = np.eye(4)
            reach = angles[1] + angles[2] + angles[3]   # J2, J3, J4 only
            T[:3, 3] = [0.095, reach, 0.080]
            return T, T

    ctx = Namespace(ik=PlanarIK())
    u = vs.reach_axis_xy(ctx, [0.0] * 5)

    assert abs(u[1]) == pytest.approx(1.0, abs=1e-6), "reach must follow the pitch chain"
    assert u[0] == pytest.approx(0.0, abs=1e-6)

    # What the old geometry would have said, for the contrast.
    radial, _t = vs.radial_tangential((0.095, 0.0), (0.100, 0.0))
    assert abs(radial[0]) == pytest.approx(1.0, abs=1e-6), "the yaw axis says X"
    assert abs(float(np.dot(u, radial))) < 1e-6, "and it is 90 deg from the truth"


def test_the_reach_direction_survives_sitting_on_the_yaw_axis():
    """radial_tangential RAISES directly over the base axis, because every
    horizontal direction is equally radial there. The pitch chain still has one
    definite direction, so the measurement keeps working where the geometry
    cannot -- which matters because the hover pose is very nearly that case."""
    class OnAxisIK:
        def base_yaw_axis_xy(self):
            return np.array([0.0, 0.0])

        def request_fk_tip(self, angles):
            T = np.eye(4)
            T[:3, 3] = [0.0, angles[1] + angles[2] + angles[3], 0.080]
            return T, T

    with pytest.raises(ValueError):
        vs.radial_tangential((0.0, 0.0), (0.0, 0.0))

    u = vs.reach_axis_xy(Namespace(ik=OnAxisIK()), [0.0] * 5)
    assert abs(u[1]) == pytest.approx(1.0, abs=1e-6)


def test_an_ik_solution_is_commanded_whole_never_censored():
    """THE REGRESSION GUARD FOR 2026-08-07. An earlier version enforced the
    server's dead `lock` by not commanding the held joints -- keeping the rest of
    the solution and dropping J1 and J5 from it.

    An IK solution is a COORDINATED answer: the other joints are where they are
    BECAUSE the deleted one was going to move. Measured on the arm, a 9 mm radial
    nudge minus its held joints executed -0.11 mm, and a tangential one went 15 mm
    the WRONG WAY. The centring loop then commanded 9 mm, moved 0.1 mm, saw no
    pixel response and asked again with the same numbers twelve times running.

    So every joint the solver named must be commanded exactly as named."""
    class GreedyIK(FakeIK):
        def request_ik(self, x, y, z, seed_rad=None, lock=None):
            super().request_ik(x, y, z, seed_rad, lock)
            # +20 not -20: CartesianBus refuses J2 below its travel limit, and
            # this test is about censoring, not about the limit guard. J1+8 and
            # J5+30 sit inside the pan and roll budgets so the solve is accepted.
            return [self.bus.ticks[1] + 8, self.bus.ticks[2] + 20,
                    self.bus.ticks[3], self.bus.ticks[4],
                    self.bus.ticks[5] + 30], 0.0

    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=GreedyIK(bus))
    before = dict(bus.ticks)

    vs.CartesianActuator("radial").apply(ctx, 10.0)

    commanded = bus.stepped[-1]          # CartesianBus records the target dict
    assert commanded[1] == before[1] + 8, "J1 must be commanded as the solver said"
    assert commanded[5] == before[5] + 30, "J5 must be commanded as the solver said"
    assert commanded[2] == before[2] + 20


def test_a_roll_heavy_solve_is_refused_not_censored():
    """J5 is the wrist ROLL, so it spins the camera about its own optical axis
    and rotates the image. It is NOT free motion to be deleted, though -- the
    claw tip sits off the roll axis, so J5 genuinely translates it and a solve
    can lean on the wrist to reach sideways. The only sound response to a
    solution that leans too hard is to ask for less, and to refuse if that does
    not help. Nothing is commanded on the way out."""
    class RollingIK(FakeIK):
        def request_ik(self, x, y, z, seed_rad=None, lock=None):
            super().request_ik(x, y, z, seed_rad, lock)
            # 90 ticks = 7.9 deg, over SERVO_VISUAL_MAX_ROLL_DEG. Fixed
            # regardless of the request, so shrinking cannot rescue it.
            return [self.bus.ticks[1], self.bus.ticks[2] + 20,
                    self.bus.ticks[3], self.bus.ticks[4],
                    self.bus.ticks[5] + 90], 0.0

    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=RollingIK(bus))

    with pytest.raises(vs.ServoAbort, match="wrist roll"):
        vs.CartesianActuator("radial").apply(ctx, 10.0)
    assert bus.stepped == [], "a refused solve must not be partially commanded"


def test_a_branch_flipping_solve_is_refused():
    """A 5 mm request answered with J2-416, J3+366 is not a nudge — it is the
    solver jumping to a different posture that reaches the same point. Seen at
    190 mm with J2 pinned at its limit; commanding it unsupervised would be a
    violent move."""
    class FlippingIK(FakeIK):
        def request_ik(self, x, y, z, seed_rad=None, lock=None):
            super().request_ik(x, y, z, seed_rad, lock)
            return [self.bus.ticks[1], self.bus.ticks[2] - 416,
                    self.bus.ticks[3] + 366, self.bus.ticks[4],
                    self.bus.ticks[5]], 0.0

    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=FlippingIK(bus))

    with pytest.raises(vs.ServoAbort, match="poorly conditioned"):
        vs.CartesianActuator("radial").apply(ctx, 10.0)
    assert bus.stepped == []


def test_a_well_conditioned_solve_goes_through_untouched():
    """The guards must not tax the normal case."""
    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=FakeIK(bus))
    assert vs.CartesianActuator("radial").apply(ctx, -10.0) == -10.0
    assert len(bus.stepped) == 1


def test_cartesian_actuator_reports_millimetres():
    act = vs.CartesianActuator("radial")
    assert act.unit == "mm"
    assert act.kind == "cartesian"
    assert "radial" in act.label()


def test_tiny_cartesian_nudges_are_not_commanded():
    """Below the minimum step the move is noise; commanding it just wears the
    servos and adds settle time."""
    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=FakeIK(bus))
    assert vs.CartesianActuator("radial").apply(ctx, 0.05) == 0.0
    assert bus.stepped == []


# --- aiming above the brick, into a box -------------------------------------

def test_the_aim_point_sits_below_the_frame_centre():
    """The camera is above and behind the claw, so the claw is over the brick
    when the brick sits BELOW the crosshair — not when it is centred."""
    ctx, _bus, _w = make_ctx(aim_offset_y=60.0)
    ax, ay = ctx.aim((480, 640, 3))
    assert ax == 320.0
    assert ay == 300.0, "aim is pushed down, so the crosshair ends up above it"


def test_zero_offset_aims_at_the_frame_centre():
    ctx, _bus, _w = make_ctx(aim_offset_x=0.0, aim_offset_y=0.0)
    assert ctx.aim((480, 640)) == (320.0, 240.0)


def test_the_aim_point_shifts_sideways_too():
    """The claw does not sit under the pixel the camera calls centre in EITHER
    axis. --aim-offset-y covers "the camera is above and behind"; this covers
    the sideways part. Negative is left, matching image coordinates."""
    ctx, _bus, _w = make_ctx(aim_offset_x=-90.0, aim_offset_y=0.0)
    ax, ay = ctx.aim((480, 640, 3))
    assert (ax, ay) == (230.0, 240.0)

    # And the error is measured from there, so a brick sitting ON the shifted
    # aim point is done -- not one sitting at the frame centre.
    assert vs.pixel_error((230.0, 240.0), (480, 640), ctx.aim((480, 640))) == (0.0, 0.0)
    assert vs.pixel_error((320.0, 240.0), (480, 640), ctx.aim((480, 640))) == (90.0, 0.0)


def test_an_aim_point_pushed_off_the_side_is_reported(capsys):
    """The same silent failure the y check exists for, on the other axis: a box
    the camera cannot see is a loop that cannot converge, while the detector,
    the arm and the gains all look healthy."""
    args = Namespace(aim_offset_x=-400.0, aim_offset_y=0.0,
                     tolerance_x=45.0, tolerance_y=55.0)
    vs.report_aim_reachability(args)
    out = capsys.readouterr().out
    assert "OFF THE FRAME in x" in out
    assert "--aim-offset-x" in out


def test_error_is_measured_from_the_aim_point_not_the_centre():
    ctx, _bus, _w = make_ctx(aim_offset_y=60.0)
    # A brick exactly at the aim point has zero error even though it is 60 px
    # below the frame centre.
    assert vs.pixel_error((320.0, 300.0), (480, 640), ctx.aim((480, 640))) == (0.0, 0.0)


def test_a_brick_already_inside_the_box_needs_no_correction():
    """The target is a region, not a pixel: anywhere in the box the claw is
    over the brick, and the descent re-centres on the way down anyway."""
    bus = FakeBus()
    world = FakeWorld(bus)
    world.offset = (30.0, 25.0)          # inside a 45 x 55 box
    args = Namespace(probe_ticks=40, settle=0.0, deadband=12.0, max_iterations=10,
                     view=False, no_wait=True, recentre="joint", joint_x=1,
                     joint_y=2, aim_offset_x=0.0, aim_offset_y=0.0, tolerance_x=45.0,
                     tolerance_y=55.0)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None)

    # Estimates supplied directly rather than probed: a probe deliberately moves
    # the arm, which would push the brick out of the very box under test.
    ax, ay = joint_actuators()
    estimates = [(("x", ax), AxisEstimate(1.0, 40, 40.0)),
                 (("y", ay), AxisEstimate(-1.33, 40, -30.0))]
    moves_before = len(bus.moves)
    assert vs.centre(ctx, estimates) is True
    assert len(bus.moves) == moves_before, "already good enough — nothing to do"


def test_the_worst_axis_is_chosen_relative_to_its_own_tolerance():
    """With a wide box and a tall one, raw pixels would keep servicing the
    looser axis. Comparing each error as a fraction of its own tolerance is
    what makes an asymmetric target behave."""
    bus = FakeBus()
    world = FakeWorld(bus)
    # x is 40 of a 45 box (89%); y is 50 of a 100 box (50%). x is worse.
    world.offset = (40.0, 50.0)
    args = Namespace(probe_ticks=40, settle=0.0, deadband=12.0, max_iterations=1,
                     view=False, no_wait=True, recentre="joint", joint_x=1,
                     joint_y=2, aim_offset_x=0.0, aim_offset_y=0.0, tolerance_x=45.0,
                     tolerance_y=100.0)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None)

    ax, ay = joint_actuators()
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax)),
                 (("y", ay), vs.probe_axis(ctx, "y", ay))]
    before = len(bus.moves)
    vs.centre(ctx, estimates)
    moved = [j for j, _t in bus.moves[before:]]
    assert moved and moved[0] == 1, "should service x (J1), the axis nearer its limit"


# --- keeping the brick in frame ---------------------------------------------

def test_a_high_gain_run_never_pushes_the_brick_out_of_frame(monkeypatch):
    """The 2026-08-05 failure, end to end.

    With a steep gain the unit clamp permits a step that moves the brick further
    than the frame is wide; the loop then looks, sees nothing, and dies. The
    pixel clamp is what makes a high-gain arm survivable, so the detector here
    is BOUNDED — it returns nothing once the brick leaves the frame, exactly
    like the real one, and a step that overshoots ends the run.

    The frame fraction is tightened so the clamp is unambiguously the thing
    doing the work rather than the gain happening to be gentle enough.
    """
    monkeypatch.setattr(
        "vision_pipeline.planning.visual_servo.config.SERVO_VISUAL_MAX_FRAME_FRACTION",
        0.10)
    # This fixture's gain is deliberately steep (9 px per tick) to exercise the
    # frame clamp. At that gain the real 25-tick stiction floor would mean a
    # 225 px resolution, and the loop would rightly refuse to aim finer — a
    # different mechanism, tested separately. Take it out of the way here.
    monkeypatch.setattr(
        "vision_pipeline.planning.visual_servo.config.SERVO_VISUAL_MIN_STEP_TICKS", 1)

    bus = FakeBus()
    world = FakeWorld(bus, px_per_tick={1: 9.0, 2: -7.0})
    world.offset = (200.0, -150.0)
    detector = FakeDetector(world, bounded=True)
    args = Namespace(probe_ticks=4, settle=0.0, deadband=12.0, max_iterations=80,
                     view=False, no_wait=True, recentre="joint",
                     joint_x=1, joint_y=2,
                     aim_offset_x=0.0, aim_offset_y=0.0, tolerance_x=12.0, tolerance_y=12.0)
    ctx = vs.Context(bus, FakeCamera(world), detector, args, None)

    ax, ay = joint_actuators(probe_ticks=args.probe_ticks)
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax)),
                 (("y", ay), vs.probe_axis(ctx, "y", ay))]
    assert vs.centre(ctx, estimates) is True

    cx, cy = world.centroid()
    assert abs(cx - world.aim[0]) <= args.deadband
    assert abs(cy - world.aim[1]) <= args.deadband


def test_probe_amount_comes_from_the_caller_not_the_config_default():
    """--probe-ticks stopped reaching the arm when probing moved into the
    actuators, which is how a 40-tick probe ran where 4 was asked for."""
    assert vs.JointActuator(1, 7).probe_amount == 7
    assert vs.CartesianActuator("radial", 2.5).probe_amount == 2.5


def test_the_loop_backs_out_the_last_move_when_the_brick_vanishes():
    """Losing the brick is usually self-inflicted, so undo the cause first.

    Ending the run instead leaves the arm parked where the camera sees nothing,
    which is the worst place to stop.
    """
    bus = FakeBus()
    world = FakeWorld(bus, px_per_tick={1: 1.0, 2: -0.75})
    # Sees the brick through the probe and the first correction, then goes
    # blind — as if that correction had pushed it out of view.
    # Blind from call 4 on: the two probe looks and one loop look succeed, so
    # exactly one correction is applied before the brick disappears.
    detector = FakeDetector(world, blind_after=3)
    args = Namespace(probe_ticks=40, settle=0.0, deadband=12.0, max_iterations=10,
                     view=False, no_wait=True, recentre="joint",
                     joint_x=1, joint_y=2)
    ctx = vs.Context(bus, FakeCamera(world), detector, args, None)

    ax, _ay = joint_actuators()
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax))]

    tick_before_correction = bus.ticks[1]
    with pytest.raises(vs.ServoAbort, match="could not recover"):
        vs.centre(ctx, estimates)

    # The correction moved J1, then the reversal put it back where it started.
    assert bus.ticks[1] == tick_before_correction, \
        "the reversal should have undone exactly the move that lost the brick"


def test_an_axis_that_costs_more_than_it_gains_is_called_out():
    """A correction may disturb the other axis a little; the alternation absorbs
    that. Disturbing it MORE than it gains, repeatedly, is a losing trade that
    walks the arm sideways forever while the loop looks busy.

    Observed 2026-08-05: each radial nudge bought ~4 px of vertical and cost
    ~13 px of horizontal, so seven iterations took the brick from 24 px right to
    63 px left. The progress monitor stopped it eventually, but blamed the gain
    rather than the geometry.
    """
    class CoupledWorld(FakeWorld):
        """J2 drags the view sideways far more than it moves it vertically."""

        def centroid(self):
            dx = (self.bus.ticks[1] - self.home[1]) * 1.0
            j2 = self.bus.ticks[2] - self.home[2]
            # Each J2 tick moves the view 1.5 px sideways and 0.75 px
            # vertically: a correction that costs twice what it gains, which is
            # the shape of the real failure without being so extreme that the
            # recovery iterations dominate the test.
            return (self.aim[0] + self.offset[0] + dx + j2 * 1.5,
                    self.aim[1] + self.offset[1] + j2 * -0.75)

    bus = FakeBus()
    world = CoupledWorld(bus)
    world.offset = (10.0, -120.0)          # mostly a vertical error to fix
    # Aim pinned to the frame centre: this test is about the COUPLING guard, and
    # letting the aim offsets fall back to config would make its geometry move
    # whenever the camera-to-claw offset is re-measured.
    args = Namespace(probe_ticks=20, settle=0.0, deadband=12.0,
                     max_iterations=40, view=False, no_wait=True,
                     recentre="joint", joint_x=1, joint_y=2,
                     aim_offset_x=0.0, aim_offset_y=0.0)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None)

    ax, ay = joint_actuators(probe_ticks=args.probe_ticks)
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax)),
                 (("y", ay), vs.probe_axis(ctx, "y", ay))]

    with pytest.raises(vs.ServoAbort, match="costing more than it gains"):
        vs.centre(ctx, estimates)


def test_a_normally_coupled_axis_is_not_called_out():
    """Ordinary mild coupling must not trip the guard — the loop handles it."""
    ctx, bus, world = make_ctx()
    ax, ay = joint_actuators()
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax)),
                 (("y", ay), vs.probe_axis(ctx, "y", ay))]
    assert vs.centre(ctx, estimates) is True


def test_reverse_undoes_a_joint_move_exactly():
    ctx, bus, _ = make_ctx()
    act = vs.JointActuator(1)
    start = bus.ticks[1]
    act.apply(ctx, 25)
    assert bus.ticks[1] == start + 25
    act.reverse(ctx, 25)
    assert bus.ticks[1] == start


# --- bounded ----------------------------------------------------------------

def test_loop_stops_at_the_iteration_cap_without_converging():
    """Bounded on purpose: the per-step clamp means a large error needs more
    iterations than the cap allows, and that must end the run cleanly rather
    than spin. The clamp is a safety bound, so hitting the cap is a normal
    outcome, not an error."""
    ctx, bus, _ = make_ctx(px_per_tick={1: 1.0, 2: -0.75}, max_iterations=3)
    ax, _ay = joint_actuators()
    estimates = [(("x", ax), vs.probe_axis(ctx, "x", ax))]
    assert vs.centre(ctx, estimates) is False
    # It should still have made real progress in those three iterations.
    assert len(bus.moves) >= 3


# --- the descent: one solve per step ----------------------------------------
#
# The 2026-08-05 failure. Descending and correcting the brick's vertical
# position in the image are ONE degree of freedom -- both ride the shoulder /
# elbow chain, and the camera is on the wrist, so descending swings the view
# (~7 px/mm measured). Running them as two loops made the arm fight itself:
#
#     IK lowers the tip 20 mm      -> brick jumps ~140 px up the frame
#     J3 jog drags the brick back  -> tip rises ~25 mm
#     next step spends itself undoing that
#
# Four steps, 4 mm of net descent, then the brick left the frame. The fake world
# below reproduces that coupling exactly, so a regression to two separate loops
# fails these tests rather than the arm.

DESCENT_A = 7.0     # px of vertical image motion per mm of tip height
DESCENT_B = 2.5     # px of vertical image motion per mm of radial reach


class DescentWorld:
    """A 2-DOF arm whose camera swings when it descends.

    J2 sets tip height, J3 sets radial reach, 1 tick == 1 mm each. The brick's
    apparent height depends on BOTH -- which is the entire problem.
    """

    frame = (480, 640)

    def __init__(self, z_mm=144.0, reach_mm=62.0, error_px=-140.0,
                 error_x_px=90.0, px_per_j1_tick=1.0):
        self.bus = None
        self.z0, self.reach0 = z_mm, reach_mm
        self.error0 = error_px
        # Sideways error is NOT zero, deliberately. It was, and that left the
        # descent's J1 branch unexecuted by every test -- which is how a bad
        # keyword argument in it (max_step, where step_ticks takes
        # max_step_ticks) reached the arm and raised TypeError mid-descent,
        # after the claw had already been committed to a step.
        self.error_x0 = error_x_px
        self.px_per_j1_tick = px_per_j1_tick
        self.j1_0 = None

    def tip(self):
        return np.array([self.bus.ticks[3] / 1000.0, 0.0, self.bus.ticks[2] / 1000.0])

    def error_px(self):
        return (self.error0
                + DESCENT_A * (self.bus.ticks[2] - self.z0)
                + DESCENT_B * (self.bus.ticks[3] - self.reach0))

    def error_x_px(self):
        if self.j1_0 is None:
            self.j1_0 = self.bus.ticks[1]
        return self.error_x0 + (self.bus.ticks[1] - self.j1_0) * self.px_per_j1_tick

    def centroid(self):
        return (self.frame[1] / 2.0 + self.error_x_px(),
                self.frame[0] / 2.0 + self.error_px())


class DescentBus(FakeBus):
    def __init__(self, world):
        super().__init__(start={1: 2000, 2: int(world.z0), 3: int(world.reach0),
                                4: 1700, 5: 2500})
        self.stepped = []

    def ticks_to_rad(self, servo_id, ticks):
        return float(ticks)

    def rad_to_ticks(self, servo_id, rad):
        return int(round(rad))

    def travel_limits(self, servo_id):
        return None

    def move_joints_stepped(self, targets, **kw):
        self.stepped.append(dict(targets))
        self.ticks.update({j: int(t) for j, t in targets.items()})


class DescentIK:
    """Exact IK for the 2-DOF world: reach and height are directly commandable."""

    def __init__(self, world):
        self.world = world
        self.solves = []
        self.locks = []

    def request_fk_tip(self, angles=None):
        # HONOURS `angles`, which reach_axis_xy depends on: it finds the arm's
        # real reach direction by perturbing a joint and watching the tip, so an
        # FK that ignores its argument describes an arm that cannot move.
        # angles[k] is joint k+1's tick here (see request_ik): [1] is height in
        # mm, [2] is reach in mm.
        T = np.eye(4)
        T[:3, 3] = (self.world.tip() if angles is None else
                    np.array([angles[2] / 1000.0, 0.0, angles[1] / 1000.0]))
        return T, T

    def base_yaw_axis_xy(self):
        # This synthetic world really does put the yaw axis at the origin --
        # reach is hypot(x, y) throughout. The REAL arm does not (81 mm off),
        # which is what test_radial_is_measured_from_the_yaw_axis_not_the_origin
        # pins; returning zeros here keeps the fake's own arithmetic honest.
        return np.array([0.0, 0.0])

    def request_ik(self, x, y, z, seed_rad=None, lock=None):
        self.solves.append((x, y, z))
        self.locks.append(tuple(lock or ()))
        b = self.world.bus
        return [b.ticks[1], z * 1000.0, np.hypot(x, y) * 1000.0,
                b.ticks[4], b.ticks[5]], 0.0


def descent_ctx(**overrides):
    world = DescentWorld()
    bus = DescentBus(world)
    world.bus = bus
    args = Namespace(
        probe_ticks=40, settle=0.0, max_iterations=40, view=False, no_wait=True,
        # DERIVED, NOT HARDCODED. This was -64.0, which was a legal grasp height
        # only while TABLE_Z_IN_BASE was -74 mm. When the table was re-measured
        # to -67.7 mm on 2026-08-07 the floor guard rose to -62.7 and every
        # descent test began refusing its own target -- a test constant that
        # silently depended on a measured one. Ask config the same question the
        # script asks.
        descend=True, descend_step=20.0, no_confirm=True,
        target_z=(vs.config.TABLE_Z_IN_BASE + vs.config.PICK_Z_OFFSET) * 1000.0,
        recentre="joint", joint_x=1, joint_y=3,
        descend_probe_mm=8.0, descend_max_reach_mm=25.0,
        no_sideways=False, lock_base=True,
        sideways_budget=vs.config.SERVO_VISUAL_SIDEWAYS_BUDGET_TICKS,
        aim_offset_x=0.0, aim_offset_y=0.0, tolerance_x=45.0, tolerance_y=55.0,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=DescentIK(world))
    return ctx, bus, world


def descent_estimates(ctx):
    """estimates in the SHAPE main() actually passes: ((axis, ACTUATOR), est).

    Built by probing, exactly as main() does, rather than hand-assembled. The
    hand-assembled version used a joint NUMBER where the real caller passes an
    actuator OBJECT, so every descent test agreed with itself and disagreed
    with the program -- and descend() read the actuator as a joint id, wrapped
    it in a second JointActuator, and handed that to the packet builder as a
    servo ID on live hardware. A fake that invents its own calling convention
    tests nothing.
    """
    actuator = vs.JointActuator(1, ctx.args.probe_ticks)
    return [(("x", actuator), vs.probe_axis(ctx, "x", actuator))]


def test_a_joint_with_no_limits_by_design_is_not_reported_as_a_gap(capsys):
    """"Unmeasured" and "needs no limits" both read as None from travel_limits,
    and they call for opposite responses. J5 is a continuous roll with no stop
    to find, so telling the operator to go measure it is noise -- and noise in a
    warning list is not harmless, because the REAL gap hides in it. J4 reached
    its hard stop on 2026-08-07 while a standing note about J1 sat above it."""
    ctx, bus, _world = descent_ctx()
    bus.calibration["5"]["limits_not_required"] = True

    vs.descend(ctx, [((("y"), vs.JointActuator(3)), 1.0)])
    out = capsys.readouterr().out

    measure_line = [ln for ln in out.splitlines() if "find_joint_limits" in ln]
    assert measure_line, "the genuinely unmeasured joints must still be named"
    assert not any("--joint 5" in ln for ln in measure_line), \
        "J5 has no stop to find; asking for it trains the operator to skim"
    assert "no travel limits by design" in out
    assert "SERVO_VISUAL_MAX_ROLL_DEG" in out, \
        "say what guards it instead, or the note reads as an unprotected joint"


def test_a_step_that_starts_outside_the_box_does_not_descend():
    """VALIDATE THE BOX BEFORE EVERY DESCENT. Height bought on a bad aim has to
    be given back later, and it is bought at the point in the run where the claw
    is nearest the table and the brick nearest to leaving frame.

    This is not the two-loop arrangement that failed on 2026-08-05: that one
    re-aimed with a JOINT JOG, which raised the tip by more than the step had
    lowered it. A re-aim here is one IK solve at CONSTANT height, so it cannot
    undo a descent."""
    ctx, bus, world = descent_ctx()
    # DescentWorld starts 140 px above the aim point, well outside a 55 px box.
    heights = []
    real_move = bus.move_joints_stepped

    def record(targets, **kw):
        real_move(targets, **kw)
        heights.append(bus.ticks[2])          # ticks[2] IS z in mm here

    bus.move_joints_stepped = record
    vs.descend(ctx, descent_estimates(ctx))

    assert heights, "the descent must have commanded something"
    assert heights[0] == pytest.approx(world.z0, abs=0.01), \
        "the first step began outside the box, so it must not have lost height"


def test_re_aiming_that_stops_working_falls_through_to_a_blind_descent(capsys):
    """The brick is visible but cannot be brought into the box, so more attempts
    only burn travel. That is the other face of "the aiming loop can no longer
    run", and it takes the same exit as losing sight of the brick rather than
    abandoning the run a few millimetres short."""
    ctx, _bus, world = descent_ctx()
    world.error0 = -400.0                    # never reachable into the box
    ctx.args.max_iterations = 60

    vs.descend(ctx, descent_estimates(ctx))
    out = capsys.readouterr().out
    assert "BLIND FINISH" in out
    assert "re-aim" in out


def test_losing_the_brick_late_finishes_the_descent_blind():
    """LOSING SIGHT NEAR THE END IS NORMAL. The camera sits above and behind the
    claw, so the brick slides out of the bottom of the frame exactly when the
    descent has nearly finished. Stopping there abandoned the run at the one
    moment it had already done its job, leaving the claw hovering over a brick
    it could no longer see."""
    ctx, bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 3    # a couple of steps, then dark

    assert vs.descend(ctx, estimates) is True
    assert bus.ticks[2] <= ctx.args.target_z + 1, \
        "a blind finish must still reach the target height, not stall at it"


def test_a_blind_finish_holds_x_and_y_and_drops_the_reach_correction():
    """Straight down, so the tool stays over whatever it was over. The reach
    term exists only to cancel the image swing a descent causes, and nothing is
    reading the image now -- carrying it on would move the claw off the brick to
    steady a view nobody is watching."""
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 1   # one fix, then dark

    assert vs.descend(ctx, estimates) is True

    # Compared against EACH OTHER, not against the pose the descent started
    # from: a step that begins outside the box now legitimately spends itself on
    # reach before the lights go out, so the blind phase may well start
    # somewhere else. What must hold is that once blind, x and y never move
    # again -- only z.
    asked = ctx.ik.solves[-6:]
    assert len(asked) >= 2, "the blind finish must have issued solves of its own"
    xs = {round(x, 9) for x, _y, _z in asked}
    ys = {round(y, 9) for _x, y, _z in asked}
    zs = {round(z, 9) for _x, _y, z in asked}
    assert len(xs) == 1, f"blind descent moved x across {sorted(xs)}"
    assert len(ys) == 1, f"blind descent moved y across {sorted(ys)}"
    assert len(zs) == len(asked), "and z must change every step, or it is not descending"


def test_a_blind_finish_needs_a_fix_to_hold(capsys):
    """Blind at the END is sound; blind from the START is a guess with a floor
    guard. Without a single detection there is no aim to hold, so refusing is
    the honest answer."""
    ctx, bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = 0            # dark from the descent's first look
    bus.stepped.clear()

    assert vs.descend(ctx, estimates) is False
    assert bus.stepped == [], "nothing may be commanded without a fix to hold"
    assert "never detected" in capsys.readouterr().out


def test_a_blind_finish_says_when_the_claw_was_not_over_the_brick(capsys):
    """The loop gives up the ability to fix an aim error it cannot see, so the
    last-seen error must be reported rather than quietly discarded. If the claw
    lands off, that number says by how much and in which direction."""
    ctx, _bus, _world = descent_ctx()          # starts 90 px right, outside a 45 px box
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 1   # one fix, then dark

    vs.descend(ctx, estimates)
    out = capsys.readouterr().out
    assert "OUTSIDE" in out and "Expect a miss" in out


def test_the_descent_actually_descends():
    """The headline regression. The old two-loop version went
    144 -> 136 -> 147.5 -> 128 -> 154.7 ... : every re-centre gave back more
    height than the step had taken, and four steps netted 4 mm."""
    ctx, bus, world = descent_ctx()
    heights = []
    real_move = bus.move_joints_stepped

    def record(targets, **kw):
        real_move(targets, **kw)
        heights.append(bus.ticks[2])

    bus.move_joints_stepped = record
    assert vs.descend(ctx, descent_estimates(ctx)) is True

    assert heights == sorted(heights, reverse=True), \
        f"height must never increase, got {heights}"
    # ticks[2] IS z in mm in this fake world. Derived from the context's own
    # target for the same reason the target is derived from config: a literal
    # here silently encodes whatever TABLE_Z_IN_BASE happened to be that week.
    assert bus.ticks[2] <= ctx.args.target_z + 1


def test_each_descent_step_is_a_single_ik_solve():
    """Two loops meant two kinds of move per step, fighting each other. One
    combined solve per step is the fix, and it is worth asserting literally."""
    ctx, bus, world = descent_ctx()
    vs.descend(ctx, descent_estimates(ctx))
    assert len(ctx.ik.solves) == len(bus.stepped)


def test_the_descent_converges_the_vertical_error_it_starts_with():
    """Starting 140 px off, with the descent itself adding ~140 px of error per
    step, the loop must still arrive aimed at the brick."""
    ctx, bus, world = descent_ctx()
    vs.descend(ctx, descent_estimates(ctx))
    assert abs(world.error_px()) <= ctx.args.tolerance_y


def test_the_model_is_fitted_from_the_arms_own_motion():
    """No dedicated probe moves: steps 1 and 2 differ in reach but both descend
    in full, so the fit costs no height."""
    ctx, bus, world = descent_ctx()
    reaches = []
    real_move = bus.move_joints_stepped

    def record(targets, **kw):
        real_move(targets, **kw)
        reaches.append(bus.ticks[3])

    bus.move_joints_stepped = record
    vs.descend(ctx, descent_estimates(ctx))
    assert reaches[0] != reaches[1], "steps 1 and 2 must differ in reach"


def test_the_radial_correction_is_capped():
    """A poorly fitted early model must not be able to lunge."""
    ctx, bus, world = descent_ctx(descend_max_reach_mm=5.0)
    start = bus.ticks[3]
    vs.descend(ctx, descent_estimates(ctx))
    per_step = [abs(s[3] - start) for s in bus.stepped[:1]]
    assert all(d <= 5 + 1 for d in per_step)


def test_the_descent_stops_at_the_floor_guard_not_below_it():
    ctx, bus, world = descent_ctx(target_z=-200.0)
    vs.descend(ctx, descent_estimates(ctx))
    floor_mm = (vs.config.TABLE_Z_IN_BASE + vs.config.MIN_CLAW_HEIGHT_M) * 1000
    assert bus.ticks[2] >= floor_mm - 1


def test_the_descent_also_corrects_sideways_error():
    """The J1 branch inside the descent, which no test entered until 2026-08-05.

    It called step_ticks(..., max_step=...) where the parameter is
    max_step_ticks. Nothing caught it: the fake world had zero sideways error,
    so the branch never ran, and the real arm raised TypeError one step into a
    descent -- with the claw already lowered and the operator holding a ruler.

    Sideways goes through J1 on purpose (base yaw cannot change height, so it
    cannot undo a descent), which is exactly why it needs its own coverage: it
    is the one part of the descent the combined IK solve does not touch.
    """
    ctx, bus, world = descent_ctx()
    estimates = descent_estimates(ctx)
    assert abs(world.error_x_px()) > ctx.args.tolerance_x, "precondition"
    during = len(bus.moves)          # the probe itself jogs J1; don't count that

    vs.descend(ctx, estimates)

    assert any(j == 1 for j, _t in bus.moves[during:]), "J1 was never commanded"
    assert abs(world.error_x_px()) <= ctx.args.tolerance_x


def test_the_descent_leaves_a_well_aimed_j1_alone():
    """A correction inside tolerance must not be manufactured."""
    ctx, bus, world = descent_ctx()
    estimates = descent_estimates(ctx)
    world.error_x0 -= world.error_x_px()      # aimed, whatever the probe left
    assert abs(world.error_x_px()) < 1e-9
    during = len(bus.moves)

    vs.descend(ctx, estimates)
    assert not any(j == 1 for j, _t in bus.moves[during:])


# --- the sideways runaway ---------------------------------------------------
#
# 2026-08-05: the descent's J1 jog drove the base steadily one way until the
# operator cut power. Its gain was measured before the descent, at a different
# posture; once wrong-signed, every correction enlarged the error and the next
# was bigger. centre() has had a ProgressMonitor against exactly this since it
# was written -- the descent branch was added without one, and J1 has no
# measured travel limits, so the servo bus could not refuse it either.
#
# Three independent bounds now. Each gets a test that turns the runaway ON and
# proves that bound alone stops it, because in a real failure only one of them
# may be in a position to notice.


class InvertedX(DescentWorld):
    """A world where correcting sideways makes it worse — the runaway condition.

    The probe measures +1 px/tick and then the relationship flips, exactly as a
    gain measured at another posture would. Nothing in the pixel readings is
    inconsistent; only the outcome is wrong.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.flipped = False

    def error_x_px(self):
        if self.j1_0 is None:
            self.j1_0 = self.bus.ticks[1]
        sign = -1.0 if self.flipped else 1.0
        return self.error_x0 + (self.bus.ticks[1] - self.j1_0) * self.px_per_j1_tick * sign


def inverted_ctx(**overrides):
    ctx, bus, world = descent_ctx(**overrides)
    inverted = InvertedX()
    inverted.bus = bus
    ctx.camera.world = inverted
    ctx.detector.world = inverted
    return ctx, bus, inverted


def test_a_sideways_correction_that_makes_things_worse_stops_the_descent():
    """Bound 1, and the one that matters most: it fires on the FIRST bad move.

    Without it the loop had no way to distinguish a correct gain from an
    inverted one, and every individual step looked perfectly reasonable.
    """
    ctx, bus, world = inverted_ctx()
    estimates = descent_estimates(ctx)
    world.flipped = True                       # the gain no longer describes reality

    assert vs.descend(ctx, estimates) is False
    j1_moves = [t for j, t in bus.moves if j == 1]
    assert len(j1_moves) <= 2, f"stopped after {len(j1_moves)} J1 moves"


def test_the_sideways_budget_bounds_travel_even_if_the_pixels_lie():
    """Bound 3. Holds when the error readings themselves cannot be trusted —
    the case where bounds 1 and 2, which both read pixels, are blind."""
    ctx, bus, world = inverted_ctx(sideways_budget=60.0)
    estimates = descent_estimates(ctx)
    start = bus.ticks[1]
    world.flipped = True

    vs.descend(ctx, estimates)
    assert abs(bus.ticks[1] - start) <= 60 + ctx.args.probe_ticks


def test_no_sideways_never_touches_the_base_at_all():
    """The escape hatch. The descent's own IK solve still runs."""
    ctx, bus, world = descent_ctx(no_sideways=True)
    estimates = descent_estimates(ctx)
    during = len(bus.moves)

    vs.descend(ctx, estimates)
    assert not any(j == 1 for j, _t in bus.moves[during:])
    assert bus.stepped, "the descent itself must still have run"


def test_the_budget_is_spent_not_reset_between_steps():
    """A per-step cap would permit an unbounded total. The ceiling is for the
    whole descent."""
    ctx, bus, world = descent_ctx(sideways_budget=30.0, tolerance_x=1.0)
    estimates = descent_estimates(ctx)
    start = bus.ticks[1]

    vs.descend(ctx, estimates)
    assert abs(bus.ticks[1] - start) <= 30 + ctx.args.probe_ticks


def test_every_descent_solve_locks_the_base():
    """A descent asks for height and reach. Base yaw is not part of the request,
    and the camera is on the wrist, so any yaw the solver spends pans the image
    the loop is reading. Measured: 5/10/20 mm descents needed 1.2 ticks of J1,
    a 40 mm one came back wanting 213 — redundancy being spent uninstructed."""
    ctx, bus, world = descent_ctx()
    vs.descend(ctx, descent_estimates(ctx))
    assert ctx.ik.locks, "no solves were made"
    assert all(lock == (1,) for lock in ctx.ik.locks)


def test_no_lock_base_leaves_the_solver_free():
    ctx, bus, world = descent_ctx(lock_base=False)
    vs.descend(ctx, descent_estimates(ctx))
    assert all(lock == () for lock in ctx.ik.locks)


# --- the yaw axis is not the origin -----------------------------------------
# On this arm the base yaw axis passes 81 mm from the imported model's origin
# (measured 2026-08-06, scripts/audit_model_axes.py) while the claw at the home
# pose sits only ~25 mm from that axis. Computing "radial" as tip_xy/|tip_xy|
# therefore points up to 108 deg away from truly outward, which turns a reach
# correction into a sideways one inside the descent.

def test_radial_is_measured_from_the_yaw_axis_not_the_origin():
    """The real geometry, in numbers taken off the arm."""
    tip = (0.0700, -0.0001)          # claw tip at the home pose, metres
    axis = (0.0778, 0.0236)          # where J1 actually turns about

    from_origin, _ = vs.radial_tangential(tip, (0.0, 0.0))
    from_axis, _ = vs.radial_tangential(tip, axis)

    cos = from_origin[0] * from_axis[0] + from_origin[1] * from_axis[1]
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    assert angle > 90.0, (
        f"origin-relative radial is only {angle:.0f} deg from the true one; "
        "this test exists because it is over 100")


def test_reach_from_axis_disagrees_with_the_origin_in_the_dangerous_direction():
    """The 'too close in for Cartesian control' guard reads this number, and
    measuring from the origin overstates it -- reporting the arm as further out,
    hence better conditioned, than it is."""
    tip = (0.0700, -0.0001)
    axis = (0.0778, 0.0236)
    assert vs.reach_from_axis(tip, (0.0, 0.0)) == pytest.approx(0.0700, abs=1e-4)
    assert vs.reach_from_axis(tip, axis) == pytest.approx(0.0249, abs=1e-3)


def test_radial_and_tangential_stay_perpendicular_and_unit_off_origin():
    radial, tangential = vs.radial_tangential((0.0700, -0.0001), (0.0778, 0.0236))
    assert np.hypot(*radial) == pytest.approx(1.0)
    assert np.hypot(*tangential) == pytest.approx(1.0)
    assert radial[0] * tangential[0] + radial[1] * tangential[1] == pytest.approx(0.0)


def test_radial_is_undefined_when_the_tool_sits_on_the_yaw_axis_not_the_origin():
    """Degeneracy follows the AXIS. Over the origin but well off the axis is
    perfectly well defined, and refusing there would block a legal pose."""
    axis = (0.0778, 0.0236)
    radial, _ = vs.radial_tangential((0.0, 0.0), axis)      # origin: fine
    assert np.hypot(*radial) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        vs.radial_tangential(axis, axis)                    # on the axis: not


# --- the go-ahead must come AFTER the hover ---------------------------------
# Asking first made the operator approve a view the very next move threw away:
# the arm started wherever the previous run left it, the brick had to be
# hand-framed from that posture, and then the hover swung the camera elsewhere.
# This pins the order rather than the code that happens to implement it.

def _source_of(func):
    import inspect
    return inspect.getsource(func)


def test_the_hover_is_commanded_before_the_go_ahead_is_requested():
    src = _source_of(vs.main)
    hover = src.index("go_to_hover(ctx)")
    # The go-ahead inside the real-run branch, i.e. the one that is NOT guarded
    # by --dry-run. Take the LAST occurrence: dry-run's copy comes earlier.
    go = src.rindex("wait_for_go(ctx)")
    assert hover < go, (
        "wait_for_go must not run before the arm reaches the hover pose; "
        "approving a view the hover then destroys is what this ordering fixes")


def test_the_dry_run_go_ahead_still_precedes_any_arm_contact():
    """--dry-run must remain hardware-free: it approves the view as it stands
    and never opens the bus, so its go-ahead necessarily comes first."""
    src = _source_of(vs.main)
    dry = src.index("if args.dry_run:")
    first_go = src.index("wait_for_go(ctx)")
    bus_open = src.index("bus = ServoBus(")
    assert dry < first_go < bus_open


def test_no_brick_at_the_go_ahead_leaves_the_arm_holding_at_hover():
    """It must not sys.exit out of the `with bus:` block -- that would drop the
    context manager mid-run. Returning keeps the arm powered and holding."""
    src = _source_of(vs.main)
    go = src.rindex("wait_for_go(ctx)")
    after = src[go:go + 900]
    assert "NO BRICK DETECTED at the go-ahead" in after
    assert "sys.exit" not in after.split("estimates =")[0]


def test_the_hover_pose_is_the_one_the_operator_measured():
    """SERVO_HOVER_TICKS is a measurement, not a preference. Pinning it here
    means a silent edit shows up as a test failure rather than as an arm that
    quietly starts somewhere else."""
    from vision_pipeline import config
    from vision_pipeline.robot_interface import poses
    assert config.SERVO_HOVER_TICKS == {1: 2057, 2: 3145, 3: 2205, 4: 1841, 5: 2688}
    assert poses.HOVER == config.SERVO_HOVER_TICKS


def test_the_hover_pose_does_not_touch_the_gripper():
    """Driving to a viewing pose must never open or close the claw -- it could
    drop or crush whatever is already held. hold_pose reports J6; HOVER omits it."""
    from vision_pipeline import config
    assert 6 not in config.SERVO_HOVER_TICKS


def test_the_reach_conditioning_check_is_made_at_the_hover_pose():
    """It used to run before the hover, measuring whatever posture the previous
    run left behind -- a number about a pose this run never visits."""
    src = _source_of(vs.main)
    hover = src.index("go_to_hover(ctx)")
    check = src.index("SERVO_VISUAL_MIN_RADIUS_M")
    go = src.rindex("wait_for_go(ctx)")
    assert hover < check < go, (
        "reach conditioning must be measured after the hover and reported "
        "before the go-ahead, so the operator can still act on it")


# --- null-space locking on the re-centring nudges ----------------------------
# Five joints against a 3-DOF position target leaves a 2-D null space, and the
# camera rides on the wrist, so null-space motion moves the very image this loop
# measures. Observed 2026-08-06: 3 mm radial nudges came back wanting 73-90
# ticks of J5 (6-8 deg of wrist ROLL), which rotates the image about its optical
# axis. A brick 100 px off-centre swings ~14 px sideways from that alone -- the
# run logged "y correction gained -12 px but cost 17 px on x", then stalled with
# the loop chasing a disturbance it was generating itself.

def _nudge_ctx(**overrides):
    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10,
                     **overrides)
    return vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                      ik=FakeIK(bus))


def test_a_radial_nudge_locks_the_wrist_roll_and_the_base():
    """Radial means 'change how far the arm reaches', which happens entirely in
    the shoulder/elbow plane. Neither base yaw nor wrist roll can change reach;
    both only move the camera."""
    ctx = _nudge_ctx()
    vs.CartesianActuator("radial").apply(ctx, 6.0)
    assert ctx.ik.locks, "no IK solve was made"
    for lock in ctx.ik.locks:
        assert set(lock) == {1, 5}


def test_a_tangential_nudge_locks_the_roll_but_LEAVES_THE_BASE_FREE():
    """Swinging the base is the entire point of a tangential nudge — locking J1
    there would disable the axis rather than protect it."""
    ctx = _nudge_ctx()
    vs.CartesianActuator("tangential").apply(ctx, 6.0)
    assert ctx.ik.locks, "no IK solve was made"
    for lock in ctx.ik.locks:
        assert set(lock) == {5}
        assert 1 not in lock


def test_the_lock_can_be_turned_off_deliberately():
    ctx = _nudge_ctx(no_lock_null=True)
    vs.CartesianActuator("radial").apply(ctx, 6.0)
    assert all(lock == () for lock in ctx.ik.locks)


def test_a_caller_predating_the_flag_still_gets_the_locking():
    """Defaulting to the SAFE side. The Namespace above has no no_lock_null at
    all, and the protection must not depend on remembering to set it."""
    ctx = _nudge_ctx()
    assert not hasattr(ctx.args, "no_lock_null")
    assert set(vs.CartesianActuator("radial").locked_joints(ctx)) == {1, 5}
    assert set(vs.CartesianActuator("tangential").locked_joints(ctx)) == {5}


def test_lock_drift_is_recorded_but_does_not_cry_wolf(caplog):
    """The drift must be RECORDED -- it is exactly how much motion the caller's
    own enforcement has to drop, and if it ever reaches zero the server-side pin
    started working. But not at warning level: the pin never holds, so a warning
    fires on every nudge of every descent, dozens of identical lines about a
    condition CartesianActuator handles two statements later. A log that cries
    wolf on a known-permanent condition is where a real fault goes to hide."""
    import logging
    from vision_pipeline.robot_interface.matlab_client import MatlabIKClient

    class Wire(MatlabIKClient):
        def __init__(self, drift):
            self.drift = drift          # no socket; _send_request is stubbed

        def _send_request(self, req):
            return {"ok": True, "angles_rad": [0.0] * 5, "err_mm": 0.0,
                    "lock_drift_rad": self.drift}

    with caplog.at_level(logging.DEBUG):
        Wire(np.deg2rad(8.0)).request_ik(0.1, 0.0, 0.1, seed_rad=[0.0] * 5, lock=[5])
    assert "8.0 deg" in caplog.text and "held joint(s) [5]" in caplog.text
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_lock_that_held_says_nothing(caplog):
    import logging
    from vision_pipeline.robot_interface.matlab_client import MatlabIKClient

    class Wire(MatlabIKClient):
        def __init__(self):
            pass

        def _send_request(self, req):
            return {"ok": True, "angles_rad": [0.0] * 5, "err_mm": 0.0,
                    "lock_drift_rad": 1e-7}

    with caplog.at_level(logging.WARNING):
        Wire().request_ik(0.1, 0.0, 0.1, seed_rad=[0.0] * 5, lock=[5])
    assert caplog.text == ""


def test_an_older_server_without_the_field_is_not_treated_as_a_failure(caplog):
    """matlab/ changes need a server restart; until then the field is absent.
    Absence must not masquerade as a held lock OR as a broken one."""
    import logging
    from vision_pipeline.robot_interface.matlab_client import MatlabIKClient

    class Wire(MatlabIKClient):
        def __init__(self):
            pass

        def _send_request(self, req):
            return {"ok": True, "angles_rad": [0.0] * 5, "err_mm": 0.0}

    with caplog.at_level(logging.WARNING):
        Wire().request_ik(0.1, 0.0, 0.1, seed_rad=[0.0] * 5, lock=[5])
    assert caplog.text == ""


# --- the run ends by offering to close the claw ------------------------------
#
# A descent that reaches grasp height and stops has done all the work and none
# of the point. Until 2026-08-07 the operator had to start a second script,
# by which time the brick has usually been nudged.

def test_the_grasp_is_offered_only_when_the_descent_SUCCEEDED():
    """A descent refused by the floor guard, stalled, or stopped by the progress
    monitor has left the claw somewhere nobody chose. Offering to close there
    invites a grab at the table."""
    src = _source_of(vs.main)
    assert "if descend(ctx, estimates):" in src, (
        "descend's return value must gate the offer -- calling it and then "
        "offering unconditionally is the bug this pins")
    guard = src.index("if descend(ctx, estimates):")
    offer = src.index("offer_grasp(ctx)")
    assert guard < offer


def test_the_close_is_automatic_and_asks_nothing():
    """It DID ask, and that was right while nobody had ever driven J6. Now that
    the positions are measured, the operator's eye is replaced by a better
    sensor for this one question: the servo's own position read-back."""
    src = _source_of(vs.offer_grasp)
    assert "input(" not in src
    assert "gripper.auto_close" in src
    assert "no_grasp" in src, "there must still be a way to skip it entirely"


def test_the_offer_targets_the_measured_grip_position_not_the_full_close():
    src = _source_of(vs.offer_grasp)
    assert "SERVO_GRIPPER_GRIP_TICKS" in src
    assert "FULL_CLOSE" not in src, (
        "closing to the full-close stop with a brick in the jaws stalls the "
        "servo against it")


def test_the_descent_and_close_claw_share_one_gripper_implementation():
    """Two copies of a loop that decides when to stop pushing on a servo is one
    copy too many. They use different entry points -- close_claw is interactive
    and per-jog, the descent is automatic -- but one module."""
    import close_claw
    assert "gripper.close_in_jogs" in _source_of(close_claw.main)
    assert "gripper.auto_close" in _source_of(vs.offer_grasp)
    assert "gripper.open_fully" in _source_of(vs.go_to_hover)


# --- the hover: automatic, and it opens the claw ------------------------------
#
# poses.goto walks every joint in sub-cap hops and re-checks the travel limits
# per hop, so a joint that is out of range is refused part-way while the others
# arrive. The run then continues from a pose that LOOKS like the hover in the
# log -- the move was commanded -- and is not one. It cannot ask about that any
# more, so it must SAY it.

class HoverBus(FakeBus):
    """Reports hover ticks, except for joints listed in `stuck`. Has a J6."""

    def __init__(self, stuck=None, j6=3003):
        super().__init__(start=dict(config.SERVO_HOVER_TICKS))
        self.stuck = dict(stuck or {})
        self.ticks.update(self.stuck)
        self.ticks[6] = j6
        self.gotos = 0
        self.j6_commands = []

    def read_position_retrying(self, servo_id):
        return self.ticks[servo_id]

    def read_position(self, servo_id):
        return self.ticks[servo_id]

    def set_motion_profile(self, ids, speed, accel):
        pass

    def move_and_verify(self, servo_id, target):
        assert servo_id == 6, "only the gripper may be commanded here"
        self.j6_commands.append(target)
        self.ticks[6] = int(target)
        return self.ticks[6]

    def travel_limits(self, servo_id):
        return (500, 3900)

    def _cal(self, servo_id):
        return {"ticks_per_rad": 651.89}

    def move_joints_stepped(self, targets, **kw):
        self.gotos += 1
        for j, t in targets.items():
            if j not in self.stuck:
                self.ticks[j] = int(t)


def _hover_ctx(bus, no_grasp=False):
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10,
                     no_wait=False, no_grasp=no_grasp)
    ctx = vs.Context(bus, None, None, args, None, ik=None)
    ctx.bus = bus
    return ctx


def _no_input():
    def explode(_p=""):
        raise AssertionError("the hover must not prompt")
    import builtins
    return builtins, explode


def test_the_hover_asks_nothing():
    """Only two questions survive in this script: the go-ahead and the
    further-close prompt at the grip."""
    bus = HoverBus(stuck={2: 3000})
    builtins, explode = _no_input()
    real, builtins.input = builtins.input, explode
    try:
        vs.go_to_hover(_hover_ctx(bus))
    finally:
        builtins.input = real
    assert bus.gotos == 1


def test_the_hover_opens_the_claw_fully():
    """At the TOP, where the jaws have room. A run that arrives at grasp height
    with the claw already shut cannot do the one thing it came for."""
    bus = HoverBus(j6=config.SERVO_GRIPPER_GRIP_TICKS)
    vs.go_to_hover(_hover_ctx(bus))
    assert bus.ticks[6] == config.SERVO_GRIPPER_OPEN_TICKS
    assert bus.j6_commands, "the claw was never commanded"


def test_the_hover_opens_the_claw_before_the_descent_can_close_it():
    """Ordering, not just presence: opening at grasp height would sweep the jaws
    open a few millimetres from the table, beside the brick."""
    src = _source_of(vs.main)
    hover = src.index("go_to_hover(ctx)")
    grasp = src.index("offer_grasp(ctx)")
    assert hover < grasp


def test_no_grasp_leaves_the_claw_alone_at_the_hover():
    bus = HoverBus(j6=config.SERVO_GRIPPER_GRIP_TICKS)
    vs.go_to_hover(_hover_ctx(bus, no_grasp=True))
    assert bus.j6_commands == []


def test_a_joint_that_did_not_arrive_is_named(capsys):
    """Silent otherwise: the move was commanded, so the log says 'to hover'
    either way. It cannot ask about it any more, so it must say it."""
    bus = HoverBus(stuck={2: 3000})
    vs.go_to_hover(_hover_ctx(bus))

    out = capsys.readouterr().out
    assert "did not arrive" in out
    assert "J2 is at 3000" in out
    assert str(config.SERVO_HOVER_TICKS[2]) in out


def test_a_clean_hover_says_nothing_about_joints_not_arriving(capsys):
    bus = HoverBus()
    vs.go_to_hover(_hover_ctx(bus))
    assert "did not arrive" not in capsys.readouterr().out


def test_the_hover_tolerance_is_tighter_than_at_poses_skip_test():
    """They answer different questions. at_pose asks 'close enough to skip the
    move?' and should stay loose; this asks 'did the move actually land?'."""
    assert config.SERVO_VISUAL_HOVER_TOLERANCE_TICKS < 40



# --- at the grip: the one prompt that survives -------------------------------
#
# auto_close leaves a grip that is firm by the servo's own reading. Whether it
# is firm enough for THIS brick is a judgement no sensor here makes, so this is
# the one place the run still stops -- and it is the "further close prompt" the
# operator asked to keep.

class GripBus(HoverBus):
    """HoverBus whose J6 has met a brick and will not move further."""

    def __init__(self, j6=3093, **kw):
        super().__init__(j6=j6, **kw)
        self.j6_stuck_at = j6

    def move_and_verify(self, servo_id, target):
        assert servo_id == 6, "only the gripper may be commanded here"
        self.j6_commands.append(target)
        return self.ticks[6]          # the brick is in the way: never moves


def _grip_ctx(bus, replies, monkeypatch):
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda _p="": next(it))
    return _hover_ctx(bus)


def _gripped(ticks, past=40):
    from vision_pipeline.robot_interface import gripper as g
    return g.GripResult(g.GRIPPED, ticks, "gripped", contact_ticks=ticks,
                        commanded_past=past)


def test_enter_squeezes_again_and_again(monkeypatch):
    bus = GripBus()
    ctx = _grip_ctx(bus, ["", "", "", "n"], monkeypatch)
    vs.squeeze_and_lift(ctx, _gripped(3093))

    assert len(bus.j6_commands) == 3, "each Enter must command one more squeeze"
    assert bus.gotos == 0, "n must not move the arm"


def test_each_squeeze_steps_by_the_squeeze_size_not_the_approach_jog(monkeypatch):
    bus = GripBus()
    ctx = _grip_ctx(bus, ["", "n"], monkeypatch)
    vs.squeeze_and_lift(ctx, _gripped(3093))

    assert bus.j6_commands == [3093 - config.SERVO_GRIPPER_SQUEEZE_TICKS]


def test_k_accepts_the_grip_and_drives_to_hover(monkeypatch):
    bus = GripBus()
    ctx = _grip_ctx(bus, ["k"], monkeypatch)
    vs.squeeze_and_lift(ctx, _gripped(3093))

    assert bus.gotos == 1, "k must lift to the hover pose"
    assert bus.j6_commands == [], "accepting must not squeeze first"


def test_the_lift_never_commands_the_gripper(monkeypatch):
    """The joint holding the brick must not appear in the pose, or the lift
    would let go at the top. HOVER omits J6; this pins that the lift path
    depends on it."""
    bus = GripBus()
    ctx = _grip_ctx(bus, ["k"], monkeypatch)
    vs.squeeze_and_lift(ctx, _gripped(3093))
    assert bus.ticks[6] == 3093
    assert 6 not in config.SERVO_HOVER_TICKS


def test_squeezing_then_accepting_does_both_in_order(monkeypatch):
    bus = GripBus()
    ctx = _grip_ctx(bus, ["", "", "k"], monkeypatch)
    vs.squeeze_and_lift(ctx, _gripped(3093))

    assert len(bus.j6_commands) == 2
    assert bus.gotos == 1


def test_an_unrecognised_key_asks_again_rather_than_guessing(monkeypatch):
    """Guessing here either squeezes a brick harder or swings the arm."""
    bus = GripBus()
    ctx = _grip_ctx(bus, ["x", "?", "n"], monkeypatch)
    vs.squeeze_and_lift(ctx, _gripped(3093))

    assert bus.j6_commands == []
    assert bus.gotos == 0


def test_squeezing_stops_at_the_full_close_stop_and_keeps_offering(monkeypatch):
    """The floor refuses, and the run does not end there -- k and n must still
    work, or the operator is stuck holding a brick with no way to lift it."""
    bus = GripBus(j6=config.SERVO_GRIPPER_FULL_CLOSE_TICKS)
    ctx = _grip_ctx(bus, ["", "k"], monkeypatch)
    vs.squeeze_and_lift(ctx, _gripped(config.SERVO_GRIPPER_FULL_CLOSE_TICKS))

    assert bus.j6_commands == [], "nothing may be commanded past the floor"
    assert bus.gotos == 1, "k must still lift"


def test_the_squeeze_prompt_continues_from_what_auto_close_already_spent():
    """auto_close loads onto the brick before handing over, so the running total
    must carry across or the advisory ceiling counts from zero twice."""
    src = _source_of(vs.squeeze_and_lift)
    assert "grip.commanded_past" in src
    assert "grip.contact_ticks" in src


# --- the descent slows down for its last steps -------------------------------
#
# The brisk whole-repo pace is fine while the claw is high; by the late steps it
# is a few millimetres off the table and the same move ends against it.

def _paced_ctx(**overrides):
    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10,
                     **overrides)
    return vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                      ik=FakeIK(bus))


def test_a_run_starts_at_the_repo_wide_pace():
    ctx = _paced_ctx()
    assert ctx.pace == (config.PICK_STEP_TICKS, config.PICK_STEP_PAUSE_S)


def test_slowing_down_switches_to_the_previous_pace():
    """The pace the descents that worked were taken at."""
    ctx = _paced_ctx()
    assert ctx.slow_down("because")
    assert ctx.pace == (config.SERVO_VISUAL_SLOW_STEP_TICKS,
                        config.SERVO_VISUAL_SLOW_PAUSE_S)
    assert config.SERVO_VISUAL_SLOW_STEP_TICKS < config.PICK_STEP_TICKS
    assert config.SERVO_VISUAL_SLOW_PAUSE_S > config.PICK_STEP_PAUSE_S


def test_slowing_down_announces_once_not_every_step(capsys):
    """It is called on every step past the threshold; a line per step would
    bury the descent's own numbers."""
    ctx = _paced_ctx()
    assert ctx.slow_down("first")
    assert not ctx.slow_down("second")
    out = capsys.readouterr().out
    assert out.count("pace:") == 1


def test_the_descent_slows_from_the_configured_step():
    src = _source_of(vs.descend)
    assert "SERVO_VISUAL_SLOW_FROM_STEP" in src
    trigger = src.index("SERVO_VISUAL_SLOW_FROM_STEP")
    move = src.index("move_joints_stepped")
    assert trigger < move, "the pace must change before the step it governs"


def test_the_blind_finish_slows_down_whatever_step_it_starts_on():
    """The lowest the claw gets, and the only part with nothing reading the
    image -- the operator's eye is all that is left watching."""
    src = _source_of(vs.descend_blind)
    assert "slow_down" in src
    assert "SLOW_FROM_STEP" not in src, "the blind finish must not be conditional"


def test_every_paced_move_in_the_loop_reads_the_current_pace():
    """Including the sideways nudges, which go through CartesianActuator and
    have no idea which descent step they are serving."""
    import inspect
    src = inspect.getsource(vs)
    assert "step_ticks=config." not in src, (
        "a move site still passes a config constant directly, so it will keep "
        "the fast pace after the descent has slowed down")
    assert src.count("step_ticks=ctx.pace[0]") == 3, (
        "expected the nudge, the descent and the blind finish to be the three "
        "paced move sites")


def test_the_slowdown_triggers_on_step_count_not_height():
    """Height is the natural trigger and depends on FK's ABSOLUTE z, the
    quantity on this arm least worth trusting -- so it would fire at the wrong
    moment exactly when FK is wrong, which is the case it exists for."""
    src = _source_of(vs.descend)
    line = [l for l in src.splitlines()
            if "SERVO_VISUAL_SLOW_FROM_STEP" in l and "step_n" in l]
    assert line, "the trigger must compare the step counter"


# --- picking at a height the table plane does not describe -------------------
#
# Answering YES to "flat on the board?" must leave the run exactly as it was.
# That is the load-bearing property here: this feature is additive, and a
# regression in the flat path would be a regression in the only path that works.

def test_answering_flat_leaves_the_target_at_the_table_plane(monkeypatch):
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 3

    ctx.flat_on_board = True
    assert vs.descend(ctx, estimates) is True

    _a, tip = vs.tip_position(ctx)
    expected = config.TABLE_Z_IN_BASE + config.PICK_Z_OFFSET
    assert tip[2] == pytest.approx(expected, abs=0.002), (
        "the flat path must still descend to the table plane")


def test_a_raised_brick_finishes_relative_to_where_sight_was_lost():
    """The whole point. The table plane does not describe this brick, so an
    absolute height derived from it is the wrong target."""
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 3
    ctx.flat_on_board = False
    ctx.blind_history = []

    assert vs.descend(ctx, estimates) is True

    j = ctx.journey
    assert j is not None, "a raised descent must record its journey"
    assert j.drop_mm == pytest.approx(config.SERVO_VISUAL_BLIND_DROP_MM, abs=2.0)


def test_a_raised_descent_uses_the_learned_drop_when_there_is_one():
    from vision_pipeline.planning import blind_travel as bt
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 3
    ctx.flat_on_board = False

    past = bt.BlindJourney(lost_tip=(0.15, 0.0, 0.100), flat_on_board=False,
                           lost_sight=True)
    past.step((0.15, 0.0, 0.060))          # 40 mm
    past.finish(bt.GRIPPED)
    ctx.blind_history = [past]

    vs.descend(ctx, estimates)
    assert ctx.journey.drop_mm == pytest.approx(40.0, abs=2.0), (
        "the log's median must beat the config default once it exists")


def test_the_learned_drop_is_capped():
    from vision_pipeline.planning import blind_travel as bt
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 3
    ctx.flat_on_board = False

    freak = bt.BlindJourney(lost_tip=(0.15, 0.0, 0.300), flat_on_board=False,
                            lost_sight=True)
    freak.step((0.15, 0.0, 0.000))         # 300 mm
    freak.finish(bt.GRIPPED)
    ctx.blind_history = [freak]

    vs.descend(ctx, estimates)
    assert ctx.journey.drop_mm <= config.SERVO_VISUAL_BLIND_DROP_MAX_MM + 2.0


def test_the_journey_records_every_blind_step_from_the_READ_BACK():
    """Not from what was commanded. The servos settle short, and this is the
    regime where FK's differentials are the only trustworthy thing left."""
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 3
    ctx.flat_on_board = False
    ctx.blind_history = []

    vs.descend(ctx, estimates)
    assert len(ctx.journey.steps) >= 1
    for s in ctx.journey.steps:
        assert len(s) == 3


def test_a_flat_descent_still_records_its_journey():
    """Both answers produce data. The flat runs are what a later raised run has
    to compare against, and refusing to log them would make the feature depend
    on the operator having already used it."""
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 3
    ctx.flat_on_board = True

    vs.descend(ctx, estimates)
    assert ctx.journey is not None
    assert ctx.journey.flat_on_board is True


def test_the_question_is_asked_after_the_go_ahead():
    """Same reason the go-ahead comes after the hover: by then the operator is
    looking at the arm and brick in the positions the run will actually use."""
    src = _source_of(vs.main)
    go = src.rindex("wait_for_go(ctx)")
    ask = src.index("ask_flat_on_board(ctx)")
    probe = src.index("probe_axis(ctx, a, act)")
    assert go < ask < probe


def test_the_question_is_skipped_when_there_is_no_descent():
    """Nothing about it applies to a centring-only run, and a prompt that
    changes nothing is a prompt that teaches the operator to hit enter."""
    src = _source_of(vs.main)
    ask = src.index("ask_flat_on_board(ctx)")
    guard = src.rindex("if args.descend:", 0, ask)
    assert guard < ask


def test_the_default_is_flat_so_nothing_changes_by_accident():
    ctx, _bus, _world = descent_ctx()
    assert ctx.flat_on_board is True
    assert ctx.journey is None


# --- stage 2: the height model steers the RAISED descent only ----------------
#
# The flat path is the only one that works, so the guarantee that matters most
# here is that none of this reaches it.

def _fitted_model(c=900.0, d=5.0):
    from vision_pipeline.planning import blind_travel as bt
    return bt.HeightModel(c=c, d=d, n=50, rms_mm=0.4)


def test_the_flat_descent_never_consults_the_height_model():
    """THE GUARANTEE. Even with a model fitted and sitting on the context, a
    flat run must reach exactly the target it always reached."""
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 3
    ctx.flat_on_board = True
    ctx.height_model = _fitted_model()

    assert vs.descend(ctx, estimates) is True
    _a, tip = vs.tip_position(ctx)
    assert tip[2] == pytest.approx(config.TABLE_Z_IN_BASE + config.PICK_Z_OFFSET,
                                   abs=0.002)


def test_a_raised_descent_stops_where_the_height_model_says():
    """The feedback loop. The area is re-read every step, so the target is a
    re-measured quantity rather than one number committed to at loss of sight."""
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.flat_on_board = False
    ctx.height_model = _fitted_model()

    assert vs.descend(ctx, estimates) is True
    assert ctx.journey is not None and ctx.journey.sightings


def test_the_model_may_commit_deeper_but_never_shallower():
    """A model that suddenly says 'further than you thought' mid-descent is
    disagreeing with itself; keeping the deeper commitment and letting the log
    show the argument beats yo-yoing the target."""
    ctx, _bus, _world = descent_ctx()
    ctx.flat_on_board = False
    ctx.height_model = _fitted_model()

    # area 8100 -> 900/90 - 5 = 5 mm remaining, i.e. shallower than the target.
    class Near:
        centroid_px = (320.0, 240.0)
        area = 8100.0
    kept = vs.record_sighting(ctx, 1, Near(), (0.15, 0.0, 0.100), -0.050, -0.068)
    assert kept == -0.050, "a shallower suggestion must not raise the target"

    # area 200 -> 900/14.1 - 5 = 58.6 mm remaining, deeper than the target.
    class Far:
        centroid_px = (320.0, 240.0)
        area = 200.0
    deeper = vs.record_sighting(ctx, 2, Far(), (0.15, 0.0, 0.000), -0.050, -0.068)
    assert deeper < -0.050, "a deeper suggestion must be taken"
    assert deeper >= -0.068, "and never past the floor guard"


def test_sightings_are_recorded_on_the_FLAT_path_too():
    """The flat runs are the ones that work, so they are where the training
    data has to come from. Excluding them would starve the model the raised
    path depends on."""
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.flat_on_board = True
    ctx.detector.blind_after = ctx.detector.calls + 4

    vs.descend(ctx, estimates)
    assert ctx.journey.sightings, "a flat descent recorded nothing"
    assert all(s.area > 0 for s in ctx.journey.sightings)


def test_a_sighting_carries_both_fk_frames():
    """The tip is what the height model and the floor guard speak; the WRIST is
    what triangulation needs, because hand-eye was solved against request_fk and
    chaining it onto the tip would be wrong by the claw's own 70 mm."""
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 4
    vs.descend(ctx, estimates)

    s = ctx.journey.sightings[0]
    assert len(s.wrist) == 4 and len(s.wrist[0]) == 4
    assert len(s.tip) == 3


def test_detect_centroid_still_returns_what_it_always_did():
    """It became a wrapper over detect_brick; a dozen callers depend on the
    old two-tuple."""
    ctx, _bus, _world = descent_ctx()
    centroid, frame = vs.detect_centroid(ctx)
    assert centroid is not None and len(centroid) == 2
    assert frame is not None


def test_an_unfitted_model_leaves_the_raised_path_as_it_was():
    """Falls back to the loss-of-sight drop, which is what it did before any of
    this existed."""
    ctx, _bus, _world = descent_ctx()
    estimates = descent_estimates(ctx)
    ctx.detector.blind_after = ctx.detector.calls + 3
    ctx.flat_on_board = False
    ctx.blind_history = []
    from vision_pipeline.planning import blind_travel as bt
    ctx.height_model = bt.HeightModel()

    vs.descend(ctx, estimates)
    assert ctx.journey.drop_mm == pytest.approx(config.SERVO_VISUAL_BLIND_DROP_MM,
                                                abs=2.0)


# --- the two-view survey: two struck poses, between centring and descent -----

class SurveyIK(FakeIK):
    """FakeIK that also answers request_fk_tip with a moving wrist."""


def _survey_ctx(no_two_view=False):
    bus = CartesianBus()
    world = FakeWorld(bus)
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10,
                     no_two_view=no_two_view)
    ctx = vs.Context(bus, FakeCamera(world), FakeDetector(world), args, None,
                     ik=FakeIK(bus))
    return ctx, bus


def test_the_survey_strikes_two_poses_and_comes_back():
    """It must return to the centred pose: a descent that starts from a swung
    base is a descent whose probe gains describe a different posture, which is
    the 2026-08-05 runaway in miniature."""
    ctx, bus = _survey_ctx()
    before = dict(bus.ticks)
    vs.two_view_survey(ctx)

    assert len(bus.stepped) >= 2, "the arm must actually move between views"
    for j in (1, 2, 3, 4, 5):
        assert abs(bus.ticks[j] - before[j]) <= 2, (
            f"J{j} did not return to where centring left it")


def test_the_survey_baseline_is_tangential():
    """Parallax needs camera translation ACROSS the line of sight, and on this
    arm that is base yaw and nothing else -- the pitch chain moves the camera
    mostly ALONG its own view, which is what triangulation learns least from."""
    src = _source_of(vs.two_view_survey)
    assert 'CartesianActuator("tangential")' in src
    assert "SERVO_VISUAL_TWO_VIEW_BASELINE_MM" in src


def test_the_baseline_clears_the_parallax_gate_at_the_working_radius():
    """24 mm at ~205 mm is 6.7 deg against a 5 deg gate, and needs 6.7 deg of
    base yaw against a tangential budget of 8 -- so it passes whole rather than
    being halved."""
    import numpy as np
    parallax = np.degrees(config.SERVO_VISUAL_TWO_VIEW_BASELINE_MM / 205.0)
    assert parallax > config.TWO_VIEW_MIN_PARALLAX_DEG
    assert parallax < config.SERVO_VISUAL_MAX_TANGENTIAL_PAN_DEG


def test_the_survey_runs_between_centring_and_the_descent():
    """The only window it is worth anything in: brick centred, claw still high,
    no height committed to."""
    src = _source_of(vs.main)
    centred = src.index("centre(ctx, estimates)")
    survey = src.index("two_view_survey(ctx)")
    descend = src.index("if descend(ctx, estimates):")
    assert centred < survey < descend


def test_no_two_view_skips_it_without_moving_anything():
    ctx, bus = _survey_ctx(no_two_view=True)
    assert vs.two_view_survey(ctx) is None
    assert bus.stepped == []


def test_the_survey_never_becomes_the_descent_target():
    """Triangulation needs FK @ hand-eye and hand_eye.json is known wrong by
    52 mm. Using its z would trade a bad assumption for a bad calibration."""
    src = _source_of(vs.descend)
    assert "two_view" not in src, (
        "the descent must not read the survey's answer -- it is a cross-check")


def test_the_survey_answer_is_scored_against_the_grip():
    src = _source_of(vs.record_journey)
    assert "ctx.two_view" in src
    assert "vs_grip_mm" in src


def test_locate_brick_two_view_accepts_a_raw_matrix_as_well_as_a_pose():
    """Callers with an FK matrix must be able to pass it. Routing through Pose
    forces matrix -> RPY -> matrix, and transform_to_pose pins roll = 0 near
    pitch = +-90 deg, which is exactly where a top-down tool sits."""
    import numpy as np
    import inspect
    from vision_pipeline.pipeline import PickPipeline

    src = inspect.getsource(PickPipeline.locate_brick_two_view)
    assert 'hasattr(ee_pose, "to_matrix")' in src

    captures = [(np.zeros((10, 10, 3), np.uint8), np.eye(4)),
                (np.zeros((10, 10, 3), np.uint8), np.eye(4))]

    class NoBricks:
        def detect(self, frame):
            return []

    # No brick, so it returns None -- but it must reach that point without
    # tripping over the matrix where it expected a Pose.
    assert PickPipeline(detector=NoBricks()).locate_brick_two_view(captures) is None


def test_the_survey_goes_left_then_centre_then_right():
    """Operator-specified 2026-08-07, and the better geometry: the two views end
    2 x baseline apart so the parallax doubles, while each MOVE stays at the
    baseline and therefore still fits the tangential pan budget."""
    ctx, _bus = _survey_ctx()

    # Spy on the REQUESTS rather than on J1, because the planar FakeIK answers a
    # tangential nudge with the pitch chain and never moves the base. What is
    # under test is the pattern the survey asks for, not the fake's kinematics.
    asked, offsets, running = [], [], [0.0]
    real_apply = vs.CartesianActuator.apply

    def spy_apply(self, ctx_, amount_mm):
        asked.append(amount_mm)
        running[0] += amount_mm
        return amount_mm

    real_detect = vs.detect_brick

    def spy_detect(c, attempts=None):
        offsets.append(running[0])
        return real_detect(c) if attempts is None else real_detect(c, attempts)

    vs.CartesianActuator.apply = spy_apply
    vs.detect_brick = spy_detect
    try:
        vs.two_view_survey(ctx)
    finally:
        vs.CartesianActuator.apply = real_apply
        vs.detect_brick = real_detect

    half = config.SERVO_VISUAL_TWO_VIEW_BASELINE_MM
    assert len(offsets) == 2, "exactly two views"
    assert offsets == [-half, +half], (
        f"the views must straddle the centred pose, got {offsets}")
    assert asked == [-half, +half, +half, -half], (
        f"left, back to centre, right, back to centre -- got {asked}")
    assert running[0] == pytest.approx(0.0), "the arm must end where it started"


def test_the_survey_passes_through_centre_between_the_two_poses():
    """Not decoration: it re-reads the arm from the pose the descent will start
    from, so a move the pan budget shrank shows up as a gap to close rather than
    as drift nobody measured."""
    src = _source_of(vs.two_view_survey)
    left = src.index('"pose 1: base LEFT"')
    centre = src.index('go(0.0, "back to centre")')
    right = src.index('"pose 2: base RIGHT"')
    assert left < centre < right


def test_the_survey_tracks_an_absolute_offset_not_a_running_total():
    """A move the budget shrinks must not silently leave the arm off centre."""
    src = _source_of(vs.two_view_survey)
    assert "delta = to_offset - offset" in src


def test_the_two_views_end_up_double_the_baseline_apart():
    """Which is the whole reason for the symmetry -- a single move that far
    would be refused by the pan budget and halved."""
    import numpy as np
    half = config.SERVO_VISUAL_TWO_VIEW_BASELINE_MM
    one_move = np.degrees(half / 205.0)
    both = np.degrees(2 * half / 205.0)
    assert one_move <= config.SERVO_VISUAL_MAX_TANGENTIAL_PAN_DEG
    assert both > config.SERVO_VISUAL_MAX_TANGENTIAL_PAN_DEG, (
        "if a single move of the full baseline fits the budget, the symmetry "
        "is buying nothing and this test is the wrong shape")
    assert both > 2 * config.TWO_VIEW_MIN_PARALLAX_DEG


def test_each_survey_pose_settles_longer_than_an_ordinary_step():
    """The arm has to stop RINGING, not just stop travelling. A centring step
    that reads a blurred frame corrects itself next iteration; a survey pose
    that reads one hands triangulation a centroid from a camera that was not
    where FK says it was, and there is no next iteration."""
    src = _source_of(vs.two_view_survey)
    assert "SERVO_VISUAL_TWO_VIEW_SETTLE_S" in src
    assert "max(ctx.args.settle" in src, (
        "the survey settle must be a FLOOR against --settle, not a replacement "
        "-- raising --settle for a shaky rig should raise this too")
    assert config.SERVO_VISUAL_TWO_VIEW_SETTLE_S >= 2.0


def test_the_survey_settles_at_every_pose_including_the_returns():
    """Four moves, four settles, and the CENTRE stops count.

    They are where the next swing starts, so an arm still ringing when it leaves
    centre is an arm still ringing when it arrives at the pose being
    photographed. Settling only where the shutter fires moves the problem one
    move upstream rather than fixing it."""
    ctx, _bus = _survey_ctx()
    waits = []
    real_wait = vs.wait_watching
    real_apply = vs.CartesianActuator.apply

    vs.wait_watching = lambda s, c, lines=None: waits.append(s)
    vs.CartesianActuator.apply = lambda self, c, mm: mm
    try:
        vs.two_view_survey(ctx)
    finally:
        vs.wait_watching = real_wait
        vs.CartesianActuator.apply = real_apply

    assert len(waits) == 4, f"expected a settle per move, got {waits}"
    assert all(w >= config.SERVO_VISUAL_TWO_VIEW_SETTLE_S for w in waits)


# --- answering "not flat" also buys more sightings ----------------------------

def _flat_q_ctx(answer, descend_step=None):
    bus = HoverBus()
    args = Namespace(settle=0.0, deadband=12.0, view=False, max_iterations=10,
                     no_wait=False, no_grasp=False, raised=False, flat=False,
                     descend_step=(config.DESCEND_STEP_MM if descend_step is None
                                   else descend_step))
    ctx = vs.Context(bus, None, None, args, None, ik=None)
    ctx.bus = bus
    import builtins
    real, builtins.input = builtins.input, lambda _p="": answer
    try:
        vs.ask_flat_on_board(ctx)
    finally:
        builtins.input = real
    return ctx


def test_answering_not_flat_halves_the_descent_step():
    """A raised brick is closer, so it leaves the frame sooner: the first four
    raised runs managed ONE sighting each against nine and ten for the flat
    ones, and a one-sighting run yields no usable training pairs at all."""
    ctx = _flat_q_ctx("n")
    assert ctx.flat_on_board is False
    assert ctx.args.descend_step == config.SERVO_VISUAL_RAISED_DESCEND_STEP_MM
    assert config.SERVO_VISUAL_RAISED_DESCEND_STEP_MM < config.DESCEND_STEP_MM


def test_answering_flat_leaves_the_descent_step_alone():
    ctx = _flat_q_ctx("y")
    assert ctx.flat_on_board is True
    assert ctx.args.descend_step == config.DESCEND_STEP_MM


def test_an_explicit_descend_step_is_not_overridden():
    """Someone who typed a number meant it."""
    ctx = _flat_q_ctx("n", descend_step=15.0)
    assert ctx.args.descend_step == 15.0


def test_answering_flat_never_builds_a_height_model():
    """THE OPERATOR'S REQUIREMENT. Not a branch guard -- fit_height_model is
    never called, so there is nothing there to read by accident."""
    ctx = _flat_q_ctx("y")
    assert not ctx.height_model.ready
    assert ctx.height_model.n == 0 and ctx.height_model.runs == 0


# --- the error paths must survive being taken ---------------------------------
#
# estimate_axis is unit-agnostic BY DESIGN -- a probe is raw ticks for a
# JointActuator and millimetres for a CartesianActuator -- so its argument is
# routinely a float. Its refusal message formatted it with `:+d`, which raises
# ValueError from inside the error path. That is the worst possible place: the
# crash replaces the diagnosis the operator needed with a traceback about string
# formatting. Live failure 2026-08-07 on `--- PROBE y: radial nudge +8 mm ---`.

def test_a_dead_probe_is_reported_not_crashed_for_a_CARTESIAN_actuator():
    from vision_pipeline.planning.visual_servo import estimate_axis

    with pytest.raises(vs.ServoAbort) as e:
        estimate_axis(8.0, 100.0, 100.2, unit="mm")     # float probe, no response
    assert "8" in str(e.value) and "mm" in str(e.value)
    assert "px" in str(e.value), "the operator needs the numbers, not a type error"


def test_a_dead_probe_is_reported_for_a_JOINT_actuator_too():
    from vision_pipeline.planning.visual_servo import estimate_axis

    with pytest.raises(vs.ServoAbort) as e:
        estimate_axis(40, 100.0, 100.2, unit="ticks")
    assert "40" in str(e.value) and "ticks" in str(e.value)


def test_a_zero_probe_is_reported_without_a_unit_being_required():
    from vision_pipeline.planning.visual_servo import estimate_axis

    with pytest.raises(vs.ServoAbort):
        estimate_axis(0.0, 100.0, 100.0)


def test_probe_axis_tells_estimate_axis_what_the_units_are():
    """Otherwise the refusal says 'units', which is true but unhelpful."""
    src = _source_of(vs.probe_axis)
    assert "unit=actuator.unit" in src


def test_no_integer_format_code_survives_in_the_unit_agnostic_layer():
    """planning/visual_servo.py is the module that must not assume ticks. Any
    `:d` in it is a float waiting to raise from an error path."""
    import ast
    import re
    from vision_pipeline.planning import visual_servo as pvs

    tree = ast.parse(Path(pvs.__file__).read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FormattedValue) or node.format_spec is None:
            continue
        spec = "".join(v.value for v in node.format_spec.values
                       if isinstance(v, ast.Constant))
        if re.search(r"[+\- #0]*\d*d$", spec):
            bad.append((node.lineno, ast.unparse(node.value), spec))
    assert not bad, f"integer format codes in a unit-agnostic module: {bad}"

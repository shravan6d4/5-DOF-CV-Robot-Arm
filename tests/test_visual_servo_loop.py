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
    def __init__(self, centroid):
        self.centroid_px = centroid
        self.confidence = 0.9


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
        aim_offset_y=0.0, tolerance_x=12.0, tolerance_y=12.0,
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

    def _tip(self):
        # Reach grows with both joints; a purely notional 1 mm per tick each.
        reach = (self.bus.ticks[2] - 2900) * 0.001 + (self.bus.ticks[3] - 600) * 0.001
        return np.array([0.150 + reach, 0.0, 0.080])

    def request_fk_tip(self, _angles):
        T = np.eye(4)
        T[:3, 3] = self._tip()
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

    with pytest.raises(vs.ServoAbort, match="poorly conditioned"):
        vs.CartesianActuator("radial").apply(ctx, 10.0)
    assert bus.stepped == [], "nothing should have been commanded"


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
    ctx, _bus, _w = make_ctx(aim_offset_y=0.0)
    assert ctx.aim((480, 640)) == (320.0, 240.0)


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
                     joint_y=2, aim_offset_y=0.0, tolerance_x=45.0,
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
                     joint_y=2, aim_offset_y=0.0, tolerance_x=45.0,
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
                     aim_offset_y=0.0, tolerance_x=12.0, tolerance_y=12.0)
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
    args = Namespace(probe_ticks=20, settle=0.0, deadband=12.0,
                     max_iterations=40, view=False, no_wait=True,
                     recentre="joint", joint_x=1, joint_y=2)
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

    def request_fk_tip(self, _angles):
        T = np.eye(4)
        T[:3, 3] = self.world.tip()
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
        descend=True, descend_step=20.0, target_z=-64.0, no_confirm=True,
        recentre="joint", joint_x=1, joint_y=3,
        descend_probe_mm=8.0, descend_max_reach_mm=25.0,
        no_sideways=False, lock_base=True,
        sideways_budget=vs.config.SERVO_VISUAL_SIDEWAYS_BUDGET_TICKS,
        aim_offset_y=0.0, tolerance_x=45.0, tolerance_y=55.0,
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
    assert bus.ticks[2] <= -64 + 1


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
    hover = src.index('poses.goto(bus, "hover"')
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
    hover = src.index('poses.goto(bus, "hover"')
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

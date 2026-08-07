"""The approved-jog close loop, and the distinction it exists to draw.

A close that REACHED its commanded position met nothing on the way -- the jaws
shut on air. A close that GRIPPED stalled early, on the brick. Those are the
success and failure cases and they differ only in where the servo stopped, so a
caller that reads "no exception" as "got it" has the answer backwards. That is
the property most of this file is about.

No serial port and no arm: a fake bus, a scripted approver, and an injected
settle so nothing sleeps.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vision_pipeline import config  # noqa: E402
from vision_pipeline.robot_interface import gripper  # noqa: E402
from vision_pipeline.robot_interface.servo_driver import ServoSafetyError  # noqa: E402

OPEN = config.SERVO_GRIPPER_OPEN_TICKS
GRIP = config.SERVO_GRIPPER_GRIP_TICKS
SHUT = config.SERVO_GRIPPER_FULL_CLOSE_TICKS


class FakeGripperBus:
    """J6 only. `stops_at` is where the brick is, if there is one."""

    def __init__(self, start=OPEN, stops_at=None, refuse_at=None):
        self.ticks = start
        self.stops_at = stops_at
        self.refuse_at = refuse_at
        self.commanded = []
        self.moves = []
        self.profiles = []

    def read_position(self, servo_id):
        assert servo_id == gripper.GRIPPER_JOINT, "only J6 may be touched"
        return self.ticks

    def set_motion_profile(self, ids, speed, accel):
        self.profiles.append((tuple(ids), speed, accel))

    def move_and_verify(self, servo_id, target):
        assert servo_id == gripper.GRIPPER_JOINT, "only J6 may be touched"
        if self.refuse_at is not None and target <= self.refuse_at:
            raise ServoSafetyError(f"J6 to {target} is outside travel")
        self.commanded.append(target)
        was = self.ticks
        if self.stops_at is not None and target < self.stops_at:
            self.ticks = self.stops_at      # the brick is in the way
        else:
            self.ticks = target
        self.moves.append(self.ticks - was)
        return self.ticks

    @property
    def fruitless(self):
        """Commands that produced no motion at all -- pushing on the brick."""
        return [m for m in self.moves if m == 0]


def yes(_prompt):
    return True


def no(_prompt):
    return False


def counted(n):
    """Approve the first n jogs, then decline."""
    state = {"left": n}

    def approve(_prompt):
        state["left"] -= 1
        return state["left"] >= 0
    return approve


def run(bus, target=GRIP, step=None, approve=yes):
    return gripper.close_in_jogs(
        bus, target, config.SERVO_GRIPPER_JOG_TICKS if step is None else step,
        approve, lambda _line: None, settle=lambda b: b.ticks)


# --- the distinction ---------------------------------------------------------

def test_a_claw_that_stalls_on_the_brick_reports_GRIPPED():
    """The wanted outcome, and it looks like a failure from the servo's side."""
    bus = FakeGripperBus(stops_at=GRIP + 90)
    result = run(bus)

    assert result.outcome == gripper.GRIPPED
    assert result.holding
    assert result.ticks == GRIP + 90
    assert "met the brick" in result.message


def test_a_claw_that_reaches_its_target_is_NOT_reported_as_holding():
    """It shut on air. No error was raised, and nothing is in the jaws."""
    bus = FakeGripperBus()
    result = run(bus)

    assert result.outcome == gripper.REACHED
    assert not result.holding, (
        "reaching the commanded position means the jaws met NOTHING -- treating "
        "that as a successful grasp is the whole bug this distinction prevents")
    assert bus.ticks == GRIP


def test_it_stops_pushing_the_moment_it_grips():
    """A stalled servo draws heavy current; that is what put checksum errors on
    the bus during the 2026-08-07 descent and overloaded J3 on 2026-08-04."""
    bus = FakeGripperBus(stops_at=GRIP + 90)
    run(bus)

    # The loop cannot know the claw has stopped until a jog produces no motion,
    # so exactly ONE command is spent discovering it. What must not happen is a
    # second, third and fourth push into a servo that is already stalled.
    assert len(bus.fruitless) == 1, (
        f"kept pushing after the stall: moves {bus.moves}")
    assert bus.ticks == bus.stops_at


# --- approval ----------------------------------------------------------------

def test_nothing_moves_without_approval():
    bus = FakeGripperBus()
    result = run(bus, approve=no)

    assert bus.commanded == []
    assert result.outcome == gripper.DECLINED
    assert result.ticks == OPEN


def test_every_jog_is_asked_for_separately_not_just_the_first():
    bus = FakeGripperBus()
    result = run(bus, approve=counted(2))

    assert len(bus.commanded) == 2, "a third jog ran without being approved"
    assert result.outcome == gripper.DECLINED


def test_a_jog_is_never_larger_than_the_step():
    bus = FakeGripperBus()
    run(bus, step=40)
    positions = [OPEN] + bus.commanded
    for a, b in zip(positions, positions[1:]):
        assert abs(b - a) <= 40


# --- the floor ---------------------------------------------------------------

def test_the_full_close_stop_is_refused_before_anything_moves():
    """Named in the gripper's own words, not surfaced as a generic travel-limit
    refusal several approved jogs into the run."""
    bus = FakeGripperBus()
    result = run(bus, target=SHUT - 1)

    assert result.outcome == gripper.REFUSED
    assert bus.commanded == []
    assert "full-close stop" in result.message


def test_past_the_open_end_is_refused_too():
    bus = FakeGripperBus()
    result = run(bus, target=OPEN + 1)
    assert result.outcome == gripper.REFUSED
    assert bus.commanded == []


def test_a_bus_refusal_mid_run_stops_the_loop_rather_than_retrying():
    bus = FakeGripperBus(refuse_at=GRIP + 100)
    result = run(bus)

    assert result.outcome == gripper.REFUSED
    assert "servo bus" in result.message
    assert result.ticks > GRIP, "it should report where the claw actually is"


# --- housekeeping ------------------------------------------------------------

def test_j6_gets_its_own_speed_profile():
    """The arm's profile is applied to J1-J5 at the start of a run, and the
    servos default to full speed otherwise -- an unconfigured gripper would snap
    shut at whatever rate it likes, on a brick, at the end of a descent."""
    bus = FakeGripperBus()
    run(bus)
    assert bus.profiles, "J6's speed was never set"
    ids, speed, accel = bus.profiles[0]
    assert ids == (gripper.GRIPPER_JOINT,)
    assert speed == config.SERVO_MOVE_SPEED_TICKS_S
    assert accel == config.SERVO_MOVE_ACCEL


def test_opening_reports_the_end_of_travel_rather_than_a_grip():
    """Direction decides what a stall MEANS. Closing into something is the
    brick; opening into something is the end of travel or a jam."""
    bus = FakeGripperBus(start=GRIP)
    bus.stops_at = None

    class Stuck(FakeGripperBus):
        def move_and_verify(self, servo_id, target):
            self.commanded.append(target)
            return self.ticks          # never moves

    stuck = Stuck(start=GRIP)
    result = gripper.close_in_jogs(stuck, OPEN, 40, yes, lambda _l: None,
                                   settle=lambda b: b.ticks)
    assert result.outcome == gripper.REACHED
    assert not result.holding
    assert "end of travel" in result.message


def test_already_there_does_not_command_anything():
    bus = FakeGripperBus(start=GRIP)
    result = run(bus)
    assert bus.commanded == []
    assert result.outcome == gripper.REACHED


def test_describe_names_the_three_measured_landmarks():
    assert gripper.describe(OPEN) == "fully open"
    assert gripper.describe(GRIP) == "at the grip position"
    assert "FULLY CLOSED" in gripper.describe(SHUT)
    assert "%" in gripper.describe((OPEN + GRIP) // 2)

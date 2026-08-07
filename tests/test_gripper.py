"""The gripper's open/closed constants, against the ticks they were measured as.

config.SERVO_GRIPPER_OPEN_RAD / CLOSE_RAD are what RobotInterface.set_gripper
commands, but they were measured as TICKS -- the operator hand-positioned the
claw and read hold_pose.py. Those are two representations of one measurement,
and the conversion between them runs through data/servo_calibration.json, so a
change to J6's home_tick or ticks_per_rad silently moves both.

WHY THIS FILE EXISTS. Until 2026-08-07 the two constants were wrong AND
inverted: OPEN_RAD = 0.0 commanded J6 to home, which is 9 ticks off the jaws
being shut, and CLOSE_RAD = +0.2 rad opened it slightly from there. Nothing
caught it because nothing had ever driven J6 on hardware -- the gripper is the
one joint the pick path never exercised, so the placeholder survived every
check. The same class of bug as home_tick being 268 ticks out: a constant that
is never executed is never tested by being executed.

No serial port and no arm.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vision_pipeline import config  # noqa: E402
from vision_pipeline.robot_interface.servo_calibration import (  # noqa: E402
    load_calibration_file,
    rad_to_ticks,
)

CAL = load_calibration_file(ROOT / "data" / "servo_calibration.json")
GRIPPER = 6


def test_open_and_close_angles_are_the_measured_ticks():
    """The radian constants must convert to the positions actually measured."""
    assert rad_to_ticks(CAL, GRIPPER, config.SERVO_GRIPPER_OPEN_RAD) == \
        config.SERVO_GRIPPER_OPEN_TICKS
    assert rad_to_ticks(CAL, GRIPPER, config.SERVO_GRIPPER_CLOSE_RAD) == \
        config.SERVO_GRIPPER_GRIP_TICKS


def test_closing_the_claw_moves_it_toward_the_full_close_stop():
    """The direction check the inverted placeholder would have failed.

    Ticks decrease as the claw closes, so 'closed' must sit between 'open' and
    the full-close stop -- not past it, and not on the far side of open.
    """
    assert (config.SERVO_GRIPPER_FULL_CLOSE_TICKS
            < config.SERVO_GRIPPER_GRIP_TICKS
            < config.SERVO_GRIPPER_OPEN_TICKS)


def test_the_grip_position_is_inside_j6s_recorded_travel():
    """Both commanded positions must be reachable through ServoBus."""
    cal6 = CAL[str(GRIPPER)]
    lo, hi = int(cal6["min_tick"]), int(cal6["max_tick"])
    for ticks in (config.SERVO_GRIPPER_OPEN_TICKS, config.SERVO_GRIPPER_GRIP_TICKS):
        assert lo <= ticks <= hi


def test_the_full_close_stop_is_the_recorded_minimum():
    """One number, two homes: the damage limit scripts name and the one the bus
    enforces have to be the same, or a refusal message points at the wrong tick.
    """
    assert int(CAL[str(GRIPPER)]["min_tick"]) == config.SERVO_GRIPPER_FULL_CLOSE_TICKS


def test_a_jog_is_a_fraction_of_the_travel_not_the_whole_of_it():
    """The approval loop is only meaningful if there is more than one approval."""
    travel = config.SERVO_GRIPPER_OPEN_TICKS - config.SERVO_GRIPPER_GRIP_TICKS
    assert 0 < config.SERVO_GRIPPER_JOG_TICKS <= travel / 3

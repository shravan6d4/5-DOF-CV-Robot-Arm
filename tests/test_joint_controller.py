"""Tests for the joint-space control seam used by the arm observation UI.

MockJointController is exercised directly (pure in-memory, no hardware, no
serial port). ServoJointController is exercised against a tiny duck-typed
fake standing in for ServoBus — its wire protocol is already covered exhaustively
by test_servo_driver.py, so here we only need to check that ServoJointController
computes the right target (current + delta) and reports back degrees/ticks
correctly, not re-verify Feetech packet framing.
"""

import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.robot_interface import servo_calibration
from vision_pipeline.robot_interface.joint_controller import (
    GRIPPER_JOINT_ID,
    NUM_JOINTS,
    MockJointController,
    ServoJointController,
)
from vision_pipeline.robot_interface.servo_driver import ServoBus

NO_CAL_FILE = "__nonexistent_servo_cal__.json"  # forces config fallback calibration


# --- MockJointController -------------------------------------------------

def test_mock_seeds_all_joints_at_home():
    controller = MockJointController(calibration_path=NO_CAL_FILE)
    states = controller.read_all()
    assert len(states) == NUM_JOINTS
    for state in states:
        cal = config.SERVO_CALIBRATION_FALLBACK[str(state.joint_id)]
        assert state.ticks == cal["home_tick"]
        assert state.degrees == pytest.approx(0.0, abs=1e-6)


def test_mock_jog_moves_by_delta_ticks():
    controller = MockJointController(calibration_path=NO_CAL_FILE)
    before = controller.read_joint(1).ticks

    after = controller.jog(1, 50)
    assert after.ticks == before + 50

    after2 = controller.jog(1, -20)
    assert after2.ticks == before + 30


def test_mock_jog_clamps_at_tick_bounds():
    controller = MockJointController(calibration_path=NO_CAL_FILE)

    low = controller.jog(1, -100_000)
    assert low.ticks == ServoBus.TICK_MIN

    high = controller.jog(2, 100_000)
    assert high.ticks == ServoBus.TICK_MAX


def test_mock_degrees_ticks_roundtrip_matches_fallback_calibration():
    controller = MockJointController(calibration_path=NO_CAL_FILE)
    # J1 fallback: home 2048, 651.89 ticks/rad, dir +1. Jog by exactly 1 rad worth
    # of ticks and confirm the reported degrees match rad->deg of 1.0.
    one_rad_ticks = round(651.89)
    state = controller.jog(1, one_rad_ticks)
    assert state.degrees == pytest.approx(math.degrees(1.0), abs=0.05)


def test_mock_set_gripper_open_and_close():
    controller = MockJointController(calibration_path=NO_CAL_FILE)
    cal = config.SERVO_CALIBRATION_FALLBACK[str(GRIPPER_JOINT_ID)]

    closed_state = controller.set_gripper(True)
    assert closed_state.joint_id == GRIPPER_JOINT_ID
    expected_closed_ticks = round(
        cal["home_tick"] + cal["dir_sign"] * config.SERVO_GRIPPER_CLOSE_RAD * cal["ticks_per_rad"]
    )
    assert closed_state.ticks == expected_closed_ticks

    open_state = controller.set_gripper(False)
    expected_open_ticks = round(
        cal["home_tick"] + cal["dir_sign"] * config.SERVO_GRIPPER_OPEN_RAD * cal["ticks_per_rad"]
    )
    assert open_state.ticks == expected_open_ticks


def test_mock_read_joint_does_not_move_it():
    controller = MockJointController(calibration_path=NO_CAL_FILE)
    before = controller.read_joint(3)
    again = controller.read_joint(3)
    assert before == again


def test_mock_degrees_per_tick_matches_fallback_calibration():
    controller = MockJointController(calibration_path=NO_CAL_FILE)
    # J1 fallback: 651.89 ticks/rad, dir +1 -> 1 tick = (1/651.89) rad.
    expected = math.degrees(1.0 / 651.89)
    assert controller.degrees_per_tick(1) == pytest.approx(expected, rel=1e-6)


def test_mock_degrees_per_tick_independent_of_current_position():
    controller = MockJointController(calibration_path=NO_CAL_FILE)
    before = controller.degrees_per_tick(2)
    controller.jog(2, 300)
    after = controller.degrees_per_tick(2)
    assert before == pytest.approx(after)


# --- ServoJointController -------------------------------------------------

class _FakeBus:
    """Duck-typed stand-in for ServoBus, driven by explicit tick state.

    Only implements what ServoJointController calls. The real wire protocol
    (packet framing, checksums, read-back) is covered by test_servo_driver.py;
    this fake exists so ServoJointController's own logic (current + delta,
    clamping via move_and_verify, tick<->degree reporting) can be tested without
    re-simulating serial bytes.
    """

    def __init__(self, present_ticks: dict, calibration: dict):
        self._present = dict(present_ticks)
        self._calibration = calibration

    def read_position(self, servo_id: int) -> int:
        return self._present[servo_id]

    def move_and_verify(self, servo_id: int, target_ticks: int) -> int:
        clamped = max(ServoBus.TICK_MIN, min(ServoBus.TICK_MAX, int(target_ticks)))
        self._present[servo_id] = clamped
        return clamped

    def rad_to_ticks(self, servo_id: int, angle_rad: float) -> int:
        return servo_calibration.rad_to_ticks(self._calibration, servo_id, angle_rad)

    def ticks_to_rad(self, servo_id: int, ticks: int) -> float:
        return servo_calibration.ticks_to_rad(self._calibration, servo_id, ticks)


def _fake_bus_controller() -> ServoJointController:
    calibration = servo_calibration.load_calibration(NO_CAL_FILE)
    present = {j: calibration[str(j)]["home_tick"] for j in range(1, NUM_JOINTS + 1)}
    return ServoJointController(_FakeBus(present, calibration))


def test_servo_controller_read_joint():
    controller = _fake_bus_controller()
    state = controller.read_joint(1)
    assert state.ticks == config.SERVO_CALIBRATION_FALLBACK["1"]["home_tick"]
    assert state.degrees == pytest.approx(0.0, abs=1e-6)


def test_servo_controller_jog_adds_delta_to_current_position():
    controller = _fake_bus_controller()
    before = controller.read_joint(2).ticks

    after = controller.jog(2, 75)
    assert after.ticks == before + 75

    after2 = controller.jog(2, -25)
    assert after2.ticks == before + 50


def test_servo_controller_jog_clamps_at_tick_bounds():
    controller = _fake_bus_controller()
    low = controller.jog(1, -1_000_000)
    assert low.ticks == ServoBus.TICK_MIN


def test_servo_controller_set_gripper():
    controller = _fake_bus_controller()
    state = controller.set_gripper(True)
    assert state.joint_id == GRIPPER_JOINT_ID
    cal = config.SERVO_CALIBRATION_FALLBACK[str(GRIPPER_JOINT_ID)]
    expected_ticks = round(
        cal["home_tick"] + cal["dir_sign"] * config.SERVO_GRIPPER_CLOSE_RAD * cal["ticks_per_rad"]
    )
    assert state.ticks == expected_ticks


def test_servo_controller_degrees_per_tick_matches_fallback_calibration():
    controller = _fake_bus_controller()
    expected = math.degrees(1.0 / 651.89)
    assert controller.degrees_per_tick(1) == pytest.approx(expected, rel=1e-6)

"""Joint-space control seam for the arm-observation UI.

RobotInterface (base.py) only speaks Cartesian Pose — there is no way to move a
single joint through it, and SimRobot has no joint model at all. Individual joint
control exists today only at the low-level ServoBus (raw ticks over serial).

JointController is a small abstraction in between: joint-in, joint-out, so a UI
can jog J1..J6 without knowing whether it's talking to real servos or a mock.
Two implementations:

- MockJointController: pure in-memory, no hardware, for developing/testing the
  UI before the bench is wired up.
- ServoJointController: thin wrapper over ServoBus for the real arm.

Both use the shared servo_calibration helpers so tick<->degree conversion is
identical in mock and hardware mode.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from vision_pipeline import config
from vision_pipeline.robot_interface import servo_calibration
from vision_pipeline.robot_interface.servo_driver import ServoBus

NUM_JOINTS = config.NUM_JOINTS
GRIPPER_JOINT_ID = NUM_JOINTS  # J6: highest-numbered joint is the gripper


@dataclass
class JointState:
    """One joint's position, in both raw ticks and degrees relative to home."""

    joint_id: int
    ticks: int
    degrees: float


class JointController(ABC):
    """Abstract per-joint control: read and jog J1..J6 (J6 = gripper)."""

    @abstractmethod
    def read_joint(self, joint_id: int) -> JointState:
        """Read one joint's current position."""

    @abstractmethod
    def jog(self, joint_id: int, delta_ticks: int) -> JointState:
        """Move one joint by delta_ticks (relative), clamped to the servo's tick range.

        Returns the resulting (read-back, for hardware) state.
        """

    @abstractmethod
    def set_gripper(self, closed: bool) -> JointState:
        """Open or close the gripper (J6) using config.SERVO_GRIPPER_*_RAD."""

    @abstractmethod
    def degrees_per_tick(self, joint_id: int) -> float:
        """Rate of change: degrees per raw tick, for previewing a jog step's size.

        A pure function of calibration (ticks_per_rad, dir_sign), independent of
        the joint's current position - used by the UI to show "this many ticks"
        as "this many degrees" without actually moving anything.
        """

    def read_all(self) -> list[JointState]:
        """Read all joints J1..J6, in order."""
        return [self.read_joint(j) for j in range(1, NUM_JOINTS + 1)]


class MockJointController(JointController):
    """In-memory joint controller: no serial port, no hardware required.

    Seeds every joint at its calibrated home_tick and just updates a dict on jog,
    clamped the same way ServoBus.move_and_verify clamps a real command. Lets the
    web UI (and its tests) run with nothing plugged in.
    """

    TICK_MIN = ServoBus.TICK_MIN
    TICK_MAX = ServoBus.TICK_MAX

    def __init__(self, calibration_path: Optional[str] = None):
        self._calibration = servo_calibration.load_calibration(calibration_path)
        self._ticks: dict[int, int] = {
            j: self._calibration[str(j)]["home_tick"] for j in range(1, NUM_JOINTS + 1)
        }

    def _state(self, joint_id: int) -> JointState:
        ticks = self._ticks[joint_id]
        rad = servo_calibration.ticks_to_rad(self._calibration, joint_id, ticks)
        return JointState(joint_id=joint_id, ticks=ticks, degrees=math.degrees(rad))

    def read_joint(self, joint_id: int) -> JointState:
        return self._state(joint_id)

    def jog(self, joint_id: int, delta_ticks: int) -> JointState:
        target = self._ticks[joint_id] + int(delta_ticks)
        self._ticks[joint_id] = max(self.TICK_MIN, min(self.TICK_MAX, target))
        return self._state(joint_id)

    def set_gripper(self, closed: bool) -> JointState:
        angle_rad = config.SERVO_GRIPPER_CLOSE_RAD if closed else config.SERVO_GRIPPER_OPEN_RAD
        ticks = servo_calibration.rad_to_ticks(self._calibration, GRIPPER_JOINT_ID, angle_rad)
        self._ticks[GRIPPER_JOINT_ID] = max(self.TICK_MIN, min(self.TICK_MAX, ticks))
        return self._state(GRIPPER_JOINT_ID)

    def degrees_per_tick(self, joint_id: int) -> float:
        # ticks_to_rad is affine in ticks (dir_sign*(ticks-home)/ticks_per_rad), so
        # a one-tick difference isolates the slope and cancels out home_tick.
        rad_delta = servo_calibration.ticks_to_rad(
            self._calibration, joint_id, 1
        ) - servo_calibration.ticks_to_rad(self._calibration, joint_id, 0)
        return math.degrees(rad_delta)


class ServoJointController(JointController):
    """Joint controller backed by a real ServoBus (physical arm)."""

    def __init__(self, servo_bus: ServoBus):
        self.servo_bus = servo_bus

    def _state(self, joint_id: int, ticks: int) -> JointState:
        rad = self.servo_bus.ticks_to_rad(joint_id, ticks)
        return JointState(joint_id=joint_id, ticks=ticks, degrees=math.degrees(rad))

    def read_joint(self, joint_id: int) -> JointState:
        ticks = self.servo_bus.read_position(joint_id)
        return self._state(joint_id, ticks)

    def jog(self, joint_id: int, delta_ticks: int) -> JointState:
        current_ticks = self.servo_bus.read_position(joint_id)
        target_ticks = current_ticks + int(delta_ticks)
        actual_ticks = self.servo_bus.move_and_verify(joint_id, target_ticks)
        return self._state(joint_id, actual_ticks)

    def set_gripper(self, closed: bool) -> JointState:
        angle_rad = config.SERVO_GRIPPER_CLOSE_RAD if closed else config.SERVO_GRIPPER_OPEN_RAD
        target_ticks = self.servo_bus.rad_to_ticks(GRIPPER_JOINT_ID, angle_rad)
        actual_ticks = self.servo_bus.move_and_verify(GRIPPER_JOINT_ID, target_ticks)
        return self._state(GRIPPER_JOINT_ID, actual_ticks)

    def degrees_per_tick(self, joint_id: int) -> float:
        rad_delta = self.servo_bus.ticks_to_rad(joint_id, 1) - self.servo_bus.ticks_to_rad(joint_id, 0)
        return math.degrees(rad_delta)

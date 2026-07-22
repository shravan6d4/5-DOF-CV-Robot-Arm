"""
Phase 3 placeholder: the seam where this vision pipeline hands off a target
pose to your teammate's robot control / kinematics code.

`RobotInterface` is an abstract base class — it defines *what* any robot
backend must be able to do (send_target_pose, open/close gripper) without
saying *how*. This lets Phase 3 plug in a simulated backend first (e.g. one
that just prints/logs poses, or talks to a simulator), then later swap in a
real backend (ROS topic, serial, or TCP socket to the arm's controller)
without changing any vision code that depends on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    """A target end-effector pose in the robot's base coordinate frame.

    Units: meters for position, degrees for orientation. Adjust if your
    teammate's kinematics code expects different units/conventions —
    this is the contract to align with them on.
    """

    x: float
    y: float
    z: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    def to_matrix(self) -> np.ndarray:
        """Return this pose as a 4x4 homogeneous transform (T_base_gripper).

        This is the bridge the vision code needs: the eye-in-hand calibrator
        consumes the gripper pose as a matrix. Uses the project's RPY convention
        (see calibration.geometry). Imported lazily to keep this module free of
        a hard dependency on the calibration package.
        """
        from vision_pipeline.calibration import geometry

        return geometry.make_transform(
            self.x, self.y, self.z, self.roll_deg, self.pitch_deg, self.yaw_deg
        )


class RobotInterface(ABC):
    """Abstract interface every robot backend (sim or real) must implement.

    The three methods are the complete contract the vision pipeline relies on:
    ask where the gripper is now (needed to turn a pixel into a world point for
    an eye-in-hand camera), command a move, and work the gripper. A real arm
    backend implements these against ROS/serial/TCP; the SimRobot implements
    them in memory for testing.
    """

    @abstractmethod
    def get_end_effector_pose(self) -> Pose:
        """Return the gripper's CURRENT pose in the base frame (forward kinematics).

        The vision pipeline calls this at capture time: because the camera is on
        the arm, the pixel-to-world mapping depends on where the arm was when the
        frame was taken. A real backend computes this from joint encoders.
        """
        raise NotImplementedError

    @abstractmethod
    def send_target_pose(self, pose: Pose) -> None:
        """Command the arm to move its end effector to the given pose."""
        raise NotImplementedError

    @abstractmethod
    def set_gripper(self, closed: bool) -> None:
        """Open (closed=False) or close (closed=True) the gripper."""
        raise NotImplementedError

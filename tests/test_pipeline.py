"""
End-to-end tests for the PickPipeline against the simulated robot.

These prove the seams line up: a synthetic brick frame flows through detection,
pixel-to-world, pick planning, and out to a RobotInterface as a concrete
sequence of poses + gripper actions — with no hardware. This is the regression
guard for "the vision side is merge-ready".
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.pipeline import PickPipeline
from vision_pipeline.robot_interface.base import Pose
from vision_pipeline.robot_interface.sim import SimRobot

# Gripper hovering above the table, camera pointing down (matches the calibration
# tests' geometry). Default placeholder intrinsics/hand-eye are fine here — we're
# testing plumbing and sequencing, not calibrated accuracy.
DOWNWARD_EE_POSE = Pose(x=0.0, y=0.0, z=0.40, roll_deg=180.0)


def _brick_frame() -> np.ndarray:
    """A gray frame with a red rectangle + fake studs: a detectable brick."""
    frame = np.full((480, 640, 3), fill_value=(120, 120, 120), dtype=np.uint8)
    cv2.rectangle(frame, (260, 180), (420, 320), color=(0, 0, 255), thickness=-1)
    for row_y in (220, 280):
        for col_x in (290, 340, 390):
            cv2.circle(frame, (col_x, row_y), 16, color=(0, 0, 200), thickness=-1)
            cv2.circle(frame, (col_x, row_y), 16, color=(0, 0, 120), thickness=2)
    return frame


def _plain_frame() -> np.ndarray:
    return np.full((480, 640, 3), fill_value=(120, 120, 120), dtype=np.uint8)


def test_run_once_commands_a_full_pick_sequence():
    robot = SimRobot(ee_pose=DOWNWARD_EE_POSE)
    pipeline = PickPipeline(robot=robot)

    target = pipeline.run_once(_brick_frame())

    assert target is not None, "A detectable brick should yield a PickTarget"
    # plan_pick_sequence issues 4 moves (hover, descend, close, lift) and 2
    # gripper actions (open at hover, close at grasp).
    assert len(robot.log.poses) == 4
    assert robot.log.gripper_states == [False, True]
    assert robot.gripper_closed is True, "Gripper should end closed, holding the brick"


def test_pick_target_z_is_above_the_table():
    from vision_pipeline import config

    robot = SimRobot(ee_pose=DOWNWARD_EE_POSE)
    target = PickPipeline(robot=robot).run_once(_brick_frame())

    assert target is not None
    # Table is at TABLE_Z_IN_BASE; grasp sits PICK_Z_OFFSET above it.
    assert target.z == pytest.approx(config.TABLE_Z_IN_BASE + config.PICK_Z_OFFSET, abs=1e-9)


def test_hover_poses_are_above_the_grasp():
    robot = SimRobot(ee_pose=DOWNWARD_EE_POSE)
    PickPipeline(robot=robot).run_once(_brick_frame())

    # Sequence is [hover, grasp, grasp, hover]; hover z must exceed grasp z.
    hover_z = robot.log.poses[0].z
    grasp_z = robot.log.poses[1].z
    assert hover_z > grasp_z


def test_locate_brick_returns_none_on_empty_frame():
    pipeline = PickPipeline()  # no robot needed for the vision-only half
    assert pipeline.locate_brick(_plain_frame(), DOWNWARD_EE_POSE) is None


def test_run_once_without_robot_raises():
    pipeline = PickPipeline()  # robot=None
    with pytest.raises(RuntimeError, match="RobotInterface"):
        pipeline.run_once(_brick_frame())

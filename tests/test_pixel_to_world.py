"""
Tests for the eye-in-hand pixel-to-world calibration.

The core test is a full round trip: place a known point on the table, PROJECT it
through the camera model into a pixel, then run that pixel back through
PixelToWorldCalibrator and confirm we recover the original table coordinate.
If the projection and back-projection agree, the whole chain (intrinsics +
transforms + ray/plane intersection) is self-consistent.

Setup: hand-eye = identity (camera at the gripper), and the gripper hovers above
the table pointing straight down (roll=180), so the camera's forward axis (+Z)
points down at the table plane z=0.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.calibration import geometry
from vision_pipeline.calibration.camera_model import CameraIntrinsics
from vision_pipeline.calibration.pixel_to_world import PixelToWorldCalibrator

# A clean, known camera: 550 px focal length, principal point at a 640x480 center.
INTRINSICS = CameraIntrinsics(fx=550.0, fy=550.0, cx=320.0, cy=240.0)

# Gripper (== camera, since hand-eye is identity) hovering 0.40 m above the base
# origin, tool pointing straight down at the table. This is T_base_gripper.
CAMERA_HEIGHT = 0.40
EE_POSE = geometry.make_transform(0.0, 0.0, CAMERA_HEIGHT, roll_deg=180.0)


def _project_world_point_to_pixel(point_base: np.ndarray) -> tuple[float, float]:
    """Project a base-frame point to a pixel, using the same camera pose."""
    t_camera_base = geometry.invert_transform(EE_POSE)  # hand-eye = identity
    point_cam = geometry.transform_point(t_camera_base, point_base)
    return INTRINSICS.project_point(point_cam)


def _calibrator() -> PixelToWorldCalibrator:
    return PixelToWorldCalibrator(
        intrinsics=INTRINSICS,
        t_gripper_camera=np.eye(4),
        table_z=0.0,
    )


@pytest.mark.parametrize(
    "world_point",
    [
        (0.00, 0.00, 0.0),   # directly under the camera -> image center
        (0.05, 0.03, 0.0),
        (-0.04, 0.06, 0.0),
        (0.08, -0.02, 0.0),
    ],
)
def test_pixel_to_world_round_trip(world_point):
    world_point = np.array(world_point, dtype=float)
    pixel = _project_world_point_to_pixel(world_point)

    recovered = _calibrator().pixel_to_world(pixel, EE_POSE)

    assert recovered is not None
    assert recovered == pytest.approx(world_point, abs=1e-6)


def test_point_under_camera_maps_to_image_center():
    # Sanity check on the projection helper itself.
    pixel = _project_world_point_to_pixel(np.array([0.0, 0.0, 0.0]))
    assert pixel == pytest.approx((320.0, 240.0), abs=1e-6)


def test_pixel_to_world_returns_none_when_ray_misses_table():
    # Point the camera UP, away from the table, so no back-projected ray can
    # reach z=0.
    ee_pose_up = geometry.make_transform(0.0, 0.0, CAMERA_HEIGHT, roll_deg=0.0)
    result = _calibrator().pixel_to_world((320.0, 240.0), ee_pose_up)
    assert result is None


def test_yaw_along_image_x_axis_is_zero_in_base():
    # With the camera looking straight down (roll=180), image +x aligns with base
    # +x, so a brick whose axis is along image x (angle 0) yields ~0 deg yaw.
    yaw = _calibrator().pixel_angle_to_world_yaw((320.0, 240.0), angle_deg=0.0, t_base_gripper=EE_POSE)
    assert yaw == pytest.approx(0.0, abs=1.0)


def test_yaw_along_image_y_axis_maps_to_negative_ninety():
    # image +y (down) maps to base -y under the roll=180 flip, so angle 90 -> -90.
    yaw = _calibrator().pixel_angle_to_world_yaw((320.0, 240.0), angle_deg=90.0, t_base_gripper=EE_POSE)
    assert yaw == pytest.approx(-90.0, abs=1.0)

"""
Tests for two-view (moving-camera stereo) triangulation.

Three layers, cheapest first:

1. geometry.triangulate_rays — exact-answer math on hand-built rays (intersecting,
   skew, parallel, over-determined). Locks the closed-form closest-point solver.
2. PixelToWorldCalibrator.triangulate_pixels — the full projection ROUND TRIP: place
   a known base-frame point, PROJECT it into two (and three) known camera poses to
   get pixels, then triangulate those pixels back and confirm recovery. Mirrors
   test_pixel_to_world's round trip, but recovering depth from two views instead of
   the table-plane assumption. Includes a NON-identity hand-eye to prove the
   gripper->camera compose is applied correctly.
3. PickPipeline.locate_brick_two_view — detection -> triangulation -> PickTarget
   wiring and the trust gates (parallax / residual), driven by a stub detector so
   the pixels are controlled exactly.

None of this needs a camera, arm, or MATLAB — it's all synthetic, like the rest of
the geometry/calibration suite.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.calibration import geometry
from vision_pipeline.calibration.camera_model import CameraIntrinsics
from vision_pipeline.calibration.pixel_to_world import PixelToWorldCalibrator
from vision_pipeline.detection.types import Detection
from vision_pipeline.pipeline import PickPipeline
from vision_pipeline.robot_interface.base import Pose

# A clean, known camera (same as test_pixel_to_world).
INTRINSICS = CameraIntrinsics(fx=550.0, fy=550.0, cx=320.0, cy=240.0)

# A brick point on the table, and two camera poses looking down at it from above,
# shifted along base +x so their rays cross at the point with healthy parallax.
# roll=180 => camera +Z (forward) points down at the table (base -Z).
BRICK = np.array([0.40, 0.02, 0.00])
CAM1 = geometry.make_transform(0.40, 0.0, 0.50, roll_deg=180.0)
CAM2 = geometry.make_transform(0.50, 0.0, 0.50, roll_deg=180.0)


def _project(point_base: np.ndarray, t_base_camera: np.ndarray) -> tuple[float, float]:
    """Project a base-frame point into a camera's image, given that camera's pose."""
    t_camera_base = geometry.invert_transform(t_base_camera)
    point_cam = geometry.transform_point(t_camera_base, point_base)
    return INTRINSICS.project_point(point_cam)


def _pose(t_base_gripper: np.ndarray) -> Pose:
    """Wrap a 4x4 gripper transform as a Pose (round-trips via transform_to_pose)."""
    x, y, z, roll, pitch, yaw = geometry.transform_to_pose(t_base_gripper)
    return Pose(x=x, y=y, z=z, roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)


# --------------------------------------------------------------------------
# Layer 1: geometry.triangulate_rays exact-answer cases
# --------------------------------------------------------------------------

def test_triangulate_exact_intersection():
    target = np.array([1.0, 2.0, 3.0])
    o1 = np.array([0.0, 0.0, 0.0])
    o2 = np.array([2.0, 0.0, -1.0])
    result = geometry.triangulate_rays([o1, o2], [target - o1, target - o2])
    assert result is not None
    point, residual = result
    assert point == pytest.approx(target, abs=1e-9)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_triangulate_directions_need_not_be_unit():
    # Same as above but with directions scaled by arbitrary positive factors.
    target = np.array([0.5, -0.3, 1.2])
    o1 = np.array([0.0, 0.0, 0.0])
    o2 = np.array([1.0, 1.0, 0.0])
    result = geometry.triangulate_rays([o1, o2], [(target - o1) * 7.3, (target - o2) * 0.02])
    assert result is not None
    point, residual = result
    assert point == pytest.approx(target, abs=1e-9)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_triangulate_skew_rays_midpoint_and_residual():
    # x-axis at z=0, and the y-direction line through (0,0,2). Closest point is the
    # midpoint (0,0,1) of the common perpendicular; each ray is 1.0 away from it.
    o1 = np.array([0.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0])
    o2 = np.array([0.0, 0.0, 2.0])
    d2 = np.array([0.0, 1.0, 0.0])
    result = geometry.triangulate_rays([o1, o2], [d1, d2])
    assert result is not None
    point, residual = result
    assert point == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert residual == pytest.approx(1.0, abs=1e-9)


def test_triangulate_parallel_rays_returns_none():
    o1 = np.array([0.0, 0.0, 0.0])
    o2 = np.array([0.0, 1.0, 0.0])
    d = np.array([1.0, 0.0, 0.0])
    assert geometry.triangulate_rays([o1, o2], [d, d.copy()]) is None


def test_triangulate_three_views_overdetermined():
    target = np.array([0.30, -0.10, 0.50])
    origins = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.3])]
    directions = [target - o for o in origins]
    result = geometry.triangulate_rays(origins, directions)
    assert result is not None
    point, residual = result
    assert point == pytest.approx(target, abs=1e-9)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_triangulate_raises_on_single_ray():
    with pytest.raises(ValueError):
        geometry.triangulate_rays([np.zeros(3)], [np.array([1.0, 0.0, 0.0])])


def test_triangulate_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        geometry.triangulate_rays(
            [np.zeros(3), np.ones(3)], [np.array([1.0, 0.0, 0.0])]
        )


def test_triangulate_raises_on_zero_direction():
    with pytest.raises(ValueError):
        geometry.triangulate_rays(
            [np.zeros(3), np.ones(3)], [np.array([1.0, 0.0, 0.0]), np.zeros(3)]
        )


# --------------------------------------------------------------------------
# Layer 2: full projection round trip through the calibrator
# --------------------------------------------------------------------------

def _calibrator(t_gripper_camera: np.ndarray) -> PixelToWorldCalibrator:
    return PixelToWorldCalibrator(
        intrinsics=INTRINSICS,
        t_gripper_camera=t_gripper_camera,
        table_z=0.0,
    )


@pytest.mark.parametrize(
    "brick",
    [
        (0.40, 0.00, 0.00),   # straight under cam1
        (0.40, 0.02, 0.00),
        (0.45, -0.03, 0.01),  # slightly off the table plane -> only two-view can see this
        (0.38, 0.05, -0.02),
    ],
)
def test_triangulate_pixels_round_trip_identity_handeye(brick):
    brick = np.array(brick, dtype=float)
    # Hand-eye identity: gripper pose == camera pose.
    px1 = _project(brick, CAM1)
    px2 = _project(brick, CAM2)

    result = _calibrator(np.eye(4)).triangulate_pixels([(px1, CAM1), (px2, CAM2)])

    assert result is not None
    assert result.point_base == pytest.approx(brick, abs=1e-6)
    assert result.residual_m == pytest.approx(0.0, abs=1e-9)
    assert result.parallax_deg > config.TWO_VIEW_MIN_PARALLAX_DEG
    assert result.num_views == 2


def test_triangulate_pixels_round_trip_nonidentity_handeye():
    # A real off-axis camera mount: offset + a 90 deg yaw between gripper and camera.
    t_gripper_camera = geometry.make_transform(0.03, -0.01, 0.02, yaw_deg=90.0)
    # Derive the gripper poses that put the cameras exactly at CAM1/CAM2.
    gp1 = CAM1 @ geometry.invert_transform(t_gripper_camera)
    gp2 = CAM2 @ geometry.invert_transform(t_gripper_camera)

    px1 = _project(BRICK, CAM1)
    px2 = _project(BRICK, CAM2)

    result = _calibrator(t_gripper_camera).triangulate_pixels([(px1, gp1), (px2, gp2)])

    assert result is not None
    assert result.point_base == pytest.approx(BRICK, abs=1e-6)
    assert result.residual_m == pytest.approx(0.0, abs=1e-9)


def test_triangulate_pixels_three_views_round_trip():
    cam3 = geometry.make_transform(0.45, 0.08, 0.50, roll_deg=180.0)
    views = [
        (_project(BRICK, CAM1), CAM1),
        (_project(BRICK, CAM2), CAM2),
        (_project(BRICK, cam3), cam3),
    ]
    result = _calibrator(np.eye(4)).triangulate_pixels(views)
    assert result is not None
    assert result.point_base == pytest.approx(BRICK, abs=1e-6)
    assert result.num_views == 3


def test_triangulate_pixels_inconsistent_views_have_large_residual():
    # Two views that saw DIFFERENT physical points -> rays skew -> big residual.
    other = BRICK + np.array([0.05, 0.05, 0.0])
    px1 = _project(BRICK, CAM1)
    px2 = _project(other, CAM2)
    result = _calibrator(np.eye(4)).triangulate_pixels([(px1, CAM1), (px2, CAM2)])
    assert result is not None
    assert result.residual_m > config.TWO_VIEW_MAX_RESIDUAL_M


def test_triangulate_pixels_raises_on_single_view():
    with pytest.raises(ValueError):
        _calibrator(np.eye(4)).triangulate_pixels([((320.0, 240.0), CAM1)])


# --------------------------------------------------------------------------
# Layer 3: pipeline wiring + trust gates
# --------------------------------------------------------------------------

class _StubDetector:
    """Returns one Detection per detect() call, with pre-set centroids (or []).

    Lets pipeline tests control the exact pixel each view yields, decoupled from
    real color/stud detection (covered by its own tests).
    """

    def __init__(self, centroids):
        self._centroids = list(centroids)
        self._i = 0

    def detect(self, frame_bgr):
        centroid = self._centroids[self._i]
        self._i += 1
        if centroid is None:
            return []
        return [
            Detection(
                centroid_px=centroid,
                area=1000.0,
                bbox=(0, 0, 10, 10),
                angle_deg=0.0,
                contour=np.zeros((1, 1, 2), dtype=np.int32),
                num_studs=4,
            )
        ]


_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)  # content ignored by the stub


def _two_view_pipeline(centroids):
    return PickPipeline(
        detector=_StubDetector(centroids),
        calibrator=_calibrator(np.eye(4)),
        robot=None,
    )


def test_locate_brick_two_view_recovers_position():
    px1 = _project(BRICK, CAM1)
    px2 = _project(BRICK, CAM2)
    pipeline = _two_view_pipeline([px1, px2])

    target = pipeline.locate_brick_two_view([(_FRAME, _pose(CAM1)), (_FRAME, _pose(CAM2))])

    assert target is not None
    assert (target.x, target.y) == pytest.approx((BRICK[0], BRICK[1]), abs=1e-6)
    # z gets the same grasp offset above the detected point as the single-view path.
    assert target.z == pytest.approx(BRICK[2] + config.PICK_Z_OFFSET, abs=1e-6)
    assert target.yaw_deg == 0.0
    assert target.num_studs == 4


def test_locate_brick_two_view_none_when_brick_missing_in_one_view():
    px1 = _project(BRICK, CAM1)
    pipeline = _two_view_pipeline([px1, None])  # brick absent from second frame
    target = pipeline.locate_brick_two_view([(_FRAME, _pose(CAM1)), (_FRAME, _pose(CAM2))])
    assert target is None


def test_locate_brick_two_view_none_on_low_parallax():
    # Cameras almost coincident -> baseline ~4 mm -> parallax well under the gate.
    cam_near = geometry.make_transform(0.404, 0.0, 0.50, roll_deg=180.0)
    px1 = _project(BRICK, CAM1)
    px2 = _project(BRICK, cam_near)
    pipeline = _two_view_pipeline([px1, px2])
    target = pipeline.locate_brick_two_view([(_FRAME, _pose(CAM1)), (_FRAME, _pose(cam_near))])
    assert target is None


def test_locate_brick_two_view_none_on_high_residual():
    # Views that saw different points -> rays don't meet -> residual gate trips.
    other = BRICK + np.array([0.05, 0.05, 0.0])
    px1 = _project(BRICK, CAM1)
    px2 = _project(other, CAM2)
    pipeline = _two_view_pipeline([px1, px2])
    target = pipeline.locate_brick_two_view([(_FRAME, _pose(CAM1)), (_FRAME, _pose(CAM2))])
    assert target is None


def test_locate_brick_two_view_raises_on_single_capture():
    pipeline = _two_view_pipeline([(320.0, 240.0)])
    with pytest.raises(ValueError):
        pipeline.locate_brick_two_view([(_FRAME, _pose(CAM1))])

"""
Tests for calibration/hand_eye.py — the shared hand-eye sampling + solve module.

All synthetic (no camera, no arm), mirroring test_triangulation.py's style:

1. HandEyeAccumulator / solve_board — plant a KNOWN T_gripper_camera and a
   stationary board, generate the (T_base_gripper, T_cam_board) pairs a real
   calibration session would produce, and confirm cv2.calibrateHandEye recovers
   the planted transform. This is the test that would have caught the
   Pose-round-trip gimbal-lock bug fixed in calibrate_hand_eye.py: samples are
   fed as raw FK 4x4s, exactly as callers (the web UI, the fixed terminal
   script) must do.
2. select_best / cross_board_agreement_mm — the multi-board bookkeeping.
3. detect_board_poses — one lightweight round trip against a real rendered
   ChArUco board image, proving the detect -> matchImagePoints -> solvePnP
   wiring returns a well-formed transform.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.calibration import charuco, geometry
from vision_pipeline.calibration.camera_model import CameraIntrinsics
from vision_pipeline.calibration.hand_eye import (
    HandEyeAccumulator,
    cross_board_agreement_mm,
    detect_board_poses,
    select_best,
)

# A stationary board sitting flat on the table at a known base-frame pose.
T_BASE_BOARD = geometry.make_transform(0.20, 0.05, 0.0)

# Gripper poses with varied rotation (roll near 180 = tool pointing down, plus
# pitch/yaw wobble) as well as translation — calibrateHandEye needs rotation
# diversity about non-parallel axes to be well-conditioned, which is exactly
# what plain Cartesian-only sampling (the bug this module replaces) would NOT
# have given.
_GRIPPER_POSES_RPY_XYZ = [
    (175, -5, -30, 0.15, -0.05, 0.10),
    (180, 0, -20, 0.18, -0.03, 0.12),
    (185, 5, -10, 0.20, 0.00, 0.14),
    (180, 10, 0, 0.22, 0.02, 0.16),
    (170, -10, 10, 0.19, 0.04, 0.13),
    (190, 0, 20, 0.17, -0.02, 0.11),
    (180, 3, 30, 0.21, 0.01, 0.15),
    (178, -3, -25, 0.16, -0.04, 0.10),
    (182, 8, 15, 0.23, 0.03, 0.17),
    (180, -8, 5, 0.19, -0.01, 0.12),
    (175, 0, -15, 0.20, 0.05, 0.14),
    (185, 4, 25, 0.18, -0.05, 0.16),
]


def _synthetic_samples(t_gripper_camera: np.ndarray):
    """Yield (t_base_gripper, t_cam_board) pairs a real session would record."""
    for roll, pitch, yaw, x, y, z in _GRIPPER_POSES_RPY_XYZ:
        t_base_gripper = geometry.make_transform(x, y, z, roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)
        t_base_camera = t_base_gripper @ t_gripper_camera
        t_cam_board = geometry.invert_transform(t_base_camera) @ T_BASE_BOARD
        yield t_base_gripper, t_cam_board


# --------------------------------------------------------------------------
# HandEyeAccumulator / solve_board
# --------------------------------------------------------------------------

def test_solve_board_recovers_known_hand_eye_transform():
    t_gripper_camera_true = geometry.make_transform(0.02, -0.01, 0.03, roll_deg=0.0, pitch_deg=0.0, yaw_deg=15.0)

    acc = HandEyeAccumulator(min_samples=10)
    for t_base_gripper, t_cam_board in _synthetic_samples(t_gripper_camera_true):
        acc.add({0: t_cam_board}, t_base_gripper)

    assert acc.counts() == {0: len(_GRIPPER_POSES_RPY_XYZ)}
    assert acc.solvable_boards() == [0]

    result = acc.solve_board(0)

    assert result.board_index == 0
    assert result.n_samples == len(_GRIPPER_POSES_RPY_XYZ)
    assert result.t_gripper_camera_tsai[:3, 3] == pytest.approx(t_gripper_camera_true[:3, 3], abs=1e-6)
    assert result.t_gripper_camera_tsai[:3, :3] == pytest.approx(t_gripper_camera_true[:3, :3], abs=1e-6)
    # Exact synthetic data -> TSAI and PARK should agree almost perfectly.
    assert result.tsai_park_disagreement_mm < 0.1
    # The board never moved -> every sample should imply the same base position.
    assert result.board_spread_mm < 0.1
    assert result.mean_board_origin_base == pytest.approx(T_BASE_BOARD[:3, 3], abs=1e-6)


def test_solve_board_raises_below_min_samples():
    t_gripper_camera_true = np.eye(4)
    acc = HandEyeAccumulator(min_samples=10)
    for i, (t_base_gripper, t_cam_board) in enumerate(_synthetic_samples(t_gripper_camera_true)):
        if i >= 5:
            break
        acc.add({0: t_cam_board}, t_base_gripper)

    assert acc.solvable_boards() == []
    with pytest.raises(ValueError):
        acc.solve_board(0)


def test_add_buckets_samples_by_board_independently():
    acc = HandEyeAccumulator(min_samples=3)
    t_base_gripper = geometry.make_transform(0.2, 0.0, 0.1, roll_deg=180.0)
    board_poses = {0: np.eye(4), 2: np.eye(4)}

    counts = acc.add(board_poses, t_base_gripper)

    assert counts == {0: 1, 2: 1}
    counts = acc.add({0: np.eye(4)}, t_base_gripper)
    assert counts == {0: 2, 2: 1}
    assert acc.solvable_boards() == []  # neither has reached min_samples=3 yet


# --------------------------------------------------------------------------
# select_best / cross_board_agreement_mm
# --------------------------------------------------------------------------

def test_select_best_prefers_most_samples():
    t_gripper_camera_true = geometry.make_transform(0.01, 0.0, 0.02, yaw_deg=5.0)
    acc = HandEyeAccumulator(min_samples=8)
    samples = list(_synthetic_samples(t_gripper_camera_true))
    for t_base_gripper, t_cam_board in samples:
        acc.add({0: t_cam_board}, t_base_gripper)
    for t_base_gripper, t_cam_board in samples[:8]:  # fewer samples -> board 1
        acc.add({1: t_cam_board}, t_base_gripper)

    results = acc.solve_all()
    assert set(results) == {0, 1}
    best = select_best(results)
    assert best.board_index == 0  # the board with more samples


def test_select_best_raises_on_empty():
    with pytest.raises(ValueError):
        select_best({})


def test_cross_board_agreement_mm_none_with_single_board():
    t_gripper_camera_true = np.eye(4)
    acc = HandEyeAccumulator(min_samples=10)
    for t_base_gripper, t_cam_board in _synthetic_samples(t_gripper_camera_true):
        acc.add({0: t_cam_board}, t_base_gripper)

    results = acc.solve_all()
    assert cross_board_agreement_mm(results) is None


def test_cross_board_agreement_mm_near_zero_when_boards_agree():
    t_gripper_camera_true = geometry.make_transform(0.015, 0.0, 0.025, yaw_deg=-10.0)
    acc = HandEyeAccumulator(min_samples=10)
    for t_base_gripper, t_cam_board in _synthetic_samples(t_gripper_camera_true):
        acc.add({0: t_cam_board}, t_base_gripper)
        acc.add({1: t_cam_board}, t_base_gripper)  # second board, identical geometry

    results = acc.solve_all()
    assert set(results) == {0, 1}
    assert cross_board_agreement_mm(results) < 0.1


# --------------------------------------------------------------------------
# detect_board_poses — real ChArUco detection round trip
# --------------------------------------------------------------------------

def test_detect_board_poses_recovers_a_well_formed_transform():
    board = charuco.build_board(0)
    width_px, height_px = 900, 540
    img = board.generateImage((width_px, height_px), marginSize=0, borderBits=1)

    intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=width_px / 2.0, cy=height_px / 2.0)
    detectors = charuco.build_detectors(1)  # board 0 only, matches what was rendered

    result = detect_board_poses(img, intrinsics, detectors)

    assert 0 in result
    T = result[0]
    assert T.shape == (4, 4)
    # Rotation block must be a valid rotation matrix (orthonormal, det +1).
    R = T[:3, :3]
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-4)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-4)
    # The board must land in FRONT of the camera (positive depth), not behind it.
    assert T[2, 3] > 0


def test_detect_board_poses_empty_when_board_absent():
    blank = np.full((480, 640), 255, dtype=np.uint8)  # no markers anywhere
    intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    detectors = charuco.build_detectors(1)

    result = detect_board_poses(blank, intrinsics, detectors)

    assert result == {}

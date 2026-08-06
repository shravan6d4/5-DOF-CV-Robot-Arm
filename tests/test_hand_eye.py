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

import cv2

from vision_pipeline import config
from vision_pipeline.calibration import charuco, geometry, hand_eye
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

def test_select_best_prefers_most_samples_when_it_cannot_judge_consistency():
    """Without an accumulator there is nothing to judge but count."""
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


# --- judging the CAPTURE, not the solve -------------------------------------
#
# Added 2026-08-05 after every existing check passed on a transform that was
# wrong. Five independent solvers (TSAI, PARK, HORAUD, DANIILIDIS, and the
# separate AX=ZB robot-world formulation) agreed on a camera offset of 80 mm
# while a ruler said 24 mm. Solvers fed the same badly-conditioned data agree
# with each other and are wrong together; only something OUTSIDE the data can
# catch that.
#
# Root cause was the capture geometry: a median 15.5 deg of rotation between
# poses, against the >= 30 deg (60 better) the hand-eye literature calls for.
# Camera translation is recovered from how far the camera SWINGS, so small
# rotations leave it barely determined while every residual still looks fine.

def _acc_with_rotations(deg_per_step, n=10, axis=(0.0, 0.0, 1.0)):
    """An accumulator whose consecutive gripper poses differ by a fixed rotation."""
    acc = hand_eye.HandEyeAccumulator(min_samples=3)
    axis = np.array(axis, dtype=float)
    for i in range(n):
        rvec = axis * np.radians(deg_per_step * i)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [0.2, 0.0, 0.1]
        acc._samples.setdefault(0, []).append(
            hand_eye.HandEyeSample(T_base_gripper=T, T_cam_board=np.eye(4)))
    return acc


def test_small_rotations_are_flagged_as_a_capture_problem():
    """The 2026-08-04 session in one assertion: 15 deg between poses."""
    warnings = hand_eye.capture_health(_acc_with_rotations(15.0), 0)
    assert any("median rotation" in w for w in warnings)
    assert any("CAMERA OFFSET" in w for w in warnings)


def test_generous_rotations_pass():
    assert not any("median rotation" in w
                   for w in hand_eye.capture_health(_acc_with_rotations(60.0), 0))


def test_too_few_poses_is_flagged():
    assert any("only 4 poses" in w
               for w in hand_eye.capture_health(_acc_with_rotations(60.0, n=4), 0))


def test_rotation_magnitudes_measure_consecutive_motion():
    rots = hand_eye.rotation_magnitudes_deg(_acc_with_rotations(20.0, n=5), 0)
    assert len(rots) == 4
    assert all(r == pytest.approx(20.0, abs=1e-6) for r in rots)


def test_coaxial_rotations_are_flagged_as_unobservable():
    """Every motion about one axis leaves camera translation along it free."""
    assert any("axes span" in w
               for w in hand_eye.capture_health(_acc_with_rotations(60.0), 0))


# --- the ruler, the only check outside the data -----------------------------

def _result_with_offset(mm):
    T = np.eye(4)
    T[:3, 3] = [mm / 1000.0, 0.0, 0.0]
    return hand_eye.BoardResult(
        board_index=0, n_samples=10, t_gripper_camera_tsai=T,
        t_gripper_camera_park=T, tsai_park_disagreement_mm=0.0,
        tsai_park_rotation_deg=0.0, board_spread_mm=1.0,
        mean_board_origin_base=np.zeros(3))


def test_a_solve_that_contradicts_the_ruler_is_rejected():
    """80 mm solved against 24 mm measured — the exact 2026-08-05 case."""
    msg = hand_eye.offset_disagrees_with_ruler(_result_with_offset(80.0))
    assert msg is not None and "80" in msg and "24" in msg


def test_a_solve_matching_the_ruler_passes():
    assert hand_eye.offset_disagrees_with_ruler(
        _result_with_offset(config.HAND_EYE_EXPECTED_OFFSET_MM)) is None


def test_the_ruler_tolerance_is_generous_enough_for_frame_ambiguity():
    """The model's wrist-body origin need not sit where a ruler naturally goes,
    so the gate must catch a 56 mm error without tripping on a 20 mm one."""
    assert hand_eye.offset_disagrees_with_ruler(
        _result_with_offset(config.HAND_EYE_EXPECTED_OFFSET_MM + 20.0)) is None
    assert hand_eye.offset_disagrees_with_ruler(
        _result_with_offset(config.HAND_EYE_EXPECTED_OFFSET_MM + 56.0)) is not None


def test_offset_mm_reports_the_solved_camera_distance():
    assert _result_with_offset(42.0).offset_mm == pytest.approx(42.0)


def test_solve_survives_a_solver_that_refuses_to_converge():
    """ANDREFF raises cv2.error on ill-posed data instead of guessing, which is
    the behaviour we WANT -- but the handler for it referenced an undefined
    `logger` and turned a useful warning into a NameError that killed a capture
    session at the compute step, after 70 samples. The samples survived only
    because they are written after every record.

    Degenerate-but-solvable input: enough rotation for TSAI to return something,
    all of it about one axis so at least one solver gives up.
    """
    acc = hand_eye.HandEyeAccumulator(min_samples=3)
    rng = np.random.default_rng(0)
    for i in range(8):
        R, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(25.0 * i)]))
        Tg = np.eye(4)
        Tg[:3, :3] = R
        Tg[:3, 3] = [0.2, 0.0, 0.1]
        Tc = np.eye(4)
        Tc[:3, :3] = R.T
        Tc[:3, 3] = rng.normal(0, 1e-4, 3) + [0.0, 0.0, 0.25]
        acc._samples.setdefault(0, []).append(
            hand_eye.HandEyeSample(T_base_gripper=Tg, T_cam_board=Tc))

    result = acc.solve_board(0)          # must not raise
    assert "TSAI" in result.solutions
    assert result.n_samples == 8


def test_solutions_are_reported_per_method():
    """The point of running five solvers is being able to compare them."""
    acc = hand_eye.HandEyeAccumulator(min_samples=3)
    for i in range(8):
        rvec = np.array([np.radians(20.0 * i), np.radians(15.0 * i), 0.0])
        R, _ = cv2.Rodrigues(rvec)
        Tg = np.eye(4)
        Tg[:3, :3] = R
        Tg[:3, 3] = [0.2 + 0.01 * i, 0.01 * i, 0.1]
        Tc = np.eye(4)
        Tc[:3, :3] = R.T
        Tc[:3, 3] = [0.0, 0.0, 0.25]
        acc._samples.setdefault(0, []).append(
            hand_eye.HandEyeSample(T_base_gripper=Tg, T_cam_board=Tc))
    r = acc.solve_board(0)
    assert len(r.solutions) >= 2
    assert r.method_spread_mm >= 0.0


# --- capture conditioning ----------------------------------------------------
# A capture can pin the camera's ORIENTATION to a fraction of a degree and leave
# its POSITION free along a whole axis. Rotating about an axis n satisfies
# (R - I) n = 0, so repeating the same rotation axis hides the offset along it
# however many samples are taken. Measured on real data 2026-08-06: 14 poses,
# median rotation 35 deg, axis spread 82 deg -- every existing check passed,
# while nine of thirteen pose changes shared one axis to within 1 degree.

def _gripper_sequence(axes, angle_deg=35.0):
    """Samples whose consecutive motions rotate about the given axes in turn."""
    samples = []
    pose = np.eye(4)
    for i, axis in enumerate(axes):
        samples.append(
            hand_eye.HandEyeSample(T_base_gripper=pose.copy(),
                                   T_cam_board=np.eye(4)))
        step = np.eye(4)
        step[:3, :3] = geometry.rotation_matrix(
            *[(angle_deg if np.allclose(axis, u) else 0.0)
              for u in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                        np.array([0, 0, 1.0]))][::-1])
        # Alternate sign so the arm does not simply wind up in one direction.
        if i % 2:
            step[:3, :3] = step[:3, :3].T
        pose = pose @ step
    return samples


def _accumulator(samples):
    acc = hand_eye.HandEyeAccumulator(min_samples=3)
    acc._samples[0] = samples
    return acc


def test_conditioning_flags_rotations_that_all_share_one_axis():
    """The real failure: every pose change is a J5 wrist roll."""
    z = np.array([0.0, 0.0, 1.0])
    acc = _accumulator(_gripper_sequence([z] * 10))
    cond = hand_eye.translation_conditioning(acc, 0)
    assert cond > config.CALIB_HAND_EYE_MAX_CONDITION
    assert any("CLUSTERED" in w for w in hand_eye.capture_health(acc, 0))


def test_conditioning_accepts_rotations_spread_over_three_axes():
    x, y, z = np.eye(3)
    acc = _accumulator(_gripper_sequence([x, y, z] * 4))
    cond = hand_eye.translation_conditioning(acc, 0)
    assert cond < config.CALIB_HAND_EYE_MAX_CONDITION
    assert not any("CLUSTERED" in w for w in hand_eye.capture_health(acc, 0))


def test_axis_spread_alone_does_not_catch_a_clustered_capture():
    """Why a new metric was needed rather than tightening the old one.

    Spread reads the WIDEST gap between any two axes, so a few unusual poses
    make a clustered set look varied. This mirrors the measured capture: mostly
    one axis, a couple of outliers, spread reported as healthy.
    """
    x, z = np.array([1.0, 0, 0]), np.array([0.0, 0, 1.0])
    acc = _accumulator(_gripper_sequence([z] * 9 + [x] + [z] * 3))
    assert hand_eye.rotation_axis_spread_deg(acc, 0) > 45.0     # looks fine...
    assert hand_eye.translation_conditioning(acc, 0) > \
        config.CALIB_HAND_EYE_MAX_CONDITION                     # ...but is not


def test_conditioning_returns_none_when_there_is_too_little_data():
    assert hand_eye.translation_conditioning(_accumulator([]), 0) is None


# --- screw congruence (Chen 1991) --------------------------------------------
# AX = XB makes A and B conjugate, so they share BOTH screw invariants: the
# rotation angle and the pitch (translation along that same axis). Neither
# depends on X, so both are testable before any solve -- the only checks here
# a solver cannot flatter by agreeing with its own assumptions.

def _screw(axis, angle_deg, pitch_m, point=(0.0, 0.0, 0.0)):
    """A rigid motion with exactly the given screw parameters."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    a = np.deg2rad(angle_deg)
    r = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)
    p = np.asarray(point, dtype=float)
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = p - r @ p + pitch_m * axis
    return t


def _pair_accumulator(motions_a, motions_b):
    """Samples whose consecutive motions are exactly the given A's and B's."""
    g, c = [np.eye(4)], [np.eye(4)]
    for a, b in zip(motions_a, motions_b):
        g.append(g[-1] @ a)
        # B is defined as T_cam_board(i) @ inv(T_cam_board(i+1)), so invert.
        c.append(geometry.invert_transform(b) @ c[-1])
    acc = hand_eye.HandEyeAccumulator(min_samples=3)
    acc._samples[0] = [hand_eye.HandEyeSample(T_base_gripper=gi, T_cam_board=ci)
                       for gi, ci in zip(g, c)]
    return acc



def _stationary_board_set(x, gripper_poses, t_base_board=None):
    """Samples generated the way reality does: one FIXED board, a camera rigidly
    offset from the gripper by x, and T_cam_board computed from the geometry.

    Chaining relative MOTIONS instead (see _pair_accumulator) cannot express a
    single bad sample: every pose after the corrupted motion inherits the error,
    which is a drifting chain, not one misdetected frame.
    """
    if t_base_board is None:
        t_base_board = geometry.make_transform(0.25, 0.05, -0.07, yaw_deg=12.0)
    acc = hand_eye.HandEyeAccumulator(min_samples=8)
    acc._samples[0] = [
        hand_eye.HandEyeSample(
            T_base_gripper=g,
            T_cam_board=geometry.invert_transform(g @ x) @ t_base_board)
        for g in gripper_poses
    ]
    return acc


def _gripper_arc(n):
    """n gripper poses rotating about cycling axes, with real translation."""
    axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    out = []
    for i in range(n):
        t = geometry.make_transform(
            0.10 + 0.01 * (i % 4), 0.02 * ((i % 3) - 1), 0.15 + 0.008 * (i % 5))
        r = _screw(axes[i % 3], 40.0 + 7.0 * i, 0.0)
        t[:3, :3] = r[:3, :3]
        out.append(t)
    return out


def test_screw_pitch_recovers_translation_along_the_axis():
    t = _screw([0.0, 0.0, 1.0], 40.0, pitch_m=0.025)
    axis, _, _ = geometry.screw_axis(t)
    assert geometry.screw_pitch(t, axis) == pytest.approx(0.025, abs=1e-9)


def test_screw_pitch_ignores_translation_across_the_axis():
    """Sliding the axis sideways is not pitch -- it is where the axis IS."""
    t = _screw([0.0, 0.0, 1.0], 40.0, pitch_m=0.0, point=(0.3, -0.2, 0.0))
    axis, _, _ = geometry.screw_axis(t)
    assert geometry.screw_pitch(t, axis) == pytest.approx(0.0, abs=1e-9)


def test_congruent_motions_report_no_error():
    """A and B conjugate by some X: both invariants must match exactly."""
    x = _screw([0.3, -0.5, 1.0], 25.0, pitch_m=0.02, point=(0.01, 0.02, 0.03))
    a_list = [_screw([0, 0, 1.0], 35.0, 0.004), _screw([1.0, 0, 0], 40.0, -0.003)]
    b_list = [geometry.invert_transform(x) @ a @ x for a in a_list]
    for p in hand_eye.screw_congruence(_pair_accumulator(a_list, b_list), 0):
        assert p.angle_error_deg == pytest.approx(0.0, abs=1e-8)
        assert p.pitch_error_mm == pytest.approx(0.0, abs=1e-6)


def test_pitch_catches_a_fault_the_angle_check_is_blind_to():
    """The whole reason pitch was added.

    A motion that turns the RIGHT amount about the RIGHT axis but slides the
    wrong way along it satisfies the angle test perfectly. Only pitch sees it.
    """
    a_list = [_screw([0, 0, 1.0], 35.0, pitch_m=0.004)]
    b_list = [_screw([0, 0, 1.0], 35.0, pitch_m=-0.040)]   # 44 mm of pitch error
    p = hand_eye.screw_congruence(_pair_accumulator(a_list, b_list), 0)[0]
    assert p.angle_error_deg == pytest.approx(0.0, abs=1e-8)   # angle: clean
    assert p.pitch_error_mm > config.CALIB_HAND_EYE_CONGRUENCE_PITCH_MM


def test_one_bad_sample_is_named_once_not_its_neighbours():
    """Attribution by consensus over ALL pairs.

    A bad sample corrupts both CONSECUTIVE pairs touching it, so neighbour-only
    logic implicates three samples for one fault and cannot say which. Scored
    against every other sample, the culprit disagrees with nearly all of them
    while its neighbours disagree only with it.
    """
    x = _screw([0.2, -0.3, 1.0], 20.0, 0.015)
    acc = _stationary_board_set(x, _gripper_arc(9))
    # One misdetected frame: this sample's board pose alone is wrong.
    acc._samples[0][4].T_cam_board = (
        acc._samples[0][4].T_cam_board @ _screw([0, 1.0, 0], 25.0, 0.06))

    scores = hand_eye.congruence_disagreement(acc, 0)
    assert scores[4] > config.CALIB_HAND_EYE_MAX_DISAGREEMENT
    assert scores[3] < config.CALIB_HAND_EYE_MAX_DISAGREEMENT
    assert scores[5] < config.CALIB_HAND_EYE_MAX_DISAGREEMENT
    assert hand_eye.congruence_outliers(acc, 0) == [4]


def test_a_clean_set_has_no_congruence_outliers():
    x = _screw([0.3, -0.5, 1.0], 25.0, 0.02)
    a_list = [_screw([0, 0, 1.0], 35.0, 0.004), _screw([1.0, 0, 0], 40.0, -0.003),
              _screw([0, 1.0, 0], 38.0, 0.002)]
    b_list = [geometry.invert_transform(x) @ a @ x for a in a_list]
    assert hand_eye.congruence_outliers(_pair_accumulator(a_list, b_list), 0) == []


# --- observability indices ---------------------------------------------------

def test_o3_catches_small_rotations_that_the_condition_number_calls_perfect():
    """Why O3 and not the condition number alone.

    Tiny rotations spread evenly over three perpendicular axes are perfectly
    CONDITIONED -- every direction equally weak -- and determine nothing. A
    scale-invariant index cannot see that; O3 can.
    """
    x, y, z = np.eye(3)
    tiny = [_screw(ax, 2.0, 0.001) for ax in (x, y, z)] * 4
    acc = _pair_accumulator(tiny, tiny)
    obs = hand_eye.observability(acc, 0)
    assert obs.condition_number < config.CALIB_HAND_EYE_MAX_CONDITION   # "perfect"
    assert obs.o3_min_singular < config.CALIB_HAND_EYE_MIN_O3           # ...but blind
    assert any("O3" in w for w in hand_eye.capture_health(acc, 0))


def test_a_good_capture_passes_every_observability_gate():
    x, y, z = np.eye(3)
    big = [_screw(ax, 45.0, 0.004) for ax in (x, y, z)] * 4
    acc = _pair_accumulator(big, big)
    obs = hand_eye.observability(acc, 0)
    assert obs.condition_number < config.CALIB_HAND_EYE_MAX_CONDITION
    assert obs.o3_min_singular > config.CALIB_HAND_EYE_MIN_O3
    assert obs.o1_product > 0 and obs.o4_noise_amplification > 0


def test_translation_conditioning_still_agrees_with_the_observability_object():
    x, z = np.array([1.0, 0, 0]), np.array([0.0, 0, 1.0])
    acc = _pair_accumulator([_screw(z, 35.0, 0.004)] * 9 + [_screw(x, 35.0, 0.004)],
                            [_screw(z, 35.0, 0.004)] * 9 + [_screw(x, 35.0, 0.004)])
    assert hand_eye.translation_conditioning(acc, 0) == pytest.approx(
        hand_eye.observability(acc, 0).condition_number)


# --- per-board quality, and which board to believe ---------------------------
# Sample count decided this until 2026-08-06, when a real capture put 22
# mutually contradictory samples on one board and 9 consistent ones on another.
# Counting picks the 22 and no rigid transform fits them.

def test_congruence_quality_separates_the_arm_from_the_camera():
    """Angle depends only on rotations, pitch also on translations. A fault that
    moves pitch while leaving angle clean is on the camera side."""
    x = _screw([0.3, -0.5, 1.0], 25.0, 0.02)
    motions = [_screw([0, 0, 1.0], 35.0, 0.004), _screw([1.0, 0, 0], 40.0, -0.003),
               _screw([0, 1.0, 0], 38.0, 0.002), _screw([1.0, 1.0, 0], 36.0, 0.005)]
    clean = _pair_accumulator(motions, [geometry.invert_transform(x) @ m @ x
                                        for m in motions])
    q = hand_eye.congruence_quality(clean, 0)
    assert q.angle_median_deg < 0.01
    assert q.pitch_rms_mm < 0.01
    assert not q.mirrored


def test_congruence_quality_flags_mirrored_translations():
    """solvePnP's planar two-fold ambiguity reflects a flat board's normal: the
    rotation magnitude survives, the translation is mirrored. So the camera
    appears to slide OPPOSITE to the arm along the same axis, which no rigid
    transform can do. Board 3 scored -0.663 for real on 2026-08-06."""
    spec = [([0, 0, 1.0], 35.0, 0.030), ([1.0, 0, 0], 40.0, -0.020),
            ([0, 1.0, 0], 38.0, 0.025), ([1.0, 1.0, 0], 36.0, -0.015),
            ([1.0, 0, 1.0], 42.0, 0.018)]
    a_list = [_screw(ax, ang, pitch) for ax, ang, pitch in spec]
    mirrored = [_screw(ax, ang, -pitch) for ax, ang, pitch in spec]
    q = hand_eye.congruence_quality(_pair_accumulator(a_list, mirrored), 0)
    assert q.angle_median_deg < 0.01        # angle is clean — the trap
    assert q.pitch_correlation < 0
    assert q.mirrored


def test_select_best_prefers_the_consistent_board_over_the_bigger_one():
    """The 2026-08-06 case, in miniature."""
    x = _screw([0.2, -0.3, 1.0], 20.0, 0.015)
    axes = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    good = [_screw(axes[i % 3], 35.0 + i, 0.004) for i in range(12)]

    acc = hand_eye.HandEyeAccumulator(min_samples=8)
    consistent = _pair_accumulator(good, [geometry.invert_transform(x) @ m @ x
                                          for m in good])
    acc._samples[0] = consistent._samples[0]                      # 13 samples, clean
    # A bigger board whose camera pitches are corrupted: more data, less truth.
    noisy_b = [_screw(axes[i % 3], 35.0 + i, 0.004 + 0.08 * ((-1) ** i))
               for i in range(16)]
    bigger = _pair_accumulator(good + good[:4], noisy_b)
    acc._samples[1] = bigger._samples[1 - 1] if False else bigger._samples[0]

    assert len(acc._samples[1]) > len(acc._samples[0])            # board 2 is bigger
    q0 = hand_eye.congruence_quality(acc, 0)
    q1 = hand_eye.congruence_quality(acc, 1)
    assert q0.pitch_rms_mm < q1.pitch_rms_mm                      # ...and worse

    results = acc.solve_all()
    assert select_best(results).board_index == 1                  # count alone: wrong
    assert select_best(results, acc).board_index == 0             # consistency: right

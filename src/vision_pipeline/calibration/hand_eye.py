"""
Hand-eye calibration: per-board sampling + solve, shared by scripts/calibrate_hand_eye.py
(terminal, Cartesian-jog era) and the arm dashboard's calibration panel
(webui/app.py, joint-jog driven). Kept in one place so the sampling/solve math
exists exactly once, unit-testable with no camera or arm attached.

WHY PER-BOARD BUCKETING. The workspace tiles config.CALIB_BOARD_COUNT distinct
ChArUco boards flat on the table (see CLAUDE.md / config.py's CALIB_* comments),
so a single frame can see several at once. Each board is independently a valid
stationary hand-eye target, so samples are kept in separate per-board buckets
and solved separately — agreement between boards is a much stronger trust
signal than the existing TSAI-vs-PARK check alone, because TSAI and PARK share
the same input data and so cannot catch a systematically bad sample set (e.g.
a board that isn't actually stationary, or a bad detector match).

WHY NOT A Pose ROUND-TRIP. Earlier code went FK 4x4 -> Pose (RPY) -> 4x4 to get
the gripper transform. geometry.transform_to_pose documents that near pitch =
+-90 deg roll/yaw are ambiguous and it forces roll=0 — and a top-down tool
orientation sits close to exactly that singularity. Rotation error is what
poisons calibrateHandEye, so callers here must pass the FK 4x4 straight through
(e.g. from MatlabIKClient.request_fk), never routed through a Pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import charuco
from vision_pipeline.calibration.camera_model import CameraIntrinsics

BoardDetectors = list[tuple[int, "cv2.aruco.CharucoBoard", "cv2.aruco.CharucoDetector"]]


def _mat_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation and 3-vector."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def detect_board_poses(
    frame_gray: np.ndarray,
    intrinsics: CameraIntrinsics,
    detectors: BoardDetectors,
    min_corners: int = config.CALIB_CHARUCO_MIN_CORNERS,
) -> dict[int, np.ndarray]:
    """Return {board_index: T_cam_board} for every board resolvable in this frame.

    Uses charuco.detect (not detector.detectBoard directly) so a frame whose
    corner/ID counts disagree — a case board.matchImagePoints does NOT check —
    is dropped rather than silently mispaired. A board below min_corners (too
    few points for a trustworthy solvePnP, matching the floor
    calibrate_camera_intrinsics.py applies) or that fails solvePnP is simply
    absent from the returned dict.

    Args:
        detectors: pre-built via charuco.build_detectors() — pass one in rather
            than rebuilding per frame; each CharucoDetector construction walks
            the dictionary, and this runs once per captured frame in a hot loop.
    """
    K = intrinsics.matrix
    dist = intrinsics.dist_coeffs
    out: dict[int, np.ndarray] = {}
    for idx, board, det in detectors:
        corners, ids = charuco.detect(det, frame_gray)
        if corners is None or len(corners) < min_corners:
            continue
        try:
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
        except cv2.error:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        out[idx] = _mat_from_Rt(R, tvec)
    return out


@dataclass
class HandEyeSample:
    """One accepted (gripper pose, board pose) observation for one board."""

    T_base_gripper: np.ndarray  # 4x4, wrist pose in base frame (FK, PHYSICAL frame)
    T_cam_board: np.ndarray  # 4x4, board pose in camera frame (solvePnP)


@dataclass
class BoardResult:
    """Solve output for one board's accumulated samples."""

    board_index: int
    n_samples: int
    t_gripper_camera_tsai: np.ndarray  # 4x4 — the value actually saved
    t_gripper_camera_park: np.ndarray  # 4x4 — computed only as a trust cross-check
    tsai_park_disagreement_mm: float
    board_spread_mm: float
    mean_board_origin_base: np.ndarray  # (3,) — doubles as a TABLE_Z_IN_BASE cross-check


class HandEyeAccumulator:
    """Buckets (gripper pose, board pose) samples per board and solves per board.

    Usage (terminal script or web UI, same object either way):
        acc = HandEyeAccumulator()
        board_poses = detect_board_poses(gray, intrinsics, detectors)
        counts = acc.add(board_poses, t_base_gripper)   # after each accepted move
        ...
        results = acc.solve_all()                        # boards with enough samples
        best = select_best(results)
        save_hand_eye(best.t_gripper_camera_tsai, config.HAND_EYE_PATH)
    """

    def __init__(self, min_samples: int = config.CALIB_HAND_EYE_MIN_SAMPLES):
        self.min_samples = min_samples
        self._samples: dict[int, list[HandEyeSample]] = {}

    def add(self, board_poses: dict[int, np.ndarray], t_base_gripper: np.ndarray) -> dict[int, int]:
        """Record one sample per board visible in this capture.

        Args:
            board_poses: output of detect_board_poses for the just-captured frame.
            t_base_gripper: 4x4 wrist pose in the base frame at that same instant
                (FK of the angles the arm was actually at — NOT routed through Pose).

        Returns:
            Updated per-board sample counts (see counts()).
        """
        t_base_gripper = np.asarray(t_base_gripper, dtype=float)
        for idx, t_cam_board in board_poses.items():
            self._samples.setdefault(idx, []).append(
                HandEyeSample(T_base_gripper=t_base_gripper.copy(), T_cam_board=t_cam_board)
            )
        return self.counts()

    def counts(self) -> dict[int, int]:
        """Current sample count per board seen so far."""
        return {idx: len(s) for idx, s in self._samples.items()}

    def solvable_boards(self) -> list[int]:
        """Boards that have reached min_samples and can be solved."""
        return [idx for idx, s in self._samples.items() if len(s) >= self.min_samples]

    def solve_board(self, board_index: int) -> BoardResult:
        """Solve hand-eye for one board's accumulated samples.

        Raises:
            ValueError: fewer than min_samples recorded for this board.
        """
        samples = self._samples.get(board_index, [])
        if len(samples) < self.min_samples:
            raise ValueError(
                f"board {board_index} has {len(samples)} samples, "
                f"need >= {self.min_samples}"
            )

        R_gripper2base = [s.T_base_gripper[:3, :3] for s in samples]
        t_gripper2base = [s.T_base_gripper[:3, 3] for s in samples]
        R_target2cam = [s.T_cam_board[:3, :3] for s in samples]
        t_target2cam = [s.T_cam_board[:3, 3] for s in samples]

        # calibrateHandEye returns cam->gripper, which IS T_gripper_camera (the
        # camera's pose in the gripper frame) — exactly what pixel_to_world loads.
        R_te, t_te = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )
        R_pk, t_pk = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_PARK,
        )
        T_tsai = _mat_from_Rt(R_te, t_te)
        T_park = _mat_from_Rt(R_pk, t_pk)
        disagreement_mm = float(np.linalg.norm(t_te.reshape(3) - t_pk.reshape(3))) * 1000.0

        # Consistency check: the board was stationary in base, so its implied
        # base-frame position should be identical across every sample. Low
        # spread => trustworthy; also doubles as a table-height measurement.
        board_origins = np.array([
            (s.T_base_gripper @ T_tsai @ s.T_cam_board)[:3, 3] for s in samples
        ])
        spread_mm = float(np.linalg.norm(board_origins.std(axis=0))) * 1000.0

        return BoardResult(
            board_index=board_index,
            n_samples=len(samples),
            t_gripper_camera_tsai=T_tsai,
            t_gripper_camera_park=T_park,
            tsai_park_disagreement_mm=disagreement_mm,
            board_spread_mm=spread_mm,
            mean_board_origin_base=board_origins.mean(axis=0),
        )

    def solve_all(self) -> dict[int, BoardResult]:
        """Solve every board that has reached min_samples."""
        return {idx: self.solve_board(idx) for idx in self.solvable_boards()}


def select_best(results: dict[int, BoardResult]) -> BoardResult:
    """Pick the board result to actually save.

    Most-sampled board wins (more constraints on the solve); ties broken by the
    lower TSAI-vs-PARK disagreement.

    Raises:
        ValueError: results is empty.
    """
    if not results:
        raise ValueError("no board results to select from")
    return max(results.values(), key=lambda r: (r.n_samples, -r.tsai_park_disagreement_mm))


def cross_board_agreement_mm(results: dict[int, BoardResult]) -> float | None:
    """Max pairwise disagreement (mm) between boards' T_gripper_camera translations.

    None if fewer than two boards solved — there is nothing independent to compare.
    This is the strongest trust signal available: two boards, sampled from
    overlapping-but-different frames, converging on the same physical camera
    offset rules out a systematically bad sample set in a way TSAI-vs-PARK cannot
    (they share the same inputs).
    """
    translations = [r.t_gripper_camera_tsai[:3, 3] for r in results.values()]
    if len(translations) < 2:
        return None
    worst = 0.0
    for i in range(len(translations)):
        for j in range(i + 1, len(translations)):
            worst = max(worst, float(np.linalg.norm(translations[i] - translations[j])) * 1000.0)
    return worst

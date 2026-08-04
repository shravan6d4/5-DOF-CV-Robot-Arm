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

import json
from dataclasses import dataclass, field
from pathlib import Path

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
    tsai_park_rotation_deg: float
    board_spread_mm: float
    mean_board_origin_base: np.ndarray  # (3,) — doubles as a TABLE_Z_IN_BASE cross-check


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle (degrees) of the rotation taking R_a to R_b.

    Exists because comparing only TRANSLATION between two hand-eye methods is
    not enough to trust a solve: an ill-conditioned or ambiguous sample set can
    drive TSAI and PARK to the same badly-wrong ORIENTATION while their
    translations agree to millimetres. Observed on hardware 2026-08-04 — a solve
    reporting 3.0 mm TSAI-vs-PARK had the camera's optical axis pointing ~180
    deg away from where it physically points, which put every board 430 mm off.
    """
    R = np.asarray(R_a, dtype=float).T @ np.asarray(R_b, dtype=float)
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


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
            tsai_park_rotation_deg=rotation_angle_deg(R_te, R_pk),
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


def save_samples(accumulator: HandEyeAccumulator, path: str | Path) -> None:
    """Write every accumulated raw sample to JSON.

    A hand-eye session is 20+ minutes of hand-jogging an arm, and the solve can
    fail in ways only visible in the raw samples (rotation-axis degeneracy, a
    board that shifted mid-session). Without this the samples die with the
    process and a failed run leaves nothing to diagnose or resume from — which
    is exactly what happened on 2026-08-04. Call after every recorded sample;
    the files are small and rewriting is cheaper than losing a session.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "boards": {
            str(idx): [
                {
                    "T_base_gripper": s.T_base_gripper.tolist(),
                    "T_cam_board": s.T_cam_board.tolist(),
                }
                for s in samples
            ]
            for idx, samples in accumulator._samples.items()
        }
    }
    path.write_text(json.dumps(payload, indent=2))


def load_samples(
    path: str | Path, min_samples: int = config.CALIB_HAND_EYE_MIN_SAMPLES
) -> HandEyeAccumulator:
    """Rebuild an accumulator from a file written by save_samples.

    Raises:
        FileNotFoundError: no such file (callers decide whether that's fatal).
    """
    payload = json.loads(Path(path).read_text())
    acc = HandEyeAccumulator(min_samples=min_samples)
    for idx_str, samples in payload.get("boards", {}).items():
        acc._samples[int(idx_str)] = [
            HandEyeSample(
                T_base_gripper=np.array(s["T_base_gripper"], dtype=float),
                T_cam_board=np.array(s["T_cam_board"], dtype=float),
            )
            for s in samples
        ]
    return acc


def rotation_axis_spread_deg(accumulator: HandEyeAccumulator, board_index: int) -> float | None:
    """Max angle between the rotation AXES of this board's relative motions.

    calibrateHandEye needs at least two motions whose rotation axes are NOT
    parallel; with every motion about one axis the camera translation along it
    is unobservable and the solve returns a confident, wrong answer. This
    reports how much axis diversity the samples actually contain, so a
    degenerate set is visible BEFORE trusting the result rather than after.

    Returns None if there are fewer than two usable motions.
    """
    samples = accumulator._samples.get(board_index, [])
    axes = []
    for a, b in zip(samples, samples[1:]):
        R_rel = a.T_base_gripper[:3, :3].T @ b.T_base_gripper[:3, :3]
        rvec, _ = cv2.Rodrigues(R_rel)
        angle = float(np.linalg.norm(rvec))
        if angle < np.radians(2.0):  # too small to define an axis reliably
            continue
        axes.append(rvec.reshape(3) / angle)
    if len(axes) < 2:
        return None
    worst = 0.0
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            cos = abs(float(np.dot(axes[i], axes[j])))  # abs: +/-axis is the same axis
            worst = max(worst, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    return worst


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

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
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import charuco, geometry
from vision_pipeline.calibration.camera_model import CameraIntrinsics

logger = logging.getLogger(__name__)

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

    # Every solver OpenCV offers, and how far apart they land. TSAI, PARK and
    # HORAUD are SEPARABLE -- they solve rotation first and then translation, so
    # any rotation error propagates straight into the translation. DANIILIDIS
    # (dual quaternion) and ANDREFF solve both at once, which the literature
    # reports as the most robust to noise. Agreement across the two families is
    # a far stronger trust signal than TSAI-vs-PARK, which are both separable
    # and can be wrong together -- as they were on 2026-08-04, agreeing with
    # each other to 2 mm on a transform that was 52 mm out.
    solutions: dict = field(default_factory=dict)     # {method name: 4x4}
    method_spread_mm: float = 0.0                     # widest disagreement in |t|

    @property
    def offset_mm(self) -> float:
        """How far the solve puts the camera from the wrist, in mm.

        The one number a ruler can check, which is why it is surfaced. A solve
        can have a small residual and still be wrong -- on 2026-08-05 five
        independent methods agreed on 80 mm against a measured 24 mm.
        """
        return float(np.linalg.norm(self.t_gripper_camera_tsai[:3, 3])) * 1000.0


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
        best = select_best(results, acc)
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

        # Run every method OpenCV has. Two separable solvers agreeing proves
        # little -- they share the same rotation-then-translation weakness. A
        # simultaneous solver (DANIILIDIS, ANDREFF) agreeing with them is real
        # evidence. ANDREFF in particular refuses to converge on ill-posed data
        # rather than returning a confident wrong answer, so its failure is
        # itself a useful signal and must not be fatal here.
        solutions = {"TSAI": T_tsai, "PARK": T_park}
        for name, flag in (("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
                           ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
                           ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS)):
            try:
                R_m, t_m = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
                    method=flag)
                solutions[name] = _mat_from_Rt(R_m, t_m)
            except cv2.error:
                logger.warning(f"board {board_index}: {name} did not converge — "
                               f"usually means the motions are too small or too "
                               f"nearly co-axial to determine the transform")
        norms = [float(np.linalg.norm(T[:3, 3])) * 1000.0 for T in solutions.values()]
        method_spread_mm = float(max(norms) - min(norms)) if norms else 0.0

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
            solutions=solutions,
            method_spread_mm=method_spread_mm,
        )

    def solve_all(self) -> dict[int, BoardResult]:
        """Solve every board that has reached min_samples."""
        return {idx: self.solve_board(idx) for idx in self.solvable_boards()}


def select_best(
    results: dict[int, BoardResult],
    accumulator: HandEyeAccumulator | None = None,
) -> BoardResult:
    """Pick the board result to actually save.

    CONSISTENCY FIRST, COUNT SECOND — and pass the accumulator, or it cannot
    judge consistency and falls back to the old count-only rule.

    Sample count used to decide this outright, on the reasoning that more
    constraints make a better solve. That is true only while the constraints
    agree with each other. Measured 2026-08-06 in a single capture: board 1 held
    9 samples obeying screw congruence to 6.1 mm rms (pitch correlation +0.995)
    while board 2 held 22 obeying it to 63.8 mm (+0.631). Counting picks board 2
    — twice the samples, and no rigid transform fits them. Twenty-two mutually
    contradictory observations are worth less than nine consistent ones, and the
    solver cannot tell you which it has because it returns an answer either way.

    Boards below CALIB_HAND_EYE_MIN_SAMPLES are excluded by the accumulator
    before this ever sees them, so the count tiebreak still protects against
    preferring a board that is consistent merely for being small.

    Raises:
        ValueError: results is empty.
    """
    if not results:
        raise ValueError("no board results to select from")
    if accumulator is None:
        return max(results.values(),
                   key=lambda r: (r.n_samples, -r.tsai_park_disagreement_mm))

    def rank(r: BoardResult):
        q = congruence_quality(accumulator, r.board_index)
        # Pitch rms is in millimetres and negated so smaller is better; the
        # count only breaks ties between boards of comparable consistency.
        return (-(q.pitch_rms_mm if q else float("inf")), r.n_samples)

    return max(results.values(), key=rank)


@dataclass
class CongruenceQuality:
    """One board's overall obedience to screw congruence, over ALL pose pairs."""

    n_pairs: int
    angle_median_deg: float
    pitch_rms_mm: float
    pitch_correlation: float

    @property
    def mirrored(self) -> bool:
        """Negative correlation: the camera's translations run OPPOSITE to the
        arm's. A rigid transform cannot do that, but solvePnP's two-fold planar
        ambiguity can -- it reflects the board's normal, preserving rotation
        magnitude while mirroring the translation. Board 3 scored -0.663 on
        2026-08-06."""
        return self.pitch_correlation < 0.0


def congruence_quality(
    accumulator: HandEyeAccumulator, board_index: int
) -> CongruenceQuality | None:
    """Summarise a board's screw congruence over every pose pair.

    Separates the two halves deliberately, because on this project they point at
    different subsystems. Angle depends only on rotations -- joint angles and
    the camera's rotation -- so it indicts the KINEMATICS. Pitch additionally
    depends on translations, so with the angles clean it indicts the CAMERA side.

    That split localised the 2026-08-06 failure in one reading: within a single
    capture every board agreed on angle to about 1 degree (same arm, same FK,
    as it must be) while pitch rms ran 6.1 mm, 63.8 mm and 147.3 mm across three
    boards. A fault that varies board to board while the arm is held constant
    cannot be in the arm.

    Returns None when there are too few pairs to say anything.
    """
    samples = accumulator._samples.get(board_index, [])
    if len(samples) < 4:
        return None
    pa, pb, angles = [], [], []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a = (geometry.invert_transform(samples[i].T_base_gripper)
                 @ samples[j].T_base_gripper)
            b = samples[i].T_cam_board @ geometry.invert_transform(
                samples[j].T_cam_board)
            axis_a, _, ang_a = geometry.screw_axis(a)
            axis_b, _, ang_b = geometry.screw_axis(b)
            # Below a few degrees the screw axis is numerically meaningless and
            # so is the pitch measured along it.
            if np.degrees(ang_a) < 5.0:
                continue
            pa.append(geometry.screw_pitch(a, axis_a))
            pb.append(geometry.screw_pitch(b, axis_b))
            angles.append(abs(np.degrees(ang_a - ang_b)))
    if len(pa) < 3:
        return None
    pa_arr, pb_arr = np.array(pa), np.array(pb)
    corr = float(np.corrcoef(pa_arr, pb_arr)[0, 1]) if np.std(pa_arr) > 1e-9 else 0.0
    return CongruenceQuality(
        n_pairs=len(pa),
        angle_median_deg=float(np.median(angles)),
        pitch_rms_mm=float(np.std(pb_arr - pa_arr) * 1000.0),
        pitch_correlation=corr,
    )


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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_samples(
    path: str | Path, min_samples: int = config.CALIB_HAND_EYE_MIN_SAMPLES
) -> HandEyeAccumulator:
    """Rebuild an accumulator from a file written by save_samples.

    Raises:
        FileNotFoundError: no such file (callers decide whether that's fatal).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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


def rotation_magnitudes_deg(accumulator: HandEyeAccumulator,
                            board_index: int) -> list[float]:
    """Rotation angle of each consecutive relative motion, in degrees.

    The quantity that decides whether TRANSLATION is observable at all. The
    camera's offset from the wrist is recovered from how far the camera swings
    when the wrist rotates, so a small rotation carries almost no information
    about it -- the residual stays small because the equations are nearly
    trivially satisfied, not because the answer is right.
    """
    samples = accumulator._samples.get(board_index, [])
    out = []
    for a, b in zip(samples[:-1], samples[1:]):
        M = np.linalg.inv(a.T_base_gripper) @ b.T_base_gripper
        cos = np.clip((np.trace(M[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        out.append(float(np.degrees(np.arccos(cos))))
    return out


def relative_motions(
    accumulator: HandEyeAccumulator, board_index: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(A, B) for each consecutive pose pair: gripper motion and camera motion.

    These are the quantities hand-eye actually solves over -- the absolute
    poses never enter. That is worth stating plainly because it rules out a
    whole family of explanations: any FIXED change of base frame, or of the
    board's own frame, cancels here exactly, so neither can explain a set that
    will not solve. Two sessions with a MOVED board are a different matter, and
    only the pair spanning the join carries the evidence.
    """
    samples = accumulator._samples.get(board_index, [])
    out = []
    for s0, s1 in zip(samples[:-1], samples[1:]):
        a = geometry.invert_transform(s0.T_base_gripper) @ s1.T_base_gripper
        b = s0.T_cam_board @ geometry.invert_transform(s1.T_cam_board)
        out.append((a, b))
    return out


@dataclass
class PairCongruence:
    """How well one pose pair obeys the screw congruence theorem."""

    index: int                  # this is the motion from sample `index` to `index + 1`
    angle_a_deg: float
    angle_b_deg: float
    pitch_a_mm: float
    pitch_b_mm: float

    @property
    def angle_error_deg(self) -> float:
        return abs(self.angle_a_deg - self.angle_b_deg)

    @property
    def pitch_error_mm(self) -> float:
        return abs(self.pitch_a_mm - self.pitch_b_mm)


def screw_congruence(
    accumulator: HandEyeAccumulator, board_index: int
) -> list[PairCongruence]:
    """Test each pose pair against the invariants X cannot change.

    AX = XB makes A and B conjugate, and conjugation preserves both screw
    invariants, so for EVERY pair

        angle(A) == angle(B)        (how far it turned)
        pitch(A) == pitch(B)        (how far it slid along that same axis)

    regardless of what X is. Two independent checks, both available before any
    solve. That matters more than it sounds: every other number this module
    produces is downstream of a solver, and a solver fed inconsistent data
    returns a confident wrong answer rather than an error. On 2026-08-05 five
    solvers plus a different problem formulation agreed to within 11 mm on a
    camera offset a ruler put 56 mm away.

    Until 2026-08-06 only the ANGLE half was tested here. Angle alone cannot see
    a fault that turns the right amount about the right axis but slides the
    wrong way along it, which is what a misdetected board or a stale frame
    produces.
    """
    out = []
    for i, (a, b) in enumerate(relative_motions(accumulator, board_index)):
        axis_a, _, ang_a = geometry.screw_axis(a)
        axis_b, _, ang_b = geometry.screw_axis(b)
        out.append(PairCongruence(
            index=i,
            angle_a_deg=float(np.degrees(ang_a)),
            angle_b_deg=float(np.degrees(ang_b)),
            pitch_a_mm=1000.0 * geometry.screw_pitch(a, axis_a),
            pitch_b_mm=1000.0 * geometry.screw_pitch(b, axis_b),
        ))
    return out


def congruence_disagreement(
    accumulator: HandEyeAccumulator,
    board_index: int,
    angle_tol_deg: float | None = None,
    pitch_tol_mm: float | None = None,
) -> dict[int, float]:
    """For each sample, the FRACTION of other samples it is inconsistent with.

    ALL PAIRS, NOT JUST NEIGHBOURS, and that is the whole point. Congruence
    holds between ANY two poses -- consecutive ordering is a capture artefact,
    not a property of the maths -- so restricting the test to neighbours throws
    away almost all of the evidence and leaves attribution ambiguous: an
    isolated bad consecutive pair implicates BOTH its endpoints equally, with
    nothing to separate them.

    Testing every pair resolves it by consensus. A genuinely bad sample is
    inconsistent with nearly every other sample, so it scores near 1.0. A good
    sample that merely sat next to a bad one is inconsistent with just that one,
    so it scores near 0. That is the same logic RANSAC applies to hand-eye
    (ethz-asl/hand_eye_calibration), reduced to a form that needs no model
    fitting at all because screw congruence is model-free.

    Cost is O(n^2) pairs, which for a capture of a few dozen poses is nothing.
    """
    if angle_tol_deg is None:
        angle_tol_deg = config.CALIB_HAND_EYE_CONGRUENCE_ANGLE_DEG
    if pitch_tol_mm is None:
        pitch_tol_mm = config.CALIB_HAND_EYE_CONGRUENCE_PITCH_MM

    samples = accumulator._samples.get(board_index, [])
    n = len(samples)
    if n < 3:
        return {}

    bad = {i: 0 for i in range(n)}
    total = {i: 0 for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            a = (geometry.invert_transform(samples[i].T_base_gripper)
                 @ samples[j].T_base_gripper)
            b = samples[i].T_cam_board @ geometry.invert_transform(
                samples[j].T_cam_board)
            axis_a, _, ang_a = geometry.screw_axis(a)
            axis_b, _, ang_b = geometry.screw_axis(b)
            # A near-zero rotation carries no information either way: its axis
            # is undefined, so pitch is meaningless and the pair must not vote.
            if np.degrees(ang_a) < 1.0:
                continue
            off = (abs(np.degrees(ang_a - ang_b)) > angle_tol_deg
                   or abs(1000.0 * (geometry.screw_pitch(a, axis_a)
                                    - geometry.screw_pitch(b, axis_b))) > pitch_tol_mm)
            for k in (i, j):
                total[k] += 1
                bad[k] += int(off)
    return {i: (bad[i] / total[i] if total[i] else 0.0) for i in range(n)}


def congruence_outliers(
    accumulator: HandEyeAccumulator,
    board_index: int,
    angle_tol_deg: float | None = None,
    pitch_tol_mm: float | None = None,
    max_disagreement: float | None = None,
) -> list[int]:
    """SAMPLE indices that no rigid transform can reconcile with the rest.

    Named by consensus over every pair (see congruence_disagreement), not by
    consecutive ones: a sample is an outlier when it disagrees with MORE THAN
    `max_disagreement` of the others, so the majority decides and one bad frame
    cannot condemn its neighbours.

    Verified against the real failure: board 2's sample 11 broke pairs 10->11
    and 11->12 while 9->10 and 12->13 were clean, and removing that one sample
    took board spread from 156 mm to 60 mm and TSAI-vs-PARK from 78 deg to 7.8.
    """
    if max_disagreement is None:
        max_disagreement = config.CALIB_HAND_EYE_MAX_DISAGREEMENT
    scores = congruence_disagreement(
        accumulator, board_index, angle_tol_deg, pitch_tol_mm)
    return sorted(i for i, s in scores.items() if s > max_disagreement)


@dataclass
class Observability:
    """Whether a capture can determine the answer, independent of any solve.

    Indices follow the robot-calibration literature (Sun & Hollerbach, ICRA
    2008; Joubair et al.), applied to the matrix whose conditioning governs
    hand-eye TRANSLATION. All are computed from the singular values of the
    stacked (R_a - I).
    """

    singular_values: np.ndarray
    o1_product: float           # D-optimality: volume of the data scatter
    o2_inverse_condition: float  # sigma_min / sigma_max — SHAPE only
    o3_min_singular: float      # E-optimality: the literature's pick
    o4_noise_amplification: float  # sigma_min^2 / sigma_max

    @property
    def condition_number(self) -> float:
        return float("inf") if self.o2_inverse_condition <= 0 else 1.0 / self.o2_inverse_condition


def observability(
    accumulator: HandEyeAccumulator, board_index: int
) -> Observability | None:
    """Score how well this capture pins the camera POSITION. None if too small.

    Once X's rotation is known its translation solves from

        (R_a - I) t_x = R_x @ t_b - t_a

    stacked over pairs, so the singular values of the stacked (R_a - I) ARE the
    observability of t_x. A rotation about axis n satisfies (R_a - I) n = 0, so
    repeating one axis leaves the offset along it invisible no matter how many
    samples are taken or how large the rotations.

    WHY FOUR INDICES AND NOT ONE. They fail differently, and the pair that
    matters here is O2 against O3:

      * O2 (inverse condition number) is scale-invariant. It sees CLUSTERING --
        one direction weak relative to the others -- and is blind to everything
        being uniformly weak. A capture of tiny rotations spread evenly over
        three axes scores a perfect O2 and determines nothing.
      * O3 (smallest singular value) is not scale-invariant, so it catches both
        failures at once: ||(R - I)v|| = 2 sin(theta/2) * |v_perp|, which is
        small when the rotations are small OR when they share an axis. The
        literature settles on it (E-optimality) as the best single predictor of
        end-effector pose uncertainty, and it subsumes the separate
        "median rotation >= 30 deg" and "axes not clustered" checks this module
        used to make independently.

    O1 (product, D-optimality) and O4 (noise amplification) are carried for
    reporting; O4 is the one usually described as least noise-sensitive.
    """
    samples = accumulator._samples.get(board_index, [])
    if len(samples) < 3:
        return None
    blocks = [a[:3, :3] - np.eye(3)
              for a, _ in relative_motions(accumulator, board_index)]
    sv = np.linalg.svd(np.vstack(blocks), compute_uv=False)
    smin, smax = float(sv[-1]), float(sv[0])
    return Observability(
        singular_values=sv,
        # Normalised by the pair count so a long capture is not flattered
        # purely for being long -- this is a per-observation figure of merit.
        o1_product=float(np.prod(sv) ** (1.0 / len(sv)) / np.sqrt(len(blocks))),
        o2_inverse_condition=(smin / smax) if smax > 1e-12 else 0.0,
        o3_min_singular=smin,
        o4_noise_amplification=(smin ** 2 / smax) if smax > 1e-12 else 0.0,
    )


def capture_health(accumulator: HandEyeAccumulator, board_index: int) -> list[str]:
    """Judge the CAPTURE, not the solve. Returns human-readable warnings.

    Written because every check the repo had scored the solve against itself,
    and a badly-conditioned capture produces a solve that scores beautifully and
    is wrong. On 2026-08-04, five independent solvers agreed to within 4 mm on a
    camera offset that a ruler put 56 mm away.

    Thresholds follow the standard guidance for articulated arms (MVTec HALCON
    and the hand-eye literature): >= 8 poses, rotations of at least 30 deg
    between them (60 is better), and at least two non-parallel rotation axes.
    """
    warnings = []
    n = len(accumulator._samples.get(board_index, []))
    if n < 8:
        warnings.append(
            f"only {n} poses (want >= 8): too few to average out pose noise")

    rots = rotation_magnitudes_deg(accumulator, board_index)
    if rots:
        median = float(np.median(rots))
        if median < config.CALIB_HAND_EYE_MIN_ROTATION_DEG:
            warnings.append(
                f"median rotation between poses is {median:.1f} deg, under the "
                f"{config.CALIB_HAND_EYE_MIN_ROTATION_DEG:.0f} deg minimum "
                f"(60 is better). THE CAMERA OFFSET IS THE PART THIS RUINS: it "
                f"is recovered from how far the camera swings, so small "
                f"rotations leave it barely determined while every residual "
                f"still looks healthy.")

    spread = rotation_axis_spread_deg(accumulator, board_index)
    if spread is not None and spread < 10.0:
        warnings.append(
            f"rotation axes span only {spread:.1f} deg — near-coaxial motion "
            f"leaves camera translation along that axis unobservable")

    obs = observability(accumulator, board_index)
    if obs is not None:
        if obs.condition_number > config.CALIB_HAND_EYE_MAX_CONDITION:
            warnings.append(
                f"rotation axes are CLUSTERED (conditioning "
                f"{obs.condition_number:.1f}, want under "
                f"{config.CALIB_HAND_EYE_MAX_CONDITION:.0f}): camera position "
                f"along the crowded axis is barely observable. Axis SPREAD does "
                f"not catch this — it reads the widest gap between any two "
                f"poses, so a handful of odd ones make a set look varied while "
                f"the bulk sit on top of each other. Vary a DIFFERENT joint; "
                f"more of the same poses cannot fix it.")
        if obs.o3_min_singular < config.CALIB_HAND_EYE_MIN_O3:
            warnings.append(
                f"observability O3 (smallest singular value) is "
                f"{obs.o3_min_singular:.2f}, under {config.CALIB_HAND_EYE_MIN_O3:.1f}: "
                f"the weakest direction carries almost no information about the "
                f"camera offset. O3 is low when rotations are SMALL or when they "
                f"SHARE AN AXIS, so read it with the two checks above — it is the "
                f"one number that catches both.")

    outliers = congruence_outliers(accumulator, board_index)
    if outliers:
        warnings.append(
            f"sample(s) {outliers} break screw congruence: their gripper and "
            f"camera motions disagree on rotation angle or on how far the motion "
            f"slid along its own axis. AX=XB makes those equal whatever X is, so "
            f"these describe motions that no rigid transform can reconcile. "
            f"Remove them before solving — one bad sample took board spread from "
            f"156 mm to 60 mm on 2026-08-06.")
    return warnings


def translation_conditioning(
    accumulator: HandEyeAccumulator, board_index: int
) -> float | None:
    """How well this capture pins the camera's POSITION. Lower is better; 1 is
    perfect and anything past ~3 means one direction is barely constrained.

    Rotation and translation fail differently, and no residual distinguishes
    them: a capture can determine the camera's orientation to a fraction of a
    degree while leaving its position free along a whole axis. Once X's rotation
    is known, its translation solves

        (R_a - I) t_x = R_x @ t_b - t_a

    stacked over consecutive pose pairs, so the conditioning of the stacked
    (R_a - I) IS the observability of t_x. A rotation about an axis n satisfies
    (R_a - I) n = 0, so rotating repeatedly about the SAME axis makes the offset
    along it invisible, however many samples are taken and however large the
    rotations are.

    Found 2026-08-06: a 14-pose set passed every existing check (median rotation
    35 deg, axis spread 82 deg) while nine of its thirteen pose changes rotated
    about the same axis to within 1 deg -- all J5 wrist roll. Rotation solved to
    0.8 deg; the solvers split 24 mm vs 50 mm on position, disagreeing almost
    entirely along that axis. Condition number 3.6.

    Returns None when there are too few samples to say.

    This is Observability.condition_number, kept as a named function because it
    is the number quoted on screen during a capture and in CLAUDE.md. See
    `observability` for the other three indices and for why O3 -- not this --
    is the better single gate: a condition number is scale-invariant, so a
    capture of uniformly tiny rotations scores a perfect 1.0 while determining
    nothing at all.
    """
    obs = observability(accumulator, board_index)
    return None if obs is None else obs.condition_number


def solve_complaints(result: BoardResult) -> list[str]:
    """Every reason this solve should not be trusted. Empty means accept.

    THE RULER IS NECESSARY BUT NOT SUFFICIENT, which cost a session to learn.
    On 2026-08-05 a sample file that had silently accumulated records from two
    different calibration frames solved to |t| = 33 mm -- comfortably inside the
    measured 24 mm -- while its board spread was 183 mm and TSAI disagreed with
    PARK by 170 degrees. A single plausible number is not evidence; it is one
    number that happened to land in range.

    Each gate catches a different failure, so they are all checked:
      * ruler        -- the solve contradicts a physical measurement
      * board spread -- the chain cannot place a stationary board consistently
      * method spread-- the solvers do not agree on an answer
      * TSAI vs PARK -- rotation is not determined at all
    """
    out = []
    ruler = offset_disagrees_with_ruler(result)
    if ruler:
        out.append(ruler)
    if result.board_spread_mm > config.HAND_EYE_MAX_BOARD_SPREAD_MM:
        out.append(
            f"board spread {result.board_spread_mm:.0f} mm (max "
            f"{config.HAND_EYE_MAX_BOARD_SPREAD_MM:.0f}): the board never moved, "
            f"so the chain should put it in one place for every sample. This "
            f"much scatter usually means the samples do not all describe the "
            f"same arm -- e.g. a file mixing captures from two calibrations.")
    if result.method_spread_mm > config.HAND_EYE_MAX_METHOD_SPREAD_MM:
        out.append(
            f"solvers disagree by {result.method_spread_mm:.0f} mm (max "
            f"{config.HAND_EYE_MAX_METHOD_SPREAD_MM:.0f}) on where the camera is")
    if result.tsai_park_rotation_deg > config.HAND_EYE_MAX_TSAI_PARK_ROT_DEG:
        out.append(
            f"TSAI and PARK disagree by {result.tsai_park_rotation_deg:.0f} deg "
            f"on camera ORIENTATION (max "
            f"{config.HAND_EYE_MAX_TSAI_PARK_ROT_DEG:.0f}): rotation is not "
            f"determined by this data at all")
    return out


def offset_disagrees_with_ruler(result: BoardResult) -> str | None:
    """Compare the solved camera offset against the measured one.

    The only check here that is independent of the data being solved. Everything
    else -- residuals, TSAI vs PARK, even two boards agreeing -- is computed
    from the same samples and can be confidently wrong together.

    Returns a message when it disagrees, None when it is fine.
    """
    got = result.offset_mm
    want = config.HAND_EYE_EXPECTED_OFFSET_MM
    if abs(got - want) <= config.HAND_EYE_OFFSET_TOLERANCE_MM:
        return None
    return (f"solve puts the camera {got:.0f} mm from the wrist; the ruler says "
            f"{want:.0f} mm (tolerance {config.HAND_EYE_OFFSET_TOLERANCE_MM:.0f}). "
            f"A solve cannot be trusted past a physical measurement, whatever "
            f"its residual says.")

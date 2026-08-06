"""Hand-eye solved from EXPLICIT motion pairs, pooled across boards.

WHAT THIS BUYS, AND WHY IT IS NOT WHAT cv2.calibrateHandEye DOES.

OpenCV takes ABSOLUTE poses -- a list of gripper poses and a list of board
poses -- and forms the motion pairs internally. That interface cannot express
two things this data needs:

  1. POOLING ACROSS BOARDS. The equation is A X = X B where B is the CAMERA's
     motion between two views. Written from board poses,

         B = T_cam_board(i) @ inv(T_cam_board(j))

     the board's own placement cancels: the same rigid camera motion comes out
     whichever stationary board measured it. So a pair from board 1 and a pair
     from board 2 constrain the SAME X and belong in the SAME solve, provided
     each pair's two ends see one board. Because OpenCV wants absolute poses in
     one frame, the repo had been solving each board separately and then
     throwing away every board but one.

  2. NON-CONSECUTIVE PAIRS. Congruence and AX=XB hold between ANY two poses;
     capture order is an artefact of how the operator jogged. Using only
     neighbours takes n-1 constraints from a set that offers n(n-1)/2.

    Measured on the 2026-08-06 capture: 11 consecutive pairs on the chosen
    board, against 390 available across all boards. Roughly 97% of the
    constraints were being discarded, which is why a capture that looked thin
    was never actually thin -- it was under-used.

WHAT IT DOES NOT BUY. Pooling adds constraints, not information that was never
recorded. If every pose in the file rotates about one axis, every pair does too,
and no amount of pairing makes the camera's offset along that axis observable.
Selection below maximises what is there; it cannot manufacture what is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import geometry


@dataclass
class MotionPair:
    """One (gripper motion, camera motion) constraint on X, with provenance."""

    a: np.ndarray            # 4x4 gripper motion, base frame
    b: np.ndarray            # 4x4 camera motion
    board: int               # which board measured b (0-based)
    i: int                   # sample indices within that board
    j: int

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(geometry.screw_axis(self.a)[2]))

    def congruence_error(self) -> tuple[float, float]:
        """(angle error in deg, pitch error in mm) — see hand_eye.screw_congruence."""
        axis_a, _, ang_a = geometry.screw_axis(self.a)
        axis_b, _, ang_b = geometry.screw_axis(self.b)
        return (abs(float(np.degrees(ang_a - ang_b))),
                abs(1000.0 * (geometry.screw_pitch(self.a, axis_a)
                              - geometry.screw_pitch(self.b, axis_b))))


def all_pairs(
    accumulator,
    min_angle_deg: float = 5.0,
    filter_congruence: bool = True,
) -> list[MotionPair]:
    """Every usable motion pair in the file, from every board.

    Pairs below `min_angle_deg` are dropped rather than filtered: a near-zero
    rotation has an undefined screw axis, so its pitch is meaningless and its
    contribution to the rotation solve is pure noise direction.
    """
    out: list[MotionPair] = []
    for board, samples in accumulator._samples.items():
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                a = (geometry.invert_transform(samples[i].T_base_gripper)
                     @ samples[j].T_base_gripper)
                b = (samples[i].T_cam_board
                     @ geometry.invert_transform(samples[j].T_cam_board))
                pair = MotionPair(a=a, b=b, board=board, i=i, j=j)
                if pair.angle_deg < min_angle_deg:
                    continue
                if filter_congruence:
                    ang_err, pitch_err = pair.congruence_error()
                    if (ang_err > config.CALIB_HAND_EYE_CONGRUENCE_ANGLE_DEG
                            or pitch_err > config.CALIB_HAND_EYE_CONGRUENCE_PITCH_MM):
                        continue
                out.append(pair)
    return out


def _log_rotation(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    axis, _, angle = geometry.screw_axis(
        np.block([[r, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    return axis * angle


def solve_rotation(pairs: list[MotionPair]) -> np.ndarray:
    """R_x from A X = X B, by Park & Martin's reduction to Procrustes.

    A_r R_x = R_x B_r means R_x carries B's rotation AXIS onto A's, so with
    alpha = log(R_a) and beta = log(R_b) the problem is exactly

        minimise  sum || alpha_i - R_x beta_i ||^2

    which is orthogonal Procrustes and has the closed form below. No iteration,
    no initial guess, and every pair contributes equally regardless of which
    board measured it -- which is the whole point of pooling.
    """
    alpha = np.array([_log_rotation(p.a[:3, :3]) for p in pairs])
    beta = np.array([_log_rotation(p.b[:3, :3]) for p in pairs])
    u, _, vt = np.linalg.svd(beta.T @ alpha)
    # Force a proper rotation; a reflection is never the answer and SVD will
    # happily return one when the data is contradictory.
    d = float(np.sign(np.linalg.det(u @ vt)))
    return (u @ np.diag([1.0, 1.0, d]) @ vt).T


def solve_translation(pairs: list[MotionPair], r_x: np.ndarray) -> np.ndarray:
    """t_x from (R_a - I) t_x = R_x t_b - t_a, stacked and least-squares.

    THE ILL-CONDITIONED HALF. (R_a - I) annihilates its own rotation axis, so
    pairs that all turn about one axis leave t_x free along it and lstsq
    returns the minimum-norm answer -- a confident number chosen by the
    regulariser rather than by the data. hand_eye.observability scores exactly
    this matrix; read it before trusting the result.
    """
    lhs = np.vstack([p.a[:3, :3] - np.eye(3) for p in pairs])
    rhs = np.concatenate([r_x @ p.b[:3, 3] - p.a[:3, 3] for p in pairs])
    t_x, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return t_x


def solve(pairs: list[MotionPair]) -> np.ndarray:
    """The 4x4 gripper->camera transform implied by these pairs."""
    if len(pairs) < 2:
        raise ValueError("need at least 2 motion pairs to solve")
    r_x = solve_rotation(pairs)
    t = np.eye(4)
    t[:3, :3] = r_x
    t[:3, 3] = solve_translation(pairs, r_x)
    return t


def information_matrix(pairs: list[MotionPair]) -> np.ndarray:
    """J^T J for the translation system — what the design criteria score."""
    lhs = np.vstack([p.a[:3, :3] - np.eye(3) for p in pairs])
    return lhs.T @ lhs


def d_optimality(pairs: list[MotionPair]) -> float:
    """log det of the information matrix. Higher is better.

    D-optimality maximises the determinant, i.e. shrinks the VOLUME of the
    uncertainty ellipsoid. The robot-calibration literature settles on O3
    (smallest singular value, E-optimality) as the better predictor of final
    accuracy, but reports determinant-based criteria as the right choice when
    the selection is GREEDY -- log det is near-submodular, so greedy addition
    behaves, whereas greedily maximising a min-singular-value plateaus and
    stalls. So: select on D, report and gate on O3.

    Log rather than raw determinant because these span many orders of magnitude
    and the raw product underflows on small subsets.
    """
    sign, logdet = np.linalg.slogdet(information_matrix(pairs))
    return float(logdet) if sign > 0 else -np.inf


def select_d_optimal(
    pairs: list[MotionPair],
    k: int,
    exchange_rounds: int = 12,
    seed: int = 0,
) -> list[MotionPair]:
    """Choose k pairs maximising D-optimality: greedy, then Fedorov exchange.

    Two stages because greedy alone lands in a local optimum -- it commits to
    early choices that a later, better-informed selection would not have made.
    The exchange step (Fedorov) then repeatedly swaps one chosen pair for one
    rejected pair whenever that raises the criterion, which is the standard
    repair and is what DETMAX and its descendants are built around.

    Deterministic given `seed`, because a calibration that changes when re-run
    on identical inputs is not one anybody can debug.
    """
    if k >= len(pairs):
        return list(pairs)
    rng = np.random.default_rng(seed)

    # --- greedy: seed with the single most informative pair, then add ---------
    remaining = list(range(len(pairs)))
    chosen: list[int] = []
    # A determinant needs 3 rows minimum to be non-degenerate, so start from a
    # random triple rather than trying to rank singletons by a criterion that
    # is -inf for all of them.
    start = list(rng.choice(len(pairs), size=min(3, len(pairs)), replace=False))
    for s in start:
        chosen.append(int(s))
        remaining.remove(int(s))

    while len(chosen) < k and remaining:
        scores = [(d_optimality([pairs[c] for c in chosen] + [pairs[r]]), r)
                  for r in remaining]
        _, best = max(scores)
        chosen.append(best)
        remaining.remove(best)

    # --- Fedorov exchange: repair the greedy path ----------------------------
    current = d_optimality([pairs[c] for c in chosen])
    for _ in range(exchange_rounds):
        improved = False
        for ci in range(len(chosen)):
            trial_base = [pairs[c] for idx, c in enumerate(chosen) if idx != ci]
            best_gain, best_r = 0.0, None
            for r in remaining:
                score = d_optimality(trial_base + [pairs[r]])
                if score > current + 1e-9 and score - current > best_gain:
                    best_gain, best_r = score - current, r
            if best_r is not None:
                remaining.append(chosen[ci])
                chosen[ci] = best_r
                remaining.remove(best_r)
                current += best_gain
                improved = True
        if not improved:
            break
    return [pairs[c] for c in chosen]


def observability_of(pairs: list[MotionPair]) -> tuple[float, float]:
    """(O3 = smallest singular value, condition number) for a pair set."""
    lhs = np.vstack([p.a[:3, :3] - np.eye(3) for p in pairs])
    sv = np.linalg.svd(lhs, compute_uv=False)
    smin, smax = float(sv[-1]), float(sv[0])
    return smin, (smax / smin if smin > 1e-12 else float("inf"))

"""Pair-based hand-eye: pooling across boards, and D-optimal pose selection.

The solver is verified against a PLANTED transform before it is ever pointed at
real data. That ordering matters here more than usual: this project has twice
been misled by a hand-eye result that was internally consistent and wrong, so a
new solver that merely agrees with the old ones proves nothing.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.calibration import geometry, hand_eye  # noqa: E402
from vision_pipeline.calibration import hand_eye_pairs as hep  # noqa: E402


def _rigid(axis, angle_deg, translation):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    a = np.deg2rad(angle_deg)
    t = np.eye(4)
    t[:3, :3] = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)
    t[:3, 3] = translation
    return t


def _gripper_poses(n, axes=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    return [
        _rigid(axes[i % len(axes)], 35.0 + 11.0 * i,
               (0.12 + 0.01 * (i % 4), 0.02 * ((i % 3) - 1), 0.15 + 0.01 * (i % 5)))
        for i in range(n)
    ]


def _accumulator(x, boards):
    """boards: {board_index: (gripper_poses, T_base_board)}."""
    acc = hand_eye.HandEyeAccumulator(min_samples=3)
    for idx, (poses, t_base_board) in boards.items():
        acc._samples[idx] = [
            hand_eye.HandEyeSample(
                T_base_gripper=g,
                T_cam_board=geometry.invert_transform(g @ x) @ t_base_board)
            for g in poses
        ]
    return acc


X_TRUE = _rigid([0.3, -0.5, 1.0], 22.0, (0.004, 0.021, 0.011))   # |t| = 24 mm


# --- the solver, against a planted answer ------------------------------------

def test_solver_recovers_a_planted_transform():
    acc = _accumulator(X_TRUE, {0: (_gripper_poses(10),
                                    geometry.make_transform(0.3, 0.05, -0.07))})
    assert np.allclose(hep.solve(hep.all_pairs(acc)), X_TRUE, atol=1e-8)


def test_pairs_from_DIFFERENT_boards_constrain_the_same_transform():
    """The claim that makes pooling legal: a board's own placement cancels out
    of B = T_cam_board(i) @ inv(T_cam_board(j)), so two boards in completely
    different places report the SAME camera motion and belong in one solve."""
    poses = _gripper_poses(8)
    acc = _accumulator(X_TRUE, {
        0: (poses, geometry.make_transform(0.30, 0.05, -0.07, yaw_deg=10.0)),
        1: (poses, geometry.make_transform(-0.10, 0.40, -0.07, yaw_deg=-75.0)),
    })
    pairs = hep.all_pairs(acc)
    board0 = [p for p in pairs if p.board == 0]
    board1 = [p for p in pairs if p.board == 1]
    for p0, p1 in zip(board0, board1):
        assert np.allclose(p0.b, p1.b, atol=1e-9)       # identical camera motion
    assert np.allclose(hep.solve(board0 + board1), X_TRUE, atol=1e-8)


def test_pooling_beats_one_board_when_the_boards_saw_different_axes():
    """The case a per-board solve structurally cannot reach. Each board alone is
    co-axial and cannot observe the camera offset; pooled, they can."""
    acc = _accumulator(X_TRUE, {
        0: (_gripper_poses(7, axes=((0, 0, 1),)),
            geometry.make_transform(0.30, 0.05, -0.07)),
        1: (_gripper_poses(7, axes=((1, 0, 0),)),
            geometry.make_transform(-0.10, 0.40, -0.07)),
    })
    pairs = hep.all_pairs(acc)
    o3_one, _ = hep.observability_of([p for p in pairs if p.board == 0])
    o3_both, _ = hep.observability_of(pairs)
    assert o3_both > 3 * o3_one


def test_non_consecutive_pairs_multiply_the_constraints():
    """12 samples offer 66 constraints, not 11. Consecutive-only ordering is an
    artefact of how the operator jogged, not a property of the maths."""
    acc = _accumulator(X_TRUE, {0: (_gripper_poses(12),
                                    geometry.make_transform(0.3, 0.05, -0.07))})
    assert len(hep.all_pairs(acc)) == 12 * 11 // 2


def test_congruence_filtering_removes_a_corrupted_sample_pairwise():
    """One bad frame poisons every pair it appears in, and only those."""
    acc = _accumulator(X_TRUE, {0: (_gripper_poses(8),
                                    geometry.make_transform(0.3, 0.05, -0.07))})
    acc._samples[0][3].T_cam_board = (
        acc._samples[0][3].T_cam_board @ _rigid([0, 1, 0], 30.0, (0.05, 0, 0)))
    kept = hep.all_pairs(acc, filter_congruence=True)
    assert all(p.i != 3 and p.j != 3 for p in kept)
    assert len(kept) == 7 * 6 // 2          # every pair NOT touching sample 3


def test_a_near_zero_rotation_is_dropped_not_kept():
    """Its screw axis is undefined, so its pitch is meaningless and its
    contribution to the rotation solve is a noise direction."""
    poses = [np.eye(4), _rigid([0, 0, 1], 0.2, (0.12, 0, 0.15)),
             _rigid([1, 0, 0], 40.0, (0.13, 0, 0.16))]
    acc = _accumulator(X_TRUE, {0: (poses,
                                    geometry.make_transform(0.3, 0.05, -0.07))})
    assert all(p.angle_deg >= 5.0 for p in hep.all_pairs(acc))


# --- D-optimal selection -----------------------------------------------------

def test_selection_prefers_varied_axes_over_a_co_axial_majority():
    """The exact shape of the real capture: mostly one axis, a few off it.
    A good selection reaches for the minority."""
    acc = _accumulator(X_TRUE, {
        0: (_gripper_poses(12, axes=((0, 0, 1),)),
            geometry.make_transform(0.30, 0.05, -0.07)),
        1: (_gripper_poses(6, axes=((1, 0, 0), (0, 1, 0))),
            geometry.make_transform(-0.10, 0.40, -0.07)),
    })
    pairs = hep.all_pairs(acc)
    chosen = hep.select_d_optimal(pairs, k=12)
    from_minority = sum(1 for p in chosen if p.board == 1)
    assert from_minority > 12 * (len([p for p in pairs if p.board == 1]) / len(pairs))


def test_selection_improves_observability_over_an_arbitrary_subset():
    acc = _accumulator(X_TRUE, {
        0: (_gripper_poses(10, axes=((0, 0, 1),)),
            geometry.make_transform(0.30, 0.05, -0.07)),
        1: (_gripper_poses(8, axes=((1, 0, 0), (0, 1, 0))),
            geometry.make_transform(-0.10, 0.40, -0.07)),
    })
    pairs = hep.all_pairs(acc)
    o3_chosen, _ = hep.observability_of(hep.select_d_optimal(pairs, k=15))
    o3_arbitrary, _ = hep.observability_of(pairs[:15])
    assert o3_chosen > o3_arbitrary


def test_selection_is_deterministic():
    """A calibration that changes when re-run on identical input is not one
    anybody can debug."""
    acc = _accumulator(X_TRUE, {0: (_gripper_poses(10),
                                    geometry.make_transform(0.3, 0.05, -0.07))})
    pairs = hep.all_pairs(acc)
    a = hep.select_d_optimal(pairs, k=8)
    b = hep.select_d_optimal(pairs, k=8)
    assert [(p.board, p.i, p.j) for p in a] == [(p.board, p.i, p.j) for p in b]


def test_selection_returns_everything_when_k_exceeds_what_exists():
    acc = _accumulator(X_TRUE, {0: (_gripper_poses(6),
                                    geometry.make_transform(0.3, 0.05, -0.07))})
    pairs = hep.all_pairs(acc)
    assert len(hep.select_d_optimal(pairs, k=999)) == len(pairs)


def test_selection_does_not_damage_a_recoverable_answer():
    """Selection must trade quantity for quality without losing correctness."""
    acc = _accumulator(X_TRUE, {0: (_gripper_poses(12),
                                    geometry.make_transform(0.3, 0.05, -0.07))})
    chosen = hep.select_d_optimal(hep.all_pairs(acc), k=10)
    assert np.allclose(hep.solve(chosen), X_TRUE, atol=1e-8)


def test_solve_refuses_a_set_too_small_to_constrain_anything():
    with pytest.raises(ValueError):
        hep.solve([])

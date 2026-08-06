"""
Tests for the 3D geometry primitives (calibration/geometry.py).

These are pure math with exact expected answers — no camera, no robot. They pin
down the transform conventions the whole pixel-to-world chain relies on, so if
someone changes the RPY convention later, these break loudly.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.calibration import geometry


def test_make_transform_and_pose_round_trip():
    # Angles chosen away from gimbal lock (pitch != +/-90).
    x, y, z, roll, pitch, yaw = 0.1, -0.2, 0.3, 10.0, 20.0, 30.0
    t = geometry.make_transform(x, y, z, roll, pitch, yaw)
    recovered = geometry.transform_to_pose(t)
    assert recovered == pytest.approx((x, y, z, roll, pitch, yaw), abs=1e-6)


def test_invert_transform_is_true_inverse():
    t = geometry.make_transform(0.5, -0.3, 0.2, 15.0, -25.0, 40.0)
    identity = t @ geometry.invert_transform(t)
    assert identity == pytest.approx(np.eye(4), abs=1e-9)


def test_transform_point_applies_translation_but_direction_does_not():
    # Pure translation, no rotation.
    t = geometry.make_transform(1.0, 2.0, 3.0)
    point = geometry.transform_point(t, np.array([0.0, 0.0, 0.0]))
    assert point == pytest.approx([1.0, 2.0, 3.0])

    # A direction ignores the translation column.
    direction = geometry.transform_direction(t, np.array([1.0, 0.0, 0.0]))
    assert direction == pytest.approx([1.0, 0.0, 0.0])


def test_transform_direction_rotates():
    # 90 deg yaw about Z maps +X -> +Y.
    t = geometry.make_transform(5.0, 5.0, 5.0, yaw_deg=90.0)
    direction = geometry.transform_direction(t, np.array([1.0, 0.0, 0.0]))
    assert direction == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_ray_plane_intersection_hits_plane():
    # Ray starting above the plane, pointing straight down, hits z=0 at (2,3,0).
    hit = geometry.ray_plane_intersection(
        origin=np.array([2.0, 3.0, 1.0]),
        direction=np.array([0.0, 0.0, -1.0]),
        plane_z=0.0,
    )
    assert hit == pytest.approx([2.0, 3.0, 0.0])


def test_ray_plane_intersection_angled():
    # 45 deg ray from (0,0,1) going +x and -z reaches z=0 at x=1.
    hit = geometry.ray_plane_intersection(
        origin=np.array([0.0, 0.0, 1.0]),
        direction=np.array([1.0, 0.0, -1.0]),
        plane_z=0.0,
    )
    assert hit == pytest.approx([1.0, 0.0, 0.0])


def test_ray_plane_intersection_parallel_returns_none():
    hit = geometry.ray_plane_intersection(
        origin=np.array([0.0, 0.0, 1.0]),
        direction=np.array([1.0, 0.0, 0.0]),  # horizontal, never meets z=0
        plane_z=0.0,
    )
    assert hit is None


def test_ray_plane_intersection_behind_returns_none():
    # Plane is below, but the ray points UP and away from it.
    hit = geometry.ray_plane_intersection(
        origin=np.array([0.0, 0.0, 1.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        plane_z=0.0,
    )
    assert hit is None


def _rotation_about(axis, point, angle_deg):
    """A rigid motion that turns `angle_deg` about the line (axis, point)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    a = np.deg2rad(angle_deg)
    r = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = np.asarray(point, dtype=float) - r @ np.asarray(point, dtype=float)
    return t


def test_screw_axis_recovers_a_planted_axis():
    axis, point, angle = geometry.screw_axis(
        _rotation_about([0.0, 0.0, 1.0], [0.3, -0.2, 0.9], 25.0))
    assert np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-9)
    assert angle == pytest.approx(np.deg2rad(25.0))
    # The z of the point is free (any point on the line will do), so only the
    # perpendicular components are pinned.
    assert np.allclose(point[:2], [0.3, -0.2], atol=1e-9)


def test_screw_axis_recovers_an_axis_that_misses_the_origin():
    """The case that matters on this arm: the base yaw axis is 81 mm from the
    model's origin, so an axis far off-origin must still come back exactly."""
    axis, point, _ = geometry.screw_axis(
        _rotation_about([0.0, 0.0, 1.0], [0.0778, 0.0236, 0.0], 4.0))
    assert np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(point[:2], [0.0778, 0.0236], atol=1e-9)


def test_screw_axis_signs_the_axis_by_the_direction_of_rotation():
    """Sign carries the sense of rotation, which is the whole point for a
    dir_sign check -- an unsigned axis cannot tell clockwise from anti."""
    pos, _, _ = geometry.screw_axis(_rotation_about([0.0, 1.0, 0.0], [0, 0, 0], 30.0))
    neg, _, _ = geometry.screw_axis(_rotation_about([0.0, 1.0, 0.0], [0, 0, 0], -30.0))
    assert np.allclose(pos, [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(neg, [0.0, -1.0, 0.0], atol=1e-9)


def test_screw_axis_on_a_tilted_off_origin_axis():
    axis_in = np.array([1.0, -2.0, 0.5])
    axis_in /= np.linalg.norm(axis_in)
    axis, point, angle = geometry.screw_axis(
        _rotation_about(axis_in, [0.11, 0.04, -0.07], 17.0))
    assert np.allclose(axis, axis_in, atol=1e-9)
    assert angle == pytest.approx(np.deg2rad(17.0))
    # The recovered point must lie ON the planted line.
    offset = point - np.array([0.11, 0.04, -0.07])
    assert np.allclose(offset - np.dot(offset, axis_in) * axis_in, 0.0, atol=1e-9)


def test_screw_axis_returns_no_angle_for_a_pure_translation():
    t = np.eye(4)
    t[:3, 3] = [0.1, 0.2, 0.3]
    _, _, angle = geometry.screw_axis(t)
    assert angle == 0.0

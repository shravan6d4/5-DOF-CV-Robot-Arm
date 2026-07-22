"""
Small 3D geometry toolkit: rigid transforms and ray/plane math, in pure numpy.

Why this exists: turning a brick's pixel into a world coordinate for an
eye-in-hand camera is a chain of coordinate-frame changes (camera -> gripper ->
robot base) plus one "where does this ray hit the table" intersection. Rather
than scatter matrix algebra through the calibration code, the primitives live
here where they can be unit-tested on their own.

Conventions used everywhere in this project:

- A "transform" is a 4x4 homogeneous matrix ``T`` that expresses a pose. When we
  name one ``T_a_b`` it means "the pose of frame b expressed in frame a", and it
  maps a point written in frame b into frame a:  ``p_a = T_a_b @ p_b``.
  Transforms chain the obvious way:  ``T_a_c = T_a_b @ T_b_c``.

- Orientation is roll/pitch/yaw in degrees, applied as intrinsic rotations about
  Z (yaw), then Y (pitch), then X (roll) — i.e. ``R = Rz(yaw) @ Ry(pitch) @
  Rx(roll)``. This is the common aerospace/robotics RPY convention. If your
  arm's kinematics use a different one, this is the single place to change it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from roll/pitch/yaw in degrees (see module doc)."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_transform(
    x: float,
    y: float,
    z: float,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Compose a 4x4 homogeneous transform from a translation + RPY orientation."""
    t = np.eye(4)
    t[:3, :3] = rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    t[:3, 3] = (x, y, z)
    return t


def transform_to_pose(t: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Decompose a 4x4 transform back into (x, y, z, roll, pitch, yaw) degrees.

    Inverse of make_transform. Near pitch = +/-90 deg the roll/yaw split is
    ambiguous (gimbal lock); we fall back to putting all the rotation into yaw,
    which is the harmless choice for a top-down pick where pitch stays near 0.
    """
    x, y, z = t[:3, 3]
    r = t[:3, :3]
    sp = -r[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = np.arcsin(sp)

    if np.isclose(abs(sp), 1.0):
        # Gimbal lock: cos(pitch) ~ 0, roll and yaw are coupled.
        roll = 0.0
        yaw = np.arctan2(-r[0, 1], r[1, 1])
    else:
        roll = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(r[1, 0], r[0, 0])

    return (float(x), float(y), float(z), *np.degrees([roll, pitch, yaw]))


def invert_transform(t: np.ndarray) -> np.ndarray:
    """Invert a rigid transform. Faster and more stable than np.linalg.inv here
    because we exploit that the rotation block is orthonormal."""
    r = t[:3, :3]
    p = t[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ p
    return inv


def transform_point(t: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a transform to a 3D point (translation included)."""
    point = np.asarray(point, dtype=float)
    homogeneous = np.array([point[0], point[1], point[2], 1.0])
    return (t @ homogeneous)[:3]


def transform_direction(t: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Apply only the rotation part of a transform to a 3D direction vector.

    Directions have no position, so the translation column must NOT be applied —
    a ray's heading doesn't change when you slide the origin around.
    """
    direction = np.asarray(direction, dtype=float)
    return t[:3, :3] @ direction


def ray_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_z: float,
) -> np.ndarray | None:
    """Intersect a ray with the horizontal plane ``z = plane_z``.

    Args:
        origin: (x, y, z) start of the ray, in the same frame as plane_z.
        direction: ray heading (need not be unit length).
        plane_z: height of the horizontal plane to hit (e.g. the tabletop).

    Returns:
        The (x, y, plane_z) intersection point, or None if the ray is parallel
        to the plane, or if the plane lies behind the ray (the camera is looking
        away from the table). Callers must treat None as "no valid target".
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)

    dz = direction[2]
    if np.isclose(dz, 0.0):
        return None  # ray runs parallel to the table, never meets it

    t = (plane_z - origin[2]) / dz
    if t <= 0:
        return None  # plane is behind the ray's start; not a real sighting

    return origin + t * direction


def triangulate_rays(
    origins: Sequence[np.ndarray],
    directions: Sequence[np.ndarray],
    min_normal_eig: float = 1e-6,
) -> tuple[np.ndarray, float] | None:
    """Least-squares closest point to two-or-more rays expressed in a shared frame.

    This is the second way to get depth from a single moving camera. ``ray_plane_
    intersection`` above assumes the point lies on a known plane (the table); this
    instead uses the SAME physical point seen from two-or-more known camera poses.
    Each view gives one ray that should pass through the point, and where the rays
    (nearly) meet is its 3D position — no plane assumption, so it works for a brick
    that is tilted, stacked, or of unknown height.

    Real rays never meet exactly (pixel noise, calibration error), so we return the
    point that minimises the sum of squared PERPENDICULAR distances to every ray —
    the maximum-likelihood estimate under isotropic Gaussian pixel noise. For a unit
    direction ``d``, the squared perpendicular distance from a point ``X`` to the line
    through ``C`` is ``|| (I - d dᵀ)(X - C) ||²``. Summing over the rays and setting the
    gradient to zero gives the normal equations ``( Σ Aᵢ ) X = Σ Aᵢ Cᵢ`` with the
    perpendicular projector ``Aᵢ = I - dᵢ dᵢᵀ`` — a single 3x3 solve.

    Args:
        origins: ray start points, each shape ``(3,)`` (e.g. camera centres in base).
        directions: ray headings, each shape ``(3,)``; need not be unit — normalised
            internally.
        min_normal_eig: degeneracy floor on the normal matrix's smallest eigenvalue.
            For two rays that eigenvalue is ``1 - |cos θ|`` (θ = angle between them), so
            it → 0 as the rays become parallel — i.e. as the stereo baseline / parallax
            vanishes and depth becomes unrecoverable. Below this we return ``None``
            instead of solving a near-singular system and returning a blown-up point.

    Returns:
        ``(point, rms_residual)`` — the ``(3,)`` closest point and the root-mean-square
        perpendicular distance from it to the rays (a direct, same-units measure of how
        well the rays actually agreed; ~0 means they genuinely intersect there). Or
        ``None`` if the rays are too parallel to triangulate (see ``min_normal_eig``).

    Raises:
        ValueError: if fewer than two rays are given, the two lists differ in length,
            or any direction is the zero vector.
    """
    if len(origins) != len(directions):
        raise ValueError("origins and directions must have the same length")
    if len(origins) < 2:
        raise ValueError("triangulation needs at least two rays")

    normal = np.zeros((3, 3))
    rhs = np.zeros(3)
    projectors: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for origin, direction in zip(origins, directions):
        c = np.asarray(origin, dtype=float).reshape(3)
        d = np.asarray(direction, dtype=float).reshape(3)
        norm = np.linalg.norm(d)
        if np.isclose(norm, 0.0):
            raise ValueError("ray direction must be a non-zero vector")
        d = d / norm
        a = np.eye(3) - np.outer(d, d)  # projector onto the plane perpendicular to d
        normal += a
        rhs += a @ c
        projectors.append(a)
        centers.append(c)

    # Degeneracy guard: the smallest eigenvalue of the symmetric PSD normal matrix
    # → 0 as the rays become parallel. Reject before solving so we never push a
    # right-hand side through a near-singular matrix and hand back a wild point.
    smallest_eig = float(np.linalg.eigvalsh(normal)[0])
    if smallest_eig < min_normal_eig:
        return None

    point = np.linalg.solve(normal, rhs)

    # RMS perpendicular distance from the solved point to each ray.
    sum_sq = 0.0
    for a, c in zip(projectors, centers):
        perp = a @ (point - c)
        sum_sq += float(perp @ perp)
    rms_residual = float(np.sqrt(sum_sq / len(centers)))

    return point, rms_residual

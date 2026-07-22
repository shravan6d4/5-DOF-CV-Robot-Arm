"""
Phase 2: map a brick's pixel location to a coordinate in the robot base frame.

The camera is mounted on the arm (eye-in-hand), so a pixel alone is not enough —
we also need to know where the camera was when the frame was taken. That arrives
at runtime as the gripper's pose (the arm's forward kinematics). Putting it
together, for one detected pixel:

  1. pixel  -> ray in the CAMERA frame            (camera_model, intrinsics)
  2. ray    -> ray in the BASE frame              (T_base_camera transform)
              T_base_camera = T_base_gripper @ T_gripper_camera
              where T_base_gripper is supplied by the robot code at capture time
              and T_gripper_camera is the fixed hand-eye calibration.
  3. ray    -> point on the table                 (intersect with z = table_z)

A single camera can't recover depth from one pixel on its own; step 3 is what
supplies it, by assuming the brick lies on the known table plane. That is exactly
true for bricks sitting on a flat table, which is our whole scenario.

The class deliberately takes the gripper pose as a 4x4 matrix (not a robot Pose
object) so this module stays independent of the robot_interface package — the
orchestrator does the Pose -> matrix conversion.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import geometry
from vision_pipeline.calibration.camera_model import CameraIntrinsics, load_intrinsics


def load_hand_eye(path: str | Path | None = None) -> np.ndarray:
    """Load the fixed gripper->camera transform (T_gripper_camera) as a 4x4.

    File format is a JSON list-of-lists (row-major 4x4). A missing file returns
    the identity, so the pipeline runs before hand-eye calibration exists — with
    the (wrong but harmless-for-plumbing) assumption that the camera sits exactly
    at the gripper origin.
    """
    if path is None:
        path = config.HAND_EYE_PATH
    path = Path(path)
    if not path.exists():
        return np.eye(4)

    matrix = np.array(json.loads(path.read_text()), dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Hand-eye file {path} must contain a 4x4 matrix, got {matrix.shape}.")
    return matrix


def save_hand_eye(t_gripper_camera: np.ndarray, path: str | Path) -> None:
    """Write a 4x4 gripper->camera transform to JSON (after calibrating it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(np.asarray(t_gripper_camera, dtype=float).tolist(), indent=2))


@dataclass
class TriangulationResult:
    """Outcome of triangulating one physical point from two-or-more camera views.

    point_base : the recovered (x, y, z) in the robot base frame, meters.
    residual_m : RMS perpendicular distance from that point to the view rays. A
                 direct 'how well did the rays actually meet' number — near zero
                 means a clean intersection, large means noise / bad calibration.
    parallax_deg : the smallest angle between any pair of view rays. This is the
                 depth-conditioning number: small angle (rays nearly parallel) =
                 weak baseline = untrustworthy depth. Callers gate on it.
    num_views  : how many views were combined (>= 2).
    """

    point_base: np.ndarray
    residual_m: float
    parallax_deg: float
    num_views: int


class PixelToWorldCalibrator:
    """Converts brick pixels to robot-base-frame coordinates for an eye-in-hand rig.

    Two ways to recover depth, both provided here:
      - pixel_to_world:     single view, assumes the point lies on the table plane.
      - triangulate_pixels: two-or-more views, recovers true depth by triangulation
                            (no plane assumption) — for tilted/stacked/unknown-height
                            bricks the single-view path can't handle.

    Usage:
        calib = PixelToWorldCalibrator()
        # t_base_gripper: 4x4 from the arm's forward kinematics at capture time
        world_xyz = calib.pixel_to_world(detection.centroid_px, t_base_gripper)
        yaw = calib.pixel_angle_to_world_yaw(
            detection.centroid_px, detection.angle_deg, t_base_gripper
        )
        # two-view depth from two arm poses:
        result = calib.triangulate_pixels([(px1, pose1), (px2, pose2)])
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        t_gripper_camera: np.ndarray | None = None,
        table_z: float = config.TABLE_Z_IN_BASE,
    ) -> None:
        self.intrinsics = intrinsics if intrinsics is not None else load_intrinsics()
        self.t_gripper_camera = (
            np.asarray(t_gripper_camera, dtype=float)
            if t_gripper_camera is not None
            else load_hand_eye()
        )
        self.table_z = table_z

    def _base_camera_transform(self, t_base_gripper: np.ndarray) -> np.ndarray:
        """Chain the arm's gripper pose with the fixed hand-eye offset."""
        return np.asarray(t_base_gripper, dtype=float) @ self.t_gripper_camera

    def pixel_to_world(
        self,
        pixel_xy: tuple[float, float],
        t_base_gripper: np.ndarray,
    ) -> np.ndarray | None:
        """Convert a pixel to an (x, y, z) point on the table, in the base frame.

        Args:
            pixel_xy: (u, v) pixel of the brick (e.g. its centroid).
            t_base_gripper: 4x4 pose of the gripper in the base frame, from the
                arm's forward kinematics at the moment the frame was captured.

        Returns:
            An (x, y, z) numpy array on the table plane, or None if the back-
            projected ray never meets the table (camera aimed away from it).
        """
        t_base_camera = self._base_camera_transform(t_base_gripper)
        ray_cam = self.intrinsics.pixel_to_ray(pixel_xy)

        origin_base = t_base_camera[:3, 3]
        direction_base = geometry.transform_direction(t_base_camera, ray_cam)

        return geometry.ray_plane_intersection(origin_base, direction_base, self.table_z)

    def pixel_angle_to_world_yaw(
        self,
        pixel_xy: tuple[float, float],
        angle_deg: float,
        t_base_gripper: np.ndarray,
        pixel_step: float = 20.0,
    ) -> float | None:
        """Convert a brick's in-image rotation to a gripper yaw in the base frame.

        The detector reports the brick's orientation as an angle in the image.
        The gripper needs a yaw about the base Z axis. We can't just reuse the
        image angle because the camera may be rotated relative to the base — so
        we take two points along the brick's axis in the image, project BOTH onto
        the table, and measure the angle of the resulting world-space segment.

        Args:
            pixel_xy: brick centroid in pixels.
            angle_deg: brick orientation in the image (e.g. minAreaRect angle).
            t_base_gripper: gripper pose in base frame (forward kinematics).
            pixel_step: how far along the axis (in pixels) to sample the second
                point. Larger is more numerically stable; must stay on the brick.

        Returns:
            Yaw in degrees in [-180, 180], or None if either point misses the
            table plane.
        """
        theta = np.radians(angle_deg)
        offset = np.array([np.cos(theta), np.sin(theta)]) * pixel_step
        p1_px = (float(pixel_xy[0]), float(pixel_xy[1]))
        p2_px = (float(pixel_xy[0] + offset[0]), float(pixel_xy[1] + offset[1]))

        p1 = self.pixel_to_world(p1_px, t_base_gripper)
        p2 = self.pixel_to_world(p2_px, t_base_gripper)
        if p1 is None or p2 is None:
            return None

        delta = p2 - p1
        return float(np.degrees(np.arctan2(delta[1], delta[0])))

    def triangulate_pixels(
        self,
        views: Sequence[tuple[tuple[float, float], np.ndarray]],
    ) -> TriangulationResult | None:
        """Recover a point's true 3D position (base frame) from the SAME point seen
        in two-or-more views — no table-plane assumption, unlike pixel_to_world.

        Because the eye-in-hand camera moves with the arm and its pose is known at
        every instant (forward kinematics @ fixed hand-eye), two shots of the same
        brick from two arm poses form a stereo pair with a known baseline. Each view
        contributes one back-projected ray in the base frame; their closest common
        point is the brick. This is the depth path for bricks that are NOT flat on
        the known table (tilted, stacked, unknown height).

        The per-view work mirrors pixel_to_world exactly — same camera-centre origin
        and same back-projected ray direction in the base frame — the only difference
        is that instead of intersecting one ray with the table plane, we intersect
        the rays with each other (geometry.triangulate_rays).

        Args:
            views: two-or-more (pixel_xy, t_base_gripper) pairs — each the point's
                pixel and the 4x4 gripper pose (arm FK) at that frame's capture
                instant. The arm MUST have moved between captures to create parallax;
                more views improve robustness.

        Returns:
            A TriangulationResult (point + residual + parallax + view count), or None
            if the views' rays are too close to parallel to triangulate (insufficient
            baseline — see geometry.triangulate_rays). Callers should additionally gate
            on result.parallax_deg / result.residual_m before trusting the point.

        Raises:
            ValueError: if fewer than two views are supplied.
        """
        if len(views) < 2:
            raise ValueError("triangulation needs at least two views")

        origins: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        for pixel_xy, t_base_gripper in views:
            t_base_camera = self._base_camera_transform(t_base_gripper)
            ray_cam = self.intrinsics.pixel_to_ray(pixel_xy)
            origins.append(t_base_camera[:3, 3])
            directions.append(geometry.transform_direction(t_base_camera, ray_cam))

        result = geometry.triangulate_rays(origins, directions)
        if result is None:
            return None
        point, residual = result

        return TriangulationResult(
            point_base=point,
            residual_m=residual,
            parallax_deg=_min_pairwise_angle_deg(directions),
            num_views=len(views),
        )


def _min_pairwise_angle_deg(directions: Sequence[np.ndarray]) -> float:
    """Smallest angle (degrees) between any pair of the given direction vectors.

    This is the worst-conditioned parallax among the views — the number to gate on
    when deciding whether a triangulation has enough baseline to be trustworthy.
    """
    dirs = [np.asarray(d, dtype=float) / np.linalg.norm(d) for d in directions]
    smallest = 180.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            cos = float(np.clip(dirs[i] @ dirs[j], -1.0, 1.0))
            smallest = min(smallest, float(np.degrees(np.arccos(cos))))
    return smallest

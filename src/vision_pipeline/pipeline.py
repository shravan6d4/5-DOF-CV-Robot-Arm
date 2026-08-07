"""
Top-level orchestration: turn a camera frame into a completed brick pick.

This is the seam your robot code merges into. It wires the four pieces together
in the order the task demands:

    detect the brick  ->  locate it in the world  ->  plan the grasp  ->  move

and it depends only on the small interfaces each piece exposes, so every piece
stays swappable:

    - detector:  LegoBrickDetector (any object with .detect(frame_bgr))
    - calibrator: PixelToWorldCalibrator (pixel + gripper pose -> world point)
    - robot:     RobotInterface (SimRobot now, your real arm later)

`locate_brick` does the vision-only half (frame -> PickTarget) and is fully
testable without a robot backend that moves. `run_once` does the full loop
including commanding the arm. To integrate the real arm you implement
RobotInterface against your hardware and pass it in here — nothing in this file
needs to change.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vision_pipeline.calibration.pixel_to_world import PixelToWorldCalibrator
from vision_pipeline.detection.lego_detector import LegoBrickDetector
from vision_pipeline.detection.types import Detection
from vision_pipeline.planning.pick import PickTarget, plan_pick_sequence
from vision_pipeline.robot_interface.base import Pose, RobotInterface


class PickPipeline:
    """Detect a brick in a frame and (optionally) command an arm to pick it."""

    def __init__(
        self,
        detector: LegoBrickDetector | None = None,
        calibrator: PixelToWorldCalibrator | None = None,
        robot: RobotInterface | None = None,
    ) -> None:
        self.detector = detector or LegoBrickDetector()
        self.calibrator = calibrator or PixelToWorldCalibrator()
        # robot may be None if you only want the vision half (locate_brick).
        self.robot = robot

    def locate_brick(
        self,
        frame_bgr: np.ndarray,
        ee_pose: Pose,
    ) -> PickTarget | None:
        """Vision half: frame + current gripper pose -> a PickTarget, or None.

        Returns None when either no brick is found OR the brick's pixel can't be
        placed on the table (its back-projected ray misses the table plane —
        e.g. the camera is aimed away from the surface). Callers must handle the
        None; there isn't always a valid pick.
        """
        bricks = self.detector.detect(frame_bgr)
        if not bricks:
            return None

        brick = bricks[0]  # largest / most confident, already sorted by detector
        t_base_gripper = ee_pose.to_matrix()

        world = self.calibrator.pixel_to_world(brick.centroid_px, t_base_gripper)
        if world is None:
            return None

        yaw = self.calibrator.pixel_angle_to_world_yaw(
            brick.centroid_px, brick.angle_deg, t_base_gripper
        )
        if yaw is None:
            yaw = 0.0  # placement worked but the axis sample missed; grasp square

        return PickTarget(
            x=float(world[0]),
            y=float(world[1]),
            z=float(world[2]) + _pick_z_offset(),
            yaw_deg=yaw,
            num_studs=brick.num_studs,
        )

    def locate_brick_two_view(
        self,
        captures: Sequence[tuple[np.ndarray, Pose]],
    ) -> PickTarget | None:
        """Two-view (moving-camera stereo) location: recover the brick's true depth
        by triangulation instead of assuming it lies flat on the table plane.

        locate_brick works only when the brick sits at the known table height — a
        single camera can't recover depth, so it assumes the plane. This instead
        detects the brick in each of two-or-more (frame, gripper-pose) captures,
        takes its centroid in each, and triangulates those centroids into a real
        base-frame point. Use it when the brick may be tilted, stacked, or of unknown
        height. The arm must MOVE between captures (a real baseline) to create the
        parallax triangulation needs.

        Assumes the same single brick is the largest red-with-studs region in every
        frame (true for one brick on a table) — the centroids across views must
        correspond to the same physical point for triangulation to be valid.

        Args:
            captures: two-or-more (frame_bgr, ee_pose) pairs — each frame paired with
                the gripper pose (forward kinematics) at that frame's capture instant.

        Returns:
            A PickTarget at the triangulated position, or None if the brick isn't
            found in every frame, or the triangulation is too weak to trust (parallax
            below TWO_VIEW_MIN_PARALLAX_DEG, or ray residual above TWO_VIEW_MAX_
            RESIDUAL_M — i.e. the rays didn't actually meet).

        Raises:
            ValueError: if fewer than two captures are supplied.
        """
        from vision_pipeline import config

        if len(captures) < 2:
            raise ValueError("two-view location needs at least two captures")

        views: list[tuple[tuple[float, float], np.ndarray]] = []
        num_studs = 0
        for frame_bgr, ee_pose in captures:
            bricks = self.detector.detect(frame_bgr)
            if not bricks:
                return None  # brick must be visible in EVERY view to triangulate
            brick = bricks[0]  # largest / most confident (detector-sorted)
            num_studs = max(num_studs, brick.num_studs)
            # ACCEPTS A RAW 4x4 AS WELL AS A Pose, and callers with a matrix
            # should pass it. Pose is (xyz, roll/pitch/yaw), so handing one in
            # forces matrix -> RPY -> matrix, and geometry.transform_to_pose
            # pins roll = 0 near pitch = +-90 deg -- which is exactly where a
            # top-down tool sits. That destroys the camera's orientation about
            # its own axis, which is precisely what a back-projected ray needs;
            # it is the same gimbal-lock trap that poisoned the hand-eye solve
            # (see CLAUDE.md). Widened rather than replaced: every existing
            # caller still passes a Pose and gets exactly what it got before.
            views.append((brick.centroid_px,
                          ee_pose.to_matrix() if hasattr(ee_pose, "to_matrix")
                          else np.asarray(ee_pose, dtype=float)))

        result = self.calibrator.triangulate_pixels(views)
        if result is None:
            return None  # rays too parallel — insufficient baseline

        # Trust gates: weak parallax or rays that never met = don't act on it.
        if result.parallax_deg < config.TWO_VIEW_MIN_PARALLAX_DEG:
            return None
        if result.residual_m > config.TWO_VIEW_MAX_RESIDUAL_M:
            return None

        point = result.point_base
        return PickTarget(
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]) + _pick_z_offset(),
            # Triangulation recovers position, not orientation, and the 5-DOF arm
            # drops yaw anyway (top-down grasp of a near-symmetric brick).
            yaw_deg=0.0,
            num_studs=num_studs,
        )

    def run_once(self, frame_bgr: np.ndarray) -> PickTarget | None:
        """Full loop on one frame: locate a brick and command the arm to pick it.

        Requires a robot backend (raises if none was provided). Reads the
        gripper's current pose from the robot (forward kinematics), locates the
        brick relative to it, then executes the planned grasp sequence. Returns
        the PickTarget it acted on, or None if there was nothing to pick.
        """
        if self.robot is None:
            raise RuntimeError(
                "run_once needs a RobotInterface backend. Construct PickPipeline "
                "with robot=SimRobot() (or your real arm backend), or call "
                "locate_brick for the vision-only result."
            )

        ee_pose = self.robot.get_end_effector_pose()
        target = self.locate_brick(frame_bgr, ee_pose)
        if target is None:
            return None

        self.execute_pick(target)
        return target

    def execute_pick(self, target: PickTarget) -> None:
        """Run the planned grasp sequence on the robot backend."""
        if self.robot is None:
            raise RuntimeError("execute_pick needs a RobotInterface backend.")

        for step in plan_pick_sequence(target):
            self.robot.send_target_pose(step.pose)
            if step.gripper_closed is not None:
                self.robot.set_gripper(step.gripper_closed)


def _pick_z_offset() -> float:
    # Imported at call time so tests that monkeypatch config still take effect.
    from vision_pipeline import config

    return config.PICK_Z_OFFSET

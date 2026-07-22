"""
Two-view (moving-camera stereo) pick demo — run BY HAND with the real arm + camera.

Unlike scripts/run_pick_demo.py (single view, assumes the brick is flat on the known
table plane), this recovers the brick's TRUE 3D position by triangulation: it grabs a
frame, shifts the arm sideways by a known baseline, grabs a second frame, and
intersects the two back-projected rays. That means it works for a brick that is tilted,
stacked, or of unknown height.

    !!! SAFETY — THIS DRIVES THE REAL ARM !!!
    It commands a sideways move and then a full pick sequence. Keep the workspace clear
    and the e-stop within reach. Use --dry-run first: it locates the brick and prints the
    triangulation WITHOUT executing the grasp.

Prereqs, in order: MATLAB ik_fk_server running (for HardwareRobot FK/IK); servo bus
wired; and — to be accurate — data/camera_intrinsics.json and data/hand_eye.json from
the two calibration scripts (it will still run on placeholders, just imprecisely).

    python scripts/run_two_view_pick.py --dry-run     # locate only, no grasp
    python scripts/run_two_view_pick.py               # locate AND pick
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.capture.camera import Camera
from vision_pipeline.pipeline import PickPipeline
from vision_pipeline.planning.pick import PickTarget
from vision_pipeline.robot_interface.base import Pose
from vision_pipeline.robot_interface.hardware import HardwareRobot

SETTLE_SECONDS = 1.0


def _capture(pipeline, camera, robot):
    """Grab one (frame, live-gripper-pose) view. Pose comes from verified FK."""
    pose = robot.get_end_effector_pose()
    frame = camera.read_frame()
    return frame, pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument(
        "--baseline", type=float, default=config.TWO_VIEW_BASELINE_M,
        help="Sideways shift (m, base +y) between the two shots (default from config).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Locate + print the triangulation, but do NOT execute the grasp.",
    )
    args = parser.parse_args()

    with HardwareRobot() as robot:
        pipeline = PickPipeline(robot=robot)
        with Camera(camera_index=args.camera_index) as camera:
            # View 1 at the arm's current pose.
            frame1, pose1 = _capture(pipeline, camera, robot)
            print(f"View 1 gripper pose: {pose1}")

            # Shift sideways by the baseline to create parallax, then View 2.
            pose2_target = Pose(
                x=pose1.x, y=pose1.y + args.baseline, z=pose1.z,
                roll_deg=pose1.roll_deg, pitch_deg=pose1.pitch_deg, yaw_deg=pose1.yaw_deg,
            )
            print(f"Shifting {args.baseline*1000:.0f} mm along +y for the second view...")
            if not robot.send_target_pose(pose2_target):
                print("Second-view move failed (unreachable?). Aborting.")
                sys.exit(1)
            time.sleep(SETTLE_SECONDS)
            frame2, pose2 = _capture(pipeline, camera, robot)
            print(f"View 2 gripper pose: {pose2}")

        # Detect + triangulate directly so we can print the quality numbers.
        views = []
        num_studs = 0
        for label, frame, pose in (("view1", frame1, pose1), ("view2", frame2, pose2)):
            bricks = pipeline.detector.detect(frame)
            if not bricks:
                print(f"No brick detected in {label}. Nothing to pick.")
                sys.exit(1)
            brick = bricks[0]
            num_studs = max(num_studs, brick.num_studs)
            views.append((brick.centroid_px, pose.to_matrix()))
            print(f"{label}: brick centroid {brick.centroid_px}, studs={brick.num_studs}")

        result = pipeline.calibrator.triangulate_pixels(views)
        if result is None:
            print("Triangulation degenerate (rays parallel — baseline too small). Aborting.")
            sys.exit(1)

        print(
            f"\nTriangulation: point(base) = "
            f"[{result.point_base[0]:.3f} {result.point_base[1]:.3f} {result.point_base[2]:.3f}] m, "
            f"parallax = {result.parallax_deg:.1f} deg, residual = {result.residual_m*1000:.1f} mm"
        )

        # Trust gates (same as PickPipeline.locate_brick_two_view).
        if result.parallax_deg < config.TWO_VIEW_MIN_PARALLAX_DEG:
            print(f"Parallax below {config.TWO_VIEW_MIN_PARALLAX_DEG} deg — depth untrustworthy. "
                  f"Increase --baseline. Aborting.")
            sys.exit(1)
        if result.residual_m > config.TWO_VIEW_MAX_RESIDUAL_M:
            print(f"Ray residual above {config.TWO_VIEW_MAX_RESIDUAL_M*1000:.0f} mm — the two views "
                  f"disagree (bad detection / calibration). Aborting.")
            sys.exit(1)

        point = result.point_base
        target = PickTarget(
            x=float(point[0]), y=float(point[1]),
            z=float(point[2]) + config.PICK_Z_OFFSET,
            yaw_deg=0.0, num_studs=num_studs,
        )
        print(f"Pick target: x={target.x:.3f} y={target.y:.3f} z={target.z:.3f} m")

        if args.dry_run:
            print("\n--dry-run: located only, no grasp executed.")
            return

        print("\nExecuting grasp sequence (hover -> descend -> close -> lift)...")
        pipeline.execute_pick(target)
        print("Done — grasp sequence sent to the arm.")


if __name__ == "__main__":
    main()

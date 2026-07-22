"""
End-to-end pick demo — runs the WHOLE pipeline against the simulated robot, so
you can watch detect -> world coordinate -> plan -> "move" happen with no arm
and (optionally) no camera.

This is the script to run to confirm the vision side is merge-ready: it exercises
every seam the real arm will plug into.

    # Fully headless: synthetic brick frame + simulated arm (no hardware needed)
    python scripts/run_pick_demo.py

    # Use a real photo as the frame instead
    python scripts/run_pick_demo.py --image "tests/sample_images/red lego brick 2.jpg"

    # Use the live webcam for the frame (still a simulated arm)
    python scripts/run_pick_demo.py --camera

When you have the real arm, replace `SimRobot()` below with your RobotInterface
backend and this same script drives real hardware.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline.capture.camera import Camera
from vision_pipeline.pipeline import PickPipeline
from vision_pipeline.robot_interface.base import Pose
from vision_pipeline.robot_interface.sim import SimRobot


def _synthetic_brick_frame() -> np.ndarray:
    """A gray frame with a red rectangle + fake studs, i.e. a detectable brick."""
    frame = np.full((480, 640, 3), fill_value=(120, 120, 120), dtype=np.uint8)
    cv2.rectangle(frame, (260, 180), (420, 320), color=(0, 0, 255), thickness=-1)
    for row_y in (220, 280):
        for col_x in (290, 340, 390):
            cv2.circle(frame, (col_x, row_y), 16, color=(0, 0, 200), thickness=-1)
            cv2.circle(frame, (col_x, row_y), 16, color=(0, 0, 120), thickness=2)
    return frame


def _get_frame(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return frame
    if args.camera:
        with Camera() as camera:
            return camera.read_frame()
    return _synthetic_brick_frame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="Use a static image as the frame.")
    source.add_argument("--camera", action="store_true", help="Grab one live webcam frame.")
    args = parser.parse_args()

    frame = _get_frame(args)

    # The simulated arm reports a fixed 'current' gripper pose. For an eye-in-hand
    # rig this is where the camera-carrying gripper is hovering as it looks down
    # at the table. The real arm reports this from its forward kinematics.
    robot = SimRobot(ee_pose=Pose(x=0.0, y=0.0, z=0.30, roll_deg=180.0), verbose=True)
    pipeline = PickPipeline(robot=robot)

    print("Running pick pipeline...\n")
    target = pipeline.run_once(frame)

    print()
    if target is None:
        print("No brick located — nothing to pick. (No detection, or ray missed the table.)")
    else:
        print(
            f"Picked brick at base-frame x={target.x:.3f} m, y={target.y:.3f} m, "
            f"z={target.z:.3f} m, yaw={target.yaw_deg:.1f} deg (studs={target.num_studs})."
        )
        print(
            f"Commanded {len(robot.log.poses)} poses and "
            f"{len(robot.log.gripper_states)} gripper actions; "
            f"gripper ended {'CLOSED' if robot.gripper_closed else 'OPEN'}."
        )


if __name__ == "__main__":
    main()

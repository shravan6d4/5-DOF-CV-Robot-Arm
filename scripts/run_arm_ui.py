"""Launch the arm observation/jog web dashboard.

A Flask page showing the live camera feed and, per joint J1-J6 (J6 = gripper),
its position (degrees + ticks) with jog buttons and a per-joint step-size slider
(in ticks). Meant for bring-up/testing: watching what the arm is doing and
nudging one joint at a time without going through the xyz IK path.

    # Mock backend (no hardware needed) - just a webcam for the camera panel
    python scripts/run_arm_ui.py

    # Mock backend, no camera either (dashboard shows a placeholder image)
    python scripts/run_arm_ui.py --no-camera

    # Real hardware: MATLAB IK/FK server not needed here (this UI never solves
    # IK - it jogs raw servo ticks directly), but the servo bus must be wired
    # and data/servo_calibration.json (or the config fallback) must be valid.
    python scripts/run_arm_ui.py --hardware --servo-port COM3

    # Overlay red-brick detection on the camera feed
    python scripts/run_arm_ui.py --overlay

Then open http://127.0.0.1:5000 (or --host/--port) in a browser.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.robot_interface.joint_controller import (
    MockJointController,
    ServoJointController,
)
from vision_pipeline.robot_interface.servo_driver import ServoBus
from vision_pipeline.webui.app import create_app


def _build_controller(args: argparse.Namespace):
    if not args.hardware:
        print("Using MockJointController (no hardware).")
        return MockJointController()

    print(f"Connecting to servo bus on {args.servo_port} @ {args.servo_baud} baud...")
    bus = ServoBus(args.servo_port, args.servo_baud)
    print("Servo bus connected.")
    return ServoJointController(bus)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hardware", action="store_true",
        help="Drive the real servo bus instead of the in-memory mock.",
    )
    parser.add_argument("--servo-port", default=config.SERVO_PORT, help="Serial port for --hardware (default: %(default)s).")
    parser.add_argument("--servo-baud", type=int, default=config.SERVO_BAUD, help="Baud rate for --hardware (default: %(default)s).")
    parser.add_argument(
        "--camera-index", type=int, default=config.CAMERA_INDEX,
        help="Webcam index for the live feed (default: %(default)s).",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Skip opening a camera; the feed panel shows a placeholder image.",
    )
    parser.add_argument("--overlay", action="store_true", help="Draw red-brick detection overlay on the camera feed.")
    parser.add_argument("--host", default=config.WEBUI_HOST, help="Host to bind (default: %(default)s).")
    parser.add_argument("--port", type=int, default=config.WEBUI_PORT, help="Port to bind (default: %(default)s).")
    args = parser.parse_args()

    controller = _build_controller(args)
    camera_index = None if args.no_camera else args.camera_index
    app = create_app(controller, camera_index=camera_index, enable_overlay=args.overlay)

    print(f"Dashboard at http://{args.host}:{args.port} (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

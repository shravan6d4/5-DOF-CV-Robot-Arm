"""
End-to-end validation of the pixel->world chain (Merge Path steps 1-3
combined) — run BY HAND with the real arm + camera, after BOTH
scripts/calibrate_camera_intrinsics.py and scripts/calibrate_hand_eye.py.

Answers the question the whole calibration exercise exists for: "can this rig
be commanded to any point the camera sees, in millimetres?" It picks ONE
ChArUco corner — a point whose true position is known from the board's own
measured geometry — and observes it from >= 2 arm poses. From that:

  1. SINGLE-VIEW check: back-projects the corner's pixel through
     PixelToWorldCalibrator.pixel_to_world (assumes the corner lies on
     config.TABLE_Z_IN_BASE) and compares it against an INDEPENDENT answer
     computed the other way — full FK -> hand-eye -> solvePnP board pose ->
     board-local corner geometry, which makes NO table-plane assumption.
     Their disagreement is the real error of intrinsics + hand-eye + table
     height, all at once.
  2. TWO-VIEW check: feeds the same views to
     PixelToWorldCalibrator.triangulate_pixels, which also assumes no table
     plane. Agreeing with (1) validates both paths independently.
  3. CROSS-POSE SPREAD: the board never moves, so every view's solvePnP-based
     T_base_board should agree. The spread is a direct calibration-quality
     number in mm, and its z also measures the table plane for free — compare
     against config.TABLE_Z_IN_BASE (currently a ruler-to-eye estimate, see
     CLAUDE.md's "Stage D" notes).
  4. Optional --goto-corner: commands the claw tip to the recovered corner
     position via HardwareRobot (solves IK + moves + reads back), so the
     final number is a ruler measurement against reality, not another
     internally-consistent computation.

Setup: tape all boards flat and rigid on the table (as for hand-eye), pick one
board + one of its corner IDs (printed by generate_charuco_board.py's
console output, or just look at the board: internal chessboard corners are
numbered row-major from the board's top-left).

    python scripts/validate_pixel_to_world.py --board 1 --corner-id 0

Controls (same jog primitive as jog_joint.py / calibrate_hand_eye.py — no IK
during capture): 1-5 = select joint, a/d = jog -/+ step, [/] = step size,
v = record a view (only when the target corner is visible), c = compute
report, q = abort.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import charuco, geometry
from vision_pipeline.calibration.camera_model import default_intrinsics, load_intrinsics
from vision_pipeline.calibration.hand_eye import detect_board_poses
from vision_pipeline.calibration.pixel_to_world import PixelToWorldCalibrator, load_hand_eye
from vision_pipeline.capture.camera import Camera
from vision_pipeline.robot_interface.base import Pose
from vision_pipeline.robot_interface.hardware import HardwareRobot
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

IK_JOINTS = range(1, 6)
DEFAULT_STEP_TICKS = 60
MIN_STEP_TICKS = 10
MAX_STEP_TICKS = 300


def _current_angles_rad(bus: ServoBus) -> list[float]:
    return [bus.ticks_to_rad(j, bus.read_position(j)) for j in IK_JOINTS]


def _detect_target(gray, board_idx, corner_id, detectors, intr):
    """Return (pixel_xy, T_cam_board) if the target board AND corner are both
    visible in this frame, else None. T_cam_board comes from solvePnP over
    the WHOLE visible board (more stable than a single point would be), and
    the corner's own pixel is read from the same detection pass so both
    numbers come from one consistent frame."""
    for idx, board, det in detectors:
        if idx != board_idx:
            continue
        corners, ids = charuco.detect(det, gray)
        if corners is None:
            return None
        ids_flat = ids.reshape(-1)
        matches = np.where(ids_flat == corner_id)[0]
        if len(matches) == 0:
            return None
        pixel = (float(corners[matches[0], 0, 0]), float(corners[matches[0], 0, 1]))
        board_poses = detect_board_poses(gray, intr, [(idx, board, det)])
        if board_idx not in board_poses:
            return None
        return pixel, board_poses[board_idx]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", type=int, default=1, help="1-based board number (default: 1)")
    parser.add_argument("--corner-id", type=int, default=0, help="ChArUco corner id on that board (default: 0)")
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--servo-port", default=config.SERVO_PORT)
    parser.add_argument("--servo-baud", type=int, default=config.SERVO_BAUD)
    parser.add_argument("--min-views", type=int, default=2, help="minimum recorded views before computing (default: 2)")
    parser.add_argument(
        "--goto-corner", action="store_true",
        help="after computing, command the claw tip to the recovered corner position for a ruler check",
    )
    args = parser.parse_args()

    board_idx = args.board - 1
    corner_id = args.corner_id

    intr = load_intrinsics()
    if np.allclose(intr.matrix, default_intrinsics().matrix):
        print("WARNING: camera intrinsics are still the config PLACEHOLDER — results are meaningless.")
    t_gripper_camera = load_hand_eye()
    if np.allclose(t_gripper_camera, np.eye(4)):
        print("WARNING: hand-eye is still the IDENTITY placeholder — results are meaningless.")

    board = charuco.build_board(board_idx)
    corner_ids_available = board.getChessboardCorners().shape[0]
    if not (0 <= corner_id < corner_ids_available):
        print(f"--corner-id must be 0..{corner_ids_available - 1} for board {args.board}.")
        sys.exit(1)
    corner_obj = board.getChessboardCorners()[corner_id]  # (3,), board-local, z=0

    detectors = charuco.build_detectors()
    calibrator = PixelToWorldCalibrator(intrinsics=intr, t_gripper_camera=t_gripper_camera)

    try:
        client = MatlabIKClient()
    except (ConnectionRefusedError, OSError):
        print(f"No MATLAB server on {config.MATLAB_SERVER_HOST}:{config.MATLAB_SERVER_PORT}.")
        print("Start it in MATLAB (matlab/ folder):  >> ik_fk_server")
        sys.exit(1)

    try:
        bus = ServoBus(args.servo_port, args.servo_baud)
    except Exception as e:
        print(f"Could not open the servo bus on {args.servo_port}: {e}")
        sys.exit(1)

    views: list[tuple[tuple[float, float], np.ndarray, np.ndarray]] = []  # (pixel, T_base_gripper, T_cam_board)
    active_joint = 1
    step_ticks = DEFAULT_STEP_TICKS

    print(f"Target: board #{args.board}, corner {corner_id} (board-local {corner_obj} m).")
    print("1-5 = select joint, a/d = jog -/+ step, [/] = step size, v = record view, c = compute, q = abort.")
    print("!!! Arm will MOVE on a/d. Keep the e-stop within reach. !!!")

    with client, bus, Camera(camera_index=args.camera_index) as camera:
        while True:
            frame = camera.read_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            target = _detect_target(gray, board_idx, corner_id, detectors, intr)

            display = frame.copy()
            if target is not None:
                px, _ = target
                cv2.drawMarker(display, (int(px[0]), int(px[1])), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(
                display,
                f"J{active_joint} step {step_ticks}  target visible: {target is not None}  views: {len(views)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
            cv2.imshow("Pixel-to-world validation", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Aborted by operator.")
                cv2.destroyAllWindows()
                sys.exit(1)

            elif key in tuple(ord(str(j)) for j in IK_JOINTS):
                active_joint = int(chr(key))
                print(f"  active joint: J{active_joint}")

            elif key == ord("["):
                step_ticks = max(MIN_STEP_TICKS, step_ticks - 10)
                print(f"  step: {step_ticks} ticks")

            elif key == ord("]"):
                step_ticks = min(MAX_STEP_TICKS, step_ticks + 10)
                print(f"  step: {step_ticks} ticks")

            elif key in (ord("a"), ord("d")):
                delta = -step_ticks if key == ord("a") else step_ticks
                current = bus.read_position(active_joint)
                try:
                    actual = bus.move_and_verify(active_joint, current + delta)
                    print(f"  J{active_joint}: {current} -> {actual} ticks ({delta:+d} requested)")
                except ServoSafetyError as e:
                    print(f"  REFUSED: {e}")

            elif key == ord("v"):
                if target is None:
                    print("  target corner not visible — not recorded.")
                    continue
                pixel, t_cam_board = target
                angles_rad = _current_angles_rad(bus)
                t_base_gripper = client.request_fk(angles_rad)
                views.append((pixel, t_base_gripper, t_cam_board))
                print(f"  recorded view {len(views)} at pixel {pixel}")

            elif key == ord("c"):
                if len(views) < args.min_views:
                    print(f"  only {len(views)} views (< {args.min_views}) — keep recording.")
                    continue
                break

    cv2.destroyAllWindows()

    # --- 1. Single-view vs the full FK->handeye->solvePnP independent answer
    print(f"\n=== Single-view check ({len(views)} views) ===")
    single_view_points = []
    independent_points = []
    for pixel, t_base_gripper, t_cam_board in views:
        single = calibrator.pixel_to_world(pixel, t_base_gripper)
        t_base_camera = t_base_gripper @ t_gripper_camera
        independent = geometry.transform_point(t_base_camera @ t_cam_board, corner_obj)
        single_view_points.append(single)
        independent_points.append(independent)
        if single is None:
            print(f"  ray never meets the table plane for this view — skipped.")
            continue
        err_mm = float(np.linalg.norm(single - independent)) * 1000.0
        print(f"  single-view {single} vs independent {independent} -> {err_mm:.1f} mm")

    # --- 2. Two-view triangulation (no table-plane assumption)
    print(f"\n=== Two-view triangulation ===")
    pixel_pose_views = [(pixel, t_base_gripper) for pixel, t_base_gripper, _ in views]
    tri = calibrator.triangulate_pixels(pixel_pose_views)
    if tri is None:
        print("  rays too close to parallel to triangulate — need more varied poses.")
    else:
        print(f"  point {tri.point_base}, residual {tri.residual_m * 1000:.1f} mm, "
              f"parallax {tri.parallax_deg:.1f} deg, {tri.num_views} views")
        print(f"  residual gate ({config.TWO_VIEW_MAX_RESIDUAL_M * 1000:.0f} mm): "
              f"{'PASS' if tri.residual_m <= config.TWO_VIEW_MAX_RESIDUAL_M else 'FAIL'}")
        print(f"  parallax gate ({config.TWO_VIEW_MIN_PARALLAX_DEG:.0f} deg): "
              f"{'PASS' if tri.parallax_deg >= config.TWO_VIEW_MIN_PARALLAX_DEG else 'FAIL'}")
        mean_independent = np.mean(independent_points, axis=0)
        err_mm = float(np.linalg.norm(tri.point_base - mean_independent)) * 1000.0
        print(f"  vs mean independent answer: {err_mm:.1f} mm")

    # --- 3. Cross-pose board spread + table-plane measurement
    print(f"\n=== Cross-pose board spread (board never moved) ===")
    board_origins = []
    for _pixel, t_base_gripper, t_cam_board in views:
        t_base_board = t_base_gripper @ t_gripper_camera @ t_cam_board
        board_origins.append(t_base_board[:3, 3])
    board_origins = np.array(board_origins)
    spread_mm = float(np.linalg.norm(board_origins.std(axis=0))) * 1000.0
    mean_origin = board_origins.mean(axis=0)
    print(f"  spread: {spread_mm:.1f} mm ({'ok' if spread_mm < 10 else 'HIGH — result is suspect'})")
    print(f"  mean board origin (base) [m]: {mean_origin}")
    print(f"  z vs config.TABLE_Z_IN_BASE ({config.TABLE_Z_IN_BASE:.4f} m): "
          f"{abs(mean_origin[2] - config.TABLE_Z_IN_BASE) * 1000:.1f} mm difference "
          f"(board/tape thickness not subtracted)")

    # --- 4. Optional physical goto
    if args.goto_corner:
        target_point = tri.point_base if tri is not None else mean_independent
        print(f"\n=== goto-corner ===")
        print(f"  commanding claw tip to {target_point} (base frame, m)")
        print("  This closes the servo bus/MATLAB connection used above and opens")
        print("  HardwareRobot fresh, so IK + servo commanding go through the merge-ready path.")
        client.close()
        bus.close()
        with HardwareRobot(servo_port=args.servo_port, servo_baud=args.servo_baud) as robot:
            pose = Pose(
                x=float(target_point[0]), y=float(target_point[1]), z=float(target_point[2]),
                roll_deg=config.PICK_ROLL_DEG, pitch_deg=config.PICK_PITCH_DEG,
            )
            print(f"  This WILL move the arm to {pose}. Claw clear of the table? Power within reach?")
            if input("  Type 'go' to proceed: ").strip().lower() != "go":
                print("  Aborted. Nothing commanded.")
                return
            ok = robot.send_target_pose(pose)
            print(f"  send_target_pose -> {ok}")
            if ok:
                print("  Now measure the physical miss with a ruler against the real corner and record it.")


if __name__ == "__main__":
    main()

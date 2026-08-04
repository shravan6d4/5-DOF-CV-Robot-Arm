"""Per-joint kinematic validation using the CAMERA as the measuring instrument.

This is Stage D's outstanding lateral probe, done with the ChArUco boards instead
of a ruler. Board spacing has been verified to reproduce the printed tiling
geometry to ~0.3%, so camera-measured motion is trustworthy ground truth.

For ONE joint at a time it: records a settled before-frame, jogs the joint by a
known tick delta, waits for the servo to actually stop, records a settled
after-frame, then compares two independent measurements of the SAME motion:

    FK      — how far MATLAB says the wrist moved (link geometry x joint angles)
    camera  — how far the board shifted within the camera, which for a stationary
              board equals how far the camera physically moved

Agreement means that joint's geometry and tick scale are right. A consistent
ratio != 1 localizes the error to that joint, which no whole-arm hand-eye solve
can do. Hand-eye calibration is meaningless until every jogged joint reads ~1.0,
because it assumes FK reports where the camera really is.

    python scripts/validate_joint_geometry.py --joint 1
    python scripts/validate_joint_geometry.py --joint 4 --ticks 120 --repeats 3

    !!! THIS DRIVES THE REAL ARM. Stay by the power cut. !!!
    Do NOT run it on J3 until its travel limits are settled (see CLAUDE.md).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import charuco
from vision_pipeline.calibration.camera_model import load_intrinsics
from vision_pipeline.calibration.hand_eye import detect_board_poses
from vision_pipeline.capture.camera import Camera
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

IK_JOINTS = range(1, 6)
SETTLE_S = 1.5  # let the servo physically stop AND the camera buffer flush


def _observe(cam, bus, client, intr, detectors):
    """Return (board poses, wrist FK) captured with the arm stationary.

    Frames are drained first: cv2.VideoCapture buffers several, so the newest
    read can lag the arm by 100ms+. Recording a stale frame against a fresh FK
    reading is exactly the mismatch this script exists to rule out.
    """
    for _ in range(6):
        frame = cam.read_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    poses = detect_board_poses(gray, intr, detectors)
    angles = [bus.ticks_to_rad(j, bus.read_position(j)) for j in IK_JOINTS]
    return poses, client.request_fk(angles)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", type=int, required=True, choices=list(IK_JOINTS))
    ap.add_argument("--ticks", type=int, default=100, help="tick delta per step (default 100)")
    ap.add_argument("--repeats", type=int, default=3, help="how many steps to measure (default 3)")
    ap.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()

    if args.joint == 3:
        print("REFUSED: J3's safe travel range is still unresolved (CLAUDE.md).")
        sys.exit(1)

    intr = load_intrinsics()
    detectors = charuco.build_detectors()

    try:
        client = MatlabIKClient()
    except (ConnectionRefusedError, OSError):
        print("No MATLAB server. Start ik_fk_server in the matlab/ folder.")
        sys.exit(1)

    bus = ServoBus(config.SERVO_PORT, config.SERVO_BAUD)
    deg = args.ticks / config.SERVO_CALIBRATION_FALLBACK[str(args.joint)]["ticks_per_rad"]
    print(f"J{args.joint}: {args.repeats} steps of {args.ticks:+d} ticks "
          f"(~{np.degrees(deg):.1f} deg each). Arm WILL move.\n")

    ratios = []
    with client, bus, Camera(camera_index=args.camera_index) as cam:
        before_poses, before_fk = _observe(cam, bus, client, intr, detectors)
        if not before_poses:
            print("No board visible. Aim the camera at the boards and retry.")
            sys.exit(1)

        for step in range(args.repeats):
            start = bus.read_position(args.joint)
            try:
                bus.move_and_verify(args.joint, start + args.ticks)
            except ServoSafetyError as e:
                print(f"  REFUSED: {e}")
                break
            time.sleep(SETTLE_S)

            after_poses, after_fk = _observe(cam, bus, client, intr, detectors)
            shared = sorted(set(before_poses) & set(after_poses))
            if not shared:
                print(f"  step {step+1}: lost sight of every board — stopping.")
                break

            fk_mm = float(np.linalg.norm(after_fk[:3, 3] - before_fk[:3, 3])) * 1000
            # Camera displacement must be measured as its position in the FIXED
            # board frame. The board's position within the camera frame changes
            # when the camera merely ROTATES -- at 9 deg with a board 400 mm out
            # that is ~60 mm of pure artifact -- so it cannot be used here.
            cam_mm = float(np.mean([
                np.linalg.norm(
                    (-after_poses[b][:3, :3].T @ after_poses[b][:3, 3])
                    - (-before_poses[b][:3, :3].T @ before_poses[b][:3, 3])
                )
                for b in shared
            ])) * 1000
            rot = np.degrees(np.linalg.norm(
                cv2.Rodrigues((np.linalg.inv(after_fk) @ before_fk)[:3, :3])[0]))

            # The SAME rotation, measured by the camera against the fixed board.
            # Every radius below is a distance divided by this angle, so if FK
            # understates it the radii inflate and the joint looks broken when
            # it is not. FK is not allowed to be the only witness to its own
            # rotation.
            cam_rot = float(np.mean([
                np.degrees(np.linalg.norm(cv2.Rodrigues(
                    after_poses[b][:3, :3] @ before_poses[b][:3, :3].T)[0]))
                for b in shared
            ]))

            # Both wrist and camera swing about the SAME joint axis, so each one's
            # travel is proportional to its radius from it. They legitimately
            # differ -- but only by however far the camera sits from the wrist,
            # which is physically bounded. Exceeding that bound is a real fault.
            chord = 2.0 * np.sin(np.radians(rot) / 2.0)
            r_wrist = fk_mm / chord if chord > 1e-6 else float("nan")
            r_cam = cam_mm / chord if chord > 1e-6 else float("nan")
            gap = abs(r_cam - r_wrist)
            ratios.append(gap)
            print(f"  step {step+1}: FK {fk_mm:6.1f} mm | camera {cam_mm:6.1f} mm")
            print(f"           rotation: FK says {rot:5.2f} deg, camera says {cam_rot:5.2f} deg"
                  f"  (ratio {cam_rot/rot if rot > 1e-6 else float('nan'):4.2f})"
                  f"{'   <-- FK UNDERSTATES ROTATION' if cam_rot > rot * 1.15 else ''}")
            print(f"           radius from joint axis: wrist {r_wrist:6.1f} mm, camera {r_cam:6.1f} mm"
                  f"  -> camera >= {gap:.0f} mm from wrist")

            before_poses, before_fk = after_poses, after_fk

    good = [r for r in ratios if np.isfinite(r)]
    if not good:
        print("\nNo usable measurements.")
        return
    mean_gap = float(np.mean(good))
    print(f"\nJ{args.joint}: camera sits ~{mean_gap:.0f} mm from this joint's axis.")
    print("\nJudge this joint on the ROTATION RATIO above, not on that distance.")
    print("  ratio ~1.0 -> FK and the camera agree on how far the joint turned, so")
    print("               this joint's tick scale and axis are correct.")
    print("  ratio > 1  -> FK understates the rotation: tick scale or gearing is wrong.")
    print("\nThe distance is informational only. It is measured from FK's Body08")
    print("origin, which is wherever the CAD import placed that body's frame -- NOT")
    print("necessarily a joint centre you can put a ruler on. Comparing it against a")
    print("hand measurement produced a false 'IMPOSSIBLE' verdict on 2026-08-04.")


if __name__ == "__main__":
    main()

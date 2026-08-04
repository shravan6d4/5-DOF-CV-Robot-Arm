"""
Eye-in-hand hand-eye calibration — run BY HAND with the real arm + camera.

This is Merge Path step 2: it solves the FIXED gripper->camera offset (T_gripper_camera)
and writes data/hand_eye.json, replacing the identity placeholder ("camera sits exactly
at the wrist"). Until this exists, every pixel->world result is off by however far the
camera really sits from the wrist. Run scripts/calibrate_camera_intrinsics.py FIRST —
this uses the intrinsics for solvePnP and is only as good as they are.

    !!! SAFETY — THIS DRIVES THE REAL ARM !!!
    Jogs one joint at a time by a raw tick delta (same primitive as
    scripts/jog_joint.py — no IK involved). Stay next to the e-stop / power
    switch, keep the workspace clear, and watch every move.

Setup: tape all CALIB_BOARD_COUNT distinct ChArUco boards (scripts/generate_charuco_board.py)
FLAT and RIGID across the tabletop the arm's wrist camera can see — they must NOT move for
the whole session, they are the stationary references the arm orbits. Run
calibrate_camera_intrinsics.py first; this uses those intrinsics for solvePnP. Then:

    python scripts/calibrate_hand_eye.py

Controls: digit keys 1-5 select the active joint; 'a'/'d' jog it by the current tick
step (down/up); '['/']' change the step size; 'r' records a sample from every board
currently visible; 'c' computes & saves; 'q' aborts.

Why JOINT-space jogs, not Cartesian IK targets (as an earlier version of this script
did): the arm works close in (tip x ~ 75 mm), where small Cartesian moves demand large
J1/J5 swings — see CLAUDE.md's "back probe" note — so a Cartesian sweep mostly gets
refused by config.SERVO_MAX_MOVE_DELTA_TICKS. Jogging J4/J5 directly also gives far
better WRIST ROTATION diversity (they rotate the wrist while barely translating it),
which is what calibrateHandEye actually needs to be well-conditioned — a translation-only
sweep leaves the rotation underconstrained even when every move succeeds.

Why the arm pose is read straight from FK, never through Pose: geometry.transform_to_pose
forces roll=0 near pitch=+-90 deg (gimbal lock), and a top-down tool orientation sits close
to exactly that singularity. Rotation error is what poisons calibrateHandEye, so this script
takes the 4x4 directly from MatlabIKClient.request_fk and never routes it through Pose.

Why boards are bucketed and solved independently: the workspace tiles CALIB_BOARD_COUNT
distinct boards, so one frame can see several. Each is an independent stationary target;
solving per board and comparing lets two boards' agreement corroborate the result in a way
TSAI-vs-PARK (which shares its input data) cannot. See calibration/hand_eye.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import charuco
from vision_pipeline.calibration.camera_model import default_intrinsics, load_intrinsics
from vision_pipeline.calibration.hand_eye import (
    HandEyeAccumulator,
    cross_board_agreement_mm,
    detect_board_poses,
    load_samples,
    rotation_axis_spread_deg,
    save_samples,
    select_best,
)
from vision_pipeline.calibration.pixel_to_world import save_hand_eye
from vision_pipeline.capture.camera import Camera
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

IK_JOINTS = range(1, 6)  # J1..J5; J6 is the gripper, not part of hand-eye
DEFAULT_STEP_TICKS = 60
MIN_STEP_TICKS = 10
MAX_STEP_TICKS = 300


def _current_angles_rad(bus: ServoBus) -> list[float]:
    """J1..J5 angles in radians, straight from a servo read-back — no Pose involved."""
    return [bus.ticks_to_rad(j, bus.read_position(j)) for j in IK_JOINTS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--servo-port", default=config.SERVO_PORT)
    parser.add_argument("--servo-baud", type=int, default=config.SERVO_BAUD)
    parser.add_argument("--out", default=config.HAND_EYE_PATH)
    parser.add_argument("--min-samples", type=int, default=config.CALIB_HAND_EYE_MIN_SAMPLES)
    parser.add_argument(
        "--samples", default="data/hand_eye_samples.json",
        help="raw samples file; rewritten after every 'r' so a crashed session is not lost",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="ignore any existing --samples file instead of resuming from it",
    )
    args = parser.parse_args()

    intr = load_intrinsics()
    if np.allclose(intr.matrix, default_intrinsics().matrix):
        print("WARNING: camera intrinsics are still the config PLACEHOLDER.")
        print("         Run scripts/calibrate_camera_intrinsics.py first — hand-eye")
        print("         accuracy is bounded by intrinsics accuracy. Continuing anyway.")

    detectors = charuco.build_detectors()

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

    acc = HandEyeAccumulator(min_samples=args.min_samples)
    if not args.fresh:
        try:
            acc = load_samples(args.samples, min_samples=args.min_samples)
            print(f"Resumed {sum(acc.counts().values())} samples from {args.samples} "
                  f"-> counts {acc.counts()}")
            print("(pass --fresh to start empty instead)")
        except FileNotFoundError:
            pass

    active_joint = 1
    step_ticks = DEFAULT_STEP_TICKS

    print(f"{config.CALIB_BOARD_COUNT} boards expected, {args.min_samples} samples/board minimum.")
    print("1-5 = select joint, a/d = jog -/+ step, [/] = step size, r = record, c = compute, q = abort.")
    print("!!! Arm will MOVE on a/d. Keep the e-stop within reach. !!!")

    with client, bus, Camera(camera_index=args.camera_index) as camera:
        while True:
            frame = camera.read_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            board_poses = detect_board_poses(gray, intr, detectors)

            display = frame.copy()
            for idx, _board, det in detectors:
                corners, ids = charuco.detect(det, gray)
                if corners is not None and len(corners) > 0:
                    try:
                        cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
                    except cv2.error:
                        pass

            counts = acc.counts()
            counts_str = ", ".join(f"#{i+1}:{counts.get(i, 0)}" for i in range(config.CALIB_BOARD_COUNT))
            cv2.putText(
                display,
                f"J{active_joint} step {step_ticks}  visible: {sorted(board_poses)}  samples [{counts_str}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
            cv2.imshow("Hand-eye calibration", display)
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

            elif key == ord("r"):
                if not board_poses:
                    print("  no board visible with enough corners — not recorded.")
                    continue
                angles_rad = _current_angles_rad(bus)
                t_base_gripper = client.request_fk(angles_rad)
                new_counts = acc.add(board_poses, t_base_gripper)
                save_samples(acc, args.samples)
                print(f"  recorded boards {sorted(board_poses)} -> counts {new_counts}")

            elif key == ord("c"):
                break

    cv2.destroyAllWindows()

    solvable = acc.solvable_boards()
    if not solvable:
        print(f"No board reached {args.min_samples} samples. Aborting without saving.")
        sys.exit(1)

    results = acc.solve_all()
    for idx, r in results.items():
        print(f"\nBoard #{idx+1}: {r.n_samples} samples")
        print(f"  TSAI vs PARK translation disagreement: {r.tsai_park_disagreement_mm:.1f} mm "
              f"({'ok' if r.tsai_park_disagreement_mm < 5 else 'HIGH — add more/varied poses'})")
        # Rotation is checked separately because translation agreement alone does
        # NOT imply the orientation is right — see hand_eye.rotation_angle_deg.
        print(f"  TSAI vs PARK ROTATION disagreement: {r.tsai_park_rotation_deg:.1f} deg "
              f"({'ok' if r.tsai_park_rotation_deg < 2 else 'HIGH — orientation is not trustworthy'})")
        axis_spread = rotation_axis_spread_deg(acc, idx)
        if axis_spread is None:
            print("  rotation-axis diversity: TOO FEW MOTIONS to assess")
        else:
            print(f"  rotation-axis diversity: {axis_spread:.0f} deg "
                  f"({'ok' if axis_spread > 30 else 'DEGENERATE — jog a different joint'})")
        print(f"  board base-frame position spread: {r.board_spread_mm:.1f} mm "
              f"({'ok' if r.board_spread_mm < 10 else 'HIGH — result is suspect'})")
        print(f"  mean board origin (base) [m]: {r.mean_board_origin_base}")

    agreement_mm = cross_board_agreement_mm(results)
    if agreement_mm is not None:
        print(f"\nCross-board agreement (max pairwise translation disagreement): "
              f"{agreement_mm:.1f} mm ({'ok' if agreement_mm < 5 else 'HIGH — boards disagree'})")
    else:
        print("\nOnly one board solved — no cross-board agreement check available.")

    best = select_best(results)

    # --- physical plausibility, independent of every internal statistic -------
    # A solve can be perfectly self-consistent and still describe a camera that
    # cannot exist on this arm. Both checks below caught a bad solve on
    # 2026-08-04 that passed the translation gate at 3.0 mm.
    print("\n=== physical plausibility ===")
    offset_mm = float(np.linalg.norm(best.t_gripper_camera_tsai[:3, 3])) * 1000.0
    print(f"  camera is {offset_mm:.0f} mm from the wrist — compare against a ruler "
          f"(claw tip sits ~70 mm out).")

    # The camera looks at the table, so its optical axis must point DOWNWARD in
    # the base frame at the poses actually sampled. An axis pointing up means the
    # solved orientation is flipped, which puts every board on the wrong side.
    samples = acc._samples[best.board_index]
    axis_z = [
        float((s.T_base_gripper[:3, :3] @ best.t_gripper_camera_tsai[:3, 2])[2])
        for s in samples
    ]
    mean_axis_z = float(np.mean(axis_z))
    print(f"  optical axis base-frame z component: {mean_axis_z:+.2f} "
          f"({'ok — camera looks down' if mean_axis_z < 0 else 'WRONG — camera looks UP; do not use this result'})")

    save_hand_eye(best.t_gripper_camera_tsai, args.out)
    print(f"\nSaved gripper->camera transform from board #{best.board_index+1} to {args.out}")
    print(f"  translation [mm]: {best.t_gripper_camera_tsai[:3, 3] * 1000}")
    print(f"  raw samples kept in {args.samples} (re-run to resume, --fresh to discard)")


if __name__ == "__main__":
    main()

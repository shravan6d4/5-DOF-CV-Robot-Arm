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
import time
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
    solve_complaints,
    translation_conditioning,
)
from vision_pipeline.calibration.pixel_to_world import save_hand_eye
from vision_pipeline.capture.camera import Camera
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

IK_JOINTS = range(1, 6)  # J1..J5; J6 is the gripper, not part of hand-eye
# Rotation between RECORDED poses is what decides whether the camera's offset
# from the wrist is observable at all -- it is recovered from how far the camera
# SWINGS about that offset, so a small rotation leaves it barely determined
# while every residual still looks healthy. The 2026-08-04 session managed a
# median of 15.5 deg and produced a transform no amount of re-solving could fix.
# The literature (MVTec HALCON, and the hand-eye papers generally) wants >= 30
# deg, ideally 60, over >= 8 poses.
#
# 651.89 ticks/rad means 60 deg is ~683 ticks. The old ceiling of 300 could not
# reach even the 30 deg minimum in one step, which is a large part of why that
# session came out the way it did.
# ...but a single command may not exceed config.SERVO_MAX_MOVE_DELTA_TICKS
# (400), so the ceiling here has to stay under it or the bus simply refuses the
# jog. 350 ticks is ~31 deg: one press clears the minimum, two clears 60.
DEFAULT_STEP_TICKS = 200        # ~17.6 deg -- two presses puts you past 30
MIN_STEP_TICKS = 10
MAX_STEP_TICKS = 350            # ~31 deg, just under the bus's per-move cap


def _current_angles_rad(bus: ServoBus) -> list[float]:
    """J1..J5 angles in radians, straight from a servo read-back — no Pose involved."""
    return [bus.ticks_to_rad(j, bus.read_position(j)) for j in IK_JOINTS]


def _rotation_since(client, bus, last_ticks) -> float:
    """Degrees the WRIST has rotated since the last recorded sample.

    Measured through FK rather than by summing jog ticks, because what the
    solve cares about is the rotation of the camera-carrying body, and several
    joints contribute to it by different amounts. Returns inf when there is no
    previous sample, so the first record is never blocked.
    """
    if last_ticks is None:
        return float("inf")
    now = [bus.ticks_to_rad(j, bus.read_position(j)) for j in IK_JOINTS]
    then = [bus.ticks_to_rad(j, t) for j, t in zip(IK_JOINTS, last_ticks)]
    M = np.linalg.inv(client.request_fk(then)) @ client.request_fk(now)
    cos = np.clip((np.trace(M[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _wait_until_still(bus: ServoBus, timeout_s: float = 6.0, tol_ticks: int = 2):
    """Block until every IK joint reports the same position on two successive reads.

    A sample pairs ONE camera frame with ONE FK reading and asserts they describe
    the same instant. Record while the arm is still settling and they do not: the
    sample stays internally plausible, so nothing rejects it, but it silently
    contradicts every other sample and no rigid transform can fit the set.

    Measured on hardware 2026-08-04: samples taken without this showed FK and
    camera rotation angles disagreeing by up to 68 deg on individual pairs, while
    scripts/validate_joint_geometry.py — which does wait — agreed to within 2%
    on the same joints. Verifying stillness is cheap; trusting the operator to
    pause long enough is not.

    Returns:
        The settled tick readings, or the last reading if it never stabilised.
    """
    deadline = time.monotonic() + timeout_s
    previous = None
    while time.monotonic() < deadline:
        current = [bus.read_position(j) for j in IK_JOINTS]
        if previous is not None and all(
            abs(a - b) <= tol_ticks for a, b in zip(current, previous)
        ):
            return current
        previous = current
        time.sleep(0.15)
    print("  WARNING: joints never settled — sample may be unreliable.")
    return previous


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
        "--resume", action="store_true",
        help="continue an existing --samples file instead of archiving it. ONLY "
             "correct if the joint calibration has not changed since those "
             "samples were taken — mixing two calibrations produces a sample "
             "set no rigid transform can fit.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="save even if the acceptance checks fail. For when you have a "
             "reason the checks cannot see.",
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

    # A NEW SESSION STARTS EMPTY. Resuming used to be the default, so that a
    # crashed session could be continued -- and save_samples rewrites this file
    # after every record, which is what made that possible. The cost showed up
    # on 2026-08-05: a fresh capture silently appended to 53 samples taken under
    # a superseded joint calibration (wrong home_tick, J5 dir_sign +1). The two
    # sets describe different arms, no rigid transform fits both, and the solve
    # came out with 162 mm of board spread while still reporting a plausible
    # 33 mm camera offset.
    #
    # Resuming is only ever correct if the calibration has not changed since,
    # which is not something this script can check -- so it is now opt-in, and
    # the old file is archived rather than overwritten.
    acc = HandEyeAccumulator(min_samples=args.min_samples)
    samples_path = Path(args.samples)
    if args.resume:
        try:
            acc = load_samples(args.samples, min_samples=args.min_samples)
            print(f"--resume: continuing {sum(acc.counts().values())} samples from "
                  f"{args.samples} -> counts {acc.counts()}")
            print("  Only correct if the joint calibration has not changed since")
            print("  those samples were taken. If it has, they cannot be mixed.")
        except FileNotFoundError:
            pass
    elif samples_path.exists() and samples_path.stat().st_size > 2:
        archived = samples_path.with_suffix(
            f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        samples_path.rename(archived)
        print(f"Archived the previous capture to {archived.name}; starting clean.")
        print("  (--resume to continue it instead)")

    active_joint = 1
    step_ticks = DEFAULT_STEP_TICKS
    last_recorded = None      # joint ticks at the previous accepted sample

    print(f"{config.CALIB_BOARD_COUNT} boards expected, {args.min_samples} samples/board minimum.")
    print("1-5 = select joint, a/d = jog -/+ step, [/] = step size, "
          "r = record, R = record anyway, c = compute, q = abort.")
    print(f"Jog at least {config.CALIB_HAND_EYE_MIN_ROTATION_DEG:.0f} deg "
          f"(60 is better) between each record — the on-screen readout says "
          f"when. Fewer, bigger poses beat many small ones.")
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
            # How far the wrist has turned since the last RECORDED sample. This
            # is the number that decides whether the capture is any good, so it
            # is on screen the whole time rather than discovered afterwards.
            swing = _rotation_since(client, bus, last_recorded)
            ok = swing >= config.CALIB_HAND_EYE_MIN_ROTATION_DEG
            cv2.putText(
                display,
                f"rotation since last sample: {swing:5.1f} deg  "
                f"({'OK to record' if ok else 'KEEP JOGGING - want >=' + str(int(config.CALIB_HAND_EYE_MIN_ROTATION_DEG))})",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if ok else (0, 165, 255), 2,
            )
            # Rotation MAGNITUDE above is only half of a good capture. Turning
            # the same joint over and over racks up plenty of degrees while
            # leaving the camera offset along that axis unobservable, because
            # (R - I) n = 0 for a rotation about n. That is exactly how the
            # 2026-08-06 capture failed: 14 poses, all the swing anyone could
            # want, nine of thirteen pose changes sharing one axis to within
            # 1 degree -- all J5. It solved the camera's orientation to 0.8 deg
            # and its position not at all. Shown live because it cannot be
            # fixed afterwards: no amount of extra poses about the same axis
            # helps, so discovering it at solve time costs the whole session.
            best = max(acc.counts(), key=lambda i: acc.counts()[i], default=None)
            cond = (translation_conditioning(acc, best)
                    if best is not None else None)
            if cond is not None:
                good = cond <= config.CALIB_HAND_EYE_MAX_CONDITION
                cv2.putText(
                    display,
                    f"axis variety: {cond:4.1f} "
                    f"({'good' if good else 'CLUSTERED - jog a DIFFERENT joint'})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if good else (0, 165, 255), 2,
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

            elif key in (ord("r"), ord("R")):
                forced = key == ord("R")   # shift-R records regardless of rotation
                # Re-capture rather than banking the frame the display loop is
                # holding. cv2.VideoCapture buffers frames, and detecting
                # CALIB_BOARD_COUNT boards per iteration makes this loop slower
                # than the camera's frame rate — so the buffer stays full and
                # the displayed frame can lag the arm by hundreds of ms. Pairing
                # a stale frame with a fresh FK reading records the camera and
                # the wrist at DIFFERENT poses, which silently corrupts the
                # solve: it stays self-consistent per sample but no single rigid
                # transform can fit the set. Diagnosed 2026-08-04, after the
                # arm itself was cleared by scripts/validate_joint_geometry.py.
                # Order matters: settle FIRST, then flush the buffer, then read
                # the joints again and require they have not moved. Only then do
                # the frame and the FK pose provably describe the same instant.
                settled = _wait_until_still(bus)
                for _ in range(6):
                    fresh = camera.read_frame()
                fresh_gray = cv2.cvtColor(fresh, cv2.COLOR_BGR2GRAY)
                fresh_poses = detect_board_poses(fresh_gray, intr, detectors)
                if not fresh_poses:
                    print("  no board visible with enough corners — not recorded.")
                    continue
                after = [bus.read_position(j) for j in IK_JOINTS]
                if settled is None or any(abs(a - b) > 2 for a, b in zip(after, settled)):
                    print("  arm MOVED while capturing — discarded. Let it settle and retry.")
                    continue
                angles_rad = [bus.ticks_to_rad(j, t) for j, t in zip(IK_JOINTS, after)]
                t_base_gripper = client.request_fk(angles_rad)
                swing = _rotation_since(client, bus, last_recorded)
                if (not forced and last_recorded is not None
                        and swing < config.CALIB_HAND_EYE_MIN_ROTATION_DEG):
                    print(f"  only {swing:.1f} deg since the last sample "
                          f"(want >= {config.CALIB_HAND_EYE_MIN_ROTATION_DEG:.0f}). "
                          f"Jog further and record again — small rotations are what")
                    print(f"  made the 2026-08-04 capture unusable. Press 'R' to force it.")
                    continue
                new_counts = acc.add(fresh_poses, t_base_gripper)
                save_samples(acc, args.samples)
                last_recorded = list(after)
                print(f"  recorded boards {sorted(fresh_poses)} -> counts {new_counts} "
                      f"({swing:.1f} deg since last)")

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

    best = select_best(results, acc)

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

    # --- the gate, which used to not exist ------------------------------------
    # Everything above this line PRINTED its concerns and then saved anyway. On
    # 2026-08-05 that would have overwritten a good transform with one solved
    # from a sample file that had silently accumulated records from two
    # different joint calibrations: 162 mm of board spread, TSAI and PARK 169
    # deg apart on camera orientation, and a camera offset of 33 mm that looked
    # entirely reasonable next to the 24 mm on the ruler. A warning nobody is
    # forced to read is not a check.
    complaints = solve_complaints(best)
    if complaints and not args.force:
        print("\n=== NOT SAVED ===")
        for c in complaints:
            print(f"  - {c}")
        print(f"\n  {args.out} is untouched. The samples are still in "
              f"{args.samples}, so nothing is lost.")
        print("  Capture more poses with larger rotations between them, or "
              "re-run with --force if you know better.")
        return

    if complaints:
        print("\n  --force: saving despite " + f"{len(complaints)} failed check(s).")

    save_hand_eye(best.t_gripper_camera_tsai, args.out)
    print(f"\nSaved gripper->camera transform from board #{best.board_index+1} to {args.out}")
    print(f"  translation [mm]: {best.t_gripper_camera_tsai[:3, 3] * 1000}")
    print(f"  raw samples kept in {args.samples}")
    print("\n  Verify on hardware next: python scripts/test_pick_dry_run.py")


if __name__ == "__main__":
    main()

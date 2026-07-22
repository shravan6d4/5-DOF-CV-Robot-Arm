"""
Eye-in-hand hand-eye calibration — run BY HAND with the real arm + camera.

This is Merge Path step 2: it solves the FIXED gripper->camera offset (T_gripper_camera)
and writes data/hand_eye.json, replacing the identity placeholder ("camera sits exactly
at the wrist"). Until this exists, every pixel->world result is off by however far the
camera really sits from the wrist. Run scripts/calibrate_camera_intrinsics.py FIRST —
this uses the intrinsics for solvePnP and is only as good as they are.

    !!! SAFETY — THIS DRIVES THE REAL ARM !!!
    It commands the arm to a series of points across its workspace. Stay next to the
    e-stop / power switch, keep the workspace clear, and watch every move. Attempt each
    target only when you can see it's safe. This is operator-paced on purpose — nothing
    moves until you press SPACE.

Setup: fix the ChArUco board (scripts/generate_charuco_board.py, printed and mounted
flat) on the table where the wrist camera can see it from many arm poses — it must NOT
move for the whole session, it's the stationary reference the arm orbits. Run
calibrate_camera_intrinsics.py first; this uses those intrinsics for solvePnP. Then:

    python scripts/calibrate_hand_eye.py

Flow per candidate target: SPACE attempts the move and, if enough board corners are
found, records a (gripper-pose, board-in-camera) sample. Because this is ChArUco, the
board does NOT need to be fully in frame — partial views (a big win here, since the
camera's angle to the board changes a lot as the arm moves) still count as long as at
least config.CALIB_CHARUCO_MIN_CORNERS corners are seen. Collect >=
config.CALIB_HAND_EYE_MIN_SAMPLES with varied arm poses, then 'c' computes & saves, 'q'
aborts.

Why varied poses work even on a 5-DOF position-only arm: we can't COMMAND wrist
orientation, but different reachable (x,y,z) still produce different wrist orientations
as a side effect of the kinematics — which is exactly the rotation diversity
calibrateHandEye needs. The limitation that lost us commanded orientation is what makes
this calibration observable.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import geometry
from vision_pipeline.calibration.camera_model import default_intrinsics, load_intrinsics
from vision_pipeline.calibration.pixel_to_world import save_hand_eye
from vision_pipeline.capture.camera import Camera
from vision_pipeline.robot_interface.base import Pose
from vision_pipeline.robot_interface.hardware import HardwareRobot

MIN_SAMPLES = config.CALIB_HAND_EYE_MIN_SAMPLES
MIN_CORNERS = config.CALIB_CHARUCO_MIN_CORNERS
SETTLE_SECONDS = 1.0  # let the arm stop shaking before grabbing the frame


def _build_board_and_detector():
    """Build the ChArUco board + detector from config.py's geometry — the SAME
    values scripts/generate_charuco_board.py used to render the printed board."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.CALIB_ARUCO_DICT))
    board = cv2.aruco.CharucoBoard(
        (config.CALIB_CHARUCO_SQUARES_X, config.CALIB_CHARUCO_SQUARES_Y),
        config.CALIB_SQUARE_SIZE_M,
        config.CALIB_MARKER_SIZE_M,
        dictionary,
    )
    return board, cv2.aruco.CharucoDetector(board)


def _candidate_targets() -> list[tuple[float, float, float]]:
    """A spread of reachable (x,y,z) targets that keep the board plausibly in view.

    Filtered to inside the arm's ~0.30 m reach. These only need to VARY the pose;
    exact values are not load-bearing. Tune to where your board actually sits."""
    targets = []
    for x in (0.12, 0.17, 0.22):
        for y in (-0.08, 0.0, 0.08):
            for z in (-0.06, -0.02, 0.02):
                if (x * x + y * y + z * z) ** 0.5 < 0.30:
                    targets.append((x, y, z))
    return targets


def _mat_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation and 3-vector."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--out", default=config.HAND_EYE_PATH)
    args = parser.parse_args()

    intr = load_intrinsics()
    if np.allclose(intr.matrix, default_intrinsics().matrix):
        print("WARNING: camera intrinsics are still the config PLACEHOLDER.")
        print("         Run scripts/calibrate_camera_intrinsics.py first — hand-eye")
        print("         accuracy is bounded by intrinsics accuracy. Continuing anyway.")
    K = intr.matrix
    dist = intr.dist_coeffs
    board, detector = _build_board_and_detector()

    # OpenCV calibrateHandEye inputs, accumulated one per accepted sample:
    R_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    R_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []
    # Kept for the built-in consistency check:
    gripper_mats: list[np.ndarray] = []
    target_in_cam_mats: list[np.ndarray] = []

    targets = _candidate_targets()
    print(f"{len(targets)} candidate targets. SPACE = attempt next, 'c' = compute, 'q' = abort.")
    print("!!! Arm will MOVE on SPACE. Keep the e-stop within reach. !!!")

    idx = 0
    with HardwareRobot() as robot, Camera(camera_index=args.camera_index) as camera:
        while True:
            frame = camera.read_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            n_corners = 0 if charuco_corners is None else len(charuco_corners)

            display = frame.copy()
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            if charuco_corners is not None and len(charuco_corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
            nxt = targets[idx] if idx < len(targets) else None
            cv2.putText(
                display,
                f"samples: {len(R_gripper2base)}/{MIN_SAMPLES}  corners: {n_corners}"
                f"  next: {nxt}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.imshow("Hand-eye calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Aborted by operator.")
                cv2.destroyAllWindows()
                sys.exit(1)

            if key == ord("c"):
                break

            if key == ord(" "):
                if idx >= len(targets):
                    print("  no more candidate targets — press 'c' to compute.")
                    continue
                x, y, z = targets[idx]
                idx += 1
                ok = robot.send_target_pose(
                    Pose(x=x, y=y, z=z,
                         roll_deg=config.PICK_ROLL_DEG, pitch_deg=config.PICK_PITCH_DEG)
                )
                if not ok:
                    print(f"  target {(x, y, z)} unreachable / move failed — skipped.")
                    continue

                time.sleep(SETTLE_SECONDS)
                frame = camera.read_frame()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                charuco_corners, charuco_ids, _mc, _mi = detector.detectBoard(gray)
                n_corners = 0 if charuco_corners is None else len(charuco_corners)
                if charuco_corners is None or n_corners < MIN_CORNERS:
                    print(f"  only {n_corners} board corners at {(x, y, z)} "
                          f"(< {MIN_CORNERS}) — skipped.")
                    continue

                obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
                ok_pnp, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok_pnp:
                    print(f"  solvePnP failed at {(x, y, z)} — skipped.")
                    continue

                R_t2c, _ = cv2.Rodrigues(rvec)
                T_base_gripper = robot.get_end_effector_pose().to_matrix()

                R_gripper2base.append(T_base_gripper[:3, :3].copy())
                t_gripper2base.append(T_base_gripper[:3, 3].copy())
                R_target2cam.append(R_t2c)
                t_target2cam.append(tvec.reshape(3))
                gripper_mats.append(T_base_gripper)
                target_in_cam_mats.append(_mat_from_Rt(R_t2c, tvec))
                print(f"  sample {len(R_gripper2base)} recorded at {(x, y, z)}.")

    cv2.destroyAllWindows()

    n = len(R_gripper2base)
    if n < MIN_SAMPLES:
        print(f"Only {n} samples (< {MIN_SAMPLES}). Aborting without saving.")
        sys.exit(1)

    # Two methods; agreement is a sanity signal on pose diversity / capture quality.
    R_te, t_te = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    R_pk, t_pk = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    disagree_mm = float(np.linalg.norm(t_te.reshape(3) - t_pk.reshape(3))) * 1000.0
    print(f"\nTSAI vs PARK translation disagreement: {disagree_mm:.1f} mm "
          f"({'ok' if disagree_mm < 5 else 'HIGH — add more/varied poses'})")

    # calibrateHandEye returns cam->gripper, which IS T_gripper_camera (pose of the
    # camera in the gripper frame) — exactly what pixel_to_world loads as hand-eye.
    T_gripper_camera = _mat_from_Rt(R_te, t_te)

    # Consistency check: the board was stationary in base, so its implied base-frame
    # position should be identical across every sample. Low spread => trustworthy.
    board_origins = []
    for T_base_gripper, T_cam_target in zip(gripper_mats, target_in_cam_mats):
        T_base_target = T_base_gripper @ T_gripper_camera @ T_cam_target
        board_origins.append(T_base_target[:3, 3])
    board_origins = np.array(board_origins)
    spread_mm = float(np.linalg.norm(board_origins.std(axis=0))) * 1000.0
    print(f"Board base-frame position spread across {n} samples: {spread_mm:.1f} mm "
          f"({'ok' if spread_mm < 10 else 'HIGH — result is suspect'})")
    print(f"  mean board origin (base) [m]: {board_origins.mean(axis=0)}")

    save_hand_eye(T_gripper_camera, args.out)
    print(f"\nSaved gripper->camera transform to {args.out}")
    print(f"  translation [mm]: {t_te.reshape(3) * 1000}")


if __name__ == "__main__":
    main()

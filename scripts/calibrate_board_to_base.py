"""Tie the ChArUco board's frame to the robot base frame, using the brick and a ruler.

    This script does not move the arm. It only reads.

Why this exists: the normal pixel -> base chain runs

    pixel -> ray(camera) -> ray(base) via FK @ HAND-EYE -> intersect the table

and the hand-eye link is, as of 2026-08-04, wrong in both rotation and
translation (measured 52 mm of vertical error against the board). Every number
downstream of it is wrong by ~400 mm.

This takes a different route to the same answer, one that never uses hand-eye:

    pixel -> ray(camera) -> intersect the BOARD PLANE  (solvePnP only)
          -> point in board coordinates
          -> point in base coordinates                 (this calibration)

The board is a rigid, flat, precisely-known object lying on the table, and
solvePnP measures the camera's pose relative to it directly from the image. So
the camera's pose in the base frame is simply not needed — which is what makes
this immune to the hand-eye problem.

What is needed instead is where the BOARD sits relative to the robot. The board
is flat on the table, so that is only three numbers: x, y, and a yaw. Two
measured points determine them exactly; a third or fourth lets the fit report a
residual, which is the only way to catch a mis-measurement.

PREFERRED: --touch, which uses the arm as its own measuring instrument.

Jog the claw tip onto a named ChArUco corner and record it. The corner's board
coordinates are known exactly from the board's printed geometry, and FK reports
the tip's base coordinates. Neither a ruler nor the operator's idea of where the
base origin sits enters anywhere -- FK *defines* the base frame, so it cannot
disagree with the frame IK will later be commanded in. That matters: a ruler
measured from an assumed origin that is off by some offset produces a transform
in the OPERATOR's frame, and the arm then misses every pick by exactly that
offset, silently and consistently.

    python scripts/calibrate_board_to_base.py --corners        # pick reachable ones
    # jog the claw tip onto corner 14, then:
    python scripts/calibrate_board_to_base.py --touch 14
    python scripts/calibrate_board_to_base.py --touch 21
    python scripts/calibrate_board_to_base.py --touch 30       # third tests the fit
    python scripts/calibrate_board_to_base.py --solve

FALLBACK: --add, which uses the brick and a ruler. Same solve, worse inputs --
it inherits the base-origin question above, plus ~5 mm of brick-height parallax
(the brick's top face sits ~9.6 mm above the board but its pixel is
back-projected onto the board PLANE), plus ~5 mm of ruler error. Use it only if
the claw cannot reach the board.

    python scripts/calibrate_board_to_base.py --add 280 35     # ruler mm, +y LEFT

    IMPORTANT: the board must not move between calibrating and picking, and
    the solve is only valid for the board it was measured against.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import charuco
from vision_pipeline.calibration.camera_model import load_intrinsics
from vision_pipeline.calibration.hand_eye import detect_board_poses
from vision_pipeline.capture.camera import Camera
from vision_pipeline.detection.lego_detector import LegoBrickDetector

SAMPLES_PATH = Path("data/board_to_base_samples.json")
TRANSFORM_PATH = Path("data/board_to_base.json")
CAMERA_WARMUP_FRAMES = 8


def brick_in_board_frame(frame_bgr, intr, board_id=None):
    """Back-project the detected brick's centroid onto the board plane.

    Returns (board_id, [x, y] metres in board coordinates, Detection), or None.
    Deliberately uses no robot pose of any kind: solvePnP puts the camera in the
    board's frame, and the brick's ray is intersected with the board's own plane
    there. That is the whole point of this script.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    poses = detect_board_poses(gray, intr, charuco.build_detectors())
    if not poses:
        return None
    if board_id is None:
        board_id = sorted(poses)[0]
    elif board_id not in poses:
        return None

    bricks = LegoBrickDetector().detect(frame_bgr)
    if not bricks:
        return None
    brick = bricks[0]

    T_cb = poses[board_id]
    R, t = T_cb[:3, :3], T_cb[:3, 3]
    origin = -R.T @ t                        # camera origin in board coordinates
    ray = R.T @ intr.pixel_to_ray(brick.centroid_px)
    if abs(ray[2]) < 1e-9:
        return None                          # ray parallel to the board plane
    point = origin + (-origin[2] / ray[2]) * ray
    return board_id, point[:2], brick


def fit_rigid_2d(board_xy: np.ndarray, base_xy: np.ndarray):
    """Least-squares rigid 2D transform taking board coords to base coords.

    Rotation + translation only, no scale: the board is a rigid printed object
    and the base frame is metric, so any apparent scale would be a measurement
    error rather than something to absorb. Standard Kabsch, with the reflection
    guard -- without it a noisy two-point fit can return a mirror, which looks
    like a fine residual and places every future brick on the wrong side.
    """
    bc, tc = board_xy.mean(axis=0), base_xy.mean(axis=0)
    h = (board_xy - bc).T @ (base_xy - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T
    return r, tc - r @ bc


def load_samples() -> list:
    return json.loads(SAMPLES_PATH.read_text()) if SAMPLES_PATH.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--touch", type=int, metavar="CORNER",
                    help="PREFERRED. Record a sample from the claw tip resting on "
                         "chessboard corner CORNER. Reads FK; commands no motion.")
    ap.add_argument("--corners", action="store_true",
                    help="list the chessboard corners and their board coordinates")
    ap.add_argument("--add", nargs=2, type=float, metavar=("X_MM", "Y_MM"),
                    help="FALLBACK. Record a sample: the brick's ruler-measured base "
                         "position in mm (+x forward, +y LEFT)")
    ap.add_argument("--solve", action="store_true", help="fit and write the transform")
    ap.add_argument("--list", action="store_true", help="show recorded samples")
    ap.add_argument("--reset", action="store_true", help="discard all samples")
    ap.add_argument("--image", help="use a static image instead of the live camera")
    ap.add_argument("--board", type=int, help="force a specific board id")
    args = ap.parse_args()

    if args.reset:
        SAMPLES_PATH.unlink(missing_ok=True)
        print("Samples discarded.")
        return

    if args.list:
        for i, s in enumerate(load_samples(), 1):
            print(f"  {i}. board{s['board']}  board({s['board_mm'][0]:+7.1f}, "
                  f"{s['board_mm'][1]:+7.1f})  ->  base({s['base_mm'][0]:+7.1f}, "
                  f"{s['base_mm'][1]:+7.1f})")
        if not load_samples():
            print("  (none)")
        return

    if args.corners:
        board = charuco.build_board(args.board if args.board is not None else 1)
        pts = np.asarray(board.getChessboardCorners(), dtype=float)
        nx = config.CALIB_CHARUCO_SQUARES_X - 1
        print(f"Board {args.board if args.board is not None else 1}: "
              f"{len(pts)} chessboard corners ({nx} across), mm in the board frame.")
        print("Pick ones the claw can actually reach, spread as widely as possible —")
        print("two corners close together fit a yaw badly.\n")
        for row_start in range(0, len(pts), nx):
            row = pts[row_start:row_start + nx]
            print("  " + "  ".join(
                f"{row_start + i:>3}:({p[0] * 1000:>5.0f},{p[1] * 1000:>5.0f})"
                for i, p in enumerate(row)))
        return

    if args.touch is not None:
        from vision_pipeline.robot_interface.matlab_client import MatlabIKClient
        from vision_pipeline.robot_interface.servo_driver import ServoBus

        board_id = args.board if args.board is not None else 1
        board = charuco.build_board(board_id)
        pts = np.asarray(board.getChessboardCorners(), dtype=float)
        if not 0 <= args.touch < len(pts):
            sys.exit(f"corner {args.touch} out of range (0..{len(pts) - 1})")
        corner = pts[args.touch]

        bus = ServoBus(config.SERVO_PORT, config.SERVO_BAUD)
        ik = MatlabIKClient(config.MATLAB_SERVER_HOST, config.MATLAB_SERVER_PORT)
        with bus:
            angles = [bus.ticks_to_rad(j, bus.read_position(j)) for j in range(1, 6)]
        _, T_tip = ik.request_fk_tip(angles)
        ik.close()
        tip = T_tip[:3, 3]

        # The tip is supposed to be ON the board, so its height is a free check
        # of whether it is actually touching -- and of TABLE_Z_IN_BASE itself.
        gap_mm = (tip[2] - config.TABLE_Z_IN_BASE) * 1000
        print(f"corner {args.touch}: board ({corner[0] * 1000:+.1f}, {corner[1] * 1000:+.1f}) mm")
        print(f"claw tip (FK):      base  ({tip[0] * 1000:+.1f}, {tip[1] * 1000:+.1f}, "
              f"{tip[2] * 1000:+.1f}) mm")
        print(f"tip height above the table: {gap_mm:+.1f} mm")
        if abs(gap_mm) > 15:
            print("  *** that is not touching the board. Sample NOT recorded — jog")
            print("      the tip down onto the corner and re-run. ***")
            return

        samples = load_samples()
        samples.append({
            "board": int(board_id),
            "board_mm": [float(corner[0] * 1000), float(corner[1] * 1000)],
            "base_mm": [float(tip[0] * 1000), float(tip[1] * 1000)],
            "source": f"claw-touch corner {args.touch}",
            "tip_gap_mm": float(gap_mm),
        })
        SAMPLES_PATH.write_text(json.dumps(samples, indent=2))
        print(f"Recorded #{len(samples)} (claw-touch, no ruler).")
        if len(samples) < 2:
            print("  Need at least one more, as far away on the board as the arm reaches.")
        return

    if args.add:
        intr = load_intrinsics()
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                sys.exit(f"could not read {args.image}")
        else:
            with Camera() as cam:
                for _ in range(CAMERA_WARMUP_FRAMES):
                    frame = cam.read_frame()

        found = brick_in_board_frame(frame, intr, args.board)
        if found is None:
            sys.exit("Need BOTH a board and a brick in the frame. Nothing recorded.")
        board_id, board_xy, brick = found

        samples = load_samples()
        samples.append({
            "board": int(board_id),
            "board_mm": [float(board_xy[0] * 1000), float(board_xy[1] * 1000)],
            "base_mm": [float(args.add[0]), float(args.add[1])],
            "pixel": [float(brick.centroid_px[0]), float(brick.centroid_px[1])],
            "confidence": float(brick.confidence),
        })
        SAMPLES_PATH.write_text(json.dumps(samples, indent=2))
        print(f"Recorded #{len(samples)}: board{board_id} "
              f"({board_xy[0] * 1000:+.1f}, {board_xy[1] * 1000:+.1f}) mm  ->  "
              f"base ({args.add[0]:+.1f}, {args.add[1]:+.1f}) mm")
        print(f"  brick at pixel {tuple(round(v) for v in brick.centroid_px)}, "
              f"confidence {brick.confidence:.2f}")
        if len(samples) < 2:
            print("  Need at least one more. Move the brick, measure, --add again.")
        return

    if not args.solve:
        ap.print_help()
        return

    samples = load_samples()
    boards = {s["board"] for s in samples}
    if len(samples) < 2:
        sys.exit(f"Need at least 2 samples to solve, have {len(samples)}.")
    if len(boards) > 1:
        sys.exit(f"Samples span boards {sorted(boards)}. Each board has its own "
                 f"frame, so they cannot be mixed. Re-run with --board.")

    board_xy = np.array([s["board_mm"] for s in samples]) / 1000.0
    base_xy = np.array([s["base_mm"] for s in samples]) / 1000.0
    r, t = fit_rigid_2d(board_xy, base_xy)

    residuals = np.linalg.norm((board_xy @ r.T + t) - base_xy, axis=1) * 1000
    print(f"Fit from {len(samples)} samples on board {boards.pop()}:")
    print(f"  yaw       {np.degrees(np.arctan2(r[1, 0], r[0, 0])):+.2f} deg")
    print(f"  translate ({t[0] * 1000:+.1f}, {t[1] * 1000:+.1f}) mm")
    for i, (s, e) in enumerate(zip(samples, residuals), 1):
        print(f"  sample {i}: residual {e:5.1f} mm   "
              f"base({s['base_mm'][0]:+.0f}, {s['base_mm'][1]:+.0f})")

    if len(samples) == 2:
        print("\n  Two samples fit exactly by construction — residuals of 0 mean")
        print("  nothing. Add a third to actually test the fit.")
    else:
        worst = residuals.max()
        print(f"\n  worst residual {worst:.1f} mm", end="")
        print("  — consistent." if worst < 15 else
              "  — TOO HIGH. Re-measure; one sample is likely mis-read.")

    payload = {
        "rotation": r.tolist(),
        "translation_m": t.tolist(),
        "table_z_in_base": config.TABLE_Z_IN_BASE,
        "board": int(samples[0]["board"]),
        "num_samples": len(samples),
        "max_residual_mm": float(residuals.max()),
    }
    TRANSFORM_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {TRANSFORM_PATH}")
    print("Valid only while the board stays exactly where it is now.")


if __name__ == "__main__":
    main()

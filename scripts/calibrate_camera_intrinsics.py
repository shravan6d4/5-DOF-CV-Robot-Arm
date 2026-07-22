"""
Camera intrinsics calibration (ChArUco board) — run BY HAND with the mounted camera.

This is Merge Path step 1: it produces data/camera_intrinsics.json (fx, fy, cx, cy,
distortion), replacing the rough placeholders in config.py. Every world coordinate
downstream inherits this, and hand-eye calibration (step 2) needs it to already be
accurate — so run this FIRST.

No arm needed — just the camera. First generate the board (writes data/charuco_board.png
using the geometry in config.py's CALIB_* constants):

    python scripts/generate_charuco_board.py

Print it, mount it FLAT and rigid (tape to cardboard/acrylic), then:

    python scripts/calibrate_camera_intrinsics.py

Hold the board at many angles/distances/positions filling the frame; the overlay turns
on when corners are found. Because this is ChArUco (not a plain chessboard), the board
does NOT need to be fully in frame — partial/angled views still contribute a valid
sample, as long as at least CALIB_CHARUCO_MIN_CORNERS corners are seen. Press 'c' to
capture that view (aim for 15-20 spread across the image, plus close-up/far/corner
positions), 'q' to finish and compute. A good result prints a reprojection error under
~1 px.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration.camera_model import CameraIntrinsics, save_intrinsics
from vision_pipeline.capture.camera import Camera

MIN_SAMPLES = config.CALIB_INTRINSICS_MIN_SAMPLES
MIN_CORNERS = config.CALIB_CHARUCO_MIN_CORNERS


def _build_board_and_detector():
    """Build the ChArUco board + detector from config.py's geometry — the SAME
    values scripts/generate_charuco_board.py used to render the printed board.
    A mismatch here would silently miscalibrate, not just fail to detect."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.CALIB_ARUCO_DICT))
    board = cv2.aruco.CharucoBoard(
        (config.CALIB_CHARUCO_SQUARES_X, config.CALIB_CHARUCO_SQUARES_Y),
        config.CALIB_SQUARE_SIZE_M,
        config.CALIB_MARKER_SIZE_M,
        dictionary,
    )
    return board, cv2.aruco.CharucoDetector(board)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument(
        "--out", default=config.CAMERA_INTRINSICS_PATH,
        help="Where to write the intrinsics JSON (default: config.CAMERA_INTRINSICS_PATH).",
    )
    args = parser.parse_args()

    board, detector = _build_board_and_detector()
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    print(
        f"ChArUco board: {config.CALIB_CHARUCO_SQUARES_X}x{config.CALIB_CHARUCO_SQUARES_Y} "
        f"squares, {config.CALIB_SQUARE_SIZE_M*1000:.1f} mm squares "
        f"({config.CALIB_ARUCO_DICT}). Partial views are OK — full board not required."
    )
    print("Fill the frame at varied angles/distances. 'c' = capture a view, 'q' = finish & compute.")

    with Camera(camera_index=args.camera_index) as camera:
        while True:
            frame = camera.read_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            n_corners = 0 if charuco_corners is None else len(charuco_corners)
            usable = charuco_corners is not None and n_corners >= MIN_CORNERS

            display = frame.copy()
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            if charuco_corners is not None and len(charuco_corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
            cv2.putText(
                display,
                f"corners: {n_corners} (need {MIN_CORNERS})  captured: {len(obj_points)}/{MIN_SAMPLES}"
                "  (c=capture q=finish)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if usable else (0, 0, 255), 2,
            )
            cv2.imshow("Intrinsics calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                if not usable:
                    print(f"  only {n_corners} corners (< {MIN_CORNERS}) — not captured.")
                else:
                    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
                    obj_points.append(obj_pts)
                    img_points.append(img_pts)
                    print(f"  captured view {len(obj_points)} ({n_corners} corners).")
            elif key == ord("q"):
                break

    cv2.destroyAllWindows()

    if len(obj_points) < MIN_SAMPLES:
        print(f"Only {len(obj_points)} views (< {MIN_SAMPLES}). Aborting without saving.")
        sys.exit(1)

    print(f"\nCalibrating from {len(obj_points)} views...")
    rms, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    print(f"Reprojection error (RMS): {rms:.3f} px  (under ~1 px is good)")

    intr = CameraIntrinsics(
        fx=float(mtx[0, 0]), fy=float(mtx[1, 1]),
        cx=float(mtx[0, 2]), cy=float(mtx[1, 2]),
        distortion=tuple(float(d) for d in dist.ravel()[:5]),
    )
    save_intrinsics(intr, args.out)
    print(f"Saved intrinsics to {args.out}")
    print(f"  fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")
    print(f"  distortion={intr.distortion}")


if __name__ == "__main__":
    main()

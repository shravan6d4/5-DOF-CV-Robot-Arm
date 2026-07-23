"""
Camera intrinsics calibration (ChArUco board) — run BY HAND with the mounted camera.

This is Merge Path step 1: it produces data/camera_intrinsics.json (fx, fy, cx, cy,
distortion), replacing the rough placeholders in config.py. Every world coordinate
downstream inherits this, and hand-eye calibration (step 2) needs it to already be
accurate — so run this FIRST.

No arm needed — just the camera. First generate the boards (writes
data/charuco_board_1.png .. _N.png using the geometry in config.py's CALIB_* constants):

    python scripts/generate_charuco_board.py

Print them, mount FLAT and rigid (tape to cardboard/acrylic), then:

    python scripts/calibrate_camera_intrinsics.py

Every board is detected independently, so laying several in the frame at once banks
several views per 'c' press. Hold them at many angles/distances/positions; the overlay
turns on when corners are found. Because this is ChArUco (not a plain chessboard), a
board does NOT need to be fully in frame — partial/angled views still contribute a
valid sample, as long as at least CALIB_CHARUCO_MIN_CORNERS corners are seen. Press 'c'
to capture (aim for 15-20 views spread across the image, plus close-up/far/corner
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
from vision_pipeline.calibration import charuco
from vision_pipeline.calibration.camera_model import CameraIntrinsics, save_intrinsics
from vision_pipeline.capture.camera import Camera

MIN_SAMPLES = config.CALIB_INTRINSICS_MIN_SAMPLES
MIN_CORNERS = config.CALIB_CHARUCO_MIN_CORNERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument(
        "--out", default=config.CAMERA_INTRINSICS_PATH,
        help="Where to write the intrinsics JSON (default: config.CAMERA_INTRINSICS_PATH).",
    )
    args = parser.parse_args()

    detectors = charuco.build_detectors()
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    print(
        f"{config.CALIB_BOARD_COUNT} distinct ChArUco boards, "
        f"{config.CALIB_CHARUCO_SQUARES_X}x{config.CALIB_CHARUCO_SQUARES_Y} squares @ "
        f"{config.CALIB_SQUARE_SIZE_M*1000:.1f} mm ({config.CALIB_ARUCO_DICT}). "
        f"Partial views are OK — full board not required."
    )
    print("Every board visible in a frame contributes its OWN view, so one 'c' on a")
    print("frame showing three boards banks three views.")
    print("Vary angles/distances. 'c' = capture, 'q' = finish & compute.")

    with Camera(camera_index=args.camera_index) as camera:
        while True:
            frame = camera.read_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

            # Run every board's detector over the frame. Each board owns a
            # disjoint slice of the dictionary, so a detector only ever matches
            # its own board and several tiled boards can be resolved at once.
            # Each detected board is an INDEPENDENT view for calibrateCamera:
            # its object points are expressed in its own board frame, and the
            # solver estimates a separate pose per view anyway, so nothing
            # needs to know how the boards are laid out relative to each other.
            # That is what makes tiling safe here — no sub-mm seam alignment
            # between sheets is required.
            found: list[tuple[int, np.ndarray, np.ndarray]] = []
            for idx, _b, det in detectors:
                corners, ids = charuco.detect(det, gray)
                if corners is not None and len(corners) >= MIN_CORNERS:
                    found.append((idx, corners, ids))

            n_corners = sum(len(c) for _, c, _ in found)
            usable = bool(found)

            # The overlay is cosmetic, so it must never be able to end the
            # session — losing 18 captured views to a drawing quirk on one
            # blurry frame would be an absurd way to fail. Draw defensively
            # and carry on; `usable` above is what actually gates capture.
            display = frame.copy()
            for _idx, corners, ids in found:
                try:
                    cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
                except cv2.error:
                    pass
            labels = ",".join(f"#{i+1}" for i, _, _ in found) if found else "none"
            cv2.putText(
                display,
                f"boards: {labels}  corners: {n_corners}  "
                f"captured: {len(obj_points)}/{MIN_SAMPLES}  (c=capture q=finish)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if usable else (0, 0, 255), 2,
            )
            cv2.imshow("Intrinsics calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                if not usable:
                    print(f"  no board with at least {MIN_CORNERS} corners — not captured.")
                else:
                    for idx, corners, ids in found:
                        # Append only if BOTH succeed. obj_points and img_points
                        # are positionally paired — calibrateCamera reads index i
                        # of each as the same view — so a half-completed append
                        # would shift every later pair by one and misalign the
                        # whole set.
                        board = detectors[idx][1]
                        try:
                            obj_pts, img_pts = board.matchImagePoints(corners, ids)
                        except cv2.error as e:
                            print(f"  board #{idx+1} rejected: matchImagePoints "
                                  f"failed ({e.err.strip()}).")
                            continue
                        obj_points.append(obj_pts)
                        img_points.append(img_pts)
                        print(f"  captured view {len(obj_points)} "
                              f"from board #{idx+1} ({len(corners)} corners).")
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

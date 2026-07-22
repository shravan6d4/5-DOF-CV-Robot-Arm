"""
Generate the printable ChArUco calibration board — run this FIRST, before either
calibration script, and don't substitute a board from anywhere else (e.g. a random
image found online). A mismatched dictionary or geometry doesn't just fail to
detect — it can silently feed WRONG 3D coordinates into solvePnP/calibrateCamera
and produce a confidently wrong calibration that looks fine but isn't.

ChArUco = chessboard + a unique ArUco marker in every other square. Unlike a plain
chessboard, every corner is individually identified by its neighboring marker IDs,
so detection works from PARTIAL / angled views — the eye-in-hand camera's view of
the board changes constantly as the arm moves, so this matters a lot here.

Board geometry is defined ONCE, in config.py (CALIB_ARUCO_DICT,
CALIB_CHARUCO_SQUARES_X/Y, CALIB_SQUARE_SIZE_M, CALIB_MARKER_SIZE_M) — this script
and both calibration scripts all read the same values, so "what got printed" and
"what the detector expects" can never drift apart.

    python scripts/generate_charuco_board.py

Writes data/charuco_board.png, sized so printing it at 100% / "actual size" (NOT
"fit to page") produces squares of exactly CALIB_SQUARE_SIZE_M. Printer scaling is
never perfectly exact though — after printing, measure a run of several squares
with a ruler (measure across e.g. 5 squares and divide by 5) and, if it's off,
correct CALIB_SQUARE_SIZE_M in config.py to the MEASURED value before running
calibrate_camera_intrinsics.py or calibrate_hand_eye.py. Mount the print FLAT and
rigid (tape to cardboard/acrylic) — a wavy board corrupts corner positions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from vision_pipeline import config

PRINT_DPI = 300
METERS_PER_INCH = 0.0254


def build_board() -> "cv2.aruco.CharucoBoard":
    """Build the ChArUco board object from config.py's geometry — the single
    source of truth every script (this one, both calibration scripts) reads."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.CALIB_ARUCO_DICT))
    return cv2.aruco.CharucoBoard(
        (config.CALIB_CHARUCO_SQUARES_X, config.CALIB_CHARUCO_SQUARES_Y),
        config.CALIB_SQUARE_SIZE_M,
        config.CALIB_MARKER_SIZE_M,
        dictionary,
    )


def main() -> None:
    if config.CALIB_MARKER_SIZE_M >= config.CALIB_SQUARE_SIZE_M:
        print("CALIB_MARKER_SIZE_M must be smaller than CALIB_SQUARE_SIZE_M — check config.py.")
        sys.exit(1)

    board = build_board()

    width_m = config.CALIB_CHARUCO_SQUARES_X * config.CALIB_SQUARE_SIZE_M
    height_m = config.CALIB_CHARUCO_SQUARES_Y * config.CALIB_SQUARE_SIZE_M
    width_px = round(width_m / METERS_PER_INCH * PRINT_DPI)
    height_px = round(height_m / METERS_PER_INCH * PRINT_DPI)

    # marginSize=0 so the pixel dimensions above map exactly to the printed
    # footprint — no hidden padding to throw off the "print at 100%" math.
    img = board.generateImage((width_px, height_px), marginSize=0, borderBits=1)

    out_path = Path("data/charuco_board.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

    # Printable interior after ~5mm margin per side. Board may print in whichever
    # orientation (portrait/landscape) fits — check by long/short edge, not by axis,
    # since a wide-grid board is meant to print landscape.
    def _fits(w_mm, h_mm, short_mm, long_mm):
        long_side, short_side = max(w_mm, h_mm), min(w_mm, h_mm)
        return long_side <= long_mm and short_side <= short_mm

    fits_a4 = _fits(width_m * 1000, height_m * 1000, 200, 287)       # A4: 210x297mm
    fits_letter = _fits(width_m * 1000, height_m * 1000, 206, 269)   # Letter: 215.9x279.4mm

    print(f"Wrote {out_path}  ({width_px}x{height_px} px @ {PRINT_DPI} DPI)")
    print(
        f"Board footprint when printed at 100%: {width_m*1000:.0f} x {height_m*1000:.0f} mm "
        f"({config.CALIB_CHARUCO_SQUARES_X}x{config.CALIB_CHARUCO_SQUARES_Y} squares "
        f"@ {config.CALIB_SQUARE_SIZE_M*1000:.0f} mm each)"
    )
    print(f"Fits on A4 with margin: {fits_a4}   Fits on Letter with margin: {fits_letter}")
    print(
        f"Dictionary: {config.CALIB_ARUCO_DICT}  |  marker size: "
        f"{config.CALIB_MARKER_SIZE_M*1000:.0f} mm"
    )
    print()
    print("Print at 100% / 'actual size' -- NOT 'fit to page'. Then measure a run of")
    print("several squares with a ruler; if it doesn't match the nominal size above,")
    print("update CALIB_SQUARE_SIZE_M in config.py to the MEASURED value before running")
    print("calibrate_camera_intrinsics.py or calibrate_hand_eye.py.")
    print("Mount flat and rigid (tape to cardboard/acrylic) -- a wavy print corrupts")
    print("corner positions.")


if __name__ == "__main__":
    main()

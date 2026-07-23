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

Writes CALIB_BOARD_COUNT distinct boards, data/charuco_board_1.png ..
data/charuco_board_N.png. They are DISTINCT — each takes its own slice of the
ArUco dictionary — so several can be laid in the camera's view at once and each
contributes an independent calibration view. Printing one board N times gives
duplicate marker IDs, which the detector cannot disambiguate; that is the
failure this replaces. Board 1 keeps the same IDs as the old single board, so
an existing printout of it stays valid.

Sized so printing at 100% / "actual size" (NOT "fit to page") produces squares
of CALIB_SQUARE_SIZE_NOMINAL_M. Printer scaling is never exact though — after
printing, measure a run of several squares with a ruler (across e.g. 5 squares,
divide by 5) and set CALIB_SQUARE_SIZE_M in config.py to the MEASURED value
before running calibrate_camera_intrinsics.py or calibrate_hand_eye.py. Mount
each print FLAT and rigid (tape to cardboard/acrylic) — a wavy board corrupts
corner positions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from vision_pipeline import config
from vision_pipeline.calibration import charuco

PRINT_DPI = 300
METERS_PER_INCH = 0.0254


def main() -> None:
    if config.CALIB_MARKER_SIZE_M >= config.CALIB_SQUARE_SIZE_M:
        print("CALIB_MARKER_SIZE_M must be smaller than CALIB_SQUARE_SIZE_M — check config.py.")
        sys.exit(1)

    try:
        charuco.check_dictionary_capacity()
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Render at the NOMINAL size, not the measured one. config.CALIB_SQUARE_SIZE_M
    # describes what the printer DELIVERED last time (26.7mm from a 25mm request);
    # feeding that back in would apply the printer's scale factor a second time.
    render_m = config.CALIB_SQUARE_SIZE_NOMINAL_M

    width_m = config.CALIB_CHARUCO_SQUARES_X * render_m
    height_m = config.CALIB_CHARUCO_SQUARES_Y * render_m
    width_px = round(width_m / METERS_PER_INCH * PRINT_DPI)
    height_px = round(height_m / METERS_PER_INCH * PRINT_DPI)

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(config.CALIB_BOARD_COUNT):
        board = charuco.build_board(i, square_size_m=render_m)
        # marginSize=0 so the pixel dimensions above map exactly to the printed
        # footprint — no hidden padding to throw off the "print at 100%" math.
        img = board.generateImage((width_px, height_px), marginSize=0, borderBits=1)
        ids = charuco.board_ids(i)
        path = out_dir / f"charuco_board_{i + 1}.png"
        cv2.imwrite(str(path), img)
        print(f"Wrote {path}   marker IDs {ids[0]}..{ids[-1]}")

    # Printable interior after ~5mm margin per side. Board may print in whichever
    # orientation (portrait/landscape) fits — check by long/short edge, not by axis,
    # since a wide-grid board is meant to print landscape.
    def _fits(w_mm, h_mm, short_mm, long_mm):
        long_side, short_side = max(w_mm, h_mm), min(w_mm, h_mm)
        return long_side <= long_mm and short_side <= short_mm

    fits_a4 = _fits(width_m * 1000, height_m * 1000, 200, 287)       # A4: 210x297mm
    fits_letter = _fits(width_m * 1000, height_m * 1000, 206, 269)   # Letter: 215.9x279.4mm

    print()
    print(f"{config.CALIB_BOARD_COUNT} DISTINCT boards, {width_px}x{height_px} px "
          f"@ {PRINT_DPI} DPI each.")
    print(
        f"Footprint when printed at 100%: {width_m*1000:.0f} x {height_m*1000:.0f} mm "
        f"({config.CALIB_CHARUCO_SQUARES_X}x{config.CALIB_CHARUCO_SQUARES_Y} squares "
        f"@ {render_m*1000:.1f} mm each, nominal)"
    )
    print(f"Fits on A4 with margin: {fits_a4}   Fits on Letter with margin: {fits_letter}")
    print(
        f"Dictionary: {config.CALIB_ARUCO_DICT}  |  "
        f"{charuco.markers_per_board()} markers per board, "
        f"{config.CALIB_BOARD_COUNT * charuco.markers_per_board()} total"
    )
    print()
    print("Board 1 carries the SAME marker IDs and layout as the old single-board")
    print("printout — OpenCV's predefined dictionaries are nested by prefix, so IDs")
    print("0..29 mean the same thing in DICT_5X5_250 as in DICT_5X5_100. An existing")
    print("print of it stays valid; you only need to print boards 2..N. (The PNG is not")
    print("byte-identical: the marker/square ratio moved by ~0.1% when the measured")
    print("square size was recorded. That is far below detection sensitivity.)")
    print()
    print("Print at 100% / 'actual size' -- NOT 'fit to page', and print every board")
    print("with the SAME settings so they end up the same scale as each other.")
    print(f"Then measure a run of several squares: the render targets "
          f"{render_m*1000:.1f} mm")
    print(f"(CALIB_SQUARE_SIZE_NOMINAL_M) and config currently records "
          f"{config.CALIB_SQUARE_SIZE_M*1000:.1f} mm")
    print("as what the printer actually delivers (CALIB_SQUARE_SIZE_M). If the new")
    print("prints measure differently, update CALIB_SQUARE_SIZE_M -- that value scales")
    print("every distance the camera reports.")
    print("Mount flat and rigid (tape to cardboard/acrylic) -- a wavy print corrupts")
    print("corner positions.")


if __name__ == "__main__":
    main()

"""ChArUco board construction — the single source of truth for board geometry.

Every script that renders a board and every script that detects one builds it
through here, so "what got printed" and "what the detector expects" cannot
drift apart. A mismatch does not fail loudly: a board built with the wrong
geometry still detects, and still returns 3D object points, they are simply
WRONG — producing a confidently bad calibration.

MULTIPLE BOARDS. The workspace uses several boards tiled to cover area. They
must be mutually distinguishable, which means each gets its own slice of the
ArUco dictionary: board 0 takes marker IDs 0..29, board 1 takes 30..59, and so
on. Printing the SAME board several times instead is the failure this module
exists to prevent — duplicate IDs make a marker's board membership ambiguous,
and the detector responds by returning mismatched corner/ID arrays or nothing
at all, neither of which reads as "you printed the wrong thing".

Note that OpenCV's predefined dictionaries are nested by prefix: the first 100
markers of DICT_5X5_250 are exactly DICT_5X5_100. Board 0 is therefore
byte-identical to a board generated against the smaller dictionary, so an
existing printout of it stays valid after the dictionary is widened.
"""

import cv2
import numpy as np

from vision_pipeline import config


def markers_per_board() -> int:
    """How many ArUco markers one board consumes.

    A ChArUco board carries a marker in every other square, so half the
    squares — hence half the dictionary slice per board.
    """
    return (config.CALIB_CHARUCO_SQUARES_X * config.CALIB_CHARUCO_SQUARES_Y) // 2


def dictionary() -> "cv2.aruco.Dictionary":
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.CALIB_ARUCO_DICT))


def board_ids(index: int) -> np.ndarray:
    """The dictionary IDs belonging to board `index` (0-based)."""
    per = markers_per_board()
    return np.arange(index * per, (index + 1) * per, dtype=np.int32)


def check_dictionary_capacity() -> None:
    """Raise if the configured dictionary cannot hold every board's markers.

    Worth failing loudly on: silently wrapping or truncating IDs would give two
    boards overlapping markers, which reintroduces exactly the ambiguity the
    per-board ID slicing is here to remove.
    """
    needed = config.CALIB_BOARD_COUNT * markers_per_board()
    available = dictionary().bytesList.shape[0]
    if needed > available:
        raise ValueError(
            f"{config.CALIB_ARUCO_DICT} holds {available} markers but "
            f"{config.CALIB_BOARD_COUNT} boards of "
            f"{config.CALIB_CHARUCO_SQUARES_X}x{config.CALIB_CHARUCO_SQUARES_Y} "
            f"squares need {needed}. Use a larger dictionary (e.g. DICT_5X5_250 "
            f"or DICT_5X5_1000) or fewer boards."
        )


def build_board(index: int = 0, square_size_m: float | None = None) -> "cv2.aruco.CharucoBoard":
    """Build board `index` from config geometry.

    Args:
        index: which board (0-based). Selects the dictionary ID slice.
        square_size_m: override the square size. Detection must use the
            MEASURED size (config.CALIB_SQUARE_SIZE_M) because that is what
            physically exists in front of the camera; rendering must use the
            NOMINAL size, because the printer applies its own scale factor on
            top. Passing the measured size to the renderer would scale the
            print a second time.
    """
    check_dictionary_capacity()
    if square_size_m is None:
        square_size_m = config.CALIB_SQUARE_SIZE_M
    # Preserve the configured marker-to-square ratio when the size is overridden,
    # so an override cannot silently change the board's proportions.
    marker_size_m = square_size_m * (config.CALIB_MARKER_SIZE_M / config.CALIB_SQUARE_SIZE_M)
    return cv2.aruco.CharucoBoard(
        (config.CALIB_CHARUCO_SQUARES_X, config.CALIB_CHARUCO_SQUARES_Y),
        square_size_m,
        marker_size_m,
        dictionary(),
        board_ids(index),
    )


def build_detectors(count: int | None = None) -> list[tuple[int, "cv2.aruco.CharucoBoard", "cv2.aruco.CharucoDetector"]]:
    """One (index, board, detector) triple per board.

    Each board needs its own detector because a CharucoDetector is bound to a
    specific board's ID set and layout. Running all of them over one frame is
    what lets several tiled boards each contribute independently.
    """
    if count is None:
        count = config.CALIB_BOARD_COUNT
    out = []
    for i in range(count):
        b = build_board(i)
        out.append((i, b, cv2.aruco.CharucoDetector(b)))
    return out


def detect(detector, gray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect one board, returning (corners, ids) or (None, None).

    Normalises shapes and rejects self-inconsistent output. OpenCV 5 returns
    charuco corners as either (N,1,2) or (N,2); those look identical to len()
    but count differently as a cv::Mat (N vs 2N), which trips size assertions
    downstream. Worse, on a marginal frame the corner and ID counts can
    genuinely disagree, and board.matchImagePoints does NOT check — it would
    pair image corners with the wrong board coordinates and yield a
    confidently wrong calibration. A frame whose own detector output is
    inconsistent is not one to calibrate from, so it is dropped entirely.
    """
    corners, ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    if corners is None or ids is None:
        return None, None
    corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    ids = np.asarray(ids, np.int32).reshape(-1, 1)
    if len(corners) != len(ids) or len(corners) == 0:
        return None, None
    return corners, ids

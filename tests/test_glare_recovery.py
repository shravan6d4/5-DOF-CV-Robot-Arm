"""Regression tests for specular-glare recovery in the red mask.

THE FAILURE THESE PIN. On 2026-08-05 a red brick on the ChArUco board flickered
in and out of detection frame to frame. The cause is not marginal thresholds: a
stud highlight bright enough to lie ACROSS the brick's silhouette cuts the red
mask in two, and the fragments individually fall under MIN_CONTOUR_AREA, so
ColorDetector emits no candidate at all. close_contour_gaps cannot help --
it repairs a contour that was already found, and here there is none.

Each test below builds the brick on a synthetic white/black checkerboard rather
than a plain background, because the board is the thing that makes naive fixes
dangerous: a white square is exactly as bright and as desaturated as a stud
highlight. A repair that keys on brightness alone passes on a grey background
and swallows the workspace on a real one.

The false-positive direction is covered by the sample-photo tests in
test_lego_detector.py (a red cup and a hand must still be rejected). Two
plausible-looking refinements to this repair were caught by exactly those,
and the rejection is recorded in config.py beside the settings.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.detection.color_detector import (  # noqa: E402
    ColorDetector,
    fill_interior_holes,
    recover_specular_regions,
)
from vision_pipeline.detection.lego_detector import LegoBrickDetector  # noqa: E402


def checkerboard(h: int = 480, w: int = 640, square: int = 40) -> np.ndarray:
    """The workspace as the camera sees it: white/black ChArUco tiling."""
    bg = np.zeros((h, w, 3), np.uint8)
    for y in range(0, h, square):
        for x in range(0, w, square):
            if ((x // square) + (y // square)) % 2 == 0:
                bg[y:y + square, x:x + square] = 235
    return bg


def brick_with_glare(bg, x=280, y=200, w=56, h=48, band=0, cross=False):
    """A red brick with studs, optionally cut by blown-out highlight bands."""
    frame = bg.copy()
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 200), -1)
    for cx in (x + w // 4, x + 3 * w // 4):
        for cy in (y + h // 4, y + 3 * h // 4):
            cv2.circle(frame, (cx, cy), max(3, w // 7), (60, 60, 225), -1)
    if band:
        cv2.rectangle(frame, (x, y + h // 2 - band // 2),
                      (x + w, y + h // 2 + band // 2), (250, 250, 250), -1)
    if cross:
        cv2.rectangle(frame, (x + w // 2 - band // 2, y),
                      (x + w // 2 + band // 2, y + h), (250, 250, 250), -1)
    return frame


# --- the failure, and the fix ----------------------------------------------

def test_glare_band_fragments_the_brick_without_recovery():
    """Establishes the failure is real before asserting it is fixed.

    Without this, the fix test could pass because the frame was never hard.
    """
    frame = brick_with_glare(checkerboard(), band=16)
    blobs = ColorDetector(glare_recovery=False).detect(frame)
    assert len(blobs) >= 2, "the highlight should have split the silhouette"
    assert max(b.area for b in blobs) < 1500, "each fragment is a fraction of the brick"


def test_recovery_reunites_a_brick_cut_by_a_highlight():
    frame = brick_with_glare(checkerboard(), band=16)
    blobs = ColorDetector(glare_recovery=True).detect(frame)
    assert len(blobs) == 1
    # The whole 56x48 silhouette, not a fragment of it.
    assert blobs[0].area > 2000


def test_cross_glare_is_detected_as_a_brick_only_with_recovery():
    """The flicker case: without recovery there is no candidate at all."""
    frame = brick_with_glare(checkerboard(), band=14, cross=True)

    without = LegoBrickDetector()
    without.color_detector.glare_recovery = False
    assert without.detect(frame) == []

    with_recovery = LegoBrickDetector()
    with_recovery.color_detector.glare_recovery = True
    found = with_recovery.detect(frame)
    assert found, "recovery should restore the brick"
    assert found[0].confidence > 0.5


def test_recovery_lifts_confidence_off_the_threshold():
    """Flicker is a confidence sitting on the threshold, not a hard failure.

    A brick scoring just above DETECTION_CONFIDENCE_THRESHOLD appears and
    disappears with ordinary frame-to-frame noise, which is what the operator
    actually saw. The repair has to move it clear, not merely over the line.
    """
    frame = brick_with_glare(checkerboard(), band=16)

    without = LegoBrickDetector()
    without.color_detector.glare_recovery = False
    low = without.detect(frame)[0].confidence

    with_recovery = LegoBrickDetector()
    with_recovery.color_detector.glare_recovery = True
    high = with_recovery.detect(frame)[0].confidence

    assert high > low + 0.3


# --- the safety property ----------------------------------------------------

def test_bare_board_yields_no_detection():
    """The board alone is bright and desaturated everywhere. Nothing may be added."""
    board = checkerboard()
    assert ColorDetector(glare_recovery=True).detect(board) == []


def test_recovery_does_not_absorb_the_board_around_the_brick():
    """Area must stay the brick's own, not grow into the white squares it sits on."""
    frame = brick_with_glare(checkerboard(), w=56, h=48, band=16)
    blob = ColorDetector(glare_recovery=True).detect(frame)[0]
    assert blob.area <= 56 * 48 * 1.15, "recovery leaked outside the brick"


def test_recovery_leaves_a_clean_frame_untouched():
    """No glare, no change — the repair must be inert when there is nothing to fix."""
    frame = brick_with_glare(checkerboard())
    plain = ColorDetector(glare_recovery=False).detect(frame)
    repaired = ColorDetector(glare_recovery=True).detect(frame)
    assert len(plain) == len(repaired) == 1
    assert abs(plain[0].area - repaired[0].area) < 1.0


def test_unenclosed_highlight_is_not_added():
    """A bright region beside red, not surrounded by it, stays out of the mask."""
    frame = np.full((200, 200, 3), 30, np.uint8)
    cv2.rectangle(frame, (40, 80), (90, 130), (40, 40, 200), -1)     # red
    cv2.rectangle(frame, (95, 80), (145, 130), (250, 250, 250), -1)  # white, adjacent
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 150, 90), (8, 255, 255))
    out = recover_specular_regions(mask, hsv)
    assert np.count_nonzero(out) == np.count_nonzero(mask)


# --- hole filling -----------------------------------------------------------

def test_fill_interior_holes_closes_a_punched_hole():
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
    cv2.circle(mask, (50, 50), 10, 0, -1)
    filled = fill_interior_holes(mask)
    assert filled[50, 50] == 255
    assert np.count_nonzero(filled) > np.count_nonzero(mask)


def test_fill_interior_holes_cannot_grow_outward():
    """Filling external contours can only add pixels they already enclose."""
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (20, 20), (40, 40), 255, -1)
    cv2.rectangle(mask, (60, 60), (80, 80), 255, -1)
    filled = fill_interior_holes(mask)
    assert np.array_equal(filled, mask)


def test_fill_interior_holes_handles_an_empty_mask():
    empty = np.zeros((50, 50), np.uint8)
    assert np.array_equal(fill_interior_holes(empty), empty)

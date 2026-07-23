"""
Unit tests for shape_detector — the geometric "is this brick-shaped?" signal
that LegoBrickDetector blends with stud count into a single confidence.

Each test builds a real OpenCV contour (via cv2.findContours on a drawn mask)
rather than hand-crafting point arrays, so the scores are exercised exactly
the way LegoBrickDetector will call them.
"""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.detection import shape_detector


def _contour_from_mask(draw_fn) -> np.ndarray:
    mask = np.zeros((300, 300), dtype=np.uint8)
    draw_fn(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    return contours[0]


def make_rectangle_contour() -> np.ndarray:
    return _contour_from_mask(
        lambda mask: cv2.rectangle(mask, (60, 100), (240, 200), color=255, thickness=-1)
    )


def make_circle_contour() -> np.ndarray:
    return _contour_from_mask(
        lambda mask: cv2.circle(mask, (150, 150), 100, color=255, thickness=-1)
    )


def make_thin_sliver_contour() -> np.ndarray:
    return _contour_from_mask(
        lambda mask: cv2.rectangle(mask, (10, 148), (290, 152), color=255, thickness=-1)
    )


def test_score_rectangularity_high_for_rectangle():
    assert shape_detector.score_rectangularity(make_rectangle_contour()) > 0.9


def test_score_rectangularity_low_for_circle():
    # A circle's fill ratio of its own bounding square is capped at pi/4
    # (~0.785), below SHAPE_RECT_SCORE_LOW (0.85) — score should be 0.
    assert shape_detector.score_rectangularity(make_circle_contour()) == 0.0


def test_score_aspect_ratio_plausible_for_square_brick():
    assert shape_detector.score_aspect_ratio(make_rectangle_contour()) == 1.0


def test_score_aspect_ratio_low_for_thin_sliver():
    # ~280:4 aspect ratio is far outside a plausible Lego footprint.
    assert shape_detector.score_aspect_ratio(make_thin_sliver_contour()) < 0.2


def test_score_shape_high_for_rectangle_low_for_circle():
    rectangle_score = shape_detector.score_shape(make_rectangle_contour())
    circle_score = shape_detector.score_shape(make_circle_contour())

    assert rectangle_score > 0.9
    assert circle_score < 0.3
    assert rectangle_score > circle_score

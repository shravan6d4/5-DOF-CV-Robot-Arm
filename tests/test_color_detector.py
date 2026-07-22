"""
Unit-style tests for ColorDetector, the pure color-thresholding stage.

ColorDetector only answers "where are the red regions?" — it does NOT decide
whether a red region is a Lego brick (that's LegoBrickDetector, tested in
test_lego_detector.py). So these tests use synthetic images with a known,
exact answer and need no camera or photos.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.detection.color_detector import ColorDetector


def make_synthetic_red_square_frame() -> np.ndarray:
    """Build a fake 'photo': a gray background with a solid red square on it.

    This lets us test the detection math (mask -> contour -> centroid) with
    a known, exact answer, without needing a real camera or photo.
    """
    frame = np.full((480, 640, 3), fill_value=(120, 120, 120), dtype=np.uint8)  # gray BGR
    # cv2.rectangle draws in BGR order: (Blue, Green, Red) = pure red is (0, 0, 255).
    top_left = (250, 150)
    bottom_right = (350, 250)
    cv2.rectangle(frame, top_left, bottom_right, color=(0, 0, 255), thickness=-1)
    expected_centroid = (
        (top_left[0] + bottom_right[0]) / 2,
        (top_left[1] + bottom_right[1]) / 2,
    )
    return frame, expected_centroid


def test_detects_synthetic_red_square():
    frame, expected_centroid = make_synthetic_red_square_frame()
    detector = ColorDetector()

    detections = detector.detect(frame)

    assert len(detections) == 1, "Expected exactly one red blob in the synthetic frame"
    detected = detections[0]

    # Allow a few pixels of tolerance for contour/moment rounding.
    assert detected.centroid_px[0] == pytest.approx(expected_centroid[0], abs=3)
    assert detected.centroid_px[1] == pytest.approx(expected_centroid[1], abs=3)


def test_no_detection_on_plain_frame():
    plain_frame = np.full((480, 640, 3), fill_value=(120, 120, 120), dtype=np.uint8)
    detector = ColorDetector()

    detections = detector.detect(plain_frame)

    assert detections == []

"""
Unit tests for stud_detector: the resolution-adaptive ROI upscale (Fix A,
recovers studs on a brick that's small in-frame because it's far from the
camera) and the specular-highlight suppression (Fix B, recovers studs whose
circular pattern is broken by a glare spot).

These operate directly on synthetic grayscale/HSV arrays with a manually
built rectangular contour, rather than going through ColorDetector — the
contour only needs to describe the region to search, and a plain 4-corner
rectangle is exact and easy to reason about.
"""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.detection import stud_detector


def make_rect_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array(
        [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32
    )


def make_tiny_stud_pattern_frame(x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    """A brick-like grayscale stud pattern shrunk to ~1/5 scale (rect 53x44,
    studs radius ~5px) — small enough that it sits well under
    STUD_ROI_REFERENCE_PX (220), simulating a brick far from the camera.
    Proportions mirror test_lego_detector.make_brick_like_frame's known-good
    240x200 rect / radius-22 studs, scaled by ~0.22.

    Returns (gray_frame, contour).
    """
    frame = np.full((300, 400), fill_value=180, dtype=np.uint8)
    w, h = 53, 44
    cv2.rectangle(frame, (x, y), (x + w, y + h), color=140, thickness=-1)
    for dy in (13, 31):
        for dx in (9, 22, 35, 44):
            cv2.circle(frame, (x + dx, y + dy), 5, color=130, thickness=-1)
            cv2.circle(frame, (x + dx, y + dy), 5, color=90, thickness=1)
    contour = make_rect_contour(x, y, w, h)
    return frame, contour


def test_resolution_adaptive_stud_detection_recovers_tiny_pattern(monkeypatch):
    frame, contour = make_tiny_stud_pattern_frame(20, 20)

    count_with_upscale = stud_detector.count_studs(frame, contour)
    assert count_with_upscale >= 2, (
        "Upscaling a too-small ROI to the reference resolution should let "
        "Hough resolve at least some of the tiny studs"
    )

    # Disabling the upscale (reference below the native ROI size, so the
    # "smaller_side < reference" check never triggers) reproduces the old,
    # unfixed behavior on this tiny pattern.
    monkeypatch.setattr(config, "STUD_ROI_REFERENCE_PX", 1)
    count_without_upscale = stud_detector.count_studs(frame, contour)

    assert count_without_upscale < count_with_upscale, (
        "Without the resolution-adaptive upscale, the tiny stud pattern "
        "should be harder (not easier) to resolve"
    )


def test_count_studs_rescales_circles_to_full_frame_coordinates():
    x, y = 100, 150
    frame, contour = make_tiny_stud_pattern_frame(x, y)
    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)

    count, circles = stud_detector.count_studs(frame, contour, return_circles=True)

    assert count >= 1
    assert circles is not None
    for cx, cy, _radius in circles:
        assert bbox_x <= cx <= bbox_x + bbox_w, "circle x should land back inside the original bbox"
        assert bbox_y <= cy <= bbox_y + bbox_h, "circle y should land back inside the original bbox"


def make_brick_pattern_with_glare(
    glare_on_one_stud: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """A brick-sized (native-resolution, no upscale needed) BGR stud pattern,
    optionally with a near-white blown highlight painted over one stud —
    simulating specular glare off a real, convex stud. Returns
    (gray_frame, hsv_frame, contour, glare_stud_center).
    """
    frame_bgr = np.full((480, 640, 3), fill_value=(120, 120, 120), dtype=np.uint8)
    top_left, bottom_right = (200, 140), (440, 340)
    cv2.rectangle(frame_bgr, top_left, bottom_right, color=(0, 0, 255), thickness=-1)

    stud_centers = [(cx, cy) for cy in (200, 280) for cx in (240, 300, 360, 400)]
    for cx, cy in stud_centers:
        cv2.circle(frame_bgr, (cx, cy), 22, color=(0, 0, 200), thickness=-1)
        cv2.circle(frame_bgr, (cx, cy), 22, color=(0, 0, 120), thickness=2)

    glare_stud_center = stud_centers[0]
    if glare_on_one_stud:
        cv2.circle(frame_bgr, glare_stud_center, 18, color=(250, 250, 250), thickness=-1)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    contour = make_rect_contour(top_left[0], top_left[1],
                                 bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    return gray, hsv, contour, glare_stud_center


def _closest_circle_distance(circles, center: tuple[int, int]) -> float:
    return min(float(np.hypot(cx - center[0], cy - center[1])) for cx, cy, _r in circles)


def test_specular_highlight_suppression_recovers_glare_covered_stud():
    gray, hsv, contour, glare_center = make_brick_pattern_with_glare(glare_on_one_stud=True)

    _, circles_without = stud_detector.count_studs(gray, contour, return_circles=True, hsv_frame=None)
    _, circles_with = stud_detector.count_studs(gray, contour, return_circles=True, hsv_frame=hsv)

    # Without suppression, Hough tends to lock onto the glare disk's own sharp
    # edge instead of the true stud rim underneath it, landing a circle that's
    # offset from the real center. Suppressing the glare first should recover
    # a circle much closer to the true, known stud center.
    dist_without = _closest_circle_distance(circles_without, glare_center)
    dist_with = _closest_circle_distance(circles_with, glare_center)

    assert dist_with < dist_without, (
        "Suppressing the glare patch before Hough should land a circle closer "
        "to the true stud center than the glare disk's own edge would"
    )
    assert dist_with <= 2.0, "with suppression, the recovered circle should be a near-exact match"


def test_suppress_specular_highlights_neutralizes_flagged_pixels():
    roi_gray = np.full((60, 60), fill_value=140, dtype=np.uint8)
    roi_gray[20:40, 20:40] = 250  # a blown-out patch on the grayscale image too

    roi_hsv = np.zeros((60, 60, 3), dtype=np.uint8)
    roi_hsv[..., 1] = 200  # baseline: well-saturated (not glare)
    roi_hsv[..., 2] = 200  # baseline: bright but not blown out
    roi_hsv[20:40, 20:40, 1] = 10    # low saturation
    roi_hsv[20:40, 20:40, 2] = 250   # very bright -> flagged as glare

    result = stud_detector.suppress_specular_highlights(roi_gray, roi_hsv)

    assert result.shape == roi_gray.shape
    # The flagged patch should be repaired toward the surrounding value, not
    # left at the blown-out 250.
    assert result[25:35, 25:35].mean() < 200
    # Untouched pixels outside the flagged patch must be unaffected.
    assert np.array_equal(result[0:15, 0:15], roi_gray[0:15, 0:15])

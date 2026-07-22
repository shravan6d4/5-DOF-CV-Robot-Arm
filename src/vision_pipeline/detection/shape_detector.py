"""
Scores how "brick-shaped" a red region is, independent of stud detection.

Why this exists: stud_detector.count_studs can legitimately return 0 even for
a real brick — the brick may be too far away for individual studs to resolve,
or the studs may be glare-corrupted. This module gives LegoBrickDetector a
second, independent signal ("is this a rectangular blob with plausible brick
proportions?") that can corroborate a detection when stud evidence is weak,
instead of studs being an all-or-nothing gate.

How it works: purely geometric, from the same contour ColorDetector already
produced — no new image processing, no new failure mode tied to lighting or
distance.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision_pipeline import config


def score_rectangularity(contour: np.ndarray) -> float:
    """How completely the contour fills its own oriented bounding rectangle.

    Computed as contourArea / minAreaRect area, then remapped from
    [SHAPE_RECT_SCORE_LOW, SHAPE_RECT_SCORE_HIGH] to [0, 1]. Any circle or
    ellipse is mathematically capped at pi/4 (~0.785) fill ratio of its own
    bounding rectangle regardless of size or elongation — the closest a
    smooth round object (cup rim, fist) can get — while a real photographed
    rectangular brick realistically reaches ~0.85-1.0. The remap turns that
    known gap into a deliberate boxy-vs-round separation.
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0

    (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
    rect_area = rect_w * rect_h
    if rect_area <= 0:
        return 0.0

    fill_ratio = area / rect_area
    lo, hi = config.SHAPE_RECT_SCORE_LOW, config.SHAPE_RECT_SCORE_HIGH
    return float(np.clip((fill_ratio - lo) / (hi - lo), 0.0, 1.0))


def score_aspect_ratio(contour: np.ndarray) -> float:
    """Plausibility of the contour's long:short side ratio as a Lego footprint.

    1.0 inside [SHAPE_ASPECT_MIN, SHAPE_ASPECT_MAX] (covers everything from a
    square brick up to a generously elongated one), decaying linearly to 0.0
    one band-width outside — partial credit near the edges rather than a hard
    cutoff.
    """
    (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
    short_side, long_side = sorted((rect_w, rect_h))
    if short_side <= 0:
        return 0.0

    ratio = long_side / short_side
    lo, hi = config.SHAPE_ASPECT_MIN, config.SHAPE_ASPECT_MAX
    if lo <= ratio <= hi:
        return 1.0
    if ratio < lo:
        return float(max(0.0, 1.0 - (lo - ratio) / lo))
    return float(max(0.0, 1.0 - (ratio - hi) / (hi - lo)))


def score_shape(contour: np.ndarray) -> float:
    """Combined "brick-shaped" score in [0, 1], independent of stud detection.

    Rectangularity times aspect-ratio plausibility, not a weighted average.
    Aspect ratio alone is a weak signal — a hand or arm silhouette can easily
    fall inside a "plausible brick" elongation band by chance — so it must
    never contribute credit to a shape that isn't already rectangular; it can
    only narrow down a candidate rectangularity has already judged as boxy.
    """
    return score_rectangularity(contour) * score_aspect_ratio(contour)

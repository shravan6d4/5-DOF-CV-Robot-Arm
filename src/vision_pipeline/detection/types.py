"""
Plain-data result type shared by every stage of the pipeline.

Why this exists: detection code (this phase) and the future calibration /
robot-control code (later phases, written by a teammate) shouldn't need to
know about each other's internals. `Detection` is the "contract" between
them — detection code produces these, downstream code consumes them, and
neither side needs to know OpenCV was involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single detected object (e.g. one Lego brick) in a camera frame.

    Attributes:
        centroid_px: (x, y) pixel coordinates of the object's center.
        area: Area of the detected contour, in pixels^2. Useful for
            filtering noise or roughly estimating distance/size.
        bbox: Axis-aligned bounding box as (x, y, width, height) in pixels.
        angle_deg: Rotation angle of the object's minimum-area bounding
            rectangle, in degrees. Useful later for orienting the gripper.
        contour: The raw OpenCV contour (array of boundary points), kept
            around in case downstream code wants more detail than the
            summary fields above provide.
        num_studs: How many Lego studs (circular bumps) were detected inside
            this region. 0 for a plain color detection; populated by
            LegoBrickDetector.
        shape_score: How "brick-shaped" (rectangular, plausible aspect ratio)
            this region is, in [0, 1], independent of stud detection. 0.0 for
            a plain color detection; populated by LegoBrickDetector.
        confidence: Blended stud + shape score in [0, 1] that LegoBrickDetector
            gated this detection on. Informational for downstream code/
            debugging, like num_studs and shape_score — never consulted by
            the pick pipeline.
    """

    centroid_px: tuple[float, float]
    area: float
    bbox: tuple[int, int, int, int]
    angle_deg: float
    contour: np.ndarray
    num_studs: int = 0
    shape_score: float = 0.0
    confidence: float = 0.0

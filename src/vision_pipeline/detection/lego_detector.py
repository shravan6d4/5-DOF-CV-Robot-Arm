"""
Phase 1 (improved): detect *Lego bricks* specifically, not just any red thing.

This builds directly on ColorDetector in a two-stage pipeline:

  Stage 1 (color):  find red regions in the frame  -> candidate blobs
  Stage 2 (studs):  keep only regions that have enough Lego studs (bumps)

Stage 1 alone can't tell a red brick from a red cup or a reddish hand — they're
all "red." Stage 2 verifies each candidate really is a brick by checking for
the studs, using stud_detector.count_studs. This is why the detector rejects a
red plastic cup (vivid red but smooth, no studs) while accepting a brick.

Keeping the two stages separate (rather than jamming everything into one class)
means the plain color stage stays simple and reusable, and the "is it really a
Lego" logic lives in one obvious place you can tune independently.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.detection import stud_detector
from vision_pipeline.detection.color_detector import ColorDetector
from vision_pipeline.detection.types import Detection


class LegoBrickDetector:
    """Detects red Lego bricks by combining color detection with stud verification.

    Usage:
        detector = LegoBrickDetector()
        bricks = detector.detect(frame)      # frame is a BGR np.ndarray
        if bricks:
            target = bricks[0]               # largest brick, has .centroid_px etc.
    """

    def __init__(
        self,
        color_detector: ColorDetector | None = None,
        min_studs: int = config.MIN_STUDS,
    ) -> None:
        # Reuse ColorDetector for stage 1. If none is supplied we build one with
        # the default (Lego-red) HSV thresholds from config.
        self.color_detector = color_detector or ColorDetector()
        self.min_studs = min_studs

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect Lego bricks in a BGR frame.

        Returns Detection objects (with `num_studs` filled in) for regions that
        are both the right color AND have at least `min_studs` studs, sorted
        largest-area first. Returns an empty list if no brick is found.
        """
        # Stage 1: color candidates (these are red blobs, brick or not).
        candidates = self.color_detector.detect(frame_bgr)

        # Stage 2 needs a grayscale copy of the frame to look for studs.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        bricks: list[Detection] = []
        for candidate in candidates:
            num_studs = stud_detector.count_studs(gray, candidate.contour)
            if num_studs < self.min_studs:
                continue  # red, but not enough studs -> not a Lego brick

            # Re-create the detection with the stud count recorded, so
            # downstream code (and debugging) can see how sure we are.
            bricks.append(
                Detection(
                    centroid_px=candidate.centroid_px,
                    area=candidate.area,
                    bbox=candidate.bbox,
                    angle_deg=candidate.angle_deg,
                    contour=candidate.contour,
                    num_studs=num_studs,
                )
            )

        bricks.sort(key=lambda d: d.area, reverse=True)
        return bricks

    def draw_debug_overlay(self, frame_bgr: np.ndarray, bricks: list[Detection]) -> np.ndarray:
        """Return a copy of the frame with detected bricks, studs, and centroids drawn."""
        overlay = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        for brick in bricks:
            cv2.drawContours(overlay, [brick.contour], -1, (0, 255, 0), 2)

            # Draw the individual studs we found, so you can see what convinced
            # the detector this was a brick.
            _, circles = stud_detector.count_studs(gray, brick.contour, return_circles=True)
            if circles is not None:
                for cx, cy, r in circles:
                    cv2.circle(overlay, (cx, cy), r, (255, 255, 0), 2)

            x, y = int(brick.centroid_px[0]), int(brick.centroid_px[1])
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                f"brick ({x},{y}) studs={brick.num_studs}",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        return overlay

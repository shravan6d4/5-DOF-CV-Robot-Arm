"""
Phase 1 (improved): detect *Lego bricks* specifically, not just any red thing.

This builds directly on ColorDetector in a two-stage pipeline:

  Stage 1 (color): find red regions in the frame -> candidate blobs
  Stage 2 (verify): score each candidate on studs AND shape, blend the two
                     into one confidence, keep it if that confidence is high
                     enough

Stage 1 alone can't tell a red brick from a red cup or a reddish hand — they're
all "red." Stage 2 first repairs each candidate's contour with
color_detector.close_contour_gaps (specular highlights on the studs can punch
holes clean through the color mask), then verifies it's really a brick using
two independent signals: stud_detector.count_studs (the studs' bumpy circular
pattern) and shape_detector.score_shape (how rectangular/brick-proportioned
the blob's silhouette is). They're blended into a single confidence rather
than gated on studs alone, because studs can be legitimately unresolvable —
the brick may be far from the camera, or its studs' specular glare may survive
suppression — in which case a strongly brick-shaped, correctly-proportioned
blob can still confirm the detection.

Keeping the stages separate (rather than jamming everything into one class)
means the plain color stage stays simple and reusable, and the "is it really a
Lego" logic lives in one obvious place you can tune independently.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.detection import shape_detector, stud_detector
from vision_pipeline.detection.color_detector import ColorDetector, close_contour_gaps
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
        confidence_threshold: float = config.DETECTION_CONFIDENCE_THRESHOLD,
    ) -> None:
        # Reuse ColorDetector for stage 1. If none is supplied we build one with
        # the default (Lego-red) HSV thresholds from config.
        self.color_detector = color_detector or ColorDetector()
        self.confidence_threshold = confidence_threshold

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect Lego bricks in a BGR frame.

        Returns Detection objects (with `num_studs`, `shape_score`, and
        `confidence` filled in) for regions that are the right color AND
        whose blended stud + shape confidence clears `confidence_threshold`,
        sorted largest-area first. Studs and shape are independent signals —
        a candidate can be confirmed by shape alone when studs aren't
        resolvable (brick far from the camera, or stud glare). Returns an
        empty list if no brick is found.
        """
        # Stage 1: color candidates (these are red blobs, brick or not).
        candidates = self.color_detector.detect(frame_bgr)

        # Stage 2 needs a grayscale copy to look for studs, and an HSV copy to
        # suppress specular highlights on them before counting.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        bricks: list[Detection] = []
        for candidate in candidates:
            # Specular highlights on the studs can punch holes clean through
            # the color contour (see close_contour_gaps), fragmenting the
            # silhouette both stud counting and shape scoring rely on. Repair
            # that once per candidate before running either signal; the
            # Detection recorded below still uses the ORIGINAL candidate
            # geometry (centroid/area/bbox/contour) for grasping.
            repaired_contour = close_contour_gaps(gray.shape, candidate.contour)

            num_studs = stud_detector.count_studs(gray, repaired_contour, hsv_frame=hsv)
            stud_score = min(1.0, num_studs / config.STUD_FULL_CREDIT_COUNT)
            shape_score = shape_detector.score_shape(repaired_contour)
            confidence = config.STUD_WEIGHT * stud_score + config.SHAPE_WEIGHT * shape_score

            if confidence < self.confidence_threshold:
                continue  # red, but neither studs nor shape support "brick"

            # Re-create the detection with the scores recorded, so downstream
            # code (and debugging) can see how sure we are and why.
            bricks.append(
                Detection(
                    centroid_px=candidate.centroid_px,
                    area=candidate.area,
                    bbox=candidate.bbox,
                    angle_deg=candidate.angle_deg,
                    contour=candidate.contour,
                    num_studs=num_studs,
                    shape_score=shape_score,
                    confidence=confidence,
                )
            )

        bricks.sort(key=lambda d: d.area, reverse=True)
        return bricks

    def draw_debug_overlay(self, frame_bgr: np.ndarray, bricks: list[Detection]) -> np.ndarray:
        """Return a copy of the frame with detected bricks, studs, and centroids drawn."""
        overlay = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        for brick in bricks:
            cv2.drawContours(overlay, [brick.contour], -1, (0, 255, 0), 2)

            # Draw the individual studs we found, so you can see what convinced
            # the detector this was a brick. Repair gaps the same way detect()
            # did, so the circles drawn here match the num_studs it recorded.
            repaired_contour = close_contour_gaps(gray.shape, brick.contour)
            _, circles = stud_detector.count_studs(
                gray, repaired_contour, return_circles=True, hsv_frame=hsv
            )
            if circles is not None:
                for cx, cy, r in circles:
                    cv2.circle(overlay, (cx, cy), r, (255, 255, 0), 2)

            x, y = int(brick.centroid_px[0]), int(brick.centroid_px[1])
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                f"brick ({x},{y}) studs={brick.num_studs} shape={brick.shape_score:.2f} "
                f"conf={brick.confidence:.2f}",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        return overlay

"""
Phase 1 core: find a colored object (e.g. a red Lego brick) in a BGR frame
using HSV thresholding + contour detection.

Concepts introduced here, if you're new to OpenCV:

- BGR vs HSV: OpenCV loads/captures color images as BGR (Blue, Green, Red)
  channels. HSV (Hue, Saturation, Value) is a different way of representing
  the same colors that separates "what color" (hue) from "how washed out"
  (saturation) and "how bright" (value). It's much easier to threshold on
  hue than to threshold on BGR, because lighting changes mostly affect
  saturation/value, not hue.

- Thresholding / masking: `cv2.inRange` produces a "mask" — a black & white
  image where white pixels are ones that fell inside your HSV range, and
  black pixels didn't. That mask is what we search for contours in.

- Morphological operations: camera noise creates tiny stray white specks in
  the mask, and can also poke small black holes inside an otherwise solid
  white blob. "Opening" (erode-then-dilate) removes small specks; "closing"
  (dilate-then-erode) fills small holes. We do both to clean up the mask.

- Contours: a contour is just a curve joining the continuous points along a
  boundary of a white region in the mask — effectively OpenCV's way of
  saying "here's the outline of a blob." We find the biggest one (assuming
  one brick in frame) and summarize it into a Detection.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.detection.types import Detection


class ColorDetector:
    """Finds colored blobs in a BGR frame using HSV thresholding.

    Usage:
        detector = ColorDetector()
        detections = detector.detect(frame)  # frame is a BGR np.ndarray
    """

    def __init__(
        self,
        hsv_lower_1: tuple[int, int, int] = config.HSV_LOWER_1,
        hsv_upper_1: tuple[int, int, int] = config.HSV_UPPER_1,
        hsv_lower_2: tuple[int, int, int] | None = config.HSV_LOWER_2,
        hsv_upper_2: tuple[int, int, int] | None = config.HSV_UPPER_2,
        min_contour_area: float = config.MIN_CONTOUR_AREA,
        morph_kernel_size: int = config.MORPH_KERNEL_SIZE,
    ) -> None:
        self.hsv_lower_1 = np.array(hsv_lower_1, dtype=np.uint8)
        self.hsv_upper_1 = np.array(hsv_upper_1, dtype=np.uint8)

        # A second HSV range is optional. Red needs two ranges because red's
        # hue sits at both ends (0 and 179) of OpenCV's hue scale. Other
        # colors (e.g. a blue or yellow brick) only need one range — pass
        # hsv_lower_2=None in that case.
        self.hsv_lower_2 = np.array(hsv_lower_2, dtype=np.uint8) if hsv_lower_2 else None
        self.hsv_upper_2 = np.array(hsv_upper_2, dtype=np.uint8) if hsv_upper_2 else None

        self.min_contour_area = min_contour_area
        self._morph_kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    def _build_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """Threshold the HSV frame into a black/white mask, then clean it up."""
        mask = cv2.inRange(hsv_frame, self.hsv_lower_1, self.hsv_upper_1)

        if self.hsv_lower_2 is not None and self.hsv_upper_2 is not None:
            mask_2 = cv2.inRange(hsv_frame, self.hsv_lower_2, self.hsv_upper_2)
            mask = cv2.bitwise_or(mask, mask_2)

        # Opening: erode then dilate — removes small speckle noise.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        # Closing: dilate then erode — fills small holes inside the blob.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

        return mask

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect colored blobs in a BGR frame.

        Returns a list of Detection objects, sorted largest-area first, so
        `detections[0]` is your best guess at "the brick" when only one is
        expected in frame. Returns an empty list if nothing was found.
        """
        hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(hsv_frame)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue  # too small — likely noise, not a real brick

            # Centroid via image moments (the standard OpenCV way to get a
            # shape's "center of mass" from its contour).
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue  # degenerate contour, avoid divide-by-zero
            centroid_x = moments["m10"] / moments["m00"]
            centroid_y = moments["m01"] / moments["m00"]

            bbox = cv2.boundingRect(contour)  # (x, y, w, h)

            # Minimum-area rotated rectangle gives us an orientation angle,
            # useful later for lining up the gripper with the brick.
            (_, _), (_, _), angle = cv2.minAreaRect(contour)

            detections.append(
                Detection(
                    centroid_px=(centroid_x, centroid_y),
                    area=area,
                    bbox=bbox,
                    angle_deg=angle,
                    contour=contour,
                )
            )

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    def draw_debug_overlay(self, frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Return a copy of the frame with detections drawn on it, for visual debugging."""
        overlay = frame_bgr.copy()
        for detection in detections:
            cv2.drawContours(overlay, [detection.contour], -1, (0, 255, 0), 2)

            x, y = int(detection.centroid_px[0]), int(detection.centroid_px[1])
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                f"({x}, {y})",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        return overlay

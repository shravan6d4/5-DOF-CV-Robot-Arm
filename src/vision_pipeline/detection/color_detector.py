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
        glare_recovery: bool = config.GLARE_RECOVERY,
    ) -> None:
        self.glare_recovery = glare_recovery
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

        # Opening: erode then dilate — removes small speckle noise. Done BEFORE
        # glare recovery so the recovery is judged against a clean red mask
        # rather than against noise specks.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        if self.glare_recovery:
            mask = recover_specular_regions(mask, hsv_frame)

        # Closing: dilate then erode — fills small holes inside the blob.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
        # Fill whatever holes remain. A hole strictly inside a red silhouette is
        # always a defect of the threshold, never information: no real brick has
        # a window through it. Cheap, and it cannot leak outward, since filling
        # external contours can only add pixels they already enclose.
        mask = fill_interior_holes(mask)

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


def fill_interior_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every hole strictly inside a masked region.

    Re-drawing the external contours filled is the whole trick: RETR_EXTERNAL
    discards interior boundaries, so painting what survives can only add pixels
    those outlines already enclose. Nothing can leak outward, which is what
    makes this safe to run unconditionally.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, color=255, thickness=-1)
    return filled


def recover_specular_regions(
    mask: np.ndarray,
    hsv_frame: np.ndarray,
    v_min: int = config.GLARE_V_MIN,
    s_max: int = config.GLARE_S_MAX,
    max_region_px: int = config.GLARE_MAX_REGION_PX,
    ring_px: int = config.GLARE_RING_PX,
    enclosure_frac: float = config.GLARE_ENCLOSURE_FRAC,
) -> np.ndarray:
    """Add blown-out highlights back into the red mask — but only where enclosed by red.

    A glossy Lego stud under direct light reflects a highlight so bright that
    its saturation collapses below the red threshold. The pixels are physically
    part of the brick; the mask simply cannot see them as red. When such a
    highlight lands mid-brick it punches a hole (which closing repairs), but
    when it lands across the silhouette it CUTS THE BRICK IN TWO, leaving
    fragments that individually fall under MIN_CONTOUR_AREA. Detection then
    reports nothing at all — the flicker this function exists to remove.

    The test for "is this glare part of the brick" is what it is SURROUNDED BY,
    not how bright it is. Brightness alone would swallow the white ChArUco board
    the workspace is tiled with, which is every bit as bright and as desaturated
    as a stud highlight. So each bright region is dilated by a few pixels and
    the resulting ring is measured: mostly red means the region sits inside the
    brick and is restored; anything else is left out. A white board square fails
    because what surrounds it is board.

    Deliberately bounded, because this is the one place in detection that ADDS
    pixels a threshold rejected:
      * regions larger than `max_region_px` are never considered (a stud
        highlight is small; a lit tabletop is not);
      * only pixels not already in the mask are candidates;
      * the enclosure fraction must be met, so an unenclosed bright region
        cannot join the brick by touching it.

    Args:
        mask: the cleaned red mask.
        hsv_frame: the same frame in HSV.

    Returns:
        A new mask with enclosed highlights restored. The input is not modified.
    """
    bright = cv2.inRange(hsv_frame, (0, 0, v_min), (179, s_max, 255))
    # Only pixels the red threshold rejected can be candidates.
    bright = cv2.bitwise_and(bright, cv2.bitwise_not(mask))
    if not bright.any():
        return mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if count <= 1:
        return mask

    out = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1,) * 2)
    h, w = mask.shape[:2]

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_region_px:
            continue

        # Work inside the region's own bounding box (plus the ring margin)
        # rather than on full frames: a board-tiled scene yields dozens of
        # bright components, and dilating a full-size mask for each is the
        # difference between a usable frame rate and a slideshow.
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        x0, y0 = max(0, x - ring_px - 1), max(0, y - ring_px - 1)
        x1, y1 = min(w, x + bw + ring_px + 1), min(h, y + bh + ring_px + 1)

        region = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        ring = cv2.bitwise_and(cv2.dilate(region, kernel),
                               cv2.bitwise_not(region))
        ring_total = int(np.count_nonzero(ring))
        if ring_total == 0:
            continue

        red_in_ring = int(np.count_nonzero(
            cv2.bitwise_and(ring, mask[y0:y1, x0:x1])))
        if red_in_ring / ring_total >= enclosure_frac:
            out[y0:y1, x0:x1] = cv2.bitwise_or(out[y0:y1, x0:x1], region)

    return out


def close_contour_gaps(frame_shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    """Repair small internal gaps in a contour caused by specular dropout.

    A glossy, curved surface (e.g. a Lego stud) can reflect a near-white
    highlight that dips below the HSV saturation floor, punching a hole
    through the middle of an otherwise-solid red contour. That hole
    fragments the contour's true silhouette, which corrupts both stud
    detection (stud_detector.count_studs relies on the region inside the
    contour) and shape scoring (shape_detector.score_shape relies on the
    contour's own fill ratio and aspect ratio) downstream.

    Closes gaps with a kernel sized as a fraction of the contour's own
    smaller dimension — proportional, so it scales with distance, and
    bounded so it can't grow large enough to merge in a genuinely separate,
    unrelated blob — then returns the single largest contour of the result.
    LegoBrickDetector calls this once per color-stage candidate before
    handing it to the stud and shape stages; it does not touch the
    centroid/area/bbox ColorDetector already reported, which stay based on
    the true (uncorrected) detection.
    """
    x, y, w, h = cv2.boundingRect(contour)
    smaller_side = min(w, h)
    kernel_size = int(smaller_side * config.STUD_REGION_CLOSE_FRAC)
    kernel_size = max(config.STUD_REGION_CLOSE_MIN_PX, min(config.STUD_REGION_CLOSE_MAX_PX, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return contour
    return max(contours, key=cv2.contourArea)

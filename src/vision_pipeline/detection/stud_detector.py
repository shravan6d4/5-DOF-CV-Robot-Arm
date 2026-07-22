"""
Detects Lego "studs" — the raised circular bumps on top of a brick — inside a
region of an image.

Why this matters: color thresholding alone can find "something red," but a red
Lego brick and a red plastic cup are both red. The studs are the giveaway that
a red blob is actually a Lego brick, so counting them lets us reject red things
that aren't bricks.

How it works — the Hough Circle Transform (cv2.HoughCircles):
  A "circle" in an image is any place where edge pixels curve around a common
  center. The Hough transform has every edge pixel "vote" for the circle
  centers it could belong to; spots that collect many votes are reported as
  circles. It's the standard classical-CV way to find circular features
  without any machine learning. Here we run it on the grayscale image, but only
  look inside the red region, so we count studs on the brick and nothing else.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision_pipeline import config


def count_studs(
    gray_frame: np.ndarray,
    contour: np.ndarray,
    return_circles: bool = False,
) -> int | tuple[int, np.ndarray | None]:
    """Count Lego studs (circular bumps) inside a single detected region.

    Args:
        gray_frame: The whole frame converted to grayscale. Hough circles work
            on a single-channel (intensity) image, and studs show up clearly as
            circular light/shadow patterns regardless of color.
        contour: The contour of the red region to search inside (one candidate
            brick), as returned by the color detector.
        return_circles: If True, also return the raw detected circles (for
            drawing debug overlays). Each circle is (x, y, radius) in full-frame
            pixel coordinates.

    Returns:
        The number of studs found, or (count, circles) if return_circles=True.
    """
    # Restrict the search to just this region: build a filled mask of the
    # contour, keep only the grayscale pixels inside it, then crop to the
    # region's bounding box so Hough only looks where the brick actually is.
    region_mask = np.zeros(gray_frame.shape, dtype=np.uint8)
    cv2.drawContours(region_mask, [contour], -1, color=255, thickness=-1)
    masked_gray = cv2.bitwise_and(gray_frame, gray_frame, mask=region_mask)

    x, y, w, h = cv2.boundingRect(contour)
    roi = masked_gray[y : y + h, x : x + w]
    if roi.size == 0:
        return (0, None) if return_circles else 0

    # A light blur removes speckle noise that would otherwise create spurious
    # tiny circles. medianBlur is good at this while keeping edges crisp.
    roi = cv2.medianBlur(roi, 5)

    # Stud size and spacing scale with how big the brick appears in the frame,
    # so we express them as fractions of the brick's smaller dimension.
    smaller_side = min(w, h)
    min_radius = max(3, int(smaller_side * config.STUD_MIN_RADIUS_FRAC))
    max_radius = max(min_radius + 2, int(smaller_side * config.STUD_MAX_RADIUS_FRAC))
    min_dist = max(8, int(smaller_side * config.STUD_MIN_DIST_FRAC))

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=config.STUD_HOUGH_DP,
        minDist=min_dist,
        param1=config.STUD_HOUGH_PARAM1,
        param2=config.STUD_HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return (0, None) if return_circles else 0

    # HoughCircles returns coordinates relative to the ROI crop; shift them back
    # into full-frame coordinates so overlays line up with the original image.
    circles = np.around(circles[0]).astype(int)
    circles[:, 0] += x
    circles[:, 1] += y

    count = len(circles)
    return (count, circles) if return_circles else count

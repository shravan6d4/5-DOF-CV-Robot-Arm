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


def suppress_specular_highlights(roi_gray: np.ndarray, roi_hsv: np.ndarray) -> np.ndarray:
    """Neutralize blown-out glare pixels (high V, low S) inside an ROI.

    A curved, glossy stud reflects a near-white highlight back at the camera;
    that highlight breaks the clean circular edge cv2.HoughCircles needs to
    lock onto. Blown-out pixels are flagged in HSV (bright AND barely
    saturated — a genuinely lit red stud stays far more saturated than that)
    and repaired with cv2.inpaint, which fills them in from surrounding real
    texture instead of leaving a bright hole in the middle of the pattern.

    roi_gray and roi_hsv must be the same crop (identical x, y, w, h).
    """
    v_channel, s_channel = roi_hsv[..., 2], roi_hsv[..., 1]
    highlight_mask = (v_channel >= config.SPECULAR_V_MIN) & (s_channel <= config.SPECULAR_S_MAX)
    # Never touch pixels the caller's contour mask already zeroed out.
    highlight_mask &= roi_gray > 0
    if not np.any(highlight_mask):
        return roi_gray

    highlight_mask = highlight_mask.astype(np.uint8) * 255
    return cv2.inpaint(roi_gray, highlight_mask, config.SPECULAR_INPAINT_RADIUS, cv2.INPAINT_TELEA)


def count_studs(
    gray_frame: np.ndarray,
    contour: np.ndarray,
    return_circles: bool = False,
    hsv_frame: np.ndarray | None = None,
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
        hsv_frame: The whole frame converted to HSV, same shape as gray_frame.
            Optional (default None skips it) — when supplied, blown specular
            highlights inside the region are suppressed before stud detection
            runs (see suppress_specular_highlights).

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

    if hsv_frame is not None:
        roi_hsv = hsv_frame[y : y + h, x : x + w]
        roi = suppress_specular_highlights(roi, roi_hsv)

    # Stud size and spacing are expressed as fractions of the brick's smaller
    # dimension, but that only holds at a "normal" ROI scale — the fractions
    # are clamped to a fixed pixel floor below, and medianBlur uses a fixed
    # kernel. Once the brick is far away, its ROI shrinks below the scale
    # those fixed numbers assume, crushing the stud pattern before Hough ever
    # runs. Fix: upscale a too-small ROI to a standard reference size first,
    # so the same fixed floor/blur behave consistently regardless of true
    # distance. Only ever upscale (never downscale), so already-workable
    # close-up cases are untouched.
    scale = 1.0
    param2 = config.STUD_HOUGH_PARAM2
    smaller_side = min(w, h)
    if smaller_side < config.STUD_ROI_REFERENCE_PX:
        scale = min(config.STUD_ROI_REFERENCE_PX / smaller_side, config.STUD_ROI_MAX_UPSCALE)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        w, h = new_w, new_h
        smaller_side = min(w, h)
        # This same "small enough to need upscaling" condition also gates a
        # more permissive accumulator threshold (see STUD_HOUGH_PARAM2_UPSCALED
        # in config.py) — a genuinely far/small brick's stud pattern is
        # already degraded, so it gets the benefit of the doubt here, while a
        # normal-sized candidate (a hand, a cup) never reaches this branch and
        # keeps the strict default below.
        param2 = config.STUD_HOUGH_PARAM2_UPSCALED

    # A light blur removes speckle noise that would otherwise create spurious
    # tiny circles. medianBlur is good at this while keeping edges crisp.
    roi = cv2.medianBlur(roi, 5)

    min_radius = max(3, int(smaller_side * config.STUD_MIN_RADIUS_FRAC))
    max_radius = max(min_radius + 2, int(smaller_side * config.STUD_MAX_RADIUS_FRAC))
    min_dist = max(8, int(smaller_side * config.STUD_MIN_DIST_FRAC))

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=config.STUD_HOUGH_DP,
        minDist=min_dist,
        param1=config.STUD_HOUGH_PARAM1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return (0, None) if return_circles else 0

    # HoughCircles returns coordinates relative to the (possibly upscaled) ROI
    # crop. Undo the upscale first — shrinking x, y, AND radius back down —
    # then shift by the ORIGINAL (unscaled) bounding-box origin, so overlays
    # line up with the original full-resolution image.
    circles = np.around(circles[0]).astype(np.float64)
    if scale != 1.0:
        circles /= scale
    circles = np.around(circles).astype(int)
    circles[:, 0] += x
    circles[:, 1] += y

    count = len(circles)
    return (count, circles) if return_circles else count

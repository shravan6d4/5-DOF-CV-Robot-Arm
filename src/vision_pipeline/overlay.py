"""Frame annotation shared by every script that puts a camera view on screen.

WHY THIS IS ONE MODULE. A webcam can be opened by exactly one process at a
time, so "watch what the camera sees while a script runs" cannot mean a second
program looking over its shoulder — the script that owns the camera has to draw
the view itself. That makes overlay drawing something several unrelated scripts
need (the standalone viewer, the visual servo loop, anything added later), and
the moment it is copied into each of them they start disagreeing about what a
marker means. One module, one visual language.

Nothing here detects anything on its own except ChArUco boards; brick detection
already has LegoBrickDetector.draw_debug_overlay and this does not duplicate it.

All functions draw IN PLACE on a BGR frame and return it, so they compose:

    frame = draw_boards(frame, detectors)
    frame = draw_aim(frame, centroid, aim)
    frame = draw_hud(frame, ["iteration 4", "error -38 px"])
"""

from __future__ import annotations

import cv2
import numpy as np

# One colour per meaning, BGR. Kept here so the viewer and the servo loop cannot
# drift into using green for opposite things.
COLOR_BRICK = (0, 255, 0)        # detected brick
COLOR_AIM = (0, 0, 255)          # where the brick should end up
COLOR_ERROR = (0, 200, 255)      # the gap between those two
COLOR_BOARD = (255, 180, 0)      # ChArUco corners
COLOR_TEXT = (255, 255, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_crosshair(frame: np.ndarray,
                   aim_px: tuple[float, float] | None = None,
                   size: int = 14) -> np.ndarray:
    """Mark the aim point — where a centred brick would sit."""
    h, w = frame.shape[:2]
    ax, ay = aim_px if aim_px is not None else (w / 2.0, h / 2.0)
    ax, ay = int(round(ax)), int(round(ay))
    cv2.line(frame, (ax - size, ay), (ax + size, ay), COLOR_AIM, 1)
    cv2.line(frame, (ax, ay - size), (ax, ay + size), COLOR_AIM, 1)
    cv2.circle(frame, (ax, ay), size + 8, COLOR_AIM, 1)
    return frame


def draw_aim(frame: np.ndarray,
             centroid_px: tuple[float, float] | None,
             aim_px: tuple[float, float] | None = None,
             deadband_px: float | None = None,
             tolerance: tuple[float, float] | None = None) -> np.ndarray:
    """Draw the aim point, the acceptance box, the brick, and the error.

    The line between brick and aim IS the quantity the servo loop is
    minimising, so drawing it literally means the operator watches the same
    thing the controller does rather than inferring it from numbers.

    The aim point is usually NOT the frame centre: the camera sits above and
    behind the claw, so the brick must come to rest below the crosshair for the
    claw to be over it. Drawing the crosshair (frame centre) and the aim point
    separately makes that offset visible instead of something to remember.

    Args:
        tolerance: (x, y) half-sizes of the acceptance box, in pixels. Drawn as
            a rectangle around the aim point — the brick anywhere inside it is
            good enough to descend from.
    """
    h, w = frame.shape[:2]
    ax, ay = aim_px if aim_px is not None else (w / 2.0, h / 2.0)

    if tolerance:
        tx, ty = tolerance
        # Clamped to the frame. A large aim offset can push the box partly (or
        # wholly) off the bottom edge, and an unclamped rectangle simply is not
        # drawn there — leaving the operator looking at a view with no target on
        # it and no indication why. Clamping shows the part that is reachable.
        cv2.rectangle(frame,
                      (int(max(ax - tx, 0)), int(max(ay - ty, 0))),
                      (int(min(ax + tx, w - 1)), int(min(ay + ty, h - 1))),
                      COLOR_ERROR, 1)
    elif deadband_px:
        cv2.circle(frame, (int(ax), int(ay)), int(deadband_px), COLOR_ERROR, 1)

    # The frame centre, always — it is where the camera is actually looking, and
    # seeing it apart from the aim point is what makes the offset legible.
    if abs(ax - w / 2.0) > 1 or abs(ay - h / 2.0) > 1:
        draw_crosshair(frame, (w / 2.0, h / 2.0), size=10)
        cv2.line(frame, (int(w / 2), int(h / 2)),
                 (int(ax), int(min(ay, h - 1))), COLOR_AIM, 1, cv2.LINE_AA)
    draw_crosshair(frame, (ax, ay))

    if centroid_px is None:
        return frame

    cx, cy = int(round(centroid_px[0])), int(round(centroid_px[1]))
    cv2.line(frame, (int(ax), int(ay)), (cx, cy), COLOR_ERROR, 2)
    cv2.circle(frame, (cx, cy), 9, COLOR_BRICK, 2)
    cv2.circle(frame, (cx, cy), 2, COLOR_BRICK, -1)

    dx, dy = centroid_px[0] - ax, centroid_px[1] - ay
    cv2.putText(frame, f"{dx:+.0f}, {dy:+.0f} px", (cx + 14, cy - 10),
                _FONT, 0.5, COLOR_ERROR, 1, cv2.LINE_AA)
    return frame


def draw_boards(frame: np.ndarray, detectors, gray: np.ndarray | None = None) -> int:
    """Detect and draw every ChArUco board visible. Returns how many were seen.

    Draws each board's corners plus its 1-based number, because the boards are
    deliberately DISTINCT (board i owns dictionary IDs i*30..i*30+29) and which
    one is which matters — hand-eye solves per board and compares them. A view
    that showed anonymous corners would hide exactly the thing worth checking.

    Args:
        detectors: the list from charuco.build_detectors().
        gray: pre-converted grayscale, if the caller already has one.
    """
    from vision_pipeline.calibration import charuco

    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    seen = 0
    for index, _board, detector in detectors:
        corners, ids = charuco.detect(detector, gray)
        if corners is None:
            continue
        seen += 1
        pts = corners.reshape(-1, 2)
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, COLOR_BOARD, -1)
        # Label at the board's centroid so several tiled boards stay legible.
        c = pts.mean(axis=0)
        cv2.putText(frame, f"board {index + 1} ({len(pts)})",
                    (int(c[0]) - 40, int(c[1])), _FONT, 0.55, COLOR_BOARD, 2,
                    cv2.LINE_AA)
    return seen


def draw_hud(frame: np.ndarray, lines: list[str], origin=(10, 10)) -> np.ndarray:
    """Status text in the top-left, on a dark panel so it stays readable.

    Camera frames of a bright table wash out plain white text exactly when
    something is going wrong and the text matters most.
    """
    if not lines:
        return frame
    x, y = origin
    pad, line_h = 8, 22
    # Clamp to the frame: a long line would otherwise size the panel past the
    # right edge, where numpy silently truncates the slice and leaves the text
    # drawn over bare pixels instead of the dark backing.
    width = min(max(len(s) for s in lines) * 10 + pad * 2, frame.shape[1] - x)

    panel = frame[y:y + line_h * len(lines) + pad, x:x + width]
    if panel.size:
        cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0, panel)

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x + pad, y + pad + line_h * i + 10),
                    _FONT, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    return frame

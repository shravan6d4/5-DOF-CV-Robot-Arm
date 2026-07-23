"""Background camera capture + MJPEG streaming for the web UI.

Camera.frames() blocks on each read, and only one thing may own the
cv2.VideoCapture at a time. CameraStreamer runs a single background thread that
owns the Camera and continuously refreshes a "latest frame" buffer; Flask
request threads just read that buffer, so multiple browser tabs/polls never
fight over the capture device.
"""

from __future__ import annotations

import logging
import threading
import time

from typing import Callable, Optional

import cv2
import numpy as np

from vision_pipeline.capture.camera import Camera
from vision_pipeline.detection.lego_detector import LegoBrickDetector

logger = logging.getLogger(__name__)

Overlay = Callable[[np.ndarray], np.ndarray]


def _make_placeholder_frame(width: int = 640, height: int = 480, text: str = "camera unavailable") -> np.ndarray:
    """A plain frame with a status message, served when no real camera feed exists."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def _brick_overlay(detector: LegoBrickDetector) -> "Overlay":
    """Wrap a LegoBrickDetector as a frame->frame overlay (the pre-existing --overlay behaviour)."""
    def _apply(frame: np.ndarray) -> np.ndarray:
        detections = detector.detect(frame)
        return detector.draw_debug_overlay(frame, detections)
    return _apply


class CameraStreamer:
    """Owns a Camera on a background thread; serves the latest frame as MJPEG.

    If the camera can't be opened (no webcam, index in use, etc.) this does not
    raise - it serves a placeholder frame instead, so the rest of the dashboard
    (joint jogging) still works with no camera attached. Pass camera_index=None
    to skip opening a camera entirely (e.g. running the dashboard on a machine
    with no webcam at all).
    """

    def __init__(
        self,
        camera_index: Optional[int],
        enable_overlay: bool = False,
        overlay: Optional[Overlay] = None,
    ):
        """
        Args:
            enable_overlay: draw the brick-detection debug overlay (unchanged
                default behaviour — see run_arm_ui.py --overlay).
            overlay: an alternative frame -> frame callable (e.g. drawing
                ChArUco corners for the calibration panel). Takes precedence
                over enable_overlay when given, so a caller doesn't have to
                also build a LegoBrickDetector it doesn't want.
        """
        self.camera_index = camera_index
        self.enable_overlay = enable_overlay
        if overlay is not None:
            self._overlay_fn: Optional[Overlay] = overlay
        elif enable_overlay:
            self._overlay_fn = _brick_overlay(LegoBrickDetector())
        else:
            self._overlay_fn = None
        self._lock = threading.Lock()
        self._frame: np.ndarray = _make_placeholder_frame()
        self._stop = threading.Event()
        if camera_index is None:
            self._frame = _make_placeholder_frame(text="no camera (disabled)")
            self._thread = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            camera = Camera(camera_index=self.camera_index)
        except RuntimeError as e:
            logger.warning(f"CameraStreamer: {e}. Serving placeholder frames.")
            with self._lock:
                self._frame = _make_placeholder_frame(text="no camera")
            return

        try:
            with camera:
                for frame in camera.frames():
                    if self._stop.is_set():
                        break
                    if self._overlay_fn is not None:
                        frame = self._overlay_fn(frame)
                    with self._lock:
                        self._frame = frame
        except RuntimeError as e:
            logger.error(f"CameraStreamer: lost camera feed: {e}")
            with self._lock:
                self._frame = _make_placeholder_frame(text="camera error")

    def latest_frame(self) -> np.ndarray:
        """Return a copy of the most recently captured (or placeholder) frame."""
        with self._lock:
            return self._frame.copy()

    def mjpeg_generator(self, fps: float = 15.0):
        """Yield multipart/x-mixed-replace JPEG chunks for a browser <img> tag."""
        period_s = 1.0 / fps
        boundary = b"--frame"
        while True:
            frame = self.latest_frame()
            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                yield (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
            time.sleep(period_s)

    def stop(self) -> None:
        self._stop.set()

"""
Thin wrapper around OpenCV's VideoCapture for reading webcam frames.

Why wrap it: cv2.VideoCapture works fine on its own, but wrapping it in a
small class means the rest of the pipeline (detection, scripts) depends on
*this* interface, not directly on OpenCV's camera API. Later, this class can
be swapped for a simulated camera (e.g. reading frames from a video file or
a simulator) without changing any detection code.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from vision_pipeline import config


class Camera:
    """Opens a webcam and yields BGR frames one at a time."""

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ) -> None:
        self.camera_index = camera_index
        self._cap = cv2.VideoCapture(camera_index)

        # These are *requests* — the camera hardware may not support the
        # exact resolution and will silently pick the closest one it can do.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {camera_index}. "
                "Check that a webcam is connected and not in use by another "
                "program, or try a different camera_index."
            )

    def read_frame(self) -> np.ndarray:
        """Grab a single frame. Raises RuntimeError if the read fails."""
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames forever, for use in a `for frame in camera.frames():` loop."""
        while True:
            yield self.read_frame()

    def release(self) -> None:
        """Release the camera so other programs (or you, on next run) can use it."""
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

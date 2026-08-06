"""
Thin wrapper around OpenCV's VideoCapture for reading webcam frames.

Why wrap it: cv2.VideoCapture works fine on its own, but wrapping it in a
small class means the rest of the pipeline (detection, scripts) depends on
*this* interface, not directly on OpenCV's camera API. Later, this class can
be swapped for a simulated camera (e.g. reading frames from a video file or
a simulator) without changing any detection code.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import cv2
import numpy as np

from vision_pipeline import config

logger = logging.getLogger(__name__)


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

        # Autofocus MUST be off for any geometry work. Refocusing physically
        # moves the lens, which changes the focal length — and fx/fy are exactly
        # what convert pixels into millimetres. With autofocus on, intrinsics
        # calibrated across a range of distances average several different
        # focal lengths (and still report a low RMS, so nothing looks wrong),
        # and every later frame is measured with whatever focus it happened to
        # settle at. Poses then disagree between viewpoints, which no
        # calibration can repair. Found 2026-08-04: the arm camera is a NexiGo
        # N930AF and this had been on for every calibration run to date.
        #
        # Best-effort: some drivers ignore these. Callers that care can check
        # autofocus_disabled.
        self.autofocus_disabled = bool(self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0))
        if self.autofocus_disabled:
            # Pin the lens too — disabling AF alone can leave it wherever it
            # last landed, which differs run to run.
            #
            # THE STREAM MUST BE RUNNING FIRST. This camera silently ignores
            # CAP_PROP_FOCUS until frames are actually flowing, and
            # get(CAP_PROP_FOCUS) returns 0.0 always, so a failed set looks
            # exactly like a successful one. Measured 2026-08-05, frame
            # sharpness (Laplacian variance) at the same scene:
            #
            #     focus never set .......... 134
            #     focus set before reading . 134   <- what this code used to do
            #     focus set while streaming  2116
            #
            # Every frame this project captured before that fix was therefore at
            # whatever focus the camera powered up with, however carefully
            # CAMERA_FOCUS was chosen. It showed up as a detector that fired on
            # ~5% of frames with the confidence pinned to its threshold, which
            # reads as a tuning problem and is not one.
            for _ in range(config.CAMERA_FOCUS_WARMUP_FRAMES):
                self._cap.read()
            self._cap.set(cv2.CAP_PROP_FOCUS, config.CAMERA_FOCUS)
            # Setting the property only STARTS the lens moving; the motor takes
            # a fair fraction of a second, and frames grabbed meanwhile are
            # focused somewhere between the old position and the new one.
            time.sleep(config.CAMERA_FOCUS_SETTLE_S)
            for _ in range(config.CAMERA_FOCUS_SETTLE_FRAMES):
                self._cap.read()
        else:
            logger.warning(
                "Could not disable autofocus on camera %d — intrinsics and every "
                "pose derived from them will drift as the lens refocuses.",
                camera_index,
            )

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

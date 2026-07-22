"""
Interactive tool for finding good HSV threshold values for your Lego brick.

Opens a window with trackbars (sliders) for the lower/upper HSV bounds, and
shows the resulting black/white mask live so you can see exactly which
pixels would be detected as you adjust the sliders. Once you find values
that cleanly isolate the brick, copy them into config.py.

Run with a webcam:
    python scripts/tune_hsv.py

Run against a static image instead (e.g. if no webcam is connected yet):
    python scripts/tune_hsv.py --image tests/sample_images/brick1.jpg

Press 'q' to quit. Press 'p' to print the current slider values.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline.capture.camera import Camera

WINDOW = "HSV Tuner"


def _nothing(_value: int) -> None:
    # cv2.createTrackbar requires a callback; we don't need to react live,
    # we just read slider positions each frame in the main loop below.
    pass


def _create_trackbars() -> None:
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("H min", WINDOW, 0, 179, _nothing)
    cv2.createTrackbar("H max", WINDOW, 179, 179, _nothing)
    cv2.createTrackbar("S min", WINDOW, 120, 255, _nothing)
    cv2.createTrackbar("S max", WINDOW, 255, 255, _nothing)
    cv2.createTrackbar("V min", WINDOW, 70, 255, _nothing)
    cv2.createTrackbar("V max", WINDOW, 255, 255, _nothing)


def _read_trackbars() -> tuple[np.ndarray, np.ndarray]:
    h_min = cv2.getTrackbarPos("H min", WINDOW)
    h_max = cv2.getTrackbarPos("H max", WINDOW)
    s_min = cv2.getTrackbarPos("S min", WINDOW)
    s_max = cv2.getTrackbarPos("S max", WINDOW)
    v_min = cv2.getTrackbarPos("V min", WINDOW)
    v_max = cv2.getTrackbarPos("V max", WINDOW)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return lower, upper


def _frame_source(image_path: str | None):
    """Yield frames forever: either repeated static image, or live camera."""
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        while True:
            yield frame
    else:
        with Camera() as camera:
            yield from camera.frames()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a static image to tune against instead of a live webcam.",
    )
    args = parser.parse_args()

    _create_trackbars()
    print("Adjust sliders until only the brick is white in the mask window.")
    print("Press 'p' to print current values, 'q' to quit.")

    for frame in _frame_source(args.image):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = _read_trackbars()
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow(WINDOW, frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            print(f"HSV_LOWER = {tuple(int(v) for v in lower)}")
            print(f"HSV_UPPER = {tuple(int(v) for v in upper)}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

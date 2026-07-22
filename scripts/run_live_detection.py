"""
Demo script: opens the webcam, runs HSV color detection on each frame, and
shows a live window with the detected brick outlined and its centroid marked.

Run with:  python scripts/run_live_detection.py
Press 'q' in the video window to quit.

Requires a webcam connected to this machine. If you don't have one yet, use
tests/test_color_detector.py with a static sample image instead.
"""

import sys
from pathlib import Path

# Allow running this script directly (without installing the package) by
# adding src/ to the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from vision_pipeline.capture.camera import Camera
from vision_pipeline.detection.lego_detector import LegoBrickDetector


def main() -> None:
    # LegoBrickDetector = color detection + stud verification, so it fires on
    # actual bricks rather than on any red object in view.
    detector = LegoBrickDetector()

    with Camera() as camera:
        print("Press 'q' to quit.")
        for frame in camera.frames():
            detections = detector.detect(frame)
            overlay = detector.draw_debug_overlay(frame, detections)

            if detections:
                best = detections[0]
                print(
                    f"Detected brick at pixel {best.centroid_px}, "
                    f"area={best.area:.0f}, studs={best.num_studs}"
                )

            cv2.imshow("Live Detection (press 'q' to quit)", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

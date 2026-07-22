"""
Browse every image in tests/sample_images/ one at a time and see whether the
detector passes (finds a brick) or fails on it, with a green bounding box
drawn around each detected brick on pass.

Run with:  python scripts/review_images.py

Controls (with the image window focused):
    n / right-arrow  -> next image
    p / left-arrow   -> previous image
    q / Esc          -> quit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from vision_pipeline.detection.color_detector import ColorDetector

SAMPLE_IMAGES_DIR = Path(__file__).resolve().parent.parent / "tests" / "sample_images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "Test Image Review (n=next, p=prev, q=quit)"

# cv2.waitKey key codes for the arrow keys differ across platforms/builds, so
# we check a handful of known codes rather than relying on just one.
KEYS_NEXT = {ord("n"), 83, 3, 2555904}
KEYS_PREV = {ord("p"), 81, 2, 2424832}
KEYS_QUIT = {ord("q"), 27}


def _find_sample_images() -> list[Path]:
    return sorted(p for p in SAMPLE_IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _build_display(image_path: Path, index: int, total: int, detector: ColorDetector) -> tuple[object, bool]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    detections = detector.detect(frame)
    passed = bool(detections)

    overlay = frame.copy()
    for detection in detections:
        x, y, w, h = detection.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = "PASS" if passed else "FAIL"
    color = (0, 255, 0) if passed else (0, 0, 255)
    header = f"{image_path.name}  ({index + 1}/{total})  {label}"
    cv2.putText(overlay, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return overlay, passed


def main() -> None:
    image_paths = _find_sample_images()
    if not image_paths:
        print(f"No sample images found in {SAMPLE_IMAGES_DIR}.")
        return

    detector = ColorDetector()
    results: dict[Path, bool] = {}

    print("Press 'n'/right-arrow for next, 'p'/left-arrow for previous, 'q'/Esc to quit.")

    index = 0
    total = len(image_paths)
    while True:
        image_path = image_paths[index]
        overlay, passed = _build_display(image_path, index, total, detector)
        results[image_path] = passed

        cv2.imshow(WINDOW, overlay)
        # waitKeyEx (not waitKey) is needed to get distinct codes for the
        # arrow keys on Windows; waitKey collapses them all to -1/0.
        key = cv2.waitKeyEx(0)

        if key in KEYS_QUIT or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in KEYS_NEXT:
            index = min(index + 1, total - 1)
        elif key in KEYS_PREV:
            index = max(index - 1, 0)

    cv2.destroyAllWindows()

    passed_count = sum(results.values())
    print(f"{passed_count}/{len(results)} passed")


if __name__ == "__main__":
    main()

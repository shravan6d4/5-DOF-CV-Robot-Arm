"""
Tests for LegoBrickDetector — the full "is this actually a Lego brick?" logic.

Two kinds of test:

1. Synthetic (always run, no files needed): draw a red rectangle *with* fake
   studs (circles) and confirm it's detected; draw a plain red rectangle with
   NO studs and confirm it's rejected. This pins down the core rule — red +
   studs = brick, red alone = not a brick — independent of any photo.

2. Real photos (run when present): a small set of labeled images in
   tests/sample_images/. Files known to be Lego bricks must be detected; files
   known NOT to be bricks (e.g. a red cup, a hand holding a phone) must be
   rejected. This is the regression test for the false positives we fixed.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.detection.lego_detector import LegoBrickDetector

SAMPLE_IMAGES_DIR = Path(__file__).parent / "sample_images"

# Ground truth for the sample photos: True = is a red Lego brick (must be
# detected), False = is NOT a brick (must be rejected). Add more entries here
# as you collect more test photos.
SAMPLE_TRUTH = {
    "red lego brick.webp": True,
    "red lego brick 2.jpg": True,
    "red lego birck 3.jpg": True,
    "hand-holding-a-red-plastic-cup-against-a-white-background-isolated-studio-shot-photo.jpg": False,
    "360_F_1185788922_fJL5kglXyiniZxGj1qpoPl5sTVsRW9Ti.jpg": False,
}


# --- Synthetic tests (always run) ------------------------------------------

def make_brick_like_frame(with_studs: bool) -> np.ndarray:
    """Build a fake photo: a red rectangle, optionally with circular 'studs'.

    with_studs=True mimics a Lego brick (red + bumps); with_studs=False mimics
    a smooth red object like a cup (red, but no studs).
    """
    frame = np.full((480, 640, 3), fill_value=(120, 120, 120), dtype=np.uint8)  # gray bg
    top_left, bottom_right = (200, 140), (440, 340)
    cv2.rectangle(frame, top_left, bottom_right, color=(0, 0, 255), thickness=-1)  # red (BGR)

    if with_studs:
        # Draw a 4x2 grid of slightly darker red circles to imitate studs. The
        # brightness difference gives Hough edges to lock onto, just like real
        # studs' light/shadow rings.
        for row_y in (200, 280):
            for col_x in (240, 300, 360, 400):
                cv2.circle(frame, (col_x, row_y), 22, color=(0, 0, 200), thickness=-1)
                cv2.circle(frame, (col_x, row_y), 22, color=(0, 0, 120), thickness=2)
    return frame


def test_detects_synthetic_brick_with_studs():
    detector = LegoBrickDetector()
    detections = detector.detect(make_brick_like_frame(with_studs=True))
    assert len(detections) >= 1, "A red rectangle with studs should be detected as a brick"
    assert detections[0].num_studs >= detector.min_studs


def test_rejects_synthetic_red_shape_without_studs():
    detector = LegoBrickDetector()
    detections = detector.detect(make_brick_like_frame(with_studs=False))
    assert detections == [], "A smooth red shape (no studs) should NOT be detected as a brick"


# --- Real-photo regression tests (run when the files are present) ----------

def _available_samples() -> list[tuple[str, bool]]:
    items = []
    for name, is_lego in SAMPLE_TRUTH.items():
        if (SAMPLE_IMAGES_DIR / name).exists():
            items.append((name, is_lego))
    return items


@pytest.mark.parametrize("name,is_lego", _available_samples() or [(None, None)])
def test_sample_photo_classification(name: str | None, is_lego: bool | None):
    if name is None:
        pytest.skip("No labeled sample images found in tests/sample_images/.")

    frame = cv2.imread(str(SAMPLE_IMAGES_DIR / name))
    assert frame is not None, f"Failed to load image: {name}"

    detector = LegoBrickDetector()
    detections = detector.detect(frame)

    if is_lego:
        assert len(detections) >= 1, (
            f"Expected to detect a Lego brick in '{name}' but found none. "
            "Lighting or brick color may differ — retune HSV with scripts/tune_hsv.py, "
            "or lower MIN_STUDS in config.py."
        )
    else:
        assert detections == [], (
            f"'{name}' is not a Lego brick but was detected as one "
            f"(studs found: {[d.num_studs for d in detections]}). "
            "Consider raising MIN_STUDS or tightening HSV in config.py."
        )

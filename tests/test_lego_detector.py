"""
Tests for LegoBrickDetector — the full "is this actually a Lego brick?" logic.

Three kinds of test:

1. Synthetic, studs+color (always run, no files needed): draw a red rectangle
   *with* fake studs (circles) and confirm it's detected; draw a plain red
   OVAL (no studs, no straight edges) and confirm it's rejected. This pins
   down the core rule — red + studs/shape = brick, red alone = not a brick —
   independent of any photo.

2. Synthetic, shape-confidence fallback (always run): a red RECTANGLE with
   zero resolvable studs (simulating "too far away" or "glare-corrupted")
   should still be accepted on shape evidence alone, since a rectangular
   silhouette with plausible brick proportions is itself strong evidence,
   even when studs can't be counted.

3. Real photos (run when present): a small set of labeled images in
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
    "real brick close.jpg": True,
    "real brick medium.jpg": True,
    "real brick far.jpg": True,
}


# --- Synthetic tests (always run) ------------------------------------------

def make_brick_like_frame(with_studs: bool) -> np.ndarray:
    """Build a fake photo: a red rectangle, optionally with circular 'studs'.

    with_studs=True mimics a Lego brick (red + bumps). with_studs=False draws
    the same rectangle with no studs at all — used by
    test_accepts_synthetic_rectangle_with_zero_studs_via_shape_confidence to
    prove the shape-confidence fallback, since a rectangular silhouette is
    exactly the shape that fallback is designed to recognize.
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


def make_round_red_blob_frame() -> np.ndarray:
    """Build a fake photo: a smooth red OVAL, no studs, no straight edges.

    A more faithful proxy for a smooth non-brick red object (a cup rim, a
    hand) than a bare rectangle would be — real distractors like that are
    round/organic, not rectangular, so this is what should get rejected by
    both the stud signal (no studs) AND the shape signal (low
    rectangularity, since an ellipse is capped well below a rectangle's
    fill ratio of its own bounding box).
    """
    frame = np.full((480, 640, 3), fill_value=(120, 120, 120), dtype=np.uint8)  # gray bg
    cv2.ellipse(frame, (320, 240), (120, 100), 0, 0, 360, color=(0, 0, 255), thickness=-1)
    return frame


def test_detects_synthetic_brick_with_studs():
    detector = LegoBrickDetector()
    detections = detector.detect(make_brick_like_frame(with_studs=True))
    assert len(detections) >= 1, "A red rectangle with studs should be detected as a brick"
    assert detections[0].num_studs >= 1
    assert detections[0].confidence >= detector.confidence_threshold


def test_rejects_synthetic_round_red_blob():
    detector = LegoBrickDetector()
    detections = detector.detect(make_round_red_blob_frame())
    assert detections == [], "A smooth round red blob (no studs, not rectangular) should NOT be detected as a brick"


def test_accepts_synthetic_rectangle_with_zero_studs_via_shape_confidence():
    detector = LegoBrickDetector()
    detections = detector.detect(make_brick_like_frame(with_studs=False))
    assert len(detections) >= 1, (
        "A rectangular red region with no resolvable studs should still be accepted "
        "on shape confidence alone (simulates 'too far away' / 'glare-corrupted')"
    )
    assert detections[0].num_studs == 0
    assert detections[0].shape_score > 0.9
    assert detections[0].confidence >= detector.confidence_threshold


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
            "or lower DETECTION_CONFIDENCE_THRESHOLD in config.py."
        )
    else:
        assert detections == [], (
            f"'{name}' is not a Lego brick but was detected as one "
            f"(studs={[d.num_studs for d in detections]} shape={[round(d.shape_score, 2) for d in detections]} "
            f"conf={[round(d.confidence, 2) for d in detections]}). "
            "Consider raising DETECTION_CONFIDENCE_THRESHOLD or tightening HSV in config.py."
        )

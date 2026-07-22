# Robotic Arm Vision

Computer vision pipeline for a 6-DOF robotic arm to detect and pick up a single Lego brick.

Pipeline phases:
1. **Phase 1** (current): Camera capture + HSV color thresholding + contour detection + centroid.
2. **Phase 2**: Pixel -> robot world-coordinate calibration.
3. **Phase 3**: Integration with the arm's control interface.
4. **Phase 4** (stretch): ML-based detector.

## Setup

```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Confirm OpenCV imports correctly
python -c "import cv2, numpy; print(cv2.__version__)"
```

## Project Layout

- `src/vision_pipeline/capture/` — webcam frame capture
- `src/vision_pipeline/detection/` — brick detection (Phase 1). Two stages:
  `color_detector.py` finds red regions; `lego_detector.py` then verifies each
  region has Lego *studs* (via `stud_detector.py`) so it only accepts real
  bricks, not other red objects like a cup or a hand.
- `src/vision_pipeline/calibration/` — pixel-to-world coordinate mapping (Phase 2, stub for now)
- `src/vision_pipeline/robot_interface/` — abstract interface for the arm's control code (Phase 3, stub for now)
- `scripts/` — runnable demos and tools (live detection viewer, HSV tuning tool)
- `tests/` — unit-style tests, run against static images in `tests/sample_images/`

## Running things

```powershell
# Tune HSV thresholds interactively (needs a webcam or a sample image)
python scripts/tune_hsv.py

# Run live detection with a webcam
python scripts/run_live_detection.py

# Run tests (works even without a camera, using tests/sample_images/)
pytest
```

## Testing without a camera

Drop a photo of a Lego brick on a plain background into `tests/sample_images/` (e.g. `brick1.jpg`)
and `pytest` will pick it up automatically. Until an image is added there, the detection test is
skipped with a clear message instead of failing.

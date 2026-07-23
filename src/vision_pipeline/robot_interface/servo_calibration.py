"""Shared per-servo calibration loading and tick<->radian conversion.

The tick<->rad math and the JSON-or-config-fallback loading were originally
inlined in ServoBus (servo_driver.py). This module factors the *pure* parts out
so code that must NOT open a serial port — notably the mock joint controller for
the web UI — can seed and convert joint positions using the exact same
calibration as the real bus, with no risk of the two drifting apart.

ServoBus keeps its own copy (its byte-level tests pin that behaviour); this is a
standalone, serial-free twin of just the calibration/conversion logic.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from vision_pipeline import config

logger = logging.getLogger(__name__)


class ServoCalibrationError(Exception):
    """Raised when servo calibration is missing or incomplete."""
    pass


def load_calibration(path: Optional[str] = None) -> dict:
    """Load per-servo calibration from JSON, falling back to config.

    Mirrors ServoBus._load_calibration: read the JSON file if present, otherwise
    use config.SERVO_CALIBRATION_FALLBACK, then validate that all of J1..J6 are
    present with the required keys.

    Args:
        path: path to servo_calibration.json, or None for config default.

    Returns:
        Calibration dict keyed by string joint IDs "1".."6".

    Raises:
        ServoCalibrationError: if a joint or a required key is missing.
    """
    calibration_path = Path(path or config.SERVO_CALIBRATION_PATH)
    if calibration_path.exists():
        try:
            with open(calibration_path) as f:
                calibration = json.load(f)
            logger.info(f"Loaded servo calibration from {calibration_path}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {calibration_path}, using fallback: {e}")
            calibration = config.SERVO_CALIBRATION_FALLBACK
    else:
        logger.warning(f"Calibration file {calibration_path} not found, using fallback")
        calibration = config.SERVO_CALIBRATION_FALLBACK

    for j in range(1, 7):
        if str(j) not in calibration:
            raise ServoCalibrationError(f"Servo J{j} missing from calibration")
        cal = calibration[str(j)]
        for key in ("home_tick", "ticks_per_rad", "dir_sign"):
            if key not in cal:
                raise ServoCalibrationError(f"Servo J{j} missing key '{key}'")

    return calibration


def _cal(calibration: dict, servo_id: int) -> dict:
    """Return one servo's calibration dict, or raise if absent."""
    if str(servo_id) not in calibration:
        raise ServoCalibrationError(f"Servo {servo_id} not calibrated")
    return calibration[str(servo_id)]


def rad_to_ticks(calibration: dict, servo_id: int, angle_rad: float) -> int:
    """Convert a joint angle (radians, relative to home) to a raw tick target."""
    cal = _cal(calibration, servo_id)
    return int(round(cal["home_tick"] + cal["dir_sign"] * angle_rad * cal["ticks_per_rad"]))


def ticks_to_rad(calibration: dict, servo_id: int, ticks: int) -> float:
    """Convert a raw present-position tick reading to a joint angle (radians)."""
    cal = _cal(calibration, servo_id)
    return cal["dir_sign"] * (ticks - cal["home_tick"]) / cal["ticks_per_rad"]

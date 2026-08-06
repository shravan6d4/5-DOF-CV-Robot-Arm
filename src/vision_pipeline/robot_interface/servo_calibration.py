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


TICK_SPAN = 4096            # the encoder's full circle: readings run 0..4095, then wrap


def unwrap_tick(tick: int, reference: int) -> int:
    """Re-express `tick` as the equivalent reading nearest `reference`.

    A Feetech encoder counts 0..4095 and wraps, so tick 4079 and tick -17 are the
    SAME physical place. Every comparison in this project (travel limits, move
    deltas, "how far is this from home") is arithmetic on raw readings, which
    silently treats those two as 4096 ticks apart. Near the seam that turns a
    17-tick move into an apparent 4079-tick leap, and a joint a hair below its
    minimum into one that appears to be 3000 ticks past its maximum.

    Unwrapping against a reference inside the joint's working range restores
    ordinary arithmetic:

        unwrap_tick(4079, 579) -> -17     # 80 ticks below a min_tick of 63
        unwrap_tick(20, 4000)  -> 4116    # the other direction, same idea

    The result is deliberately allowed outside 0..4095: it is a *continuous*
    joint coordinate, not something to write to a register. Anything sent to the
    servo must be brought back with `% TICK_SPAN`.
    """
    half = TICK_SPAN // 2
    return reference + ((tick - reference + half) % TICK_SPAN) - half


def wrapped_past_seam(tick: int, lo: int, hi: int) -> bool:
    """True when `tick` reads on the FAR side of the 0/4095 seam from [lo, hi].

    This is the difference between "the joint is a little out of range" (walk it
    back) and "the joint's reading has wrapped" (no goal-position command can
    walk it back — the servo would drive the long way round, through every hard
    stop in between). They look identical in raw ticks and need opposite
    responses, which is why it gets a name.
    """
    return unwrap_tick(tick, (lo + hi) // 2) != tick


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
            # encoding= is NOT optional. Python on Windows defaults to the
            # system ANSI codepage (cp1252 here), and this file carries prose:
            # the *_basis provenance fields hold em-dashes and degree signs. The
            # same bug was fixed in ServoBus's own loader and missed here, so
            # this twin crashed with UnicodeDecodeError the moment a basis note
            # was written with a non-ASCII character -- taking down every caller
            # that does not go through ServoBus, MockJointController included.
            with open(calibration_path, encoding="utf-8") as f:
                calibration = json.load(f)
            logger.info(f"Loaded servo calibration from {calibration_path}")
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
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


def load_calibration_file(path=None) -> dict:
    """Read servo_calibration.json as UTF-8, whatever the platform default is.

    Exists because Path.read_text() with no encoding uses the OS default --
    cp1252 on Windows -- which silently mangles every non-ASCII character in the
    file. That was harmless when the file held only numbers; it is not now that
    each joint carries a prose `limit_basis` / `dir_sign_basis` note, and a
    read-modify-write cycle bakes the damage in permanently. Bitten twice on
    2026-08-05.
    """
    p = Path(path or config.SERVO_CALIBRATION_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def save_calibration_file(calibration: dict, path=None) -> Path:
    """Write servo_calibration.json as readable UTF-8. Counterpart to the above.

    ensure_ascii=False keeps the prose notes legible in the file rather than
    escaping every accented character into \\uXXXX soup.
    """
    p = Path(path or config.SERVO_CALIBRATION_PATH)
    p.write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return p


def joint_angle_rad(calibration: dict, servo_id: int, ticks: int) -> float:
    """Tick -> angle in the MATLAB model's ABSOLUTE convention.

    Distinct from ticks_to_rad above, which returns the offset from home and is
    what the mock joint controller wants. This adds home_angle_rad, matching
    ServoBus.ticks_to_rad, because the angle limits handed to MATLAB have to be
    in the same convention the IK server solves in. Every home_angle_rad is
    currently 0.0, so the two agree numerically today -- which is exactly why
    the difference needs stating rather than discovering later.
    """
    cal = _cal(calibration, servo_id)
    return cal.get("home_angle_rad", 0.0) + ticks_to_rad(calibration, servo_id, ticks)


def angle_limits(calibration: dict) -> dict:
    """Per-joint {min_rad, max_rad} for every joint with measured tick limits.

    The single conversion from the tick limits ServoBus enforces to the angle
    limits matlab/init_arm.m loads into the IK model's PositionLimits. The two
    MUST agree: if MATLAB believes a joint can reach further than the bus will
    allow, IK returns solutions the bus then refuses, and the run dies partway
    through a move with no obvious cause.

    Sorted because dir_sign can be negative, which swaps which tick end is the
    larger angle.
    """
    out = {}
    for j in range(1, 7):
        cal = calibration.get(str(j))
        if not cal or "min_tick" not in cal or "max_tick" not in cal:
            continue
        pair = sorted((joint_angle_rad(calibration, j, cal["min_tick"]),
                       joint_angle_rad(calibration, j, cal["max_tick"])))
        out[str(j)] = {"min_rad": pair[0], "max_rad": pair[1]}
    return out


def write_angle_limits(calibration: dict, path=None) -> tuple[Path, dict]:
    """Regenerate data/joint_limits_rad.json from the tick limits."""
    p = Path(path or config.JOINT_LIMITS_RAD_PATH)
    limits = angle_limits(calibration)
    p.write_text(json.dumps(limits, indent=2) + "\n", encoding="utf-8")
    return p, limits


def dir_sign_report(calibration: dict, joints=range(1, 6)) -> list[str]:
    """Render each joint's dir_sign and how confident we are in it.

    Exists because a dir_sign can only ever be settled by a physical jog or a
    ruler -- an FK-vs-IK residual cannot see it, since rad_to_ticks and
    ticks_to_rad apply the same calibration in both directions and it cancels.
    That makes a provisionally-flipped sign invisible to every automated check
    in the repo, so the file records its own provenance ('dir_sign_confirmed',
    'dir_sign_basis') and the scripts that move the arm print it up front.
    Without this, a hypothesis flip is one forgotten evening away from being
    mistaken for a measurement.

    Returns lines ready to print (no trailing newlines).
    """
    lines = ["dir_sign:"]
    unconfirmed = []
    for j in joints:
        cal = calibration.get(str(j))
        if cal is None:
            continue
        confirmed = cal.get("dir_sign_confirmed")
        if confirmed is True:
            status = "confirmed by physical jog"
        elif confirmed is False:
            status = "*** UNCONFIRMED ***"
            unconfirmed.append(j)
        else:
            status = "provenance not recorded"
        lines.append(f"    J{j}: {cal['dir_sign']:+d}   {status}")

    for j in unconfirmed:
        basis = calibration[str(j)].get("dir_sign_basis")
        if basis:
            lines.append("")
            lines.append(f"  J{j}: {basis}")

    if unconfirmed:
        js = ", ".join(f"J{j}" for j in unconfirmed)
        lines.append("")
        lines.append(f"  {js} carry an UNVERIFIED direction. Nothing this script")
        lines.append("  prints can validate one -- only the physical arm can. Watch the")
        lines.append("  first move, and stop it (Ctrl-C) if a joint turns the other way.")
    return lines


def rad_to_ticks(calibration: dict, servo_id: int, angle_rad: float) -> int:
    """Convert a joint angle (radians, relative to home) to a raw tick target."""
    cal = _cal(calibration, servo_id)
    return int(round(cal["home_tick"] + cal["dir_sign"] * angle_rad * cal["ticks_per_rad"]))


def ticks_to_rad(calibration: dict, servo_id: int, ticks: int) -> float:
    """Convert a raw present-position tick reading to a joint angle (radians)."""
    cal = _cal(calibration, servo_id)
    return cal["dir_sign"] * (ticks - cal["home_tick"]) / cal["ticks_per_rad"]

"""
Capture the arm's CURRENT physical pose as its home (0-rad) reference.

Run this with the arm parked in a pose that approximates the MATLAB kinematic home
(init_arm.m: homeAngles = zeros(1,6)). It reads each servo's present position and
writes those ticks as `home_tick` in data/servo_calibration.json, so that "where the
arm is right now" is DEFINED as 0 rad for every joint. After this, FK of the current
pose reads ~0 rad and the startup move to MATLAB home is a near-zero move rather than
a blind swing from an unknown reference.

    !!! THIS DOES NOT MOVE ANYTHING !!!
    It only reads positions and writes a JSON file. Safe to run anytime.

    python scripts/capture_servo_home.py

What this DOES set:  home_tick (measured now, per joint).
What this does NOT verify:  ticks_per_rad and dir_sign are carried over UNCHANGED
from config.SERVO_CALIBRATION_FALLBACK. Those only affect NON-zero angle commands
(real IK targets), not the home capture or the ~0 startup move — but they must still
be verified per joint (direction + scale + reachable range) before trusting the arm
to drive to arbitrary IK targets. See the printed warnings.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import serial

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_driver import ServoBus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=config.SERVO_PORT)
    parser.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    parser.add_argument("--out", default=config.SERVO_CALIBRATION_PATH)
    args = parser.parse_args()

    try:
        bus = ServoBus(args.port, baud=args.baud)
    except serial.SerialException as e:
        print(f"Could not open {args.port}: {e}")
        sys.exit(1)

    calibration = {}
    with bus:
        print("Reading present positions (no movement commanded)...")
        for sid in range(1, 7):
            try:
                ticks = bus.read_position(sid)
            except Exception as e:
                print(f"  J{sid}: FAILED to read -- {e}. Aborting, nothing written.")
                sys.exit(1)
            fallback = config.SERVO_CALIBRATION_FALLBACK[str(sid)]
            calibration[str(sid)] = {
                "home_tick": int(ticks),                       # measured NOW
                "ticks_per_rad": fallback["ticks_per_rad"],    # UNVERIFIED placeholder
                "dir_sign": fallback["dir_sign"],              # UNVERIFIED placeholder
            }
            print(f"  J{sid}: home_tick = {ticks}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nWrote {out_path}")
    print("Current pose is now DEFINED as 0 rad (MATLAB home) for every joint.")
    print("NOTE: ticks_per_rad and dir_sign are still placeholders — verify each joint's")
    print("      direction, scale, and reachable range before commanding real IK targets.")


if __name__ == "__main__":
    main()

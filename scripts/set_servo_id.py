"""
Set a Feetech servo's bus ID — run BY HAND, ONE servo on the bus at a time.

Feetech servos usually ship all set to the SAME factory ID (often 1). Our driver
addresses J1..J6 as IDs 1..6, so each servo must be given a unique ID once, before
the whole chain is wired together — otherwise they collide on the bus.

    !!! CONNECT ONLY ONE SERVO AT A TIME !!!
    Writing an ID reprograms EVERY servo currently answering at the old ID.

Typical workflow, repeated for each servo:
    1. Plug in exactly one servo.
    2. See what ID it currently has:
         python scripts/set_servo_id.py --scan
    3. Give it its joint ID (1..6 for J1..J6):
         python scripts/set_servo_id.py --new-id 3
       (auto-detects the single connected servo as the source; or pass --old-id N)

Change persists in the servo's EEPROM — you only do this once per servo.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import serial

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_driver import ServoBus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=config.SERVO_PORT, help="Serial port (default: config.SERVO_PORT).")
    parser.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    parser.add_argument("--scan", action="store_true", help="Only list IDs answering on the bus, then exit.")
    parser.add_argument("--new-id", type=int, help="ID to assign (1..6 for J1..J6).")
    parser.add_argument("--old-id", type=int, help="Current ID (default: auto-detect the one connected servo).")
    args = parser.parse_args()

    try:
        bus = ServoBus(args.port, baud=args.baud)
    except serial.SerialException as e:
        print(f"Could not open {args.port}: {e}")
        print("Check Device Manager -> Ports (COM & LPT) for the adapter's COM number,")
        print("and that its USB-serial driver is installed.")
        sys.exit(1)

    with bus:
        present = bus.scan_ids()
        print(f"IDs answering on the bus: {present if present else '(none)'}")

        if args.scan:
            return

        if args.new_id is None:
            print("Nothing to do. Pass --new-id N to assign an ID, or --scan to just list.")
            sys.exit(1)

        if not 1 <= args.new_id <= 253:
            print(f"--new-id must be 1..253 (use 1..6 for J1..J6), got {args.new_id}.")
            sys.exit(1)

        old_id = args.old_id
        if old_id is None:
            if len(present) == 0:
                print("No servo detected. Plug ONE servo in and check power/wiring.")
                sys.exit(1)
            if len(present) > 1:
                print(f"Multiple servos detected {present}. Connect ONLY ONE at a time, "
                      f"or specify --old-id.")
                sys.exit(1)
            old_id = present[0]

        if old_id == args.new_id:
            print(f"Servo is already at ID {old_id}. Nothing to change.")
            return

        print(f"Changing servo ID {old_id} -> {args.new_id} (writing EEPROM)...")
        if bus.write_servo_id(old_id, args.new_id):
            print(f"Success. Servo now answers at ID {args.new_id}.")
        else:
            print("Failed. See the logged error above; re-scan to check the servo's state.")
            sys.exit(1)


if __name__ == "__main__":
    main()

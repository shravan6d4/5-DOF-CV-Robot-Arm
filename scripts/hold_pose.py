"""
Lock the arm where it is RIGHT NOW: read each servo's position, command that
same position back as its goal.

This is the "I hand-positioned the arm with the power off, now hold it there"
tool. Net commanded travel is ZERO by construction — each joint is told to go
exactly where it already is — so it cannot swing, and the safety cap in
move_and_verify cannot be tripped.

    !!! THIS DOES NOT CHANGE ANY CALIBRATION !!!
    It never touches data/servo_calibration.json. Ticks-to-radians, dir_sign,
    and home_tick are all left exactly as they are.

That distinction is the entire point of this script existing separately from
capture_servo_home.py, which looks superficially similar but does something
very different: it REDEFINES the current pose as 0 rad on every joint by
overwriting home_tick. Run that by accident and FK starts reporting the arm as
being somewhere it is not, which invalidates TABLE_Z_IN_BASE and every world
coordinate derived from it. Use this script to hold a pose; use that one only
when you deliberately intend to re-zero the kinematic reference.

Joints are locked base-first (J1 -> J6) because each locked joint carries the
ones outboard of it, and each joint's read is followed immediately by its own
write so the arm has the least possible time to sag in between.

Usage (from the repo root):
    python scripts/hold_pose.py              # lock all six
    python scripts/hold_pose.py --dry-run    # report positions, command nothing
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

ALL_JOINTS = range(1, 7)

# How far a joint may drift between the initial read and the final verify
# before it is worth saying out loud. A few ticks is normal settling; tens of
# ticks means the joint sagged under gravity and is now HOLDING a pose lower
# than the one you positioned it in.
DRIFT_NOTE_TICKS = 15


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    ap.add_argument("--dry-run", action="store_true",
                    help="read and report positions without commanding a hold")
    args = ap.parse_args()

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as e:
        print(f"Could not open the servo bus on {args.port}: {e}")
        print("Is the servo rail powered? USB alone powers the adapter, not the servos.")
        sys.exit(1)

    started_at = {}
    failures = []

    with bus:
        if args.dry_run:
            print("Reading positions (--dry-run: nothing will be commanded).\n")
        else:
            print("Locking each joint at its present position (zero net travel).\n")

        for j in ALL_JOINTS:
            try:
                present = bus.read_position(j)
            except Exception as e:
                print(f"  J{j}: FAILED to read — {e}")
                failures.append(j)
                continue

            started_at[j] = present

            if args.dry_run:
                print(f"  J{j}: at {present} ticks")
                continue

            try:
                # target == present, so delta is 0: no motion is possible here.
                bus.move_and_verify(j, present)
                print(f"  J{j}: holding at {present} ticks")
            except ServoSafetyError as e:
                # Should be unreachable with a zero delta; if it fires, the read
                # is unstable and locking on it would be locking onto noise.
                print(f"  J{j}: REFUSED — {e}")
                failures.append(j)
            except Exception as e:
                print(f"  J{j}: FAILED to command — {e}")
                failures.append(j)

        if not args.dry_run and started_at:
            print("\nRe-reading to confirm the hold took:")
            for j, before in started_at.items():
                try:
                    after = bus.read_position(j)
                except Exception as e:
                    print(f"  J{j}: could not re-read — {e}")
                    continue
                drift = after - before
                flag = "  <-- SAGGED" if abs(drift) >= DRIFT_NOTE_TICKS else ""
                print(f"  J{j}: {before} -> {after}  ({drift:+d} ticks){flag}")

    if failures:
        print(f"\nJoints not holding: {failures}. Support the arm before letting go.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: nothing was commanded. The arm is NOT being held.")
        return

    print("\nAll joints commanded to hold. Test by easing your support off "
          "gradually,\nnot by letting go — a joint that did not take the "
          "command will drop.")
    print("\nCalibration was not modified. To check FK agrees with where the arm "
          "physically\nis now:  python scripts/check_servo_health.py")


if __name__ == "__main__":
    main()

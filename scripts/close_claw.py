"""Close (or open) the claw in small jogs, asking before every one.

    !!! THIS DRIVES THE REAL ARM. J6 ONLY -- no other joint is touched. !!!

The gripper is the one joint the pick path never exercised on hardware, so it
gets its own script rather than a `set_gripper(True)` buried in a sequence. It
moves J6 and nothing else: whatever pose the arm is holding, it keeps.

visual_servo.py offers the same thing at the end of a descent, through the same
code (robot_interface/gripper.py). This script exists separately because the
recovery case -- "that descent ended badly, open the claw" -- cannot go through
a descent.

    python scripts/close_claw.py            # jog to the grip position (3003)
    python scripts/close_claw.py --open     # jog back out to 3219
    python scripts/close_claw.py --to 2980  # a tighter grip, if 3003 is loose
    python scripts/close_claw.py --step 20  # finer jogs near the brick
    python scripts/close_claw.py --dry-run  # print the plan, command nothing

MEASURED POSITIONS (operator, 2026-08-07, read off hold_pose.py):

    3219   fully OPEN
    3003   closed ON THE BRICK -- the working grip
    2732   fully closed, jaws touching. "It should never be this much."

Ticks DECREASE as the claw closes. 2732 is a DAMAGE limit, not a target, and
this script will not pass it: a close driven into the stop with a brick in the
jaws stalls the servo against the brick, and a stalled servo draws heavy current
-- that is what put the checksum errors on the bus during the 2026-08-07 descent
and what overloaded J3 on 2026-08-04. It is also J6's min_tick in
data/servo_calibration.json, so ServoBus refuses it independently of this file.

A STALL WHILE CLOSING IS SUCCESS. In goto_tick.py a joint that stops short is
obstructed and the script backs off; here the obstruction is the brick, which is
the entire point.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.robot_interface import gripper
from vision_pipeline.robot_interface.servo_driver import ServoBus


def ask(prompt: str) -> bool:
    """One approval. Anything but an explicit yes stops the run."""
    try:
        return input(prompt).strip().lower() in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--to", type=int, default=None,
                    help=f"target tick (default: grip at "
                         f"{config.SERVO_GRIPPER_GRIP_TICKS})")
    ap.add_argument("--open", action="store_true",
                    help=f"open instead: target {config.SERVO_GRIPPER_OPEN_TICKS}")
    ap.add_argument("--step", type=int, default=config.SERVO_GRIPPER_JOG_TICKS,
                    help=f"ticks per approved jog "
                         f"(default {config.SERVO_GRIPPER_JOG_TICKS})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the first jog, command nothing")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    if args.to is not None and args.open:
        print("--to and --open ask for two different targets. Pick one.")
        sys.exit(2)

    target = args.to
    if target is None:
        target = (config.SERVO_GRIPPER_OPEN_TICKS if args.open
                  else config.SERVO_GRIPPER_GRIP_TICKS)

    why = gripper.refusal(target)
    if why is not None:
        print(why)
        print("Refusing.")
        sys.exit(1)

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as e:
        print(f"Could not open the servo bus on {args.port}: {e}")
        print("Is the servo rail powered? USB alone powers the adapter, not the servos.")
        sys.exit(1)

    with bus:
        try:
            current = bus.read_position(gripper.GRIPPER_JOINT)
        except Exception as e:
            print(f"Could not read J6: {e}")
            sys.exit(1)

        total = target - current
        closing = total < 0
        step = max(1, min(abs(args.step), config.SERVO_MAX_MOVE_DELTA_TICKS))
        n_jogs = (abs(total) + step - 1) // step

        print("\nJ6 (gripper) only. No other joint is commanded.\n")
        print(f"  now     {current}  ({gripper.describe(current)})")
        print(f"  target  {target}  ({gripper.describe(target)})")
        print(f"  travel  {total:+d} ticks, {'CLOSING' if closing else 'OPENING'}, "
              f"up to {n_jogs} jog{'s' if n_jogs != 1 else ''} of {step}")
        print(f"  floor   {config.SERVO_GRIPPER_FULL_CLOSE_TICKS} "
              f"(full close -- this script will not pass it)")

        if total == 0:
            print("\nAlready there. Nothing to do.")
            return

        if closing and current < config.SERVO_GRIPPER_OPEN_TICKS - 60:
            print(f"\n  NOTE: starting {config.SERVO_GRIPPER_OPEN_TICKS - current} "
                  f"ticks in from fully open, so the claw is\n"
                  f"        already partly closed. Check the brick is between the "
                  f"jaws, not\n        behind them.")

        if args.dry_run:
            first = current + (-step if closing else step)
            first = max(first, target) if closing else min(first, target)
            print(f"\n--dry-run: the first jog would be {current} -> {first}. "
                  f"Nothing was commanded.")
            return

        print("\nEvery jog is approved separately. Enter = go, anything else = stop.")
        print("Ctrl-C also stops. The claw holds wherever it got to.\n")

        result = gripper.close_in_jogs(bus, target, step, ask, print)

    print()
    print(result.message)
    if result.outcome == gripper.GRIPPED:
        print("That is the wanted outcome; not pushing further.")
    elif result.outcome == gripper.REACHED and closing:
        print("NOTE: it met nothing on the way, so the jaws shut on air. If a "
              "brick was\nmeant to be there, the claw is not where the brick is.")
    if result.outcome in (gripper.REFUSED, gripper.FAILED):
        sys.exit(1)
    if not args.open:
        print("\nTo let go: python scripts/close_claw.py --open")


if __name__ == "__main__":
    main()

"""Close (or open) the claw in small jogs, asking before every one.

    !!! THIS DRIVES THE REAL ARM. J6 ONLY -- no other joint is touched. !!!

The gripper is the one joint the pick path never exercised on hardware, so it
gets its own script rather than a `set_gripper(True)` buried in a sequence. It
moves J6 and nothing else: whatever pose the arm is holding, it keeps.

Run it after hold_pose.py has the arm parked over the brick with the claw open.

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

A STALL WHILE CLOSING IS SUCCESS, and that is the one way this differs from
goto_tick.py. There, a joint that stops short is obstructed and the script backs
off. Here the obstruction is the brick, which is the entire point: if the claw
stops moving before it reaches the target, it has GRIPPED, and the right thing
is to stop and say so rather than push harder into it.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

GRIPPER = 6

# Two reads this close together mean the servo has stopped.
SETTLE_TOL_TICKS = 2

# A jog that advanced less than this did not really move. While closing that is
# the brick; while opening it is the end of travel.
STALL_TICKS = 3


def settled(bus: ServoBus, timeout_s: float = 3.0) -> int:
    """Where J6 is once it has actually stopped moving.

    move_and_verify's return value is unreliable mid-travel (its stall check can
    trip during the acceleration ramp), and here that matters more than usual:
    an early read looks exactly like the claw having gripped.
    """
    previous = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = bus.read_position(GRIPPER)
        if previous is not None and abs(current - previous) <= SETTLE_TOL_TICKS:
            return current
        previous = current
        time.sleep(0.15)
    return previous if previous is not None else bus.read_position(GRIPPER)


def describe(ticks: int) -> str:
    """Name a J6 position against the three measured landmarks."""
    if ticks >= config.SERVO_GRIPPER_OPEN_TICKS - 10:
        return "fully open"
    if ticks <= config.SERVO_GRIPPER_FULL_CLOSE_TICKS + 10:
        return "FULLY CLOSED (jaws touching)"
    if abs(ticks - config.SERVO_GRIPPER_GRIP_TICKS) <= 10:
        return "at the grip position"
    span = config.SERVO_GRIPPER_OPEN_TICKS - config.SERVO_GRIPPER_FULL_CLOSE_TICKS
    pct = 100.0 * (ticks - config.SERVO_GRIPPER_FULL_CLOSE_TICKS) / span
    return f"{pct:.0f}% open"


def ask(prompt: str) -> bool:
    """One approval. Anything but an explicit yes stops the run."""
    try:
        reply = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  stopped.")
        return False
    return reply in ("", "y", "yes")


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

    # Refuse here as well as in ServoBus. The limit is the reason this script
    # exists, so it should be named in the script's own words, not surfaced as a
    # generic travel-limit refusal three steps into the run.
    if target < config.SERVO_GRIPPER_FULL_CLOSE_TICKS:
        print(f"Target {target} is past the full-close stop at "
              f"{config.SERVO_GRIPPER_FULL_CLOSE_TICKS}.")
        print("That is the jaws shut on themselves. With a brick in the way the "
              "servo would\nstall against it. Refusing.")
        sys.exit(1)
    if target > config.SERVO_GRIPPER_OPEN_TICKS:
        print(f"Target {target} is past the open end of travel at "
              f"{config.SERVO_GRIPPER_OPEN_TICKS}. Refusing.")
        sys.exit(1)

    step = max(1, min(abs(args.step), config.SERVO_MAX_MOVE_DELTA_TICKS))

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as e:
        print(f"Could not open the servo bus on {args.port}: {e}")
        print("Is the servo rail powered? USB alone powers the adapter, not the servos.")
        sys.exit(1)

    with bus:
        try:
            current = bus.read_position(GRIPPER)
        except Exception as e:
            print(f"Could not read J6: {e}")
            sys.exit(1)

        bus.set_motion_profile([GRIPPER], config.SERVO_MOVE_SPEED_TICKS_S,
                               config.SERVO_MOVE_ACCEL)

        total = target - current
        closing = total < 0
        n_jogs = (abs(total) + step - 1) // step

        print(f"\nJ6 (gripper) only. No other joint is commanded.\n")
        print(f"  now     {current}  ({describe(current)})")
        print(f"  target  {target}  ({describe(target)})")
        print(f"  travel  {total:+d} ticks, {'CLOSING' if closing else 'OPENING'}, "
              f"up to {n_jogs} jog{'s' if n_jogs != 1 else ''} of {step}")
        print(f"  floor   {config.SERVO_GRIPPER_FULL_CLOSE_TICKS} "
              f"(full close -- this script will not pass it)")

        if abs(total) == 0:
            print("\nAlready there. Nothing to do.")
            return

        if closing and current < config.SERVO_GRIPPER_OPEN_TICKS - 60:
            print(f"\n  NOTE: starting {config.SERVO_GRIPPER_OPEN_TICKS - current} "
                  f"ticks in from fully open, so the claw is\n"
                  f"        already partly closed. Check the brick is between the "
                  f"jaws, not\n        behind them.")

        if args.dry_run:
            first = current + (-step if closing else step)
            if closing:
                first = max(first, target)
            else:
                first = min(first, target)
            print(f"\n--dry-run: the first jog would be {current} -> {first}. "
                  f"Nothing was commanded.")
            return

        print("\nEvery jog is approved separately. Enter = go, anything else = stop.")
        print("Ctrl-C also stops. The claw holds wherever it got to.\n")

        while current != target:
            remaining = target - current
            nxt = current + max(-step, min(step, remaining))

            if not ask(f"  jog {current} -> {nxt}  ({target - nxt:+d} left)  [Enter/n] "):
                print(f"\nStopped with J6 at {current} ({describe(current)}).")
                return

            try:
                bus.move_and_verify(GRIPPER, nxt)
            except ServoSafetyError as e:
                print(f"  REFUSED: {e}")
                return
            except Exception as e:
                print(f"  move failed: {e}")
                return

            arrived = settled(bus)
            moved = arrived - current
            current = arrived

            if abs(moved) < STALL_TICKS:
                # Asked for `step` ticks and got none of them.
                if closing:
                    print(f"  at {current} -- did not move.")
                    print(f"\nTHE CLAW HAS GRIPPED. It stopped {target - current:+d} "
                          f"ticks short of the\ntarget, which means it met the brick "
                          f"first. That is the wanted outcome;\nnot pushing further.")
                else:
                    print(f"  at {current} -- did not move. That is the open end of "
                          f"travel,\nor something is holding the jaws.")
                return

            print(f"  at {current}  ({describe(current)}, {target - current:+d} to go)")

    print(f"\nJ6 is at {current} ({describe(current)}).")
    if not args.open:
        print("The claw is holding this position. To let go: "
              "python scripts/close_claw.py --open")


if __name__ == "__main__":
    main()

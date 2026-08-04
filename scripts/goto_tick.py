"""Move ONE joint to a target tick value, in safe steps.

    !!! THIS DRIVES THE REAL ARM. Stay by the power cut. !!!

Exists because a legitimate long move cannot be issued as one command:
ServoBus refuses anything over SERVO_MAX_MOVE_DELTA_TICKS (400) in a single
call, which is the guard that turned the J1 wrap-seam incident into a refusal
instead of a 358-degree runaway. This walks to the target in increments under
that cap, waiting for the joint to actually stop between each.

Waiting matters: move_and_verify's return value is unreliable mid-travel
(_wait_for_settle returns early when its stall check trips during the servo's
acceleration ramp), so each step re-reads the position after it settles rather
than trusting what the move reported.

Travel limits still apply — a target outside the joint's measured range is
refused, as is any step toward it.

    python scripts/goto_tick.py --joint 2 --ticks 2883
    python scripts/goto_tick.py --joint 2 --ticks 2883 --step 150   # gentler
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

SETTLE_TOL_TICKS = 2


def settled(bus: ServoBus, joint: int, timeout_s: float = 4.0) -> int:
    """Where the joint is once it has actually stopped moving."""
    previous = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = bus.read_position(joint)
        if previous is not None and abs(current - previous) <= SETTLE_TOL_TICKS:
            return current
        previous = current
        time.sleep(0.15)
    return previous if previous is not None else bus.read_position(joint)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--ticks", type=int, required=True, help="target position")
    ap.add_argument("--step", type=int, default=200,
                    help=f"max ticks per step (default 200, cap {config.SERVO_MAX_MOVE_DELTA_TICKS})")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    step = min(abs(args.step), config.SERVO_MAX_MOVE_DELTA_TICKS)
    bus = ServoBus(args.port, args.baud)

    with bus:
        current = bus.read_position(args.joint)
        limits = bus.travel_limits(args.joint)
        total = args.ticks - current
        print(f"J{args.joint}: {current} -> {args.ticks}  ({total:+d} ticks, "
              f"{abs(total) // step + 1} steps of up to {step})")
        if limits:
            print(f"  travel limits: {limits}")
        print("  Arm WILL move. Ctrl-C or cut power to stop.\n")

        while current != args.ticks:
            remaining = args.ticks - current
            nxt = current + max(-step, min(step, remaining))
            try:
                bus.move_and_verify(args.joint, nxt)
            except ServoSafetyError as e:
                print(f"  REFUSED: {e}")
                return
            except Exception as e:
                print(f"  move failed: {e}")
                return

            arrived = settled(bus, args.joint)
            if abs(arrived - current) < 2 and abs(remaining) > 2:
                print(f"  stopped moving at {arrived} with {remaining:+d} still to go "
                      f"— obstructed or at a hard stop. Not forcing it.")
                return
            current = arrived
            print(f"  at {current}  ({args.ticks - current:+d} to go)")

    print(f"\nJ{args.joint} is at {current}.")


if __name__ == "__main__":
    main()

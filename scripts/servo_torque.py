"""Show which joints are actually holding, and re-enable one that has gone limp.

    !!! Re-enabling torque can move the joint. Support the arm first. !!!

Why this exists: a Feetech servo that trips its overload protection does not
stop answering the bus. It reports a healthy voltage, a normal temperature, and
no fault flags -- it simply stops holding. From outside that is indistinguishable
from "the servo lost power", which is what J3 looked like on 2026-08-04 after it
strained during an aborted move. The giveaway is one register: Torque Enable.

A limp joint is then moved by gravity, so by the time it is noticed the arm is
somewhere nobody commanded (J3 had sagged 54 deg). Meanwhile the servo's Goal
Position still holds the last thing it was told. Enabling torque naively makes
it snap back to that stale goal at full speed from an unplanned pose, which is
why enable_torque sets the goal to the joint's PRESENT position first.

    python scripts/servo_torque.py                 # report only, changes nothing
    python scripts/servo_torque.py --enable 3      # re-enable J3 where it sits
    python scripts/servo_torque.py --enable all    # every limp joint

A joint that trips repeatedly is telling you something real -- it is being asked
to hold more than it can. Re-enabling is not a fix for that.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_driver import ServoBus


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--enable", help="joint id (1-6) or 'all' to re-enable torque")
    ap.add_argument("--off", type=int, help="DISABLE torque on a joint (it will drop)")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    bus = ServoBus(args.port, args.baud)
    with bus:
        limp, answered, silent = [], [], []
        print("joint   volts   temp   torque   position   faults")
        for j in range(1, 7):
            try:
                if not bus.ping(j):
                    silent.append(j)
                    print(f"  J{j}    NO RESPONSE")
                    continue
                d = bus.read_diagnostics(j)
                pos = bus.read_position(j)
            except Exception as e:
                silent.append(j)
                print(f"  J{j}    read failed: {e}")
                continue
            answered.append(j)
            on = d.get("torque_enabled")
            if on is False:
                limp.append(j)
            print(f"  J{j}    {d.get('voltage_v', '?'):>4} V  {d.get('temperature_c', '?'):>3} C   "
                  f"{'ON ' if on else 'OFF'}      {pos:>5}      "
                  f"{', '.join(d.get('faults', [])) or '-'}")

        # Report what was actually established, never what was merely not
        # contradicted. An earlier version printed "all joints are holding"
        # when NOTHING answered, because no servo had reported torque=OFF --
        # a health check that says "all good" to a dead bus is worse than none.
        if not answered:
            print("\n  NOTHING ON THE BUS ANSWERED. Nothing was verified.")
            print("  The arm is powered off, the serial adapter is unplugged, or")
            print(f"  {args.port} is held by another process. Torque state UNKNOWN.")
            return
        if silent:
            print(f"\n  {', '.join(f'J{j}' for j in silent)} DID NOT ANSWER — "
                  f"state unknown for those.")
        if limp:
            print(f"\n  LIMP: {', '.join(f'J{j}' for j in limp)} — not holding position.")
            print("  Anything outboard of these is held up by friction alone.")
        elif not silent:
            print("\n  All six joints answered and all are holding.")

        if args.off:
            print(f"\nDisabling torque on J{args.off}. It WILL be moved by gravity.")
            if input("  Are you supporting the arm? [y/N] ").strip().lower() in ("y", "yes"):
                bus.disable_torque(args.off)
                print(f"  J{args.off} is now limp.")
            else:
                print("  Cancelled.")
            return

        if not args.enable:
            if limp:
                print(f"\n  Re-enable with:  python scripts/servo_torque.py "
                      f"--enable {limp[0] if len(limp) == 1 else 'all'}")
            return

        targets = limp if args.enable == "all" else [int(args.enable)]
        if not targets:
            print("\n  Nothing to enable.")
            return

        print(f"\nRe-enabling torque on {', '.join(f'J{j}' for j in targets)}.")
        print("Each will hold where it currently sits — it will NOT return to any")
        print("earlier pose. Support the arm's weight before confirming.")
        if input("  Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("  Cancelled.")
            return

        for j in targets:
            try:
                held = bus.enable_torque(j)
                print(f"  J{j}: torque ON, holding {held}")
            except Exception as e:
                print(f"  J{j}: FAILED — {e}")


if __name__ == "__main__":
    main()

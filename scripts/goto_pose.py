"""Drive the arm to a named joint-space pose. MOVES THE ARM.

    python scripts/goto_pose.py --pose hover      # where runs should start
    python scripts/goto_pose.py --pose home       # MATLAB's zero, claw near the table
    python scripts/goto_pose.py --pose hover --dry-run

HOVER is the pose every closed-loop run should begin from: the brick is in the
camera's view, so detection has something to work with, and every run starts
from the same geometry -- which is what makes the probe gains from one run
comparable with the next.

HOME is the pose matlab/init_arm.m calls home, all five joint angles zero. FK
there puts the claw tip 70 mm in front of the base, dead centre, 6.3 mm above
the table. It is also the definition of home_tick: if the arm is visibly not in
that pose when this finishes, the calibration is wrong, not the arm.

Raw ticks throughout, never IK. A recovery pose has to work when the clever
paths do not, and IK depends on a hand-eye transform that is known wrong.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.robot_interface import poses
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default="hover", choices=sorted(poses.POSES))
    ap.add_argument("--dry-run", action="store_true",
                    help="show the move and command nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as e:
        print(f"Could not open the servo bus on {args.port}: {e}")
        sys.exit(1)

    with bus:
        # Speed and acceleration live in the servo's SRAM and reset on every
        # power cycle, so they have to be re-applied per run. Without them the
        # servo travels at its full default speed and each step ends in a hard
        # stop, with peak torque far above what the pose needs statically.
        bus.set_motion_profile(sorted(poses.POSES[args.pose]),
                               config.SERVO_MOVE_SPEED_TICKS_S,
                               config.SERVO_MOVE_ACCEL)
        print(f"\nMoving to {args.pose.upper()}:")
        for line in poses.describe_move(bus, poses.POSES[args.pose]):
            print(line)

        if args.dry_run:
            print("\n--dry-run: nothing commanded.")
            return
        if poses.at_pose(bus, args.pose):
            print("\n  Already there. Nothing to do.")
            return
        if not args.yes:
            if input("\n  Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("  Aborted. Nothing commanded.")
                return

        try:
            final = poses.goto(bus, args.pose, label=args.pose,
                               progress=lambda k, n: print(f"    hop {k}/{n}",
                                                           flush=True))
        except ServoSafetyError as e:
            print(f"\n  REFUSED: {e}")
            print("  The arm is holding. Nothing further was commanded.")
            sys.exit(1)
        except KeyboardInterrupt:
            held = bus.freeze(sorted(poses.POSES[args.pose]))
            print(f"\n  Ctrl-C — froze {len(held)} joint(s) where they stand.")
            sys.exit(1)

        print(f"\n  Arrived: " + ", ".join(f"J{j}={t}" for j, t in final.items()))
        if args.pose == "home":
            print("\n  This should be MATLAB's zero pose. Check it against reality:")
            print("    claw tip ~70 mm in front of the base column, dead centre,")
            print("    ~6 mm above the table. If it is not, home_tick is wrong.")


if __name__ == "__main__":
    main()

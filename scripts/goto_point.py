"""Command the claw tip to one base-frame point, previewing every joint first.

    !!! THIS DRIVES THE REAL ARM. Ctrl-C freezes it; power cut is last resort. !!!

The point comes from YOU -- a ruler, not the camera. That is the whole purpose:
it takes vision, hand-eye, and the pixel back-projection completely out of the
loop, leaving only the parts already validated on hardware (IK, the servo bus,
FK). If the claw lands on the brick, those parts are sound. If it lands
somewhere offset, the offset itself is the measurement -- it is the difference
between where the base frame actually is and where it was assumed to be, which
no amount of camera work can reveal.

Staged on purpose. It hovers APPROACH_HEIGHT above the point, stops, and waits
for you to look before descending. Most of the travel happens on the way to
hover, well clear of the table, so anything wrong is visible while there is
still room to react.

Paced on purpose. Every move is broken into hops of at most PICK_STEP_TICKS
(60) with PICK_STEP_PAUSE_S (0.5 s) of rest between them, and the servos are
speed-capped as well, so the arm advances in short watchable increments rather
than one continuous slew. This is the same ServoBus.move_joints_stepped that
HardwareRobot uses for real picks, so the demo and the pipeline cannot drift.

Stopping it. Ctrl-C overwrites each joint's goal with its present position:
motion halts, torque stays on, nothing falls. Cutting power instead drops
holding torque on every joint at once and the arm drops under its own weight --
that is how J3 was overloaded on 2026-08-04. Use the power cut only if Ctrl-C
does not visibly stop the arm.

    python scripts/goto_point.py --x 145 --y 35 --z -64
    python scripts/goto_point.py --x 145 --y 35 --z -64 --hover-only
    python scripts/goto_point.py --x 145 --y 35 --z -64 --step 30 --pause 1.0

    +x is forward, +y is LEFT, z is height (table is TABLE_Z_IN_BASE = -74 mm).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.robot_interface.matlab_client import IKUnreachableError, MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

IK_JOINTS = (1, 2, 3, 4, 5)


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def read_angles(bus: ServoBus) -> list[float]:
    return [bus.ticks_to_rad(j, bus.read_position(j)) for j in IK_JOINTS]


def _safe_read(bus: ServoBus):
    """read_angles, but survives a bus that is itself failing.

    Called on the error path, where the most likely cause is exactly that the
    bus stopped answering, so re-reading must not raise a second time and bury
    the original failure.
    """
    try:
        return read_angles(bus)
    except Exception as e:
        print(f"    Could not re-read joint positions either: {e}")
        print("    Joint state is UNKNOWN. Power-cycle and run check_servo_health.py.")
        return None


def go_to(bus, ik, label, xyz, angles_now, step_cap, pause_s):
    """Preview, confirm, then move in paced steps.

    Returns the new joint angles, or None if the move did not happen (refused,
    unreachable, or declined).
    """
    x, y, z = xyz
    print(f"\n--- {label}: ({x * 1000:+.1f}, {y * 1000:+.1f}, {z * 1000:+.1f}) mm ---")

    floor_z = config.TABLE_Z_IN_BASE + config.MIN_CLAW_HEIGHT_M
    if z < floor_z:
        print(f"  REFUSED by the floor guard: z is below {floor_z * 1000:.1f} mm "
              f"(table {config.TABLE_Z_IN_BASE * 1000:.0f} + "
              f"{config.MIN_CLAW_HEIGHT_M * 1000:.0f} clearance).")
        return None

    try:
        target_angles, err_mm = ik.request_ik(x, y, z, seed_rad=angles_now)
    except IKUnreachableError as e:
        print(f"  UNREACHABLE: {e}")
        return None
    print(f"  IK solved, residual {err_mm:.1f} mm")

    # Preview every joint before anything moves.
    blocked, worst = [], 0.0
    for j, (now, want) in zip(IK_JOINTS, zip(angles_now, target_angles)):
        delta_deg = np.degrees(want - now)
        tick_now, tick_want = bus.rad_to_ticks(j, now), bus.rad_to_ticks(j, want)
        limits = bus.travel_limits(j)
        note = ""
        if limits and not (limits[0] <= tick_want <= limits[1]):
            note = f"  OUTSIDE LIMITS {limits} - will be refused"
            blocked.append(j)
        worst = max(worst, abs(delta_deg))
        print(f"    J{j}: {tick_now:5d} -> {tick_want:5d} ticks  "
              f"({delta_deg:+7.2f} deg){note}")

    if blocked:
        print("\n  At least one joint would leave its measured travel. Not moving.")
        return None
    if worst > config.SERVO_WATCH_POWER_MOVE_DEG:
        print(f"\n  *** LARGEST MOVE IS {worst:.1f} deg - over the "
              f"{config.SERVO_WATCH_POWER_MOVE_DEG:.0f} deg threshold. ***")

    targets = {j: bus.rad_to_ticks(j, a) for j, a in zip(IK_JOINTS, target_angles)}
    biggest = max(abs(targets[j] - bus.rad_to_ticks(j, a))
                  for j, a in zip(IK_JOINTS, angles_now))
    rounds = max(1, int(np.ceil(biggest / step_cap)))
    print(f"\n  {rounds} steps of at most {step_cap} ticks "
          f"({np.degrees(step_cap / 651.89):.1f} deg), {pause_s:.1f}s rest between")
    print(f"  roughly {rounds * (pause_s + 0.7):.0f}s of motion. "
          f"Ctrl-C freezes the arm in place.")

    if not confirm("  Execute this move? [y/N] "):
        print("  Skipped.")
        return None

    try:
        bus.move_joints_stepped(
            targets, step_ticks=step_cap, pause_s=pause_s,
            progress=lambda k, n: print(f"    step {k}/{n}", flush=True),
        )
    except KeyboardInterrupt:
        # Ctrl-C alone does NOT stop the arm: the Goal Position is already in
        # the servo, which will keep travelling there with nothing attached.
        # Overwrite it with where the joint already is.
        held = bus.freeze(IK_JOINTS)
        print(f"\n\n    *** STOPPED - frozen at {held} ***")
        print("    Torque is still on; the arm is holding, not falling.")
        print("    Cut power only if it is still moving after this.")
        return _safe_read(bus)
    except ServoSafetyError as e:
        print(f"\n    REFUSED: {e}")
        print("    Arm stopped part-way. Re-reading all joints.")
        return _safe_read(bus)
    except Exception as e:
        print(f"\n    BUS/MOVE FAILURE: {e}")
        print("    The arm stopped part-way and is holding an intermediate pose.")
        print("    Run scripts/servo_torque.py before commanding anything else.")
        return _safe_read(bus)

    new_angles = read_angles(bus)

    # Where it actually ended up, against where it was asked to go.
    _, T_tip = ik.request_fk_tip(new_angles)
    tip = T_tip[:3, 3]
    error = np.linalg.norm(tip - np.array([x, y, z])) * 1000
    print(f"  claw tip now: ({tip[0] * 1000:+.1f}, {tip[1] * 1000:+.1f}, "
          f"{tip[2] * 1000:+.1f}) mm   [{error:.1f} mm from commanded]")
    print(f"  height above the table: {(tip[2] - config.TABLE_Z_IN_BASE) * 1000:+.1f} mm")
    return new_angles


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--x", type=float, required=True, help="forward, mm")
    ap.add_argument("--y", type=float, required=True, help="LEFT, mm")
    ap.add_argument("--z", type=float, required=True, help="height, mm (table = -74)")
    ap.add_argument("--approach", type=float, default=config.APPROACH_HEIGHT * 1000,
                    help="hover height above the point, mm")
    ap.add_argument("--step", type=int, default=config.PICK_STEP_TICKS,
                    help=f"max ticks per joint per step "
                         f"(default {config.PICK_STEP_TICKS})")
    ap.add_argument("--pause", type=float, default=config.PICK_STEP_PAUSE_S,
                    help=f"seconds of rest between steps "
                         f"(default {config.PICK_STEP_PAUSE_S})")
    ap.add_argument("--speed", type=int, default=config.SERVO_MOVE_SPEED_TICKS_S,
                    help=f"servo speed limit in ticks/s, 0 = maximum "
                         f"(default {config.SERVO_MOVE_SPEED_TICKS_S})")
    ap.add_argument("--hover-only", action="store_true",
                    help="stop at hover; do not descend")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    x, y, z = args.x / 1000, args.y / 1000, args.z / 1000

    print("=" * 68)
    print("GOTO POINT - the arm will move. Ctrl-C freezes it.")
    print("=" * 68)
    print(f"target      ({args.x:+.1f}, {args.y:+.1f}, {args.z:+.1f}) mm")
    print(f"            {float(np.hypot(x, y)) * 1000:.0f} mm from the base axis, "
          f"{args.z - config.TABLE_Z_IN_BASE * 1000:+.0f} mm above the table")
    print(f"hover       {args.approach:.0f} mm above that")

    bus = ServoBus(args.port, args.baud)
    ik = MatlabIKClient(config.MATLAB_SERVER_HOST, config.MATLAB_SERVER_PORT)

    with bus:
        # Speed and acceleration live in the servos' SRAM, so they are lost on
        # every power cycle and must be re-applied per run.
        bus.set_motion_profile(IK_JOINTS, args.speed, config.SERVO_MOVE_ACCEL)
        print(f"\nspeed cap   {args.speed} ticks/s "
              f"(~{np.degrees(args.speed / 651.89):.0f} deg/s), accel "
              f"{config.SERVO_MOVE_ACCEL}"
              + ("   <-- UNLIMITED" if args.speed == 0 else ""))
        print(f"pacing      {args.step} ticks per step, {args.pause:.1f}s between")

        limp = [j for j in IK_JOINTS
                if bus.read_diagnostics(j).get("torque_enabled") is False]
        if limp:
            print(f"\n  *** {', '.join(f'J{j}' for j in limp)} HAS NO TORQUE - not "
                  f"holding position. ***")
            print("      Support the arm and run: python scripts/servo_torque.py --enable all")
            return

        angles = read_angles(bus)
        _, T_tip = ik.request_fk_tip(angles)
        tip = T_tip[:3, 3]
        print(f"\nclaw tip now: ({tip[0] * 1000:+.1f}, {tip[1] * 1000:+.1f}, "
              f"{tip[2] * 1000:+.1f}) mm")

        angles = go_to(bus, ik, "HOVER", (x, y, z + args.approach / 1000), angles,
                       args.step, args.pause) or angles

        if args.hover_only:
            print("\n--hover-only: stopping here.")
        else:
            print("\nLook at the arm. Is it directly above the brick?")
            if confirm("Descend to the point? [y/N] "):
                go_to(bus, ik, "DESCEND", (x, y, z), angles, args.step, args.pause)
            else:
                print("  Not descending.")

    ik.close()
    print("\n" + "=" * 68)
    print("Compare the claw tip against the brick. Any consistent offset is the")
    print("base-frame error - measure it and it can be corrected once, globally.")
    print("=" * 68)


if __name__ == "__main__":
    main()

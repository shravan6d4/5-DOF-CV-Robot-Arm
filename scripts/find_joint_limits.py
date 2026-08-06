"""Measure ONE joint's safe travel range and record it, watching for the
encoder wrap seam throughout.

    !!! THIS DRIVES THE REAL ARM. Stay by the power cut. !!!

Why this exists: ServoBus.move_and_verify caps how FAR one command travels but,
until the limits below are measured, has no idea WHERE a joint may go. Small
legal steps can therefore walk a joint into a hard stop — which is exactly how
J3 and J4 jammed on 2026-08-04, each time ending in a stalled servo straining
against a stop until the power was cut.

Why it steps rather than sweeps: every step is small, verified, and confirmed
by you before the next one. The joint stops at the first sign of resistance, so
the stop is *found* rather than *hit*.

Why it watches the seam: the encoder wraps 4095 -> 0. A joint parked near that
boundary reports a ~4000-tick jump for a hair of real movement, which is what
sent J1 the long way round at speed in July. Tick limits measured across a wrap
are meaningless — they are not even on the same scale — so this refuses to
record them and tells you to re-centre the servo instead.

    python scripts/find_joint_limits.py --joint 3
    python scripts/find_joint_limits.py --joint 2 --step 20   # finer steps

Per direction it steps until you say stop, then records that end. Run it for
each direction, then it writes min_tick/max_tick into
data/servo_calibration.json and the matching angle limits into
data/joint_limits_rad.json, which MATLAB's IK solver reads so it stops
producing solutions the arm cannot reach.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_calibration import (
    load_calibration_file,
    save_calibration_file,
    write_angle_limits,
)
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

# A real step is tens of ticks. Anything approaching half the encoder's range is
# the 4095->0 wrap showing up as a huge apparent jump, not real motion.
SEAM_JUMP_TICKS = 1500
# How close a RECORDED limit may sit to the 0/4095 seam before it is called out.
# A warning threshold, not an applied margin -- a limit inside it means the
# joint's ENCODER is badly placed and should be re-centred
# (scripts/recentre_joint.py), not that the limit should be pulled in.
SEAM_MARGIN_TICKS = config.SERVO_SEAM_WARN_TICKS


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _settled_position(bus: ServoBus, joint: int, timeout_s: float = 4.0) -> int:
    """Poll until the joint stops moving, then return where it actually is.

    move_and_verify's return value cannot be used to judge how far a joint
    travelled: _wait_for_settle returns early when its stall check trips, which
    on a real servo happens routinely during the acceleration ramp, so it hands
    back a position the joint has already left. Trusting it made this script
    read "moved 0 ticks" mid-travel and declare a stop that was not there.
    """
    previous = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = bus.read_position(joint)
        if previous is not None and abs(current - previous) <= 2:
            return current
        previous = current
        time.sleep(0.15)
    return previous if previous is not None else bus.read_position(joint)


def explore(bus: ServoBus, joint: int, step: int) -> int | None:
    """Step one direction until the operator stops it. Returns the end tick."""
    sign = "+" if step > 0 else "-"
    print(f"\n--- exploring J{joint} in the {sign} direction, {abs(step)} ticks per step ---")
    print("    After each step: 'y' to keep going, anything else to stop there.")
    print("    STOP THE MOMENT you see or hear the joint straining.")

    last = bus.read_position(joint)
    print(f"    starting at {last}")
    stalled = 0

    while True:
        if not _confirm(f"    step {step:+d} from {last}? [y/N] "):
            print(f"    stopped at {last}")
            return last
        try:
            bus.move_and_verify(joint, last + step)
        except ServoSafetyError as e:
            print(f"    REFUSED: {e}")
            return last
        except Exception as e:
            print(f"    move failed: {e} — stopping here.")
            return last

        # Re-read after the joint has actually stopped; move_and_verify's return
        # value is unreliable mid-travel (see _settled_position).
        actual = _settled_position(bus, joint)

        if abs(actual - last) > SEAM_JUMP_TICKS:
            print(f"\n    *** ENCODER WRAP DETECTED: {last} -> {actual} ***")
            print("    That jump is the 4095->0 boundary, not real motion. Tick limits")
            print("    measured across it are not on one scale and cannot be used.")
            print("    Re-centre this servo (Feetech one-key midpoint, as was done for")
            print("    J1 in July) so it works far from the seam, then re-run this.")
            return None

        moved = actual - last
        # Two consecutive under-travels, not one: a single short step can come
        # from a transient read or the servo still creeping, and calling the
        # limit early is exactly how this script first under-measured J2.
        if abs(moved) < abs(step) * 0.4:
            stalled += 1
            print(f"    only moved {moved:+d} of {step:+d} "
                  f"({'STOP CONFIRMED' if stalled >= 2 else 'once more to confirm'})")
            if stalled >= 2:
                print(f"    treating {actual} as the limit.")
                return actual
        else:
            stalled = 0
            print(f"    now at {actual} ({moved:+d})")

        last = actual


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--step", type=int, default=30, help="ticks per step (default 30)")
    ap.add_argument(
        "--direction", choices=("both", "plus", "minus"), default="both",
        help="which way to explore. Use plus/minus when the joint already sits "
             "AT one end: the starting position is then recorded as that end and "
             "only the other direction is explored, so the arm is never driven "
             "into a stop it is already touching.",
    )
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    ap.add_argument("--calibration", default=config.SERVO_CALIBRATION_PATH)
    args = ap.parse_args()

    bus = ServoBus(args.port, args.baud, calibration_path=args.calibration)

    # Ignore any limits already recorded for THIS joint: the job here is to
    # establish them, and a previous bad measurement would otherwise refuse the
    # very steps needed to correct it. Every other joint stays protected, and
    # each step is still operator-confirmed.
    stale = {k: bus._cal(args.joint).pop(k, None) for k in ("min_tick", "max_tick")}
    if any(v is not None for v in stale.values()):
        print(f"  (ignoring previously recorded limits {stale} while re-measuring)")

    print(f"J{args.joint}: finding travel limits. The arm WILL move on each confirmed step.")

    with bus:
        start = bus.read_position(args.joint)
        print(f"  current position: {start}")
        if min(start, 4095 - start) < 300:
            print(f"  WARNING: only {min(start, 4095-start)} ticks from the 0/4095 seam.")
            print("  Limits measured here are fragile; consider re-centring first.")

        if args.direction == "plus":
            print(f"  --direction plus: recording {start} as the LOWER limit "
                  f"(joint is already at that end) and exploring + only.")
            lo = start
            hi = explore(bus, args.joint, abs(args.step))
        elif args.direction == "minus":
            print(f"  --direction minus: recording {start} as the UPPER limit "
                  f"(joint is already at that end) and exploring - only.")
            hi = start
            lo = explore(bus, args.joint, -abs(args.step))
        else:
            hi = explore(bus, args.joint, abs(args.step))
            if hi is None:
                sys.exit(1)

            print(f"\n  returning toward the start ({start}) before the other direction...")
            while abs(bus.read_position(args.joint) - start) > abs(args.step):
                cur = bus.read_position(args.joint)
                nxt = cur + int(np.sign(start - cur)) * abs(args.step)
                try:
                    bus.move_and_verify(args.joint, nxt)
                except Exception as e:
                    print(f"  stopped returning: {e}")
                    break

            lo = explore(bus, args.joint, -abs(args.step))

        if lo is None or hi is None:
            sys.exit(1)

    lo, hi = sorted((lo, hi))
    span = hi - lo
    print(f"\n=== J{args.joint} measured travel: [{lo}, {hi}]  ({span} ticks, "
          f"{np.degrees(span / 651.89):.0f} deg) ===")

    # A limit is only useful if the joint can sit near it safely. One landing on
    # the 0/4095 seam cannot: a hair of motion past it — a commanded overshoot,
    # or gravity sag while unpowered — wraps the reading by ~4000 ticks, and the
    # servo cannot cross back. J3 was first measured to tick 6 this way.
    # Warned rather than refused: the measurement is real, it is the RECORDING
    # that needs a decision, and only the operator knows whether the joint needs
    # that end of its travel at all.
    for name, edge in (("lower", lo), ("upper", hi)):
        margin = min(edge, 4095 - edge)
        if margin < SEAM_MARGIN_TICKS:
            print(f"\n  *** the {name} limit ({edge}) is only {margin} ticks from the")
            print(f"      0/4095 encoder seam ({np.degrees(margin / 651.89):.1f} deg). A limit here is")
            print("      not usable — past it the reading wraps and the servo is stuck.")
            print("      If the arm does not need that end, pull the limit in by hand")
            print(f"      (~{SEAM_MARGIN_TICKS} ticks of margin) and note why in limit_basis.")
            print("      If it DOES need it, re-centre the servo instead.")

    if not _confirm("Write these limits into the calibration? [y/N] "):
        print("Nothing written.")
        return

    # UTF-8 explicitly: every joint now carries a prose limit_basis note, and a
    # default-encoding read-modify-write on Windows mangles them permanently.
    cal = load_calibration_file(args.calibration)
    cal[str(args.joint)]["min_tick"] = int(lo)
    cal[str(args.joint)]["max_tick"] = int(hi)
    path = save_calibration_file(cal, args.calibration)
    print(f"Wrote min_tick/max_tick for J{args.joint} to {path}")

    # Angle limits for MATLAB's IK solver, in the MODEL's own convention so
    # init_arm.m can apply them directly. Written through the shared helper so
    # the tick limits and the angle limits cannot drift apart.
    angle_path, angles = write_angle_limits(cal)
    print(f"Wrote angle limits for {len(angles)} joint(s) to {angle_path}")
    print("\nRestart the MATLAB server (ik_fk_server) so the IK solver picks them up.")


if __name__ == "__main__":
    main()

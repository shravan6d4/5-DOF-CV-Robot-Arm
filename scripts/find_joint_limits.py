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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

# A real step is tens of ticks. Anything approaching half the encoder's range is
# the 4095->0 wrap showing up as a huge apparent jump, not real motion.
SEAM_JUMP_TICKS = 1500
ANGLE_LIMITS_PATH = "data/joint_limits_rad.json"


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def explore(bus: ServoBus, joint: int, step: int) -> int | None:
    """Step one direction until the operator stops it. Returns the end tick."""
    sign = "+" if step > 0 else "-"
    print(f"\n--- exploring J{joint} in the {sign} direction, {abs(step)} ticks per step ---")
    print("    After each step: 'y' to keep going, anything else to stop there.")
    print("    STOP THE MOMENT you see or hear the joint straining.")

    last = bus.read_position(joint)
    print(f"    starting at {last}")

    while True:
        if not _confirm(f"    step {step:+d} from {last}? [y/N] "):
            print(f"    stopped at {last}")
            return last
        try:
            actual = bus.move_and_verify(joint, last + step)
        except ServoSafetyError as e:
            print(f"    REFUSED: {e}")
            return last
        except Exception as e:
            print(f"    move failed: {e} — stopping here.")
            return last

        if abs(actual - last) > SEAM_JUMP_TICKS:
            print(f"\n    *** ENCODER WRAP DETECTED: {last} -> {actual} ***")
            print("    That jump is the 4095->0 boundary, not real motion. Tick limits")
            print("    measured across it are not on one scale and cannot be used.")
            print("    Re-centre this servo (Feetech one-key midpoint, as was done for")
            print("    J1 in July) so it works far from the seam, then re-run this.")
            return None

        moved = actual - last
        if abs(moved) < abs(step) * 0.4:
            print(f"    only moved {moved:+d} of {step:+d} requested — likely AT THE STOP.")
            print(f"    treating {actual} as the limit.")
            return actual

        last = actual
        print(f"    now at {actual} ({moved:+d})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--step", type=int, default=30, help="ticks per step (default 30)")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    ap.add_argument("--calibration", default=config.SERVO_CALIBRATION_PATH)
    args = ap.parse_args()

    bus = ServoBus(args.port, args.baud, calibration_path=args.calibration)
    print(f"J{args.joint}: finding travel limits. The arm WILL move on each confirmed step.")

    with bus:
        start = bus.read_position(args.joint)
        print(f"  current position: {start}")
        if min(start, 4095 - start) < 300:
            print(f"  WARNING: only {min(start, 4095-start)} ticks from the 0/4095 seam.")
            print("  Limits measured here are fragile; consider re-centring first.")

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
        if lo is None:
            sys.exit(1)

    lo, hi = sorted((lo, hi))
    span = hi - lo
    print(f"\n=== J{args.joint} measured travel: [{lo}, {hi}]  ({span} ticks, "
          f"{np.degrees(span / 651.89):.0f} deg) ===")

    if not _confirm("Write these limits into the calibration? [y/N] "):
        print("Nothing written.")
        return

    path = Path(args.calibration)
    cal = json.loads(path.read_text())
    cal[str(args.joint)]["min_tick"] = int(lo)
    cal[str(args.joint)]["max_tick"] = int(hi)
    path.write_text(json.dumps(cal, indent=2))
    print(f"Wrote min_tick/max_tick for J{args.joint} to {path}")

    # Angle limits for MATLAB's IK solver. Written as radians in the MODEL's own
    # convention so init_arm.m can apply them directly, and kept in a separate
    # file because they are specific to this physical arm.
    angles = {}
    for j in range(1, 7):
        c = cal[str(j)]
        if "min_tick" not in c or "max_tick" not in c:
            continue
        a = sorted((bus.ticks_to_rad(j, c["min_tick"]), bus.ticks_to_rad(j, c["max_tick"])))
        angles[str(j)] = {"min_rad": a[0], "max_rad": a[1]}
    Path(ANGLE_LIMITS_PATH).write_text(json.dumps(angles, indent=2))
    print(f"Wrote angle limits for {len(angles)} joint(s) to {ANGLE_LIMITS_PATH}")
    print("\nRestart the MATLAB server (ik_fk_server) so the IK solver picks them up.")


if __name__ == "__main__":
    main()

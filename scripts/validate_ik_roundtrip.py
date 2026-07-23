"""
Stage-D validation — the real "does MATLAB IK drive this arm correctly" test.

For each target: solve IK -> command the servos -> read back what they actually
reached -> run FK on those verified positions -> compare against what was asked
for. This is the first test where IK commands the arm, and the first that
validates MAGNITUDE rather than just direction: dir_sign being correct only
proves joints move the right WAY, not that ticks_per_rad, backlash, and the
model's link lengths match physical reality.

Compares against the CLAW TIP, not the wrist. IK targets the tip; FK's primary
transform is the wrist, ~70mm away. Comparing a commanded tip against a
measured wrist would report that 70mm frame offset as if it were positioning
error. Requires the T_tip field, so the MATLAB server must have been restarted
since ik_fk_server.m gained it.

SAFETY — this moves the arm through IK-computed poses, which can shift several
joints at once (unlike the single-joint jogs of Stage C):
  * LIFTS FIRST, then does its lateral/descent tests at the raised height, so
    the tight clearance under the claw stops being the binding constraint;
  * refuses any target below a floor derived from --clearance-mm;
  * previews each move (target, IK residual, per-joint tick deltas) and waits
    for confirmation before commanding anything;
  * ServoBus's per-move tick cap and read-back verification stay active.
Keep the power cut within reach. It remains the only real e-stop.

Usage (from the repo root):
    python scripts/validate_ik_roundtrip.py --clearance-mm 10 --dry-run
    python scripts/validate_ik_roundtrip.py --clearance-mm 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.robot_interface.matlab_client import IKUnreachableError, MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

IK_JOINTS = range(1, 6)

# Clearance the script insists on keeping between the claw tip and the table.
#
# MUST stay below the real pick height (table + config.PICK_Z_OFFSET), or this
# validation would exclude the exact height the arm has to work at — the brick
# sits ON the table, so a floor above the pick height validates a region the
# robot never operates in. With PICK_Z_OFFSET at 10mm, a 5mm margin leaves the
# pick height reachable with 5mm still underneath it.
FLOOR_MARGIN_MM = 5.0

# How far to lift before running the lateral tests. Chosen so the sideways
# moves happen with real room underneath rather than fighting the (typically
# very tight) clearance at pick height.
LIFT_MM = 45.0

# The accuracy probes, as (label, dx, dy, dz) in mm, applied from the LIFTED
# pose. Deliberately modest: this is a measurement, not a workspace sweep, and
# every mm of travel is a mm the safety cap has to allow.
PROBES = [
    ("forward", 25.0, 0.0, 0.0),
    ("back", -25.0, 0.0, 0.0),
    ("left", 0.0, 25.0, 0.0),
    ("right", 0.0, -25.0, 0.0),
    ("down", 0.0, 0.0, -20.0),
    ("up", 0.0, 0.0, 20.0),
]


def read_angles(bus) -> list[float]:
    """Current MATLAB joint angles from live servo positions."""
    return [bus.ticks_to_rad(j, bus.read_position(j)) for j in IK_JOINTS]


def tip_mm(client, angles_rad) -> np.ndarray:
    """Claw-tip position in mm, base frame, for a set of joint angles."""
    _, T_tip = client.request_fk_tip(angles_rad)
    return 1000 * T_tip[:3, 3]


def preview_move(bus, client, target_mm, current_angles):
    """Solve IK and report what the move would require, without commanding it.

    Returns (angles_rad, err_mm, tick_deltas) or None if IK can't reach it.
    """
    try:
        angles_rad, err_mm = client.request_ik(*(target_mm / 1000.0))
    except IKUnreachableError as e:
        print(f"    IK could not reach this target: {e}")
        return None

    tick_deltas = {}
    for idx, j in enumerate(IK_JOINTS):
        now_tick = bus.read_position(j)
        want_tick = bus.rad_to_ticks(j, angles_rad[idx])
        tick_deltas[j] = want_tick - now_tick
    return angles_rad, err_mm, tick_deltas


def execute_move(bus, angles_rad) -> bool:
    """Command J1..J5. Returns False if any joint refused or failed."""
    for idx, j in enumerate(IK_JOINTS):
        target_tick = bus.rad_to_ticks(j, angles_rad[idx])
        try:
            bus.move_and_verify(j, target_tick)
        except ServoSafetyError as e:
            print(f"    REFUSED on J{j}: {e}")
            return False
        except RuntimeError as e:
            print(f"    FAILED on J{j}: {e}")
            return False
    return True


def attempt(bus, client, label, target_mm, floor_z, args, results):
    """Preview, confirm, execute, and measure one target."""
    print(f"\n--- {label} ---")
    print(f"  target (tip):  ({target_mm[0]:+7.1f}, {target_mm[1]:+7.1f}, {target_mm[2]:+7.1f}) mm")

    if target_mm[2] < floor_z:
        print(f"  SKIPPED: target z {target_mm[2]:+.1f} is below the safety floor "
              f"{floor_z:+.1f} mm.")
        results.append((label, None, None, "below floor"))
        return

    current_angles = read_angles(bus)
    prev = preview_move(bus, client, target_mm, current_angles)
    if prev is None:
        results.append((label, None, None, "IK unreachable"))
        return
    angles_rad, err_mm, tick_deltas = prev

    print(f"  IK residual:   {err_mm:.2f} mm (solver's own error against the target)")
    deltas = "  ".join(f"J{j}{d:+5d}" for j, d in tick_deltas.items())
    print(f"  joint moves:   {deltas}  ticks")

    biggest = max(abs(d) for d in tick_deltas.values())
    if biggest > config.SERVO_MAX_MOVE_DELTA_TICKS:
        print(f"  NOTE: largest move {biggest} ticks exceeds the "
              f"{config.SERVO_MAX_MOVE_DELTA_TICKS}-tick cap; the driver will refuse it.")

    if args.dry_run:
        print("  --dry-run: not commanded.")
        results.append((label, None, None, "dry run"))
        return

    if not args.yes:
        if input("  Type 'go' to command this move: ").strip().lower() != "go":
            print("  Skipped.")
            results.append((label, None, None, "skipped by operator"))
            return

    if not execute_move(bus, angles_rad):
        results.append((label, None, None, "move refused/failed"))
        return

    achieved = tip_mm(client, read_angles(bus))
    error = achieved - target_mm
    dist = float(np.linalg.norm(error))

    print(f"  achieved (tip):({achieved[0]:+7.1f}, {achieved[1]:+7.1f}, {achieved[2]:+7.1f}) mm")
    print(f"  error:         ({error[0]:+6.1f}, {error[1]:+6.1f}, {error[2]:+6.1f}) "
          f"-> {dist:.1f} mm")
    results.append((label, target_mm, achieved, f"{dist:.1f} mm"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clearance-mm", type=float, required=True,
                    help="measured gap under the claw RIGHT NOW (drives the safety floor)")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    ap.add_argument("--dry-run", action="store_true",
                    help="preview every target's IK solution without moving")
    ap.add_argument("--yes", action="store_true",
                    help="skip per-move confirmation (still moves!)")
    args = ap.parse_args()

    try:
        client = MatlabIKClient()
    except (ConnectionRefusedError, OSError):
        print(f"No MATLAB server on {config.MATLAB_SERVER_HOST}:{config.MATLAB_SERVER_PORT}.")
        print("Start it in MATLAB (matlab/ folder):  >> ik_fk_server")
        sys.exit(1)

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as e:
        print(f"Could not open the servo bus on {args.port}: {e}")
        sys.exit(1)

    results = []
    with client, bus:
        try:
            start_tip = tip_mm(client, read_angles(bus))
        except RuntimeError as e:
            print(f"\n{e}")
            sys.exit(1)

        table_z = start_tip[2] - args.clearance_mm
        floor_z = table_z + FLOOR_MARGIN_MM

        print(f"start (tip):   ({start_tip[0]:+7.1f}, {start_tip[1]:+7.1f}, "
              f"{start_tip[2]:+7.1f}) mm")
        print(f"table (est):   z = {table_z:+.1f} mm  "
              f"(from --clearance-mm {args.clearance_mm:.0f})")
        print(f"safety floor:  z = {floor_z:+.1f} mm  "
              f"(never command the tip below this)")

        # Lift first, so the tight clearance stops being the binding constraint
        # for everything that follows.
        lifted = start_tip + np.array([0.0, 0.0, LIFT_MM])
        attempt(bus, client, f"lift +{LIFT_MM:.0f}mm", lifted, floor_z, args, results)

        if not args.dry_run and results[-1][2] is None:
            print("\nLift did not complete — stopping rather than running the "
                  "probes from an unknown pose.")
        else:
            base = lifted if args.dry_run else tip_mm(client, read_angles(bus))
            for label, dx, dy, dz in PROBES:
                attempt(bus, client, label, base + np.array([dx, dy, dz]),
                        floor_z, args, results)
                # Return to the lifted pose between probes so errors don't
                # accumulate and each probe is measured from the same origin.
                attempt(bus, client, f"return (after {label})", base,
                        floor_z, args, results)

            # The measurement that actually matters: accuracy at the height the
            # arm has to work at. A brick sits ON the table, so every pick ends
            # here — validating only at the lifted height would leave the
            # operating point untested. Run last, so the probes above have
            # already shown whether the error is small enough to trust a
            # descent this close to the tabletop.
            pick_z = table_z + 1000 * config.PICK_Z_OFFSET
            pick_target = np.array([start_tip[0], start_tip[1], pick_z])
            print(f"\n{'=' * 62}")
            print(f"Descending to the REAL pick height "
                  f"(table {table_z:+.1f} + PICK_Z_OFFSET "
                  f"{1000 * config.PICK_Z_OFFSET:.0f}mm = {pick_z:+.1f} mm).")
            worst_so_far = [float(n.split()[0]) for _, _, a, n in results
                            if a is not None and n.endswith("mm")]
            if worst_so_far:
                print(f"Worst error so far: {max(worst_so_far):.1f} mm. If that is "
                      f"anywhere near the\n{args.clearance_mm:.0f}mm clearance, "
                      f"stop here and re-check rather than descending.")
            print("=" * 62)
            attempt(bus, client, "PICK HEIGHT", pick_target, floor_z, args, results)

    print("\n" + "=" * 62)
    print(f"{'probe':22} {'result':>12}")
    print("-" * 62)
    for label, _, _, note in results:
        print(f"{label:22} {note:>12}")
    print("=" * 62)
    errs = [float(n.split()[0]) for _, _, a, n in results
            if a is not None and n.endswith("mm")]
    if errs:
        print(f"worst error {max(errs):.1f} mm across {len(errs)} measured moves "
              f"(target: under 10 mm)")
    print("\nThese are FK-vs-IK numbers — they confirm the software chain agrees "
          "with itself.\nSpot-check at least one pose against a ruler: that is what "
          "catches a model\nthat is self-consistent but does not match the real arm.")


if __name__ == "__main__":
    main()

"""
Stage-C direction check — jogs ONE joint a small amount and compares the
physical motion against MATLAB's prediction, to settle that joint's dir_sign.

WHY RAW TICKS, NOT IK: the whole point is that we do not yet trust the
tick<->angle direction mapping. Going through IK would fold an unknown sign
into a solve and produce a target that is wrong in a way that is hard to read.
Commanding a raw tick delta keeps exactly one unknown in play at a time.

HOW IT READS: dir_sign is what converts ticks to the angle MATLAB is given, so
the FK prediction below is computed THROUGH the current dir_sign. If the arm
moves the way the prediction says, that joint's dir_sign is right. If it moves
the opposite way, it is inverted and must be flipped.

All positions/predictions here are in the PHYSICAL frame (+X forward, +Y left,
+Z up) — MatlabIKClient converts to/from the imported model's flipped frame at
the wire (model +Z is physically DOWN; see CLAUDE.md "COORDINATE FRAMES"), so
this script's "up"/"left" language matches what the operator actually sees.

SAFETY. This moves the arm. Before running:
  * claw clear of the table with a few cm underneath (J3 folding down is the
    near-miss that already happened once during bring-up);
  * power cut within reach — it remains the only real e-stop;
  * one joint per run, watch the arm, not the screen.
ServoBus's move cap (config.SERVO_MAX_MOVE_DELTA_TICKS) stays active throughout.

Usage (from the repo root):
    python scripts/jog_joint.py --joint 3 --dry-run     # predict only, no motion
    python scripts/jog_joint.py --joint 3               # +40 ticks
    python scripts/jog_joint.py --joint 3 --ticks -40   # the other way
    python scripts/jog_joint.py --joint 3 --ticks 40 --apply-flip
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

IK_JOINTS = range(1, 6)
DEFAULT_TICKS = 80           # ~7 deg at J1..J5 — enough wrist travel to see
MIN_VISIBLE_MM = 3.0         # below this the motion is too small to judge by eye
MIN_VISIBLE_DEG = 3.0        # ...or, for a joint that only rotates the wrist
CLAW_LEN_MM = 70.06          # ClawTip offset from the wrist, from init_arm.m
CLEARANCE_MARGIN_MM = 5.0    # never plan to close more than this much of the gap


def describe_motion(d_mm: np.ndarray) -> str:
    """Render a base-frame displacement as directions a human can check."""
    axes = [
        (d_mm[0], "forward", "backward"),
        (d_mm[1], "left", "right"),
        (d_mm[2], "up", "down"),
    ]
    parts = [
        f"{abs(v):.1f} mm {pos if v > 0 else neg}"
        for v, pos, neg in axes
        if abs(v) >= 0.2
    ]
    return ", ".join(parts) if parts else "no appreciable movement"


def describe_rotation(R0: np.ndarray, R1: np.ndarray) -> tuple[str, float]:
    """Describe the wrist's rotation between two poses, in base-frame terms.

    Needed for a joint like J5 that spins the wrist about its own axis: the
    wrist ORIGIN does not move at all, so a position-only prediction reports
    nothing and the operator has nothing to compare against. The claw visibly
    turns, though, so describe that instead.

    Returns (description, angle_deg).
    """
    R_rel = R0.T @ R1
    angle = np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return "no rotation", 0.0
    axis_local = np.array([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1],
    ]) / (2.0 * np.sin(angle))
    axis = R0 @ axis_local  # into base frame

    i = int(np.argmax(np.abs(axis)))
    sense = axis[i] > 0
    # Right-hand rule, read from the operator's viewpoint given +X forward,
    # +Y left, +Z up (pinned empirically in Stage B).
    naming = {
        0: ("counterclockwise seen from the front", "clockwise seen from the front"),
        1: ("tilting back/up", "tilting forward/down"),
        2: ("counterclockwise seen from above", "clockwise seen from above"),
    }
    pos, neg = naming[i]
    return f"{np.rad2deg(angle):.1f} deg, {pos if sense else neg}", float(np.rad2deg(angle))


def read_angles(bus) -> dict:
    """Current tick and MATLAB angle for each IK joint."""
    state = {}
    for j in IK_JOINTS:
        tick = bus.read_position(j)
        state[j] = {"tick": tick, "rad": bus.ticks_to_rad(j, tick)}
    return state


def predict(client, state, joint, target_tick, bus):
    """FK before/after the proposed jog. Returns (T0, T1, p0_mm, p1_mm, delta_mm)."""
    angles_now = [state[j]["rad"] for j in IK_JOINTS]
    angles_after = list(angles_now)
    angles_after[joint - 1] = bus.ticks_to_rad(joint, target_tick)

    T0 = client.request_fk(angles_now)
    T1 = client.request_fk(angles_after)
    p0 = 1000 * T0[:3, 3]
    p1 = 1000 * T1[:3, 3]
    return T0, T1, p0, p1, p1 - p0


def apply_flip(joint: int) -> None:
    """Flip one joint's dir_sign in the calibration file, in place."""
    path = Path(config.SERVO_CALIBRATION_PATH)
    if not path.exists():
        print(f"  Cannot apply: {path} does not exist (running on config fallback).")
        print(f"  Edit config.SERVO_CALIBRATION_FALLBACK['{joint}']['dir_sign'] by hand.")
        return
    cal = json.loads(path.read_text())
    old = cal[str(joint)]["dir_sign"]
    cal[str(joint)]["dir_sign"] = -old
    path.write_text(json.dumps(cal, indent=2) + "\n")
    print(f"  Wrote {path}: J{joint} dir_sign {old:+d} -> {-old:+d}")
    print(f"  Re-run this jog to confirm the prediction now matches.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", type=int, required=True, choices=list(IK_JOINTS),
                    help="which joint to jog (1-5)")
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS,
                    help=f"tick delta, signed (default +{DEFAULT_TICKS})")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    ap.add_argument("--dry-run", action="store_true",
                    help="show the prediction and exit without moving")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt (still moves!)")
    ap.add_argument("--apply-flip", action="store_true",
                    help="if you report the motion was OPPOSITE, flip dir_sign in the file")
    ap.add_argument("--clearance-mm", type=float, default=10.0,
                    help="measured gap under the claw right now (default 10)")
    args = ap.parse_args()

    joint = args.joint

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

    with client, bus:
        state = read_angles(bus)
        start_tick = state[joint]["tick"]
        target_tick = start_tick + args.ticks

        if not (ServoBus.TICK_MIN <= target_tick <= ServoBus.TICK_MAX):
            print(f"Target {target_tick} is outside 0..4095. Pick a smaller delta.")
            sys.exit(1)

        cal = bus._cal(joint)
        T0, T1, p0, p1, d = predict(client, state, joint, target_tick, bus)
        travel_mm = float(np.linalg.norm(d))
        rot_desc, rot_deg = describe_rotation(T0[:3, :3], T1[:3, :3])

        print(f"\nJ{joint} jog:  {start_tick} -> {target_tick} ticks "
              f"({args.ticks:+d}, dir_sign {cal['dir_sign']:+d})")
        print(f"  wrist now:       ({p0[0]:+7.1f}, {p0[1]:+7.1f}, {p0[2]:+7.1f}) mm")
        print(f"  wrist predicted: ({p1[0]:+7.1f}, {p1[1]:+7.1f}, {p1[2]:+7.1f}) mm")

        # A joint that only spins the wrist (J5) moves the origin ~nowhere, so
        # fall back to describing the rotation the operator can actually see.
        watch_rotation = travel_mm < MIN_VISIBLE_MM
        if watch_rotation:
            prediction = f"the claw should ROTATE {rot_desc}"
        else:
            prediction = f"the wrist should move {describe_motion(d)}"
        print(f"\n  PREDICTION: {prediction}")
        print(f"              (wrist travel {travel_mm:.1f} mm, rotation {rot_deg:.1f} deg)")

        if travel_mm < MIN_VISIBLE_MM and rot_deg < MIN_VISIBLE_DEG:
            print(f"\n  Under {MIN_VISIBLE_MM} mm and {MIN_VISIBLE_DEG} deg — too small")
            print(f"  to judge by eye. Re-run with a larger --ticks "
                  f"(e.g. {2 * abs(args.ticks)}).")
            sys.exit(1)

        # --- table clearance guard --------------------------------------
        # The claw tip hangs ~CLAW_LEN below the wrist, so downward wrist
        # motion eats the gap under the claw. J3 folding down into the table
        # is the near-miss that already happened once during bring-up.
        descent_mm = -min(0.0, d[2])
        if descent_mm > 0:
            usable = args.clearance_mm - CLEARANCE_MARGIN_MM
            print(f"\n  Clearance: claw is {args.clearance_mm:.0f} mm above the table; "
                  f"this jog descends {descent_mm:.1f} mm.")
            if descent_mm > usable:
                print(f"\n  REFUSED: that would leave under {CLEARANCE_MARGIN_MM:.0f} mm "
                      f"of clearance.")
                print(f"  Jog this joint the OTHER way instead (lifts away from the table):")
                print(f"      python scripts/jog_joint.py --joint {joint} "
                      f"--ticks {-args.ticks}")
                print(f"  Or, if the arm really has more room than "
                      f"{args.clearance_mm:.0f} mm, re-run with --clearance-mm.")
                sys.exit(1)

        if args.dry_run:
            print("\n  --dry-run: nothing commanded.")
            return

        print(f"\n  This WILL move the arm. Claw clear of the table? Power within reach?")
        if not args.yes:
            if input("  Type 'go' to proceed: ").strip().lower() != "go":
                print("  Aborted. Nothing commanded.")
                return

        try:
            actual_tick = bus.move_and_verify(joint, target_tick)
        except ServoSafetyError as e:
            print(f"\n  REFUSED: {e}")
            sys.exit(1)

        moved = actual_tick - start_tick
        print(f"\n  commanded {target_tick}, servo settled at {actual_tick} "
              f"(moved {moved:+d} ticks of {args.ticks:+d} requested)")
        if abs(moved) < abs(args.ticks) * 0.5:
            print("  WARNING: the joint moved far less than requested — obstructed,")
            print("  at a travel limit, or underpowered. Resolve before reading anything")
            print("  into the direction below.")

        print(f"\n  Predicted: {prediction}")
        print(f"  What actually happened?")
        print(f"    [m] matched the prediction")
        print(f"    [o] moved the OPPOSITE way")
        print(f"    [?] unclear / too small to tell")
        answer = input("  > ").strip().lower()

        if answer.startswith("m"):
            print(f"\n  J{joint} dir_sign {cal['dir_sign']:+d} is CORRECT. No change needed.")
        elif answer.startswith("o"):
            print(f"\n  J{joint} dir_sign {cal['dir_sign']:+d} is INVERTED — should be "
                  f"{-cal['dir_sign']:+d}.")
            if args.apply_flip:
                apply_flip(joint)
            else:
                print(f"  Re-run with --apply-flip to write it, then jog again to confirm.")
        else:
            print(f"\n  Inconclusive. Re-run with a larger --ticks so the motion is")
            print(f"  unambiguous, or sight along a single axis.")

        print(f"\n  Joint left at {actual_tick} ticks (jogged, not returned to start).")
        print(f"  To undo:  python scripts/jog_joint.py --joint {joint} "
              f"--ticks {-args.ticks}")


if __name__ == "__main__":
    main()

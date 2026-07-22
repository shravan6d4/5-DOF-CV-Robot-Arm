"""
Stage-B bring-up health check — READ-ONLY. Commands no motion whatsoever.

Run this first every time the arm is powered up, and always before any script
that moves a joint. It answers the four questions that everything downstream
depends on:

  B1  Does every servo answer on the bus?
  B2  Are the position reads STABLE at rest? (the J1 encoder wrap-seam that
      caused the bring-up runaway shows up here as a reading that jumps
      between ~0 and ~4095 while the arm is physically still)
  B3  Is the arm actually at the home pose the calibration file describes?
  B4  Does MATLAB FK agree with where the arm physically is?

B4 is the real payoff: it is the first physical proof that the MATLAB-absolute
vs servo-relative zero reconciliation (config.MATLAB_HOME_DEG / home_angle_rad)
is correct. Before that fix, an arm sitting at home reported [0,0,0,0,0] to
MATLAB — a pose off by ~179 deg on J2 — and every camera pose derived from FK
was silently wrong.

Prereqs:
  * servo bus powered from its OWN supply (USB powers only the adapter's
    serial chip, not the servos) and wired to the adapter;
  * for the B4 check, ik_fk_server.m running in MATLAB (skipped if absent).

Usage (from the repo root):
    python scripts/check_servo_health.py
    python scripts/check_servo_health.py --port COM5
    python scripts/check_servo_health.py --samples 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus

JOINTS = range(1, 7)
IK_JOINTS = range(1, 6)  # J1..J5; J6 is the gripper and never goes through MATLAB

CLAW_LEN_MM = 70.06  # ClawTip offset from the wrist (Body08), from init_arm.m

# A reading that swings more than this at rest is not sensor noise.
STABLE_SPREAD_TICKS = 3
# How far from the calibrated home tick still counts as "at home".
HOME_AGREE_TICKS = 25


def check_bus(bus) -> list[int]:
    """B1: which of J1..J6 answer."""
    print("B1  bus enumeration")
    present = []
    for j in JOINTS:
        ok = bus.ping(j)
        print(f"      J{j}: {'answering' if ok else 'NO RESPONSE'}")
        if ok:
            present.append(j)
    missing = [j for j in JOINTS if j not in present]
    if missing:
        print(f"    FAIL: no response from {['J%d' % j for j in missing]}")
        print("      Check: bus power (separate supply, not USB), daisy-chain")
        print("      seating, and that IDs were assigned with set_servo_id.py.")
    else:
        print("    PASS: all six servos answering.")
    return present


def check_stability(bus, present, samples) -> dict:
    """B2: repeated reads at rest. Catches encoder wrap-seam flicker."""
    print(f"\nB2  read stability ({samples} samples at rest — do not touch the arm)")
    readings = {j: [] for j in present}
    for _ in range(samples):
        for j in present:
            try:
                readings[j].append(bus.read_position(j))
            except RuntimeError:
                readings[j].append(None)

    stable = True
    medians = {}
    for j in present:
        vals = [v for v in readings[j] if v is not None]
        if not vals:
            print(f"      J{j}: all reads failed")
            stable = False
            continue
        lo, hi = min(vals), max(vals)
        spread = hi - lo
        medians[j] = int(np.median(vals))
        flag = ""
        if spread > STABLE_SPREAD_TICKS:
            flag = "  <-- UNSTABLE"
            stable = False
            # A spread this large usually means the reading is straddling the
            # 0/4095 encoder seam rather than the joint actually vibrating.
            if lo < 100 and hi > 3995:
                flag = "  <-- WRAP-SEAM FLICKER (recentre this servo!)"
        print(f"      J{j}: median {medians[j]:4d}  range [{lo:4d},{hi:4d}]  spread {spread:3d}{flag}")

    if stable:
        print("    PASS: all readings steady.")
    else:
        print("    FAIL: unstable reads. A joint whose home sits near the 0/4095")
        print("      seam must be re-centred (Feetech one-key midpoint: write 128")
        print("      to Torque Enable, addr 40, with the joint at home) before use.")
    return medians


def check_home(bus, medians) -> bool:
    """B3: is the arm where the calibration file says home is?"""
    print("\nB3  home agreement (arm should be parked at its home pose)")
    at_home = True
    for j in sorted(medians):
        cal = bus._cal(j)
        delta = medians[j] - cal["home_tick"]
        ok = abs(delta) <= HOME_AGREE_TICKS
        at_home &= ok
        print(f"      J{j}: read {medians[j]:4d}  home {cal['home_tick']:4d}  "
              f"delta {delta:+5d}  {'ok' if ok else 'OFF'}")
    if at_home:
        print("    PASS: arm is at the calibrated home pose.")
    else:
        print("    NOTE: arm is not at the calibrated home. That is fine if you")
        print("      moved it deliberately — but B4 below then tests FK at the")
        print("      CURRENT pose, not home. Do not 'correct' this by commanding")
        print("      a move to home: verify by hand first (that is exactly the")
        print("      blind return-to-home that caused the J1 runaway).")
    return at_home


def check_fk(bus, medians) -> None:
    """B4: does MATLAB FK agree with physical reality?"""
    print("\nB4  MATLAB FK cross-check")
    missing = [j for j in IK_JOINTS if j not in medians]
    if missing:
        print(f"    SKIP: need J1..J5 readings, missing {missing}.")
        return

    angles_rad = [bus.ticks_to_rad(j, medians[j]) for j in IK_JOINTS]
    print("      joint angles handed to MATLAB (absolute, radians):")
    for j, a in zip(IK_JOINTS, angles_rad):
        print(f"        J{j}: {a:+.4f} rad  ({np.rad2deg(a):+8.2f} deg)")

    try:
        client = MatlabIKClient()
    except (ConnectionRefusedError, OSError):
        print("    SKIP: no MATLAB server on "
              f"{config.MATLAB_SERVER_HOST}:{config.MATLAB_SERVER_PORT}.")
        print("      Start it in MATLAB (matlab/ folder):  >> ik_fk_server")
        return

    with client:
        T = client.request_fk(angles_rad)

    p = T[:3, 3]
    print(f"      FK wrist position: "
          f"x={1000*p[0]:+7.1f}  y={1000*p[1]:+7.1f}  z={1000*p[2]:+7.1f}  mm")
    # The claw tip sits CLAW_LEN beyond the wrist along a fixed local direction.
    # At home the wrist frame is ~aligned with the base and the claw hangs down,
    # so this is a good estimate of the part you can actually see and measure.
    tip_z = 1000 * p[2] - CLAW_LEN_MM
    print(f"      claw tip (est, claw hanging down): "
          f"x={1000*p[0]:+7.1f}  y={1000*p[1]:+7.1f}  z={tip_z:+7.1f}  mm")
    print()
    print("    ACTION: compare against the physical arm. The WRIST is Body08 —")
    print("      the joint J5 turns; the claw tip is ~70 mm beyond it.")
    print("      Agreement within ~10 mm means the MATLAB<->servo angle mapping")
    print("      is correct and Stage C (dir_sign jogs) can proceed.")
    print("      If the wrist lands far from where the arm visibly is — e.g.")
    print("      claiming it hangs below its own mount — then home_angle_rad in")
    print("      data/servo_calibration.json does not describe the pose the arm")
    print("      is in. Fix that before commanding any motion.")
    print()
    print(f"      Tabletop estimate: if the claw tip is H mm above the table,")
    print(f"      then TABLE_Z_IN_BASE ~ {tip_z:+.1f} - H  mm.")
    print(f"      (config currently says {1000*config.TABLE_Z_IN_BASE:+.1f} mm)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=config.SERVO_PORT,
                    help=f"serial port (default {config.SERVO_PORT})")
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    ap.add_argument("--samples", type=int, default=20,
                    help="stability samples per joint (default 20)")
    args = ap.parse_args()

    print(f"Servo health check — READ ONLY, no motion commanded.")
    print(f"Port {args.port} @ {args.baud} baud\n")

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as e:
        print(f"Could not open the servo bus on {args.port}: {e}")
        print("Check the port name (Device Manager on Windows) and that the")
        print("adapter is plugged in. Pass a different one with --port COMx.")
        sys.exit(1)

    with bus:
        present = check_bus(bus)
        if not present:
            sys.exit(1)
        medians = check_stability(bus, present, args.samples)
        if not medians:
            sys.exit(1)
        check_home(bus, medians)
        check_fk(bus, medians)

    print("\nRead-only checks complete. Nothing was commanded to move.")


if __name__ == "__main__":
    main()

"""Ask IK what it would do for small Cartesian nudges — READ-ONLY, no motion.

    Commands nothing. Reads joint positions, asks MATLAB to solve, prints.

WHY. The visual servo drives two image axes with two Cartesian directions:
TANGENTIAL (sideways, expected to be base yaw) and RADIAL (in/out, expected to
be the shoulder/elbow chain). Those are supposed to be independent — that
independence is what lets the loop correct one axis at a time and treat the
image Jacobian as diagonal.

On 2026-08-05 they were not independent. Every RADIAL nudge came back solved as
a large J1 swing:

    y via radial nudge +2.523 mm  ->  J1+44, J5+5     x error 2px -> 60px
    y via radial nudge +1.541 mm  ->  J1+23, J5+3     x error 12px -> 38px

J1 is base yaw. Yawing the base is a TANGENTIAL motion — it should not appear in
a radial solve at all, and 44 ticks at this reach is ~6 mm of sideways travel
for a 2.5 mm in/out request. The loop then spent every other iteration undoing
the disturbance its last correction created, and eventually lost the brick.

WHAT THIS SEPARATES. Two very different causes produce that symptom:

  1. IK POSTURE DRIFT. The solver is position-only on a 5-DOF arm, so many
     joint configurations satisfy the same tip position and nothing in the
     request forbids yawing the base. The solve is then "correct" but useless
     to a loop that assumed the axes were separable.

  2. A WRONG MODEL. The claw tip's position depends on wrist orientation, which
     depends on J4/J5 angles, which depend on their dir_sign. J5's is still an
     UNCONFIRMED HYPOTHESIS. If the model holds the claw at the wrong
     orientation, its idea of how to move the tip is wrong, and the solutions
     are strange for a reason no amount of solver tuning will fix.

The discriminator is BEARING. A pure radial move keeps atan2(y, x) constant by
construction, so any correct solution must leave J1 essentially untouched. If IK
returns a large J1 change for a target whose bearing did not change, the model
and the solver disagree about where the tip is — which points at cause 2.

Usage (from the repo root, MATLAB's ik_fk_server running):
    python scripts/diagnose_cartesian_axes.py
    python scripts/diagnose_cartesian_axes.py --nudge 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.planning.visual_servo import radial_tangential, reach_from_axis
from vision_pipeline.robot_interface.matlab_client import IKUnreachableError, MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus

IK_JOINTS = (1, 2, 3, 4, 5)
# A J1 change beyond this for a bearing-preserving target is not rounding.
J1_SUSPICIOUS_TICKS = 8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nudge", type=float, default=5.0,
                    help="test displacement in mm (default 5)")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    try:
        ik = MatlabIKClient(config.MATLAB_SERVER_HOST, config.MATLAB_SERVER_PORT)
    except (ConnectionRefusedError, OSError):
        print(f"No MATLAB server on {config.MATLAB_SERVER_HOST}:"
              f"{config.MATLAB_SERVER_PORT}. Start it:  >> ik_fk_server")
        sys.exit(1)

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as e:
        print(f"Could not open the servo bus on {args.port}: {e}")
        sys.exit(1)

    print("=" * 72)
    print("CARTESIAN AXIS DIAGNOSIS — read-only, nothing will move.")
    print("=" * 72)

    with bus:
        ticks = {j: bus.read_position_retrying(j) for j in IK_JOINTS}
        angles = [bus.ticks_to_rad(j, ticks[j]) for j in IK_JOINTS]
        T_wrist, T_tip = ik.request_fk_tip(angles)
        tip, wrist = T_tip[:3, 3], T_wrist[:3, 3]

        print("\ncurrent pose")
        print("  ticks      " + "  ".join(f"J{j}={ticks[j]}" for j in IK_JOINTS))
        print(f"  wrist      ({wrist[0]*1000:+7.1f}, {wrist[1]*1000:+7.1f}, "
              f"{wrist[2]*1000:+7.1f}) mm")
        print(f"  claw tip   ({tip[0]*1000:+7.1f}, {tip[1]*1000:+7.1f}, "
              f"{tip[2]*1000:+7.1f}) mm")

        # The frame-conversion invariant: in the physical frame the tip hangs
        # BELOW the wrist in any sane pose. If it does not, the model is not
        # describing this arm and nothing below is meaningful.
        if tip[2] > wrist[2]:
            print("\n  *** The claw tip reads ABOVE the wrist. That is physically")
            print("      impossible for this arm — a frame conversion is broken.")
            print("      Fix that before reading anything into the solves below.")

        axis_xy = ik.base_yaw_axis_xy()
        radius = reach_from_axis((tip[0], tip[1]), axis_xy)
        bearing = float(np.degrees(np.arctan2(tip[1], tip[0])))
        print(f"  radius {radius*1000:.1f} mm from the base axis, "
              f"bearing {bearing:+.2f} deg")

        # TWO CANDIDATE REFERENCES, and the difference is the whole question.
        #
        # Tip-referenced radial keeps the TIP's bearing constant. That sounds
        # right and is not, because the claw hangs off to one side: the tip sits
        # well off the arm's own plane, so holding its bearing while extending
        # forces the base to yaw. Base yaw pans the CAMERA, which is mounted on
        # the wrist -- so a few millimetres of tip motion swings the whole image.
        #
        # Wrist-referenced radial extends the arm along its OWN plane instead.
        # The tip goes wherever the fixed claw offset puts it, which is not a
        # clean radial line, but J1 has no reason to move -- and J1 is what the
        # camera actually cares about.
        tip_radial, tip_tangential = radial_tangential((tip[0], tip[1]), axis_xy)
        wr_radial, wr_tangential = radial_tangential((wrist[0], wrist[1]), axis_xy)
        wrist_bearing = float(np.degrees(np.arctan2(wrist[1], wrist[0])))

        print(f"\n  tip bearing   {bearing:+.2f} deg   "
              f"radial ({tip_radial[0]:+.3f}, {tip_radial[1]:+.3f})")
        print(f"  wrist bearing {wrist_bearing:+.2f} deg   "
              f"radial ({wr_radial[0]:+.3f}, {wr_radial[1]:+.3f})")
        print(f"  the claw hangs {abs(bearing - wrist_bearing):.1f} deg off the "
              f"arm's plane — that offset is why the two differ")

        for name, vec in (("RADIAL (tip-referenced)", tip_radial),
                          ("RADIAL (wrist-referenced)", wr_radial),
                          ("TANGENTIAL", tip_tangential)):
            for sign in (+1, -1):
                d = sign * args.nudge / 1000.0
                target = tip + np.array([vec[0], vec[1], 0.0]) * d
                tgt_bearing = float(np.degrees(np.arctan2(target[1], target[0])))
                d_bearing = tgt_bearing - bearing

                print(f"\n--- {name} {sign * args.nudge:+.1f} mm ---")
                print(f"    target bearing {tgt_bearing:+.2f} deg "
                      f"({d_bearing:+.2f} deg from now)")
                try:
                    solution, err_mm = ik.request_ik(*target, seed_rad=angles)
                except IKUnreachableError as e:
                    print(f"    UNREACHABLE: {e}")
                    continue

                deltas = {j: bus.rad_to_ticks(j, a) - ticks[j]
                          for j, a in zip(IK_JOINTS, solution)}
                print(f"    IK residual {err_mm:.2f} mm")
                print(f"    joints     " +
                      "  ".join(f"J{j}{deltas[j]:+d}" for j in IK_JOINTS))

                # Where the solution ACTUALLY puts the tip, and in which
                # direction -- the request and the achieved motion agreeing is
                # not something to assume when the model is under suspicion.
                _, T_after = ik.request_fk_tip(list(solution))
                moved = (T_after[:3, 3] - tip) * 1000
                along = float(np.dot(moved[:2], vec))
                across = float(np.dot(moved[:2], [-vec[1], vec[0]]))
                print(f"    tip moves  {np.linalg.norm(moved):.2f} mm total: "
                      f"{along:+.2f} along, {across:+.2f} across, "
                      f"{moved[2]:+.2f} vertical")

                # THE NUMBER THAT MATTERS TO THE LOOP. The camera rides on the
                # wrist, so J1 pans it directly: every tick of base yaw sweeps
                # the whole image sideways regardless of how far the tip went.
                # A correction is only usable if this stays small.
                pan_deg = abs(deltas[1]) / 651.89 * 180 / np.pi
                verdict = "clean" if pan_deg < 0.5 else (
                    "TOLERABLE" if pan_deg < 1.5 else "SWAMPS THE IMAGE")
                print(f"    camera pan {pan_deg:5.2f} deg from J1{deltas[1]:+d}"
                      f"   <-- {verdict}")

    ik.close()
    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("  'along/across' says whether IK solved the request. Near-perfect")
    print("  numbers there mean the model is self-consistent and the solver is")
    print("  fine — they say NOTHING about whether the move is useful.")
    print()
    print("  'camera pan' is what the servo loop actually lives or dies by.")
    print("  The camera is on the wrist, so J1 sweeps the entire image; a few")
    print("  millimetres of tip motion bought with 6 deg of yaw looks like a")
    print("  huge sideways jump to a loop that measures pixels.")
    print()
    print("  Compare the two RADIAL blocks. Whichever keeps the pan small is")
    print("  the reference the vertical image axis should use — that is the")
    print("  choice this script exists to make.")
    print("=" * 72)


if __name__ == "__main__":
    main()

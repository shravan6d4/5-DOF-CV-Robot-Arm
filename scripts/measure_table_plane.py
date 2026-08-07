"""Find the table by touching it, from several poses. Read-only: MOVES NOTHING.

    python scripts/measure_table_plane.py

Hand-position the claw so its tip rests on the tabletop, press Enter, repeat from
a DIFFERENT arm posture. Each sample is one tick reading run through FK.

WHY SEVERAL POSES AND NOT ONE. A single touch gives one number and no way to
know whether it is right. Every sample is the same physical plane, so FK must
return the same z for all of them -- that is a property of the KINEMATICS, not
of the table, and it holds no matter what the table height turns out to be. So
the spread across poses measures the calibration, and the mean measures the
table. One reading conflates the two and can only ever confirm what you already
believed.

This exists because on 2026-08-07 a single touch reading disagreed with the
stored table height by 111 mm, while FK checked out at two other poses. One
equation against ten calibration unknowns (five home_tick, five ticks_per_rad)
cannot say which is wrong. Four or five touches can.

WHAT THE OUTPUT MEANS.
  * spread within a few mm  -> FK is consistent; the mean IS the table height
                               and TABLE_Z_IN_BASE should be set to it.
  * spread of tens of mm    -> FK does not describe this arm. This script CANNOT
                               tell you which parameter is wrong -- see below --
                               only that one is, and by how much.

ONE PLANE DETECTS, IT DOES NOT IDENTIFY, and the distinction is a published
result rather than caution on this repo's part. Zhuang, Motaghedi & Roth, "Robot
Calibration with Planar Constraints" (ICRA 1999): a single-plane constraint is
NOT sufficient to calibrate a robot; a minimum of THREE planar constraints is
needed, and only if (a) the three planes are mutually non-parallel, (b) the
identification Jacobian of the unconstrained system is nonsingular, and (c) the
points measured on each plane are not collinear. With one plane the
identification matrix is rank-deficient: whole families of wrong calibrations
put every touch on the same plane, so an optimiser will happily fit one.

That is why the correlations below are labelled a HINT and not a diagnosis. A
joint whose angle tracks the residual is where to look first; it is not proof,
because with one plane the parameters are not separable even in principle. To
actually close it, repeat this against two more non-parallel surfaces -- a book
stood on edge, the side of a box -- and fit the kinematics to all three.

THE CAMERA ROUTE, AND WHY IT IS SHUT HERE. With a working hand-eye transform a
board lying flat on the table gives this for free: solvePnP puts the board plane
in camera coordinates, FK @ hand-eye puts the camera in the base frame, and the
composition is the table plane -- no touching. Better still, robot-world
hand-eye (`AX = ZB`, Zhuang/Roth/Sudhakar 1994; cv2.calibrateRobotWorldHandEye)
solves for Z = base -> world DIRECTLY, and if the board is flat on the table then
Z's translation IS the table height. Both are blocked while data/hand_eye.json is
known wrong (CLAUDE.md), and AX=ZB on our current samples returns 257 mm for a
24 mm ruler measurement. Touch needs no camera and no calibration but its own,
which is exactly why it is the fallback.

VARY THE POSTURE, NOT JUST THE SPOT. Touching five points along one arc with the
elbow at the same angle tells you almost nothing: the joints barely move between
them, so a calibration error stays constant and hides inside the mean. Fold the
elbow differently each time -- reach in close, stretch out, come at it high and
low. The report warns when the postures are too alike to be informative.

TORQUE. The arm must be limp enough to position by hand. Kill torque with
`python scripts/servo_torque.py --disable` (SUPPORT THE ARM FIRST -- it falls),
or move it against the servos if they are soft enough. Re-enable where it stands
with `python scripts/servo_torque.py --enable N` when finished.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus

IK_JOINTS = (1, 2, 3, 4, 5)

# Below this, the postures are too alike for the samples to be independent: a
# calibration error common to all of them cancels out of the spread and the
# result looks clean while being wrong in exactly the way this script exists to
# catch. Degrees of total joint travel between the two most similar poses.
MIN_POSTURE_SPREAD_DEG = 20.0


def read_pose(bus, ik):
    """One tick reading and the claw tip it implies. Commands nothing."""
    ticks = {j: bus.read_position_retrying(j) for j in IK_JOINTS}
    angles = [bus.ticks_to_rad(j, ticks[j]) for j in IK_JOINTS]
    _wrist, tip = ik.request_fk_tip(angles)
    return ticks, np.array(angles), tip[:3, 3] * 1000.0


def report(samples, cal_home):
    """Say whether these touches describe one plane, and if not, blame a joint."""
    zs = np.array([s["tip"][2] for s in samples])
    spread = float(zs.max() - zs.min())

    print("\n" + "=" * 68)
    print(f"{len(samples)} touches of the same physical plane\n")
    for i, s in enumerate(samples, 1):
        t = s["tip"]
        print(f"  {i}: tip ({t[0]:+7.1f}, {t[1]:+7.1f}, {t[2]:+7.1f}) mm    "
              + " ".join(f"J{j}{np.degrees(a):+6.0f}"
                         for j, a in zip(IK_JOINTS, s["angles"])))

    print(f"\n  mean z   {zs.mean():+7.1f} mm")
    print(f"  spread   {spread:7.1f} mm  (max - min)")
    print(f"  stored   {config.TABLE_Z_IN_BASE * 1000:+7.1f} mm  "
          f"(config.TABLE_Z_IN_BASE)")

    # Postures too alike? Then the spread proves nothing either way.
    travel = []
    for i in range(len(samples)):
        for k in range(i + 1, len(samples)):
            travel.append(np.degrees(np.abs(
                samples[i]["angles"] - samples[k]["angles"]).sum()))
    if travel and min(travel) < MIN_POSTURE_SPREAD_DEG:
        print(f"\n  *** TWO POSTURES DIFFER BY ONLY {min(travel):.0f} deg of total "
              f"joint travel.")
        print(f"      Samples that alike share whatever error they carry, so it")
        print(f"      cancels out of the spread above and this report will call a")
        print(f"      broken calibration clean. Re-take them with the elbow folded")
        print(f"      differently -- in close, stretched out, high, low.")

    if spread <= 5.0:
        print(f"\n  FK IS CONSISTENT. Every posture agrees on the same plane to "
              f"{spread:.1f} mm,")
        print(f"  which no calibration error could survive. The table is at "
              f"{zs.mean():+.1f} mm.")
        print(f"\n  Set it:  config.TABLE_Z_IN_BASE = {zs.mean() / 1000:.4f}")
        if abs(zs.mean() - config.TABLE_Z_IN_BASE * 1000) > 5:
            print(f"  That moves it by "
                  f"{zs.mean() - config.TABLE_Z_IN_BASE * 1000:+.1f} mm. Note which "
                  f"way: a LOWER table")
            print(f"  means every descent goes further before stopping.")
        return

    print(f"\n  *** FK DOES NOT DESCRIBE THIS ARM. These are all one flat plane, so")
    print(f"      a {spread:.0f} mm spread is the kinematics disagreeing with itself,")
    print(f"      not the table being uneven. Do NOT set TABLE_Z_IN_BASE from this.")

    # A HINT, NOT A DIAGNOSIS. One plane leaves the identification matrix
    # rank-deficient (Zhuang/Motaghedi/Roth, ICRA 1999), so entire families of
    # wrong calibrations put every touch on this same plane. Correlation says
    # where to look; it cannot say what to change.
    resid = zs - zs.mean()
    print(f"\n      Correlation of the z error with each joint's angle "
          f"({len(samples)} samples):")
    for k, j in enumerate(IK_JOINTS):
        a = np.array([np.degrees(s["angles"][k]) for s in samples])
        if a.std() < 1.0:
            print(f"        J{j}: barely moved ({a.std():.1f} deg) -- says nothing")
            continue
        r = float(np.corrcoef(a, resid)[0, 1])
        flag = "   <== look here first" if abs(r) > 0.8 else ""
        print(f"        J{j}: r = {r:+.2f}   over {a.min():+.0f}..{a.max():+.0f} "
              f"deg{flag}")
    print(f"\n      A joint near +/-1.00 is where to look, NOT a verdict. With a")
    print(f"      single plane the parameters are not separable even in principle:")
    print(f"      Zhuang/Motaghedi/Roth (ICRA 1999) show one plane is rank")
    print(f"      deficient and that THREE mutually non-parallel planes are the")
    print(f"      minimum for identification. Repeat this against a book on edge")
    print(f"      and a box side, then fit all three together.")
    print(f"\n      home_tick for reference: "
          + "  ".join(f"J{j}={cal_home[j]}" for j in IK_JOINTS))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as exc:                                      # noqa: BLE001
        print(f"Could not open the servo bus on {args.port}: {exc}")
        return 1
    try:
        ik = MatlabIKClient(config.MATLAB_SERVER_HOST, config.MATLAB_SERVER_PORT)
    except (ConnectionRefusedError, OSError):
        print(f"No MATLAB server on {config.MATLAB_SERVER_HOST}:"
              f"{config.MATLAB_SERVER_PORT}. Start it:  >> ik_fk_server")
        bus.close()
        return 1

    cal_home = {j: bus._cal(j)["home_tick"] for j in IK_JOINTS}
    samples = []

    print(__doc__)
    print("=" * 68)
    print("This script COMMANDS NOTHING. It only reads where you put the arm.")
    print("Enter to record a touch, 'done' to finish, Ctrl-C to abandon.\n")

    with bus, ik:
        while True:
            try:
                answer = input(f"  touch {len(samples) + 1}: claw tip ON the table, "
                               f"then Enter (or 'done'): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAbandoned; nothing was written.")
                return 1
            if answer in ("done", "d", "q"):
                break

            try:
                ticks, angles, tip = read_pose(bus, ik)
            except Exception as exc:                              # noqa: BLE001
                print(f"     could not read: {exc}")
                continue

            samples.append({"ticks": ticks, "angles": angles, "tip": tip})
            print(f"     tip ({tip[0]:+7.1f}, {tip[1]:+7.1f}, {tip[2]:+7.1f}) mm   "
                  + " ".join(f"J{j}={ticks[j]}" for j in IK_JOINTS))
            if len(samples) >= 2:
                zs = [s["tip"][2] for s in samples]
                print(f"     spread so far: {max(zs) - min(zs):.1f} mm")

    if len(samples) < 2:
        print("\nNeed at least two postures: one touch cannot tell a table height")
        print("from a calibration error, which is the whole point of this script.")
        return 1

    report(samples, cal_home)
    return 0


if __name__ == "__main__":
    sys.exit(main())

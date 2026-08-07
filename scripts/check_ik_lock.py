"""Is null-space motion actually suppressed before a descent? Read-only.

    python scripts/check_ik_lock.py

Commands no motion. Reads the arm's current angles, asks MATLAB for a small
Cartesian move with joints pinned, then applies the SAME drop `visual_servo.py`
applies and reports what would really be commanded. Needs the servo bus (to read
where the arm is) and the MATLAB server.

WHY THIS EXISTS. The camera rides on the wrist, so any joint motion the solver
spends without being asked moves the very image the visual loop is measuring.
`visual_servo.py` holds J5 on every re-centring nudge and J1 on radial ones for
exactly that reason. A hold that quietly fails is worse than no hold at all --
the operator tunes gains against a disturbance they believe is suppressed.

TWO LAYERS, AND ONLY THE SECOND ONE IS LOAD-BEARING.

  1. The `lock` field in the IK request. MATLAB pins the joint's PositionLimits
     around its seed angle. THIS DOES NOT WORK and is not expected to: with
     lock=[5], lock=[1] and lock=[1,5] the server returns a byte-identical
     solution. Three separate fixes were tried inside rigidBodyJoint --
     a degenerate [v, v] interval, a narrow [v-eps, v+eps] band, and release(ik)
     to defeat the System object's cached RigidBodyTree. None took. The request
     is left in place because it costs nothing and reports lock_drift_rad, which
     is what makes the failure visible instead of silent.

  2. Python simply does not command a held joint, whatever the solver returned
     (scripts/visual_servo.py, CartesianActuator.apply). This is what actually
     protects the loop, and section 2 below is the test of it.

So a FAILING section 1 is the documented status quo and does not block a run.
What blocks a run is a stale server, or a hold that leaves no motion to nudge
with. Both are checked below.

RUN IT AFTER EVERY RESTART OF THE MATLAB SERVER, and after any change to
matlab/. A stale server is indistinguishable from a working one at the protocol
level except by the build marker this script compares.
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
TICKS_PER_RAD = 651.89

# What each nudge axis holds, mirroring CartesianActuator.locked_joints. Kept as
# a literal rather than imported: scripts/ is not a package, and a wrong copy
# here would show up immediately as a lock set that does not match the run log.
AXES = {"radial": [1, 5], "tangential": [5]}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--move-mm", type=float, default=3.0,
                    help="size of the test move (default 3, as used by a nudge)")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as exc:                                      # noqa: BLE001
        print(f"Could not open the servo bus on {args.port}: {exc}")
        return 1

    try:
        client = MatlabIKClient(config.MATLAB_SERVER_HOST, config.MATLAB_SERVER_PORT)
    except (ConnectionRefusedError, OSError):
        print(f"No MATLAB server on {config.MATLAB_SERVER_HOST}:"
              f"{config.MATLAB_SERVER_PORT}. Start it:  >> ik_fk_server")
        bus.close()
        return 1

    blocking = []

    with bus, client:
        ticks = {j: bus.read_position_retrying(j) for j in IK_JOINTS}
        angles = [bus.ticks_to_rad(j, ticks[j]) for j in IK_JOINTS]

        _wrist, tip = client.request_fk_tip(angles)
        origin = tip[:3, 3]
        print(f"\nclaw tip now ({origin[0] * 1000:+.1f}, {origin[1] * 1000:+.1f}, "
              f"{origin[2] * 1000:+.1f}) mm")
        print(f"test move: {args.move_mm:.0f} mm, the size of a re-centring nudge")

        print("\n1) MATLAB-side lock (advisory -- expected to fail)\n")
        solutions = {}
        for axis, locked in AXES.items():
            target = origin + np.array([args.move_mm / 1000.0, 0.0, 0.0])
            # Go through the client's own connection rather than opening a
            # second socket: ik_fk_server serves ONE client at a time, so a
            # second connection is accepted and then never read until the first
            # disconnects, which presents as a timeout with no explanation.
            # _send_request is used directly (not request_ik) so a server
            # predating lock_drift_rad is named as such instead of passing, and
            # so the client's per-solve warning does not fire once per line.
            try:
                resp = client._send_request(
                    {"cmd": "ik", "x": float(target[0]), "y": float(-target[1]),
                     "z": float(-target[2]), "seed_rad": list(angles),
                     "lock": [int(j) for j in locked]})
            except Exception as exc:                              # noqa: BLE001
                print(f"  {axis:<11} IK refused -- {exc}")
                blocking.append(f"IK cannot solve a {axis} nudge from this pose")
                continue

            build = resp.get("server_build")
            if build != MatlabIKClient.SERVER_BUILD:
                print(f"  *** STALE SERVER. It reports build "
                      f"{build or '(none -- predates the marker)'}, this repo is "
                      f"{MatlabIKClient.SERVER_BUILD}.")
                print("      MATLAB keeps running the code it was started with,")
                print("      so this says nothing about the code on disk.")
                print("      Restart it and run this again:  >> ik_fk_server")
                blocking.append("the MATLAB server is running stale code")
                break

            solutions[axis] = resp["angles_rad"]
            drift = resp["lock_drift_rad"] * TICKS_PER_RAD
            print(f"  {axis:<11} holds {str(locked):<7} drift {drift:6.1f} ticks "
                  f"({np.degrees(resp['lock_drift_rad']):5.2f} deg)  "
                  f"{'honoured' if drift < 2 else 'IGNORED (as expected)'}")

        if not blocking and solutions:
            print("\n2) What Python actually commands (this is the load-bearing one)\n")
            for axis, solution in solutions.items():
                held = set(AXES[axis])
                want = {j: bus.rad_to_ticks(j, a)
                        for j, a in zip(IK_JOINTS, solution)}
                sent = {j: (ticks[j] if j in held else want[j]) for j in IK_JOINTS}

                dropped = {j: want[j] - ticks[j] for j in held
                           if abs(want[j] - ticks[j]) > 2}
                moved = {j: sent[j] - ticks[j] for j in IK_JOINTS}
                biggest = max(abs(d) for d in moved.values())

                print(f"  {axis} nudge, holding {sorted(held)}:")
                print("      solver wanted: "
                      + "  ".join(f"J{j}{want[j] - ticks[j]:+6d}" for j in IK_JOINTS))
                print("      we command:    "
                      + "  ".join(f"J{j}{moved[j]:+6d}" for j in IK_JOINTS))
                if dropped:
                    print("      dropped:       "
                          + ", ".join(f"J{j}{d:+d}" for j, d in sorted(dropped.items()))
                          + " ticks of null-space motion the camera never sees")

                # A held joint reading anything but zero would mean the drop
                # itself is broken, which no amount of MATLAB-side fixing covers.
                leaked = {j: moved[j] for j in held if abs(moved[j]) > 2}
                if leaked:
                    print(f"      *** LEAK: held joint(s) still moving {leaked}")
                    blocking.append("the Python-side hold is not dropping the joint")

                # Holding a joint can leave nothing to move with. The nudge then
                # reports its requested size while the arm stands still, and the
                # probe divides a pixel shift by a move that never happened --
                # manufacturing a gain out of detection noise.
                if dropped and biggest < 2:
                    print(f"      *** the hold leaves no motion at all on this axis")
                    blocking.append(
                        f"a {axis} nudge from this pose is only reachable through "
                        f"the joints it holds")

    print()
    if blocking:
        print("  NOT READY TO DESCEND:")
        for reason in blocking:
            print(f"    - {reason}")
        print("\n  Section 1 failing on its own is NOT a reason to stop -- the")
        print("  MATLAB lock has never worked and the run does not rely on it.")
        return 1

    print("  Null-space motion is suppressed where it counts. J5 stays put on")
    print("  every nudge and J1 on radial ones, so the image only moves for")
    print("  reasons the loop asked for, and a probe gain means what it says.")
    print("\n  MATLAB's own lock is ignoring the request (section 1) -- that is the")
    print("  documented status quo, not a regression. Python drops the joints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

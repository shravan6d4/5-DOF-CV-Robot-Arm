"""Does the IK solver actually HOLD a joint when told to? Read-only.

    python scripts/check_ik_lock.py

Commands no motion. Reads the arm's current angles, asks MATLAB for a small
Cartesian move with joints pinned, and checks whether the pinned joints stayed
put. Needs the servo bus (to read where the arm is) and the MATLAB server.

WHY THIS EXISTS. `visual_servo.py` locks J5 on every re-centring nudge and J1
on every descent, and the whole point is that the camera rides on the wrist:
null-space motion moves the image the loop is measuring, so a lock that quietly
fails is worse than no lock at all -- the operator tunes gains against a
disturbance they believe is suppressed.

It failed quietly for two rounds of fixes. First the pin was a degenerate
[v, v] interval, which looked like the obvious culprit and was not. Then a
narrow [v-eps, v+eps] band, which behaved identically. The actual cause:
inverseKinematics is a MATLAB System object, so it LOCKS on its first call and
caches its RigidBodyTree -- every PositionLimits change made afterwards is
never consulted. That is also why the limits from data/joint_limits_rad.json
DO work: init_arm.m applies them before the solver is built. The fix is
release(ik) around the pin, and this script is how you know it took.

RUN IT AFTER EVERY RESTART OF THE MATLAB SERVER, and after any change to
matlab/. A stale server is indistinguishable from a working one at the protocol
level except by this test.
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--move-mm", type=float, default=3.0,
                    help="size of the test move (default 3, as used by a nudge)")
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    try:
        with ServoBus(args.port, args.baud) as bus:
            angles = [bus.ticks_to_rad(j, bus.read_position_retrying(j))
                      for j in IK_JOINTS]
    except Exception as exc:                                  # noqa: BLE001
        print(f"Could not read the servo bus on {args.port}: {exc}")
        sys.exit(1)

    try:
        client = MatlabIKClient(config.MATLAB_SERVER_HOST, config.MATLAB_SERVER_PORT)
    except (ConnectionRefusedError, OSError):
        print(f"No MATLAB server on {config.MATLAB_SERVER_HOST}:"
              f"{config.MATLAB_SERVER_PORT}. Start it:  >> ik_fk_server")
        sys.exit(1)

    ok = True
    with client:
        _wrist, tip = client.request_fk_tip(angles)
        origin = tip[:3, 3]
        print(f"\nclaw tip now ({origin[0] * 1000:+.1f}, {origin[1] * 1000:+.1f}, "
              f"{origin[2] * 1000:+.1f}) mm")
        print(f"asking for a {args.move_mm:.0f} mm move with joints pinned\n")

        # Go through the client's own connection rather than opening a second
        # socket: ik_fk_server serves ONE client at a time, so a second
        # connection is accepted and then never read until the first
        # disconnects, which presents as a timeout with no explanation.
        # _send_request is used directly (not request_ik) so a server predating
        # lock_drift_rad is named as such instead of silently passing.
        for locked in ([5], [1], [1, 5]):
            target = origin + np.array([args.move_mm / 1000.0, 0.0, 0.0])
            try:
                resp = client._send_request(
                    {"cmd": "ik", "x": float(target[0]), "y": float(-target[1]),
                     "z": float(-target[2]), "seed_rad": list(angles),
                     "lock": [int(j) for j in locked]})
            except Exception as exc:                          # noqa: BLE001
                print(f"  lock {locked}: IK refused — {exc}")
                continue

            build = resp.get("server_build")
            if build != MatlabIKClient.SERVER_BUILD:
                print(f"  *** STALE SERVER. It reports build "
                      f"{build or '(none — predates the marker)'}, this repo is "
                      f"{MatlabIKClient.SERVER_BUILD}.")
                print("      MATLAB keeps running the code it was started with.")
                print("      Restart it and run this again:  >> ik_fk_server")
                ok = False
                break

            moved = [(np.array(resp["angles_rad"])[k] - angles[k]) * TICKS_PER_RAD
                     for k in range(5)]
            drift_ticks = resp["lock_drift_rad"] * TICKS_PER_RAD
            held = resp["lock_drift_rad"] <= MatlabIKClient.LOCK_DRIFT_WARN_RAD
            ok = ok and held
            print(f"  lock {str(locked):<7} drift {drift_ticks:6.1f} ticks "
                  f"({np.degrees(resp['lock_drift_rad']):5.2f} deg)   "
                  f"{'HOLDS' if held else '*** NOT HOLDING'}")
            print(f"           per joint: "
                  + "  ".join(f"J{k + 1}{moved[k]:+6.1f}" for k in range(5)))

    print()
    if ok:
        print("  The lock holds. Null-space motion is suppressed, so a probe gain")
        print("  measured through these solves means what it says.")
    else:
        print("  THE LOCK IS NOT HOLDING. Every gain the visual loop measures")
        print("  through a locked solve is contaminated by joint motion nobody")
        print("  asked for — and because the camera is on the wrist, that motion")
        print("  moves the very image the loop reads. Do not run a descent until")
        print("  this passes: restart the MATLAB server first (>> ik_fk_server),")
        print("  and if it still fails the release(ik) in handle_ik_request is")
        print("  not doing its job.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

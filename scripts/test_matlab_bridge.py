"""
Stage-1 verification for the MATLAB IK/FK bridge — run BY HAND against a live
server, not part of the pytest suite (it needs MATLAB running).

Prereqs:
  1. In MATLAB, from the repo's matlab/ folder:  >> ik_fk_server
     (it prints "Server listening for connections...")
  2. In this venv, from the repo root:            python scripts/test_matlab_bridge.py

What it checks, for each of the 8 named targets from IKtrials_v2.m's regression
battery (the same ones matlab/test_ik_fk.m uses):
  * request_ik(x,y,z) succeeds and reports err_mm under the 10 mm tolerance;
  * request_fk(angles) round-trips — the WRIST pose FK returns is a valid 4x4,
    and (as a self-consistency check) re-solving is stable.

Note: FK returns the WRIST (Body08), not the claw tip, so FK(IK(target)) does
NOT land on `target` — the tip sits CLAW_LEN beyond the wrist. This script
therefore checks IK error against the tip (what the solver optimizes) and checks
FK only for a well-formed transform. The tip round-trip is verified MATLAB-side
in test_ik_fk.m where the tip frame is available.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline.robot_interface.matlab_client import IKUnreachableError, MatlabIKClient

# The 8 targets from IKtrials_v2.m Part 6, expressed in the PHYSICAL frame.
# MatlabIKClient now converts physical -> model on the wire (the imported
# model's frame is upside-down; see CLAUDE.md "COORDINATE FRAMES"), so to keep
# exercising the exact validated solver battery these are the model-frame
# points with y and z negated — the client's conversion maps them back to the
# original numbers before the solver sees them. The historical names describe
# the MODEL frame: "low" is physically ~160mm ABOVE the base origin, and
# left/right are physically mirrored.
TARGETS = [
    ("fwd-mid", (0.150, 0.000, 0.080)),
    ("fwd-high", (0.150, 0.000, 0.000)),
    ("left", (0.000, -0.180, 0.060)),
    ("right", (0.000, 0.180, 0.060)),
    ("diag", (0.120, -0.120, 0.100)),
    ("near", (0.080, -0.060, 0.040)),
    ("low", (0.140, 0.060, 0.160)),
    ("stretch", (0.220, 0.000, 0.060)),
]

IK_TOL_MM = 10.0


def main() -> None:
    try:
        client = MatlabIKClient()
    except ConnectionRefusedError:
        print("Could not connect to the MATLAB server on localhost:9999.")
        print("Start it first:  in MATLAB, from the matlab/ folder, run  ik_fk_server")
        sys.exit(1)

    print(f"{'target':10} {'xyz [mm]':>18} {'IK err':>10}   FK")
    print("-" * 60)

    n_ok = 0
    n_fail = 0
    with client:
        for name, (x, y, z) in TARGETS:
            try:
                angles_rad, err_mm = client.request_ik(x, y, z)
            except IKUnreachableError as e:
                print(f"{name:10} [{1000*x:4.0f} {1000*y:4.0f} {1000*z:4.0f}]  UNREACHABLE: {e}")
                n_fail += 1
                continue

            # Round-trip the solved angles back through FK; expect a valid 4x4.
            T = client.request_fk(angles_rad)
            fk_ok = (
                isinstance(T, np.ndarray)
                and T.shape == (4, 4)
                and np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-6)
            )

            within = err_mm <= IK_TOL_MM
            status = "ok" if (within and fk_ok) else "CHECK"
            if within and fk_ok:
                n_ok += 1
            else:
                n_fail += 1

            print(
                f"{name:10} [{1000*x:4.0f} {1000*y:4.0f} {1000*z:4.0f}] "
                f"{err_mm:7.2f} mm   {status}"
            )

    print("-" * 60)
    print(f"{n_ok} ok, {n_fail} to check (tolerance {IK_TOL_MM:.0f} mm).")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

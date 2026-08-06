"""Re-solve the saved hand-eye samples in the corrected joint-angle frame.

    python scripts/reinterpret_hand_eye.py --search    # rank dir_sign hypotheses
    python scripts/reinterpret_hand_eye.py             # apply the stored one
    python scripts/reinterpret_hand_eye.py --write     # ...and save it

WHY THE OLD SAMPLES ARE SALVAGEABLE. Each sample is a pair. `T_cam_board` came
from solvePnP on a ChArUco board -- camera and board only, no arm, no FK, no
joint calibration. IT WAS NEVER WRONG. `T_base_gripper` came from FK of the
joint angles, and the calibration those angles were computed with has since
changed. So exactly one half of every pair is mislabelled, by a KNOWN per-joint
rule. That is a re-interpretation, not a re-capture: the arm really was where it
was, and only the labelling of which joint angles that pose corresponded to was
wrong.

THE CATCH. The samples stored 4x4 matrices, not the joint ticks. A joint-angle
change does NOT correspond to any fixed transform of the resulting pose -- the
correction depends on the pose -- so the stored matrices cannot be patched by
multiplying them by anything. The angles have to come back first:

    T_stored --(invert FK)--> theta_rec --(scale, offset)--> theta_true --(FK)--> T_fixed

The inversion is well posed here in a way it is NOT in general. The standing
warning (CLAUDE.md: "a 5-DOF arm has multiple joint solutions per tip pose") is
about inverting a POSITION. These samples store the full 4x4 WRIST pose, and a
full 6-DOF pose on a 5-DOF arm's reachable manifold pins the configuration to a
discrete set rather than a 2-D family. Every recovery is checked by pushing it
back through FK and requiring it to reproduce the stored matrix, and each solve
is seeded from the previous one so the branch stays continuous.

THE REFEREE IS BOARD SPREAD. The board never moved, so

    T_base_gripper @ T_gripper_camera @ T_cam_board

must land on the same base-frame point for every sample. Its scatter is pure
chain inconsistency, it needs no arm and no ruler, and it cannot be talked into
agreeing. `--search` uses it to rank all 32 dir_sign hypotheses, which is how J5
gets settled from the data rather than assumed.

Needs the MATLAB server up, for FK only. The arm can be powered off.
"""

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import hand_eye
from vision_pipeline.calibration.pixel_to_world import save_hand_eye
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient

IK_JOINTS = (1, 2, 3, 4, 5)
SAMPLES = Path("data/hand_eye_samples.json")

# The calibration as it stood WHEN THE SAMPLES WERE RECORDED (2026-08-04 17:27).
# J5 was dir_sign +1 then; it was flipped to -1 eighty minutes later.
CAPTURE_CAL = Path("data/servo_calibration.json.bak-20260804-preJ5flip")
# Same home_tick values, but already in TODAY's tick numbering: J3's encoder was
# re-centred on 2026-08-05 (+2065 on every J3 tick) and this backup postdates
# that. Re-centring preserved angles by construction, so nothing else has to
# account for it.
HOME_CAL = Path("data/servo_calibration.json.bak-20260805-preHomeFix")
NEW_CAL = Path("data/servo_calibration.json")

RECOVER_TOL_MM = 0.5
RECOVER_TOL_DEG = 0.5


def correction(dir_hypothesis=None):
    """(scale, offset) mapping a RECORDED joint angle to the TRUE one.

        theta_true = scale * theta_recorded + offset

    A calibration change is not always an offset, and assuming it was cost this
    script its first run. Angles come from

        theta = dir_sign * (tick - home_tick) / ticks_per_rad

    so changing home_tick SHIFTS the angle, but changing dir_sign REFLECTS it.
    Solving the recorded formula for the tick and substituting the current one:

        theta_true = (d_now / d_cap) * theta_rec
                   + d_now * (h_cap - h_now) / ticks_per_rad

    Both terms depend on d_now, which is why a sign flip cannot be bolted on
    afterwards as an extra offset.
    """
    cap = json.loads(CAPTURE_CAL.read_text(encoding="utf-8"))
    home = json.loads(HOME_CAL.read_text(encoding="utf-8"))
    now = json.loads(NEW_CAL.read_text(encoding="utf-8"))

    scale, offset = np.zeros(5), np.zeros(5)
    for k, j in enumerate(IK_JOINTS):
        d_cap = cap[str(j)]["dir_sign"]
        d_now = (dir_hypothesis or {}).get(j, now[str(j)]["dir_sign"])
        tpr = now[str(j)]["ticks_per_rad"]
        scale[k] = d_now / d_cap
        offset[k] = d_now * (home[str(j)]["home_tick"]
                             - now[str(j)]["home_tick"]) / tpr
    return scale, offset


def pose_error(T_a, T_b):
    """(translation mm, rotation deg) between two 4x4 poses."""
    mm = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])) * 1000.0
    R = T_a[:3, :3] @ T_b[:3, :3].T
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return mm, float(np.degrees(np.arccos(cos)))


def residual(T_have, T_want):
    """6-vector: translation error, then rotation error as a rotation vector."""
    dt = T_want[:3, 3] - T_have[:3, 3]
    R = T_want[:3, :3] @ T_have[:3, :3].T
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-9:
        dr = np.zeros(3)
    else:
        dr = angle / (2.0 * np.sin(angle)) * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.concatenate([dt, dr])


def invert_fk(client, T_target, seed, iters=60, step=1e-4):
    """Recover joint angles whose FK reproduces T_target. Gauss-Newton.

    Done in Python against the server's FK rather than through MATLAB's IK
    solver deliberately: this is a pure function inversion, and MATLAB's solver
    applies PositionLimits. Those limits describe the CORRECTED frame, so they
    would reject the very angles being recovered -- the old, mislabelled ones --
    and quietly clamp the answer instead of failing.
    """
    theta = np.array(seed, dtype=float)
    for _ in range(iters):
        T_now = client.request_fk(list(theta))
        r = residual(T_now, T_target)
        if np.linalg.norm(r[:3]) * 1000 < 1e-3 and np.linalg.norm(r[3:]) < 1e-6:
            break
        J = np.zeros((6, 5))
        for k in range(5):
            bumped = theta.copy()
            bumped[k] += step
            J[:, k] = residual(T_now, client.request_fk(list(bumped))) / step
        try:
            theta = theta + np.linalg.lstsq(J, r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
    return theta, client.request_fk(list(theta))


def rebuild(client, acc, thetas, scale, offset):
    """Apply one correction hypothesis to already-recovered angles."""
    out = hand_eye.HandEyeAccumulator(min_samples=acc.min_samples)
    for idx, samples in acc._samples.items():
        for sample, theta in zip(samples, thetas[idx]):
            if theta is None:
                continue
            T = client.request_fk(list(scale * theta + offset))
            out._samples.setdefault(idx, []).append(
                hand_eye.HandEyeSample(T_base_gripper=T,
                                       T_cam_board=sample.T_cam_board))
    return out


def mean_spread(accumulator):
    """Mean board spread in mm, or inf if nothing is solvable."""
    results = accumulator.solve_all()
    if not results:
        return float("inf")
    return float(np.mean([r.board_spread_mm for r in results.values()]))


def report(results, label):
    print(f"\n{label}")
    for idx in sorted(results):
        r = results[idx]
        print(f"  board {idx + 1}: n={r.n_samples:3d}  "
              f"spread {r.board_spread_mm:8.1f} mm   "
              f"TSAI-PARK {r.tsai_park_disagreement_mm:7.1f} mm / "
              f"{r.tsai_park_rotation_deg:5.1f} deg")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="save the winning transform to data/hand_eye.json")
    ap.add_argument("--search", action="store_true",
                    help="rank all 32 dir_sign hypotheses by board spread")
    ap.add_argument("--samples", default=str(SAMPLES))
    args = ap.parse_args()

    acc = hand_eye.load_samples(args.samples)
    total = sum(len(v) for v in acc._samples.values())
    print(f"Loaded {total} samples across {len(acc._samples)} boards")

    before = acc.solve_all()
    base = float(np.mean([r.board_spread_mm for r in before.values()]))
    report(before, f"AS RECORDED - mean board spread {base:.1f} mm")

    client = MatlabIKClient()

    print("\n--- recovering joint angles from the stored wrist poses ---")
    thetas, kept, worst = {}, 0, 0.0
    seed = np.zeros(5)
    for idx in sorted(acc._samples):
        thetas[idx] = []
        for sample in acc._samples[idx]:
            theta, T_check = invert_fk(client, sample.T_base_gripper, seed)
            mm, deg = pose_error(T_check, sample.T_base_gripper)
            worst = max(worst, mm)
            if mm > RECOVER_TOL_MM or deg > RECOVER_TOL_DEG:
                thetas[idx].append(None)
                continue
            seed = theta
            thetas[idx].append(theta)
            kept += 1
    print(f"  recovered {kept}/{total}  (worst reproduction error {worst:.4f} mm)")
    if kept < total:
        print(f"  {total - kept} could not be inverted and are dropped.")

    stored = {j: json.loads(NEW_CAL.read_text(encoding="utf-8"))[str(j)]["dir_sign"]
              for j in IK_JOINTS}
    best_hyp = stored

    if args.search:
        print("\n--- ranking dir_sign hypotheses by board spread ---")
        rows = []
        for bits in range(32):
            hyp = {j: (1 if (bits >> k) & 1 == 0 else -1)
                   for k, j in enumerate(IK_JOINTS)}
            sc, off = correction(hyp)
            rows.append((mean_spread(rebuild(client, acc, thetas, sc, off)), hyp))
        rows.sort(key=lambda r: r[0])
        print(f"  {'spread mm':>10}   J1 J2 J3 J4 J5")
        for sp, hyp in rows[:6]:
            marks = " ".join(f"{hyp[j]:+d}" for j in IK_JOINTS)
            tag = "   <- currently stored" if hyp == stored else ""
            print(f"  {sp:>10.1f}   {marks}{tag}")
        stored_score = next(sp for sp, h in rows if h == stored)
        print(f"\n  the stored signs score {stored_score:.1f} mm")
        best_hyp = rows[0][1]

    scale, offset = correction(best_hyp)
    print("\n--- correction applied ---")
    for k, j in enumerate(IK_JOINTS):
        note = "   REFLECTED (dir_sign changed since capture)" if scale[k] < 0 else ""
        print(f"  J{j}: theta_true = {scale[k]:+.0f} * theta_rec "
              f"{np.degrees(offset[k]):+8.2f} deg{note}")

    fixed = rebuild(client, acc, thetas, scale, offset)
    after = fixed.solve_all()
    spread = float(np.mean([r.board_spread_mm for r in after.values()]))
    report(after, f"CORRECTED - mean board spread {spread:.1f} mm (was {base:.1f})")

    print("\n--- verdict ---")
    if spread < base * 0.5:
        print(f"  Spread collapsed {base:.0f} -> {spread:.0f} mm. The arm was where")
        print("  it was; only the labelling was wrong. These samples are usable.")
    elif spread < base:
        print("  Improved, but not decisively. Something else is wrong too.")
    else:
        print("  NO IMPROVEMENT. Re-capture rather than trusting this.")

    if after:
        best = hand_eye.select_best(after, fixed)
        T = best.t_gripper_camera_tsai
        print(f"\n  best board {best.board_index + 1} ({best.n_samples} samples)")
        print(f"  camera offset from wrist: ({T[0, 3] * 1000:+.1f}, "
              f"{T[1, 3] * 1000:+.1f}, {T[2, 3] * 1000:+.1f}) mm")
        print(f"  implied board origin in base: "
              f"{np.round(best.mean_board_origin_base * 1000, 1)} mm")

        if args.write:
            if spread >= base * 0.5:
                print("\n  REFUSING to write: the solve did not decisively improve.")
                sys.exit(1)
            out = Path(config.HAND_EYE_PATH)
            if out.exists():
                bak = out.with_suffix(
                    f".json.bak-{date.today():%Y%m%d}-prereinterpret")
                shutil.copy(out, bak)
                print(f"\n  backed up to {bak.name}")
            save_hand_eye(T, str(out))
            print(f"  wrote {out}")
            print("  VERIFY: python scripts/test_pick_dry_run.py - it compares the")
            print("  camera height from the board (solvePnP, no FK) against FK plus")
            print("  this transform. They were 52 mm apart.")
        else:
            print("\n  Nothing written. Re-run with --write to save it.")

    client.close()


if __name__ == "__main__":
    main()

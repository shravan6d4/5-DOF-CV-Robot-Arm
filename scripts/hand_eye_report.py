"""The one hand-eye diagnostic: inventory, quality, outliers, solve, verdict.

    python scripts/hand_eye_report.py                 # every sample file found
    python scripts/hand_eye_report.py --samples FILE  # just one
    python scripts/hand_eye_report.py --fix           # write a cleaned copy

Read-only unless --fix. No arm, no camera, no MATLAB -- pure analysis of the
saved samples, so it is safe to run mid-capture.

WHY ONE SCRIPT. Diagnosing a failed hand-eye session had been spread over
check_hand_eye.py (a pass/fail gate), a throwaway pair-residual script, and a
pile of one-off inline commands. Each answered a different question and none
said which to believe, so a 156 mm board spread looked the same whether every
sample was slightly wrong or one sample was catastrophically wrong -- and those
need opposite responses (recapture everything vs delete one row).

THE ORDER OF THE SECTIONS IS THE ORDER TO TRUST THEM IN, and it is deliberate:

  1. INVENTORY      -- what data exists at all, and whether sets can be mixed.
  2. PER-BOARD QUALITY -- which board to believe. Every board in one capture
     sees the same arm through the same FK, so they must agree on rotation
     ANGLE; anything that differs board to board therefore cannot be in the arm.
     That comparison is what localises a fault to the camera side.
  3. SCREW CONGRUENCE -- the only checks that need NO SOLVE. AX = XB makes A and
     B conjugate, so every pose pair must agree on rotation ANGLE and on PITCH
     (how far the motion slid along its own axis) whatever X is (Chen 1991).
     A solver fed inconsistent data returns a confident wrong answer rather than
     an error, so anything downstream of a solve can be flattered by its own
     assumptions. These cannot.
  4. OBSERVABILITY  -- whether the capture CAN determine the answer, before
     asking what the answer is. Singular values of the stacked (R_a - I), scored
     with the indices from the robot-calibration literature. A capture that
     cannot observe the camera offset produces a beautiful residual and a wrong
     number; this is the section that says so.
  5. SOLVE          -- five methods, three formulations, and the gates.
  6. VERDICT        -- what to actually do next.

THE HISTORY THIS ENCODES. Every check here exists because something got past
the checks that came before it:
  * 2026-08-04: five solvers agreed within 11 mm on an offset a ruler put 56 mm
    away. Agreement between solvers fed the same bad data is not evidence.
  * 2026-08-05: a file silently mixed two calibrations; |t| = 33 mm sat inside
    the ruler tolerance and "passed" while board spread was 183 mm.
  * 2026-08-06: a capture passed every existing check (median rotation 35 deg,
    axis spread 82 deg) while 9 of 13 pose changes shared one axis to within
    1 deg. Rotation solved to 0.8 deg, position did not solve at all.
  * 2026-08-06: one bad sample of fifteen cost 96 mm of board spread.
  * 2026-08-06: the board with the MOST samples was the least self-consistent,
    so every tool that ranked by count had been analysing the worst data
    available while a clean board sat beside it.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import geometry, hand_eye

RULE = "=" * 78


def find_sample_files(explicit: str | None) -> list[Path]:
    """Every sample file worth looking at, newest first.

    Archives are included by default. A failed session's data is the only
    evidence of what went wrong, and on this project archived sets have twice
    turned out to contain recoverable samples -- so the default is to show what
    exists rather than to make the operator remember the filenames.
    """
    if explicit:
        return [Path(explicit)]
    data = Path("data")
    files = sorted(data.glob("hand_eye_samples.json*"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [p for p in files if p.stat().st_size > 2]


def load(path: Path):
    try:
        return hand_eye.load_samples(path)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  {path.name}: UNREADABLE ({exc})")
        return None


def best_board(acc) -> int | None:
    """The board to analyse: most CONSISTENT, not most numerous.

    Picking by sample count analyses the worst board on this data. In the
    2026-08-06 capture board 1 held 9 samples obeying screw congruence to 6.1 mm
    while board 2 held 22 obeying it to 63.8 mm -- and 22 mutually contradictory
    observations are worth less than 9 consistent ones.
    """
    counts = {i: len(v) for i, v in acc._samples.items() if v}
    if not counts:
        return None
    def rank(i):
        q = hand_eye.congruence_quality(acc, i)
        return (-(q.pitch_rms_mm if q else float("inf")), counts[i])
    return max(counts, key=rank)


def report_per_board(acc) -> None:
    """Compare every board on the checks that need no solve.

    THE COMPARISON IS THE DIAGNOSIS. Every board in one capture sees the same
    arm through the same FK, so they MUST agree on rotation angle -- and they
    do, to about a degree. Anything that then differs board to board cannot be
    in the arm, which is how the 2026-08-06 fault was localised to the camera
    side in a single reading.
    """
    print(f"\n{RULE}\n2.  PER-BOARD QUALITY — angle indicts the ARM, pitch the "
          f"CAMERA\n{RULE}\n")
    print("  Angle congruence depends only on rotations, so it is shared by every")
    print("  board and tests the kinematics. Pitch additionally depends on")
    print("  translations, which are measured per board — so with angles clean,")
    print("  a board-to-board spread in pitch is a CAMERA-side fault.\n")
    print(f"  {'board':>6} {'n':>4} {'angle med':>10} {'pitch rms':>10} "
          f"{'pitch corr':>11}   verdict")
    for i in sorted(acc._samples):
        n = len(acc._samples[i])
        q = hand_eye.congruence_quality(acc, i)
        if q is None:
            print(f"  {i + 1:6d} {n:4d} {'—':>10} {'—':>10} {'—':>11}   too few samples")
            continue
        if q.mirrored:
            verdict = "*** MIRRORED — see below"
        elif q.pitch_rms_mm > 25:
            verdict = "*** inconsistent"
        else:
            verdict = "clean"
        print(f"  {i + 1:6d} {n:4d} {q.angle_median_deg:9.2f}° "
              f"{q.pitch_rms_mm:9.1f} {q.pitch_correlation:+11.3f}   {verdict}")

    if any((q := hand_eye.congruence_quality(acc, i)) and q.mirrored
           for i in acc._samples):
        print("\n  MIRRORED means the camera's translations run OPPOSITE to the")
        print("  arm's along the same axis. No rigid transform can do that, but")
        print("  solvePnP's two-fold planar ambiguity can: it reflects a flat")
        print("  board's normal, keeping the rotation magnitude while mirroring")
        print("  the translation. That board's samples are unusable as recorded.")


def report_inventory(paths: list[Path]) -> dict[Path, object]:
    print(f"\n{RULE}\n1.  INVENTORY\n{RULE}\n")
    loaded = {}
    for p in paths:
        acc = load(p)
        if acc is None:
            continue
        loaded[p] = acc
        counts = {i + 1: len(v) for i, v in sorted(acc._samples.items())}
        age = (time.time() - p.stat().st_mtime) / 3600.0
        live = "  <- CURRENT" if p.name == "hand_eye_samples.json" else ""
        print(f"  {p.name}")
        print(f"      {sum(counts.values()):3d} samples across {len(counts)} boards "
              f"{counts}   {age:.1f}h old{live}")
    if not loaded:
        print("  No sample files found. Run scripts/calibrate_hand_eye.py first.")
    return loaded


def report_congruence(acc, board: int) -> list[int]:
    print(f"\n{RULE}\n3.  SCREW CONGRUENCE — no solve involved, so nothing can "
          f"flatter it\n{RULE}\n")
    print("  AX = XB makes the gripper motion A and the camera motion B conjugate.")
    print("  Conjugation preserves both screw invariants, so for EVERY pair:")
    print("      angle(A) == angle(B)      how far it turned")
    print("      pitch(A) == pitch(B)      how far it slid along that same axis")
    print("  whatever X is. Angle alone is the weaker half — it is blind to a")
    print("  motion that turns correctly but slides the wrong way.\n")

    pairs = hand_eye.screw_congruence(acc, board)
    if not pairs:
        print("  Too few samples to form a pair.")
        return []

    print(f"  {'pair':>7} {'ang A':>7} {'ang B':>7} {'d ang':>7}   "
          f"{'pitch A':>8} {'pitch B':>8} {'d pitch':>8}")
    for p in pairs:
        bad = (p.angle_error_deg > config.CALIB_HAND_EYE_CONGRUENCE_ANGLE_DEG
               or p.pitch_error_mm > config.CALIB_HAND_EYE_CONGRUENCE_PITCH_MM)
        print(f"  {p.index:3d}->{p.index + 1:<3d} {p.angle_a_deg:7.1f} "
              f"{p.angle_b_deg:7.1f} {p.angle_error_deg:7.2f}   "
              f"{p.pitch_a_mm:8.1f} {p.pitch_b_mm:8.1f} {p.pitch_error_mm:8.1f}"
              f"{'   <== BREAKS CONGRUENCE' if bad else ''}")

    ang = np.array([p.angle_error_deg for p in pairs])
    pit = np.array([p.pitch_error_mm for p in pairs])
    print(f"\n  angle error: median {np.median(ang):5.2f} deg, worst {ang.max():5.2f}"
          f"   (tolerance {config.CALIB_HAND_EYE_CONGRUENCE_ANGLE_DEG:.0f})")
    print(f"  pitch error: median {np.median(pit):5.1f} mm,  worst {pit.max():5.1f}"
          f"   (tolerance {config.CALIB_HAND_EYE_CONGRUENCE_PITCH_MM:.0f})")

    print("\n  The table above is CONSECUTIVE pairs, which is how the capture was")
    print("  ordered — but congruence holds between ANY two poses, so ordering")
    print("  carries no meaning and neighbours are a small slice of the evidence.")
    print("  Worse, an isolated bad consecutive pair implicates BOTH endpoints")
    print("  equally. Scoring every pair settles it by consensus: a genuinely bad")
    print("  sample disagrees with nearly all the others, while its innocent")
    print("  neighbour disagrees only with it.\n")

    scores = hand_eye.congruence_disagreement(acc, board)
    limit = config.CALIB_HAND_EYE_MAX_DISAGREEMENT
    print(f"  {'sample':>7}  disagrees with")
    for i, s in sorted(scores.items()):
        bar = "#" * int(s * 40)
        mark = "   <== OUTLIER" if s > limit else ""
        print(f"  {i:5d}    {s * 100:5.0f}%  {bar}{mark}")

    outliers = hand_eye.congruence_outliers(acc, board)
    if outliers:
        print(f"\n  OUTLIER SAMPLES: {outliers}  "
              f"(disagree with more than {limit * 100:.0f}% of the rest)")
    else:
        print("\n  No sample breaks congruence with the majority. Whatever is")
        print("  wrong, it is not one bad frame — look at observability below.")
    return outliers


def report_observability(acc, board: int) -> None:
    print(f"\n{RULE}\n4.  OBSERVABILITY — CAN this capture determine the answer?"
          f"\n{RULE}\n")
    obs = hand_eye.observability(acc, board)
    if obs is None:
        print("  Too few samples.")
        return

    print("  Rotation and translation fail differently and no residual tells")
    print("  them apart. Once X's rotation is known its translation solves from")
    print("      (R_a - I) t_x = R_x @ t_b - t_a")
    print("  so the singular values of the stacked (R_a - I) ARE the")
    print("  observability of the camera offset. (R_a - I) n = 0 for a rotation")
    print("  about n, so repeating one axis hides the offset along it however")
    print("  many samples are taken.\n")

    sv = obs.singular_values
    print(f"  singular values: {sv[0]:6.2f} {sv[1]:6.2f} {sv[2]:6.2f}\n")
    gates = [
        ("O3  smallest singular value", obs.o3_min_singular,
         config.CALIB_HAND_EYE_MIN_O3, "min", "the literature's pick "
         "(E-optimality): low when rotations are SMALL *or* SHARE AN AXIS"),
        ("1/O2  condition number", obs.condition_number,
         config.CALIB_HAND_EYE_MAX_CONDITION, "max", "clustering only — "
         "scale-invariant, so uniformly tiny rotations score a perfect 1.0"),
    ]
    for name, value, limit, sense in [(g[0], g[1], g[2], g[3]) for g in gates]:
        ok = value >= limit if sense == "min" else value <= limit
        print(f"  {name:<30} {value:7.2f}   "
              f"({'want >=' if sense == 'min' else 'want <='} {limit:.1f})  "
              f"{'OK' if ok else '*** FAILS'}")
    for g in gates:
        print(f"       {g[4]}")
    print(f"\n  O1  product (D-optimality)     {obs.o1_product:7.3f}   "
          f"volume of the data scatter, reported not gated")
    print(f"  O4  noise amplification       {obs.o4_noise_amplification:7.3f}   "
          f"least noise-sensitive of the four")

    # Name the direction that is poorly seen, and say what it means physically.
    if obs.condition_number > config.CALIB_HAND_EYE_MAX_CONDITION:
        blocks = [a[:3, :3] - np.eye(3)
                  for a, _ in hand_eye.relative_motions(acc, board)]
        _, _, vt = np.linalg.svd(np.vstack(blocks))
        weak = vt[-1]
        axes = [geometry.screw_axis(a)[0]
                for a, _ in hand_eye.relative_motions(acc, board)]
        near = sum(1 for n in axes
                   if np.degrees(np.arccos(np.clip(abs(np.dot(n, weak)), 0, 1))) < 15)
        print(f"\n  The poorly-seen direction is ({weak[0]:+.2f},{weak[1]:+.2f},"
              f"{weak[2]:+.2f}) in the gripper frame,")
        print(f"  and {near} of {len(axes)} pose changes rotate about it (within 15 deg).")
        print("  That is the cause: you cannot locate the camera ALONG the axis")
        print("  you keep rotating about, any more than you can tell where along")
        print("  an axle something sits by watching the wheel spin.")


def report_solve(acc, board: int) -> None:
    print(f"\n{RULE}\n5.  SOLVE\n{RULE}\n")
    results = acc.solve_all()
    if board not in results:
        print(f"  Board {board + 1} has too few samples to solve "
              f"(needs {acc.min_samples}).")
        return
    r = results[board]
    for name, T in r.solutions.items():
        t = T[:3, 3] * 1000
        print(f"  {name:<12} |t| = {np.linalg.norm(t):6.1f} mm   "
              f"({t[0]:+7.1f},{t[1]:+7.1f},{t[2]:+7.1f})")
    missing = {"TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"} - set(r.solutions)
    if missing:
        print(f"  {', '.join(sorted(missing))} did NOT converge — on this data that "
              f"is a signal, not a nuisance:")
        print("  a method refusing to fit means no rigid transform describes it.")
    print(f"\n  TSAI/PARK/HORAUD are SEPARABLE (rotation first, then translation),")
    print(f"  so they share a blind spot and their agreement means little.")
    print(f"  ANDREFF and DANIILIDIS solve both at once. Agreement ACROSS those")
    print(f"  two families is the part worth reading.\n")
    print(f"  method spread {r.method_spread_mm:6.1f} mm | "
          f"board spread {r.board_spread_mm:6.1f} mm | "
          f"TSAI-PARK {r.tsai_park_rotation_deg:5.1f} deg")

    cross = hand_eye.cross_board_agreement_mm(results)
    if cross is not None:
        print(f"  cross-board agreement {cross:.1f} mm across {len(results)} boards "
              f"— independent sample sets converging is the strongest signal the "
              f"data can give on its own")


def report_verdict(acc, board: int, outliers: list[int], path: Path) -> bool:
    print(f"\n{RULE}\n6.  VERDICT\n{RULE}\n")
    warnings = hand_eye.capture_health(acc, board)
    results = acc.solve_all()
    complaints = hand_eye.solve_complaints(results[board]) if board in results else \
        ["not enough samples to solve"]

    if warnings:
        print("  CAPTURE problems (no solve can fix these):")
        for w in warnings:
            print(f"    - {w}")
        print()
    if complaints:
        print("  SOLVE problems:")
        for c in complaints:
            print(f"    - {c}")
        print()

    if not warnings and not complaints:
        print("  ACCEPTED. Every gate passes: screw congruence, observability,")
        print("  the ruler, board spread and solver agreement.")
        print("\n  Next: python scripts/test_pick_dry_run.py")
        return True

    print("  DO NOT SAVE THIS. What to do, in order:\n")
    if outliers:
        print(f"    1. Drop sample(s) {outliers} — re-run with --fix to write a")
        print("       cleaned copy (the original is backed up).")
    if any("CLUSTERED" in w or "O3" in w for w in warnings):
        n = "2" if outliers else "1"
        print(f"    {n}. Recapture with more AXIS VARIETY. On this arm J2, J3 and")
        print("       J4 are all the SAME axis (parallel to 0.0 deg), so only two")
        print("       independent axes exist: the pitch chain and the J5 roll.")
        print("       ALTERNATE them — a long run of either one stalls.")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=None,
                    help="one file to analyse (default: every one found in data/)")
    ap.add_argument("--board", type=int, default=None,
                    help="1-based board number (default: the one with most samples)")
    ap.add_argument("--fix", action="store_true",
                    help="write a copy with congruence outliers removed")
    args = ap.parse_args()

    paths = find_sample_files(args.samples)
    if not paths:
        print("No sample files in data/.")
        sys.exit(1)

    loaded = report_inventory(paths)
    if not loaded:
        sys.exit(1)

    # Analyse the current file when scanning everything -- the archives are
    # listed so they are not forgotten, but a report on all of them at once is
    # unreadable and only one of them is the session in progress.
    target = Path(args.samples) if args.samples else paths[0]
    acc = loaded[target]
    board = (args.board - 1) if args.board is not None else best_board(acc)
    if board is None:
        print("\nNo board has any samples.")
        sys.exit(1)

    print(f"\n\nAnalysing {target.name}, board {board + 1} "
          f"({len(acc._samples[board])} samples)")
    if len(paths) > 1 and not args.samples:
        print(f"({len(paths) - 1} archived file(s) listed above; "
              f"--samples FILE to analyse one)")

    report_per_board(acc)
    outliers = report_congruence(acc, board)
    report_observability(acc, board)
    report_solve(acc, board)
    ok = report_verdict(acc, board, outliers, target)

    if args.fix and outliers:
        backup = target.with_suffix(
            f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}-preFix")
        shutil.copy2(target, backup)
        acc._samples[board] = [s for i, s in enumerate(acc._samples[board])
                               if i not in set(outliers)]
        hand_eye.save_samples(acc, target)
        print(f"\n  --fix: removed {outliers} from board {board + 1}.")
        print(f"  Original backed up to {backup.name}. Re-run to see the effect.")
    elif args.fix:
        print("\n  --fix: nothing to remove.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

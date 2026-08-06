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
import time
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import geometry
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

IK_JOINTS = range(1, 6)
DEFAULT_TICKS = 80           # ~7 deg at J1..J5 — enough wrist travel to see
MIN_VISIBLE_MM = 3.0         # below this the motion is too small to judge by eye
MIN_VISIBLE_DEG = 3.0        # ...or, for a joint that only rotates the wrist
CLAW_LEN_MM = 70.06          # ClawTip offset from the wrist, from init_arm.m
CLEARANCE_MARGIN_MM = 5.0    # never plan to close more than this much of the gap
SETTLE_TIMEOUT_S = 4.0       # how long to wait for the joint to stop creeping
SETTLE_POLL_S = 0.25
SETTLE_TOL_TICKS = 3         # two reads this close = it has come to rest


def base_yaw_axis_xy(client) -> np.ndarray:
    """Where the base yaw axis actually is, in base-frame XY millimetres.

    NOT the origin. The imported model's origin is a CAD artefact sitting 81 mm
    from the column the arm turns about (measured 2026-08-06 by
    scripts/audit_model_axes.py), so "radial" computed as `tip - origin` points
    108 deg away from the true radial direction at the home pose. That is the
    bug behind jog predictions announcing sideways motion for the pitch joints.

    Read out of FK rather than stored as a constant, so it cannot go stale if
    the model or the frame convention changes: jog J1 a little, and the axis it
    rotated the wrist about IS the base yaw axis.
    """
    a0 = [0.0] * 5
    a1 = [np.deg2rad(4.0), 0.0, 0.0, 0.0, 0.0]
    t0, _ = client.request_fk_tip(a0)
    t1, _ = client.request_fk_tip(a1)
    _, point, _ = geometry.screw_axis(t1 @ geometry.invert_transform(t0))
    return point[:2] * 1000.0


def describe_motion(d_mm: np.ndarray, tip_xy: np.ndarray | None = None,
                    yaw_xy: np.ndarray | None = None,
                    axis: np.ndarray | None = None) -> str:
    """Render a base-frame displacement as directions a human can check.

    DECOMPOSE ABOUT THE JOINT'S OWN AXIS. A revolute joint moves the tip
    strictly perpendicular to its axis, so the axis fixes which directions are
    even possible, and naming the impossible one is what makes a prediction
    unfalsifiable. Two cases, and they need opposite language:

      * horizontal axis (J2/J3/J4, the pitches) -- the claw moves OUT/IN and
        UP/DOWN in the arm's own vertical plane, and CANNOT move sideways.
      * vertical axis (J1 base yaw) -- the claw swings LEFT/RIGHT about that
        axis and cannot change height.

    TWO WRONG VERSIONS OF THIS CAME FIRST, both on 2026-08-06, and the second
    is the instructive one:

      1. Splitting along the fixed base Y axis, which called a pure pitch
         "1.9 mm left" purely because the arm pointed off-centre. The operator
         watched a J3 jog and said the claw plainly moved forward.
      2. Splitting radially about the base yaw axis. Better, but still wrong:
         the arm's plane is offset from that axis by the J1->J2 link (12.7 mm),
         so a perfectly clean J3 pitch still came out as "14.8 mm out AND
         5.1 mm left". The residual was real arithmetic about the wrong centre,
         not an error the operator could ever see.

    Sideways is still REPORTED for a pitch joint rather than dropped, because a
    non-zero value there is evidence the model's axes are wrong -- which is
    exactly the fault that was being hunted when this was written. It should
    read 0.0.

    `yaw_xy` (see base_yaw_axis_xy) only signs out-vs-in, so both cases agree
    on which way is "away from the base column".
    """
    d_mm = np.asarray(d_mm, dtype=float)
    axes = None

    if axis is not None and tip_xy is not None and yaw_xy is not None:
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        radial = np.asarray(tip_xy[:2], dtype=float) - np.asarray(yaw_xy, dtype=float)

        if abs(axis[2]) < 0.2 and np.linalg.norm(radial) > 1e-6:
            # Horizontal axis: a pitch. "Out" is horizontal, perpendicular to
            # the axis, signed away from the base column.
            out = np.cross(axis, [0.0, 0.0, 1.0])
            out = out / np.linalg.norm(out)
            if float(np.dot(out[:2], radial)) < 0:
                out = -out
            axes = [
                (float(np.dot(d_mm, out)), "out (away from the base column)",
                 "in (toward the base column)"),
                (float(d_mm[2]), "up", "down"),
                (float(np.dot(d_mm, axis)), "sideways +axis", "sideways -axis"),
            ]
        elif abs(axis[2]) > 0.8 and np.linalg.norm(radial) > 1e-6:
            # Vertical axis: a yaw. Sideways is the whole point here.
            r = radial / np.linalg.norm(radial)
            axes = [
                (float(np.dot(d_mm[:2], np.array([-r[1], r[0]]))), "left", "right"),
                (float(np.dot(d_mm[:2], r)), "out (away from the base column)",
                 "in (toward the base column)"),
                (float(d_mm[2]), "up", "down"),
            ]

    if axes is None:
        # A tilted axis (or no axis supplied) has no clean two-term story, so
        # say plainly what the base frame reports rather than inventing one.
        axes = [
            (float(d_mm[0]), "along +X", "along -X"),
            (float(d_mm[1]), "along +Y", "along -Y"),
            (float(d_mm[2]), "up", "down"),
        ]

    parts = [
        f"{abs(v):.1f} mm {pos if v > 0 else neg}"
        for v, pos, neg in axes
        if abs(v) >= 0.2
    ]
    return ", ".join(parts) if parts else "no appreciable movement"


def is_roll(axis: np.ndarray, claw_dir: np.ndarray | None) -> bool:
    """Does this joint SPIN the claw about its own pointing direction?

    A roll barely translates the tip -- J5 moves it 7.5 mm for a 17.6 deg turn,
    which clears MIN_VISIBLE_MM and would otherwise have the operator judging a
    sign from a 7 mm wobble instead of an unmistakable spin. Worse, that small
    translation has no clean description: the roll axis sits at no particular
    angle to anything, so describe_motion can only fall back to raw base-frame
    components. Watch the spin instead, whatever the travel.
    """
    if claw_dir is None:
        return False
    v = np.asarray(claw_dir, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return False
    return abs(float(np.dot(np.asarray(axis, dtype=float), v / n))) > 0.8


def describe_rotation(R0: np.ndarray, R1: np.ndarray,
                      claw_dir: np.ndarray | None = None) -> tuple[str, float]:
    """Describe the wrist's rotation between two poses.

    Needed for a joint like J5 that spins the wrist about its own axis: the
    wrist ORIGIN barely moves, so a position-only prediction reports almost
    nothing and the operator has nothing to compare against. The claw visibly
    turns, though, so describe that instead.

    A ROLL GETS A FRAME-INDEPENDENT DESCRIPTION, and that is the point of
    `claw_dir`. The fallback below names the rotation against the BASE frame,
    which on this arm is rotated ~89 deg from the direction the arm actually
    reaches (measured 2026-08-06, scripts/audit_model_axes.py). A J5 roll about
    the forearm therefore came out of that fallback as "tilting back/up" -- a
    pitch, which is not what a roll does and not what the operator would see.
    Spin sense about the claw's own pointing direction needs no base frame, so
    it survives that error entirely.

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
    deg = float(np.rad2deg(angle))

    if claw_dir is not None and np.linalg.norm(claw_dir) > 1e-9:
        view = np.asarray(claw_dir, dtype=float)
        view = view / np.linalg.norm(view)
        if is_roll(axis, claw_dir):
            # Right-hand rule: looking ALONG the axis (it points away from the
            # viewer), a positive rotation reads as clockwise. The viewer here
            # stands at the wrist looking out towards the claw.
            spin = "CLOCKWISE" if float(np.dot(axis, view)) > 0 else "ANTICLOCKWISE"
            return (f"{deg:.1f} deg, the claw SPINS {spin} — viewed from the "
                    f"wrist looking out along the claw", deg)

    i = int(np.argmax(np.abs(axis)))
    sense = axis[i] > 0
    # Base-frame naming, and the base frame is rotated ~89 deg from the arm's
    # own forward, so treat these words as approximate.
    naming = {
        0: ("counterclockwise seen from the front", "clockwise seen from the front"),
        1: ("tilting back/up", "tilting forward/down"),
        2: ("counterclockwise seen from above", "clockwise seen from above"),
    }
    pos, neg = naming[i]
    return f"{deg:.1f} deg, {pos if sense else neg} (base-frame naming)", deg


def read_angles(bus) -> dict:
    """Current tick and MATLAB angle for each IK joint."""
    state = {}
    for j in IK_JOINTS:
        tick = bus.read_position(j)
        state[j] = {"tick": tick, "rad": bus.ticks_to_rad(j, tick)}
    return state


def predict(client, state, joint, target_tick, bus):
    """FK before/after the proposed jog, for BOTH the wrist and the claw tip.

    Predicting the CLAW TIP matters: the operator watches the claw, which is
    the visually salient part, but the wrist is what request_fk reports. For
    the big arm joints (J1-J3) the two move together and it makes no
    difference. For the WRIST joints they diverge badly -- J4 pitches the
    wrist almost about its own axis, so the wrist barely translates while the
    tip swings ~70mm out on the lever, and J5 rolls the wrist with the tip on
    its axis. Comparing a wrist-based prediction against a claw observation
    for those joints produces a mismatched comparison, not a result. (J4's
    first confirm jog was wasted exactly this way.)

    Returns (T0_wrist, T1_wrist, tip0_mm, tip1_mm, tip_delta_mm, wrist_delta_mm).
    """
    angles_now = [state[j]["rad"] for j in IK_JOINTS]
    angles_after = list(angles_now)
    angles_after[joint - 1] = bus.ticks_to_rad(joint, target_tick)

    T0, T0_tip = client.request_fk_tip(angles_now)
    T1, T1_tip = client.request_fk_tip(angles_after)

    tip0 = 1000 * T0_tip[:3, 3]
    tip1 = 1000 * T1_tip[:3, 3]
    wrist_delta = 1000 * (T1[:3, 3] - T0[:3, 3])
    return T0, T1, tip0, tip1, tip1 - tip0, wrist_delta


def _stamp(joint: int, confirmed: bool, basis: str) -> Optional[dict]:
    """Record in the calibration file how this joint's dir_sign was decided.

    A dir_sign is only ever settled by a physical jog, so the jog's verdict is
    the one piece of evidence worth persisting — and the file is where it has to
    live, because no downstream number can re-derive it. Returns the loaded dict
    (already written back), or None if there is no file to write.
    """
    path = Path(config.SERVO_CALIBRATION_PATH)
    if not path.exists():
        print(f"  Cannot record: {path} does not exist (running on config fallback).")
        print(f"  Edit config.SERVO_CALIBRATION_FALLBACK['{joint}'] by hand.")
        return None
    cal = json.loads(path.read_text(encoding="utf-8"))
    cal[str(joint)]["dir_sign_confirmed"] = confirmed
    cal[str(joint)]["dir_sign_basis"] = basis
    # ensure_ascii=False keeps the prose notes legible in the file; the explicit
    # encoding is what stops Windows writing cp1252 where every other reader
    # expects UTF-8. Writing pure ASCII was masking a latent decode bug in
    # ServoBus rather than avoiding one.
    path.write_text(json.dumps(cal, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return cal


def confirm_sign(joint: int, sign: int, ticks: int, prediction: str) -> None:
    """Mark a dir_sign as confirmed after the arm agreed with the prediction."""
    stamped = _stamp(
        joint, True,
        f"CONFIRMED by physical jog on {date.today().isoformat()}: "
        f"{ticks:+d} ticks, operator reported the motion matched the prediction "
        f"({prediction}).",
    )
    if stamped is not None:
        print(f"  Recorded in {config.SERVO_CALIBRATION_PATH}: "
              f"J{joint} dir_sign {sign:+d} confirmed.")


def apply_flip(joint: int, ticks: int = 0, prediction: str = "") -> None:
    """Flip one joint's dir_sign in the calibration file, in place."""
    path = Path(config.SERVO_CALIBRATION_PATH)
    if not path.exists():
        print(f"  Cannot apply: {path} does not exist (running on config fallback).")
        print(f"  Edit config.SERVO_CALIBRATION_FALLBACK['{joint}']['dir_sign'] by hand.")
        return
    cal = json.loads(path.read_text(encoding="utf-8"))
    old = cal[str(joint)]["dir_sign"]
    cal[str(joint)]["dir_sign"] = -old
    # Flipped, but NOT yet confirmed: the confirming evidence is the *next* jog
    # matching its prediction, which has not happened yet.
    cal[str(joint)]["dir_sign_confirmed"] = False
    cal[str(joint)]["dir_sign_basis"] = (
        f"Flipped {old:+d} -> {-old:+d} on {date.today().isoformat()} after a "
        f"{ticks:+d}-tick jog moved OPPOSITE to the prediction ({prediction}). "
        f"NOT yet re-confirmed — jog again and check the new prediction matches."
    )
    # ensure_ascii=False keeps the prose notes legible in the file; the explicit
    # encoding is what stops Windows writing cp1252 where every other reader
    # expects UTF-8. Writing pure ASCII was masking a latent decode bug in
    # ServoBus rather than avoiding one.
    path.write_text(json.dumps(cal, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
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
        T0, T1, p0, p1, d, wrist_d = predict(client, state, joint, target_tick, bus)
        travel_mm = float(np.linalg.norm(d))
        rot_desc, rot_deg = describe_rotation(
            T0[:3, :3], T1[:3, :3], claw_dir=p0 - T0[:3, 3] * 1000.0)
        yaw_xy = base_yaw_axis_xy(client)
        # The joint's own axis, straight out of the two poses already computed
        # -- the motion between them IS a rotation about it. No extra FK calls,
        # and nothing to go stale if the model changes.
        jog_axis, _, _ = geometry.screw_axis(T1 @ geometry.invert_transform(T0))

        print(f"\nJ{joint} jog:  {start_tick} -> {target_tick} ticks "
              f"({args.ticks:+d}, dir_sign {cal['dir_sign']:+d})")
        print(f"  CLAW TIP now:       ({p0[0]:+7.1f}, {p0[1]:+7.1f}, {p0[2]:+7.1f}) mm")
        print(f"  CLAW TIP predicted: ({p1[0]:+7.1f}, {p1[1]:+7.1f}, {p1[2]:+7.1f}) mm")
        print(f"  (out/in below is measured from the BASE YAW AXIS at "
              f"({yaw_xy[0]:+.1f}, {yaw_xy[1]:+.1f}) mm, not the model origin — "
              f"the claw sits {np.linalg.norm(p0[:2] - yaw_xy):.0f} mm out from it)")

        # A joint whose axis passes through the tip (J5 roll) barely moves it,
        # so fall back to describing the rotation the operator can still see.
        watch_rotation = (travel_mm < MIN_VISIBLE_MM
                          or is_roll(jog_axis, p0 - T0[:3, 3] * 1000.0))
        if watch_rotation:
            prediction = f"the claw should ROTATE {rot_desc}"
        else:
            prediction = (f"the CLAW should move "
                          f"{describe_motion(d, p0, yaw_xy, jog_axis)}")
        print(f"\n  PREDICTION: {prediction}")
        print(f"              (claw travel {travel_mm:.1f} mm, rotation {rot_deg:.1f} deg)")

        # For the wrist joints the wrist and the claw genuinely go different
        # ways; surface that rather than letting the operator reconcile a
        # claw observation against a wrist number in their head.
        wrist_travel = float(np.linalg.norm(wrist_d))
        claw_desc = describe_motion(d, p0, yaw_xy, jog_axis)
        wrist_desc = describe_motion(wrist_d, p0, yaw_xy, jog_axis)
        if wrist_travel >= 0.5 and wrist_desc != claw_desc:
            print(f"  (the WRIST meanwhile moves {wrist_desc} — "
                  f"watch the CLAW, not the wrist)")

        if travel_mm < MIN_VISIBLE_MM and rot_deg < MIN_VISIBLE_DEG:
            print(f"\n  Under {MIN_VISIBLE_MM} mm and {MIN_VISIBLE_DEG} deg — too small")
            print(f"  to judge by eye. Re-run with a larger --ticks "
                  f"(e.g. {2 * abs(args.ticks)}).")
            sys.exit(1)

        # --- table clearance guard --------------------------------------
        # `d` is CLAW TIP motion, which is what actually closes the gap to the
        # table -- the tip is the lowest part and the thing that would strike.
        # (J3 folding the claw down into the table is the near-miss that
        # already happened once during bring-up.)
        descent_mm = -min(0.0, d[2])
        if descent_mm > 0:
            usable = args.clearance_mm - CLEARANCE_MARGIN_MM
            print(f"\n  Clearance: claw is {args.clearance_mm:.0f} mm above the table "
                  f"(--clearance-mm, YOUR figure, default 10);")
            # FK's own opinion, shown alongside. Deliberately NOT used as the
            # guard: FK's absolute height depends on home_tick and
            # TABLE_Z_IN_BASE, and this script exists precisely because the
            # joint calibration is under suspicion. But the two figures
            # disagreeing is worth knowing -- a large gap means one of them is
            # wrong, and the operator is the one who can look.
            fk_clearance = p0[2] - config.TABLE_Z_IN_BASE * 1000.0
            print(f"            FK reckons {fk_clearance:.0f} mm from the same pose.")
            if abs(fk_clearance - args.clearance_mm) > 30:
                print(f"            Those disagree by {abs(fk_clearance - args.clearance_mm):.0f} mm. "
                      f"If your figure is measured rather than\n"
                      f"            estimated, TABLE_Z_IN_BASE ({config.TABLE_Z_IN_BASE * 1000:.0f} mm) "
                      f"or home_tick is off.")
            print(f"            This jog descends {descent_mm:.1f} mm.")
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

        # WAIT FOR IT TO ACTUALLY STOP. move_and_verify's read-back happens as
        # soon as the servo is within tolerance of the goal, which under load is
        # not the same as finished: on 2026-08-06 a J3 jog reported "settled at
        # 2796" and the joint was at 2903 moments later -- it had delivered 102
        # of the requested 200 ticks at the instant of the read and then crept
        # the rest of the way. Every number downstream inherits that: the script
        # told the operator the servo had under-travelled by half, which made a
        # perfectly good move look like a 2.6x scale error in ticks_per_rad.
        #
        # Poll until two consecutive reads agree, so what gets reported is where
        # the joint came to rest rather than where it was passing through.
        settled_tick = actual_tick
        deadline = time.time() + SETTLE_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(SETTLE_POLL_S)
            now = bus.read_position_retrying(joint)
            if abs(now - settled_tick) <= SETTLE_TOL_TICKS:
                settled_tick = now
                break
            settled_tick = now
        if settled_tick != actual_tick:
            print(f"  (kept moving after the verify read: {actual_tick} -> "
                  f"{settled_tick}; using the settled value)")
        actual_tick = settled_tick

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

            # HOW FAR, not just which way. dir_sign is a direction, so a matched
            # jog says nothing about ticks_per_rad or the link lengths -- and
            # those are exactly what a differential measurement CAN check, which
            # nothing else in the repo does. Scaled by the ticks the servo
            # actually delivered, because it routinely settles short under load:
            # comparing a full-jog prediction against a half-executed move
            # manufactures a scale error that is not there.
            #
            # Prompted 2026-08-06, when a J3 jog predicted 72.3 mm for +200
            # ticks, the servo delivered +102, and the operator measured the
            # claw drop at ~95 mm -- roughly 2.6x the per-tick prediction. Worth
            # catching at the moment someone has a ruler in their hand.
            if abs(moved) > 0 and travel_mm > MIN_VISIBLE_MM:
                expected = travel_mm * abs(moved) / max(abs(args.ticks), 1)
                print(f"\n  Optional: how far did the claw ACTUALLY travel?")
                print(f"  Prediction for the {moved:+d} ticks the servo delivered: "
                      f"{expected:.1f} mm")
                print(f"  (blank to skip — this checks ticks_per_rad, which the")
                print(f"   direction test above cannot see)")
                raw = input("  observed mm > ").strip()
                if raw:
                    try:
                        seen = abs(float(raw))
                    except ValueError:
                        seen = None
                    if seen is not None and expected > 0:
                        ratio = seen / expected
                        print(f"    observed {seen:.1f} mm vs predicted {expected:.1f} "
                              f"mm  (x{ratio:.2f})")
                        if not 0.75 <= ratio <= 1.33:
                            print(f"    *** SCALE MISMATCH. The direction is right but the")
                            print(f"    AMOUNT is not, so ticks_per_rad "
                                  f"({cal['ticks_per_rad']:.1f}) or the model's link")
                            print(f"    lengths are off for this joint. dir_sign is still")
                            print(f"    confirmed — this is a separate fault.")
                        else:
                            print(f"    scale agrees; ticks_per_rad looks right for J{joint}.")
            # Persist the verdict. This is the only evidence that ever settles a
            # dir_sign, so leaving it in the terminal scrollback loses it.
            confirm_sign(joint, cal["dir_sign"], moved, prediction)
        elif answer.startswith("o"):
            print(f"\n  J{joint} dir_sign {cal['dir_sign']:+d} is INVERTED — should be "
                  f"{-cal['dir_sign']:+d}.")
            if args.apply_flip:
                apply_flip(joint, moved, prediction)
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

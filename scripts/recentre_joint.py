"""Move one joint's encoder seam away from its working range, and fix up the numbers.

    !!! THE JOINT GOES LIMP DURING THIS. SUPPORT THE ARM BEFORE RUNNING. !!!

THE PROBLEM. A Feetech encoder counts 0..4095 and wraps. Wherever that 0/4095
seam falls is a place the joint must never reach: a hair of motion past it flips
the reading from ~5 to ~4090, and the servo will not drive back across. J3's
travel was measured to tick 6 -- six ticks from the seam -- so its usable range
had to be held back by a margin that cost real workspace, and the joint still
ended up sitting below its own limit during normal work.

Moving the limit does not fix that. Moving the SEAM does. The Feetech one-key
midpoint command (write 128 to Torque Enable, addr 40) redefines the joint's
current physical position as tick 2048, shifting every tick reading by a
constant. Put the middle of the working range at 2048 and the seam ends up as
far away as the encoder allows, in both directions at once.

WHAT SHIFTS AND WHAT DOES NOT. Ticks shift; angles do not. The joint has not
moved and no physical relationship has changed -- only the encoder's numbering.
Because ticks_to_rad subtracts home_tick, the radian convention is preserved
exactly, PROVIDED home_tick shifts by the same offset as everything else. This
script applies that one offset to home_tick, min_tick and max_tick together,
which is the entire correctness argument.

    physical angle <-> radians    unchanged
    radians        <-> ticks      shifted by a constant
    home_tick, min_tick, max_tick all move together, or FK is wrong

SAFETY ORDER, WHICH MATTERS. Re-centring while torque is on is dangerous: the
Goal Position register keeps its numeric value but now means a DIFFERENT
physical position, so the servo drives there the instant it is re-interpreted.
This script therefore drops torque first, re-centres, then re-enables through
ServoBus.enable_torque, which pins the goal to the present position before
energising. The joint is limp in between -- support the arm.

IF THE READING HAS ALREADY WRAPPED, this is the only way out. Once a joint sits
on the far side of the seam there is no goal position that walks it back -- the
servo interprets the goal linearly and would drive the long way round the circle,
through every hard stop in between. It cannot be driven to the middle of its
travel first, so it is re-centred exactly where it stands (--here, which the
script also applies on its own when it detects the wrap). That costs some
symmetry in where the seam lands and buys the joint back without moving it.

Usage (from the repo root):
    python scripts/recentre_joint.py --joint 3 --dry-run   # plan only
    python scripts/recentre_joint.py --joint 3
    python scripts/recentre_joint.py --joint 3 --here      # redefine in place
"""

import argparse
import shutil
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_calibration import (
    load_calibration_file,
    save_calibration_file,
    unwrap_tick,
    write_angle_limits,
)
from vision_pipeline.robot_interface.servo_driver import ServoBus

MIDPOINT_TICK = 2048          # what the one-key midpoint command defines "here" as
ONE_KEY_MIDPOINT = 128        # written to ADDR_TORQUE_ENABLE (40)
PLACEMENT_TOLERANCE = 25      # how close to the target pose the joint must be
MIDPOINT_CONFIRM_S = 3.0      # how long to keep asking whether the EEPROM took
TICK_MAX = 4095


class Plan:
    """Everything the re-centre will do, computed before anything is touched."""

    def __init__(self, cal, joint, present_tick, here=False):
        entry = cal[str(joint)]
        for key in ("min_tick", "max_tick"):
            if key not in entry:
                raise SystemExit(
                    f"J{joint} has no {key}. Measure its travel first with "
                    f"scripts/find_joint_limits.py — re-centring without knowing "
                    f"the range cannot place the seam sensibly."
                )

        self.joint = joint
        self.entry = entry
        self.lo, self.hi = int(entry["min_tick"]), int(entry["max_tick"])
        self.mid = (self.lo + self.hi) // 2
        self.present = present_tick

        # The joint's reading may have wrapped past the seam, in which case the
        # raw number is 4096 off the numbering the stored limits live in. Every
        # comparison below is done in the UNWRAPPED coordinate; only the value
        # read from and written to the servo stays raw.
        self.present_unwrapped = unwrap_tick(present_tick, self.mid)
        self.wrapped = self.present_unwrapped != present_tick

        # Anchor = the physical pose whose tick number gets redefined as 2048.
        # Normally that is the middle of travel, which puts the seam as far from
        # both ends as one offset can. But a wrapped joint CANNOT be driven to
        # the middle first -- crossing back over the seam is the very thing the
        # servo will not do -- so it is re-centred exactly where it stands. That
        # costs some symmetry and buys the joint back.
        self.here = here or self.wrapped
        self.anchor = self.present_unwrapped if self.here else self.mid
        self.offset = MIDPOINT_TICK - self.anchor

        self.new = {
            "home_tick": int(entry["home_tick"]) + self.offset,
            "min_tick": self.lo + self.offset,
            "max_tick": self.hi + self.offset,
        }
        self.clearance = min(self.new["min_tick"], TICK_MAX - self.new["max_tick"])
        self.old_clearance = min(self.lo, TICK_MAX - self.hi)

    @property
    def in_range_after(self) -> bool:
        return self.new["min_tick"] <= MIDPOINT_TICK <= self.new["max_tick"]

    def describe(self):
        tpr = self.entry["ticks_per_rad"]
        deg = lambda t: np.degrees(t / tpr)

        print(f"\nJ{self.joint} currently reads {self.present} ticks")
        if self.wrapped:
            print(f"  *** THAT READING HAS WRAPPED past the 0/4095 seam. In the")
            print(f"      numbering its limits are written in, the joint is really")
            print(f"      at {self.present_unwrapped}, i.e. "
                  f"{self.lo - self.present_unwrapped} ticks "
                  f"({deg(self.lo - self.present_unwrapped):.1f} deg) below min_tick.")
            print(f"      No goal-position command can walk it back: the servo would")
            print(f"      drive {self.present - self.lo} ticks the long way round.")
        print(f"  measured travel   [{self.lo}, {self.hi}]  "
              f"({deg(self.hi - self.lo):.1f} deg of motion)")
        print(f"  seam clearance    {self.old_clearance} ticks "
              f"({deg(self.old_clearance):.1f} deg)   <-- the problem")

        where = ("WHERE IT STANDS (no motion)" if self.here
                 else "the middle of travel")
        print(f"\n  re-centre AT tick {self.present} — {where} — which becomes "
              f"{MIDPOINT_TICK}")
        print(f"  offset            {self.offset:+d} ticks applied to every stored tick")
        print(f"\n  home_tick         {self.entry['home_tick']:5d} -> {self.new['home_tick']:5d}")
        print(f"  min_tick          {self.lo:5d} -> {self.new['min_tick']:5d}")
        print(f"  max_tick          {self.hi:5d} -> {self.new['max_tick']:5d}")
        print(f"  seam clearance    {self.old_clearance} -> {self.clearance} ticks "
              f"({deg(self.clearance):.1f} deg)")
        print(f"\n  Angles are UNCHANGED — home_tick moves with the limits, so every")
        print(f"  radian this arm reports means what it meant before.")

        if not self.in_range_after:
            short = self.new["min_tick"] - MIDPOINT_TICK
            if short > 0:
                print(f"\n  AFTERWARDS J{self.joint} still reads {short} ticks "
                      f"({deg(short):.1f} deg) below min_tick — it is out of range")
                print(f"  now and re-centring does not move it. The difference is that")
                print(f"  the seam is no longer in the way, so an ordinary move brings")
                print(f"  it back:")
                print(f"      python scripts/goto_tick.py --joint {self.joint} "
                      f"--ticks {self.new['min_tick'] + 20}")


def apply_offset(cal, joint, offset, calibration_path, assume_yes=False,
                 note=None, backup_path=None, limits_path=None):
    """Shift one joint's stored ticks by `offset`. Touches no hardware.

    The bookkeeping half of a re-centre, split out so it can be run on its own.
    A re-centre is two independent writes -- the servo's EEPROM and this file --
    and if the second one is missed the arm reports every angle for that joint
    wrong by the offset, with nothing to indicate it. Recovering needs a way to
    apply the shift alone, WITHOUT a second midpoint write, which would move the
    numbering again and compound the error.
    """
    entry = cal[str(joint)]
    for key in ("min_tick", "max_tick"):
        if key not in entry:
            raise SystemExit(f"J{joint} has no {key} to shift.")

    new = {k: int(entry[k]) + offset for k in ("home_tick", "min_tick", "max_tick")}
    tpr = entry["ticks_per_rad"]

    print(f"\nJ{joint}: shifting every stored tick by {offset:+d} "
          f"({np.degrees(offset / tpr):.1f} deg of renumbering)")
    print(f"  home_tick         {entry['home_tick']:5d} -> {new['home_tick']:5d}")
    print(f"  min_tick          {entry['min_tick']:5d} -> {new['min_tick']:5d}")
    print(f"  max_tick          {entry['max_tick']:5d} -> {new['max_tick']:5d}")
    print(f"\n  No servo is touched and the arm does not move. Angles are")
    print(f"  unchanged — home_tick shifts with the limits.")

    outside = [k for k, v in new.items() if not 0 <= v <= TICK_MAX]
    if outside:
        raise SystemExit(
            f"\n  REFUSED: {', '.join(outside)} would land outside 0..{TICK_MAX}. "
            f"An offset that pushes a stored tick off the encoder is off by a "
            f"whole turn ({rj_span()}) or simply wrong; nothing was written."
        )

    if not assume_yes:
        if input("\n  Type 'apply' to write it: ").strip().lower() != "apply":
            print("  Aborted. Nothing changed.")
            return

    backup = backup_path
    if backup is None:
        backup = Path(calibration_path).with_suffix(
            f".json.bak-{date.today():%Y%m%d}-preJ{joint}offset")
        shutil.copy(calibration_path, backup)

    entry.update(new)
    entry["recentred"] = note or (
        f"Encoder re-centred {date.today().isoformat()}: every stored tick "
        f"shifted {offset:+d} so the 0/4095 seam sits far from the working "
        f"range. Angles are unchanged — home_tick moved with the limits. "
        f"Previous values in {backup.name}."
    )
    path = save_calibration_file(cal, calibration_path)
    angle_path, angles = write_angle_limits(cal, limits_path)
    print(f"\n  backed up to {backup.name}")
    print(f"  wrote {path}")
    print(f"  wrote {angle_path} ({len(angles)} joint(s))")
    print(f"\n  NEXT: restart the MATLAB server (>> ik_fk_server) so IK reloads")
    print(f"  the angle limits, then: python scripts/check_servo_health.py")


def rj_span() -> int:
    """The encoder span, named so the refusal message above can cite it."""
    return TICK_MAX + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan and the new numbers; change nothing")
    ap.add_argument("--here", action="store_true",
                    help="re-centre at the joint's PRESENT pose instead of driving "
                         "it to the middle of travel first. Implied automatically "
                         "when the reading has wrapped past the seam, since such a "
                         "joint cannot be driven anywhere.")
    ap.add_argument("--apply-offset", type=int, default=None, metavar="N",
                    help="REPAIR MODE. Shift this joint's stored home/min/max by "
                         "N ticks and regenerate the angle limits, WITHOUT "
                         "touching the servo. For the one state this script can "
                         "leave behind: the EEPROM write landed but the "
                         "confirmation read did not see it, so the servo is "
                         "re-centred and the file is not.")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    ap.add_argument("--calibration", default=config.SERVO_CALIBRATION_PATH)
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    joint = args.joint
    cal = load_calibration_file(args.calibration)

    if args.apply_offset is not None:
        apply_offset(cal, joint, args.apply_offset, args.calibration, args.yes)
        return

    try:
        bus = ServoBus(args.port, args.baud)
    except Exception as e:
        print(f"Could not open the servo bus on {args.port}: {e}")
        sys.exit(1)

    with bus:
        present = bus.read_position_retrying(joint)
        p = Plan(cal, joint, present, here=args.here)
        p.describe()

        if args.dry_run:
            print(f"\n--dry-run: nothing changed. To proceed:")
            if p.here:
                print(f"    python scripts/recentre_joint.py --joint {joint}"
                      f"{'' if p.wrapped else ' --here'}")
            else:
                print(f"    python scripts/goto_tick.py --joint {joint} "
                      f"--ticks {p.anchor}")
                print(f"    python scripts/recentre_joint.py --joint {joint}")
            return

        # The joint MUST be at the pose whose tick is about to be redefined.
        # Re-centring anywhere else silently applies a different offset than the
        # one computed above, and home_tick would then describe a pose the arm
        # is not in -- which is how FK starts lying. In --here mode the anchor IS
        # the present pose, so this is satisfied by construction.
        if not p.here and abs(present - p.anchor) > PLACEMENT_TOLERANCE:
            print(f"\n  REFUSED: J{joint} is at {present}, not within "
                  f"{PLACEMENT_TOLERANCE} ticks of {p.anchor}.")
            print(f"  The offset is computed from where the joint IS when the")
            print(f"  midpoint is written, so it has to be in the right place.")
            print(f"  Drive it there first:")
            print(f"      python scripts/goto_tick.py --joint {joint} "
                  f"--ticks {p.anchor}")
            print(f"  Or, to redefine the seam around the pose it is in now:")
            print(f"      python scripts/recentre_joint.py --joint {joint} --here")
            sys.exit(1)

        print(f"\n  *** J{joint} WILL GO LIMP for a moment. SUPPORT THE ARM NOW. ***")
        print(f"  This writes the servo's EEPROM. It is reversible only by")
        print(f"  re-centring again at a known pose.")
        if not args.yes:
            if input("  Type 'recentre' to proceed: ").strip().lower() != "recentre":
                print("  Aborted. Nothing changed.")
                return

        backup = Path(args.calibration).with_suffix(
            f".json.bak-{date.today():%Y%m%d}-preJ{joint}recentre")
        shutil.copy(args.calibration, backup)
        print(f"  backed up calibration to {backup.name}")

        # Torque off BEFORE the midpoint write: afterwards the Goal Position
        # register means a different physical place, and a powered servo would
        # drive to it immediately.
        bus.disable_torque(joint)
        time.sleep(0.05)

        if not bus._write_register(joint, ServoBus.ADDR_TORQUE_ENABLE,
                                   bytes([ONE_KEY_MIDPOINT])):
            print("  FAILED to write the midpoint command. Torque is OFF — "
                  "run: python scripts/servo_torque.py --enable all")
            sys.exit(1)
        # POLL for the new numbering rather than reading once after a fixed
        # sleep. The one-key command writes EEPROM, and how long the servo takes
        # to commit it and start reporting the shifted position is not specified
        # anywhere -- 0.2 s was a guess, and on 2026-08-05 it was too short. The
        # single read came back with the OLD value, the script concluded the
        # servo did not support the command and exited leaving the calibration
        # untouched... but the write HAD landed. That is the worst possible
        # outcome: servo re-centred, file not, every J3 angle silently wrong by
        # the offset, and a message saying nothing had changed.
        after = None
        deadline = time.time() + MIDPOINT_CONFIRM_S
        while time.time() < deadline:
            time.sleep(0.1)
            try:
                after = bus.read_position_retrying(joint)
            except Exception:
                continue
            if abs(after - MIDPOINT_TICK) <= PLACEMENT_TOLERANCE:
                break

        print(f"\n  J{joint} now reads {after} ticks (expected ~{MIDPOINT_TICK})")
        if after is None or abs(after - MIDPOINT_TICK) > PLACEMENT_TOLERANCE:
            print(f"\n  COULD NOT CONFIRM the midpoint after {MIDPOINT_CONFIRM_S:.0f}s.")
            print(f"  Torque is OFF — support the arm and run:")
            print(f"      python scripts/servo_torque.py --enable {joint}")
            print(f"  The calibration file was NOT modified.")
            print(f"\n  *** DO NOT SIMPLY RE-RUN THIS SCRIPT. *** The write may have")
            print(f"  landed anyway and only the confirmation failed, in which case")
            print(f"  the servo is re-centred and the file is not. A second midpoint")
            print(f"  write would compound the offset. Check first:")
            print(f"      python scripts/check_servo_health.py     (see B3a)")
            print(f"  If J{joint} now reads near {MIDPOINT_TICK}, the write DID land —")
            print(f"  finish the bookkeeping without touching the servo:")
            print(f"      python scripts/recentre_joint.py --joint {joint} "
                  f"--apply-offset {MIDPOINT_TICK - p.present_unwrapped}")
            sys.exit(1)

        # Re-energise pinned to where it sits, never to a stale goal.
        bus.enable_torque(joint)
        print(f"  torque re-enabled, holding {after}")

        # The offset is MIDPOINT_TICK - where the joint was, NOT `after` - where
        # it was. Those differ by however much the joint sagged while limp, and
        # that sag is REAL MOTION: the arm genuinely moved, and the new
        # numbering reports it correctly. Folding it into the offset instead
        # would silently rotate home_tick by the size of the sag and put a
        # permanent bias into every angle this joint reports. The one-key
        # command defines the pose AT THE INSTANT OF THE WRITE as 2048, and the
        # joint was still where it started then -- only ~50 ms of limp had
        # elapsed.
        actual_offset = MIDPOINT_TICK - p.present_unwrapped
        sag = after - MIDPOINT_TICK
        if sag:
            print(f"  J{joint} moved {abs(sag)} ticks "
                  f"({np.degrees(abs(sag) / p.entry['ticks_per_rad']):.1f} deg) "
                  f"while limp — real motion, not part of the offset.")

        apply_offset(
            cal, joint, actual_offset, args.calibration, assume_yes=True,
            note=(f"Encoder re-centred {date.today().isoformat()}: every stored "
                  f"tick shifted {actual_offset:+d} so the 0/4095 seam sits far "
                  f"from the working range"
                  + (f", after the reading wrapped past the seam to {present} "
                     f"during a visual-servo run" if p.wrapped else "")
                  + f". Angles are unchanged — home_tick moved with the limits. "
                  f"Previous values in {backup.name}."),
            backup_path=backup,
        )

        print(f"\n  NEXT, in order:")
        if not p.in_range_after:
            target = cal[str(joint)]["min_tick"] + 20
            print(f"    0. J{joint} is still below its range (reads {after}, "
                  f"min is {cal[str(joint)]['min_tick']}). Nothing is in the way")
            print(f"       of it now — walk it back in:")
            print(f"         python scripts/goto_tick.py --joint {joint} "
                  f"--ticks {target}")
        print(f"    1. Restart the MATLAB server so IK reloads the angle limits:")
        print(f"         >> ik_fk_server")
        print(f"    2. python scripts/check_servo_health.py")
        print(f"       B4's claw tip must land where it did BEFORE this change —")
        print(f"       same arm, same pose, same FK. If it moved, the offset was")
        print(f"       applied wrong and nothing downstream should be trusted.")
        print(f"    3. Optionally re-measure the travel now that the seam is far")
        print(f"       away, to recover the margin that was protecting it:")
        print(f"         python scripts/find_joint_limits.py --joint {joint}")


if __name__ == "__main__":
    main()

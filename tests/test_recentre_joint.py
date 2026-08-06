"""Encoder re-centring: the offset arithmetic and the two limit files staying in sync.

Re-centring shifts every stored tick for one joint by a constant so the 0/4095
encoder seam ends up far from the working range. The correctness argument is
short and worth testing precisely:

    home_tick, min_tick and max_tick all move by the SAME offset
        => ticks_to_rad, which subtracts home_tick, is unchanged
        => every angle the arm reports still means what it meant before

If home_tick moved and the limits did not (or vice versa), FK would start
reporting the arm somewhere it is not, which invalidates TABLE_Z_IN_BASE and
every world coordinate derived from it. That is a silent failure, so it gets a
test rather than a comment.

No serial port and no arm: the planning and file-writing halves are pure.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import recentre_joint as rj  # noqa: E402
from vision_pipeline.robot_interface.servo_calibration import (  # noqa: E402
    angle_limits,
    joint_angle_rad,
    load_calibration_file,
    save_calibration_file,
    ticks_to_rad,
    write_angle_limits,
)

TICK_MAX = 4095


def make_cal(home=1054, lo=63, hi=1096, dir_sign=-1, tpr=651.89):
    """J3-shaped calibration: travel hard against the 0 end of the encoder."""
    cal = {str(j): {"home_tick": 2048, "ticks_per_rad": tpr, "dir_sign": 1,
                    "home_angle_rad": 0.0} for j in range(1, 7)}
    cal["3"] = {"home_tick": home, "ticks_per_rad": tpr, "dir_sign": dir_sign,
                "home_angle_rad": 0.0, "min_tick": lo, "max_tick": hi}
    return cal


# --- the offset -------------------------------------------------------------

def test_plan_centres_the_working_range_on_the_encoder_midpoint():
    """Best placement a single offset can achieve: seam equidistant from both ends."""
    cal = make_cal()
    p = rj.Plan(cal, 3, present_tick=551)

    assert p.anchor == (63 + 1096) // 2
    assert p.offset == rj.MIDPOINT_TICK - p.anchor
    mid = (p.new["min_tick"] + p.new["max_tick"]) // 2
    assert abs(mid - rj.MIDPOINT_TICK) <= 1


def test_recentring_moves_the_seam_far_away():
    """The whole point. J3 sat ~63 ticks from the seam; it should end up ~1500."""
    p = rj.Plan(make_cal(), 3, present_tick=551)

    before = min(63, TICK_MAX - 1096)
    assert before < 100, "precondition: the seam really was close"
    assert p.clearance > 1400
    assert p.clearance > before * 10


def test_the_direction_is_away_from_the_seam_not_toward_it():
    """A sign error here would push the range INTO the seam — the exact failure
    this script exists to prevent, applied backwards."""
    p = rj.Plan(make_cal(), 3, present_tick=551)
    assert p.offset > 0, "travel sits near tick 0, so every tick must shift UP"
    assert p.new["min_tick"] > 63
    assert p.new["max_tick"] > 1096


def test_every_stored_tick_shifts_by_the_same_offset():
    """The correctness argument in one assertion."""
    p = rj.Plan(make_cal(), 3, present_tick=551)
    assert p.new["home_tick"] - 1054 == p.offset
    assert p.new["min_tick"] - 63 == p.offset
    assert p.new["max_tick"] - 1096 == p.offset


def test_angles_are_unchanged_by_recentring():
    """Ticks shift, radians do not — because home_tick shifted with them.

    This is what makes re-centring safe to do at all: FK, TABLE_Z_IN_BASE and
    every world coordinate keep meaning the same thing afterwards.
    """
    cal = make_cal()
    p = rj.Plan(cal, 3, present_tick=551)

    before = [ticks_to_rad(cal, 3, t) for t in (63, 551, 1096, 1054)]
    after_cal = make_cal(home=p.new["home_tick"], lo=p.new["min_tick"],
                         hi=p.new["max_tick"])
    after = [ticks_to_rad(after_cal, 3, t + p.offset)
             for t in (63, 551, 1096, 1054)]

    for b, a in zip(before, after):
        assert a == pytest.approx(b, abs=1e-12)


def test_plan_refuses_a_joint_with_no_measured_travel():
    """Without a range there is no middle, so no sensible place to put the seam."""
    cal = make_cal()
    del cal["3"]["min_tick"]
    with pytest.raises(SystemExit, match="find_joint_limits"):
        rj.Plan(cal, 3, present_tick=551)


# --- rescuing a joint whose reading has already wrapped ---------------------
#
# J3 crossed the seam during a visual-servo run on 2026-08-05 and came back
# reading 4079. That joint cannot be driven to the middle of its travel first --
# crossing back is the one thing the servo will not do -- so the seam has to be
# redefined exactly where it stands. The offset then has to be measured against
# the UNWRAPPED position, because the stored limits live in a numbering the raw
# reading is a whole encoder turn away from. Getting that wrong shifts every
# limit by 4096 and corrupts the calibration silently.

WRAPPED_J3 = 4079            # what the servo actually reported


def test_a_wrapped_reading_is_recognised_as_such():
    p = rj.Plan(make_cal(), 3, present_tick=WRAPPED_J3)
    assert p.wrapped
    assert p.present_unwrapped == WRAPPED_J3 - 4096
    assert p.here, "a wrapped joint cannot be driven anywhere first"


def test_a_wrapped_joint_is_recentred_where_it_stands():
    p = rj.Plan(make_cal(), 3, present_tick=WRAPPED_J3)
    assert p.anchor == p.present_unwrapped
    assert p.offset == rj.MIDPOINT_TICK - (WRAPPED_J3 - 4096)


def test_the_wrapped_offset_does_not_shift_the_limits_by_an_encoder_turn():
    """The trap: `2048 - 4079` is off by 4096 from the offset the limits need.

    Both are 'correct' modulo the encoder span, but only one keeps the stored
    ticks inside 0..4095 where they can be commanded.
    """
    p = rj.Plan(make_cal(), 3, present_tick=WRAPPED_J3)
    for key, value in p.new.items():
        assert 0 <= value <= TICK_MAX, f"{key} landed outside the encoder"


def test_recentring_in_place_still_clears_the_seam():
    """Less symmetric than centring on the middle of travel, but it must still
    buy back an order of magnitude of clearance — otherwise it is not worth the
    EEPROM write."""
    p = rj.Plan(make_cal(), 3, present_tick=WRAPPED_J3)
    assert p.clearance > 10 * p.old_clearance


def test_angles_survive_a_wrapped_recentre_too():
    """The same correctness argument, in the case where the arithmetic is hard."""
    cal = make_cal()
    p = rj.Plan(cal, 3, present_tick=WRAPPED_J3)
    after_cal = make_cal(home=p.new["home_tick"], lo=p.new["min_tick"],
                         hi=p.new["max_tick"])

    for tick in (63, 551, 1096, 1054):
        assert ticks_to_rad(after_cal, 3, tick + p.offset) == pytest.approx(
            ticks_to_rad(cal, 3, tick), abs=1e-12)


def test_the_joint_is_still_below_its_range_afterwards_and_says_so():
    """Re-centring moves the SEAM, not the joint. J3 was 80 ticks under its
    minimum before and is 80 ticks under it after — the difference is that
    nothing is in the way of walking it back now."""
    p = rj.Plan(make_cal(), 3, present_tick=WRAPPED_J3)
    assert not p.in_range_after
    assert p.new["min_tick"] - rj.MIDPOINT_TICK == 63 - p.present_unwrapped


def test_the_offset_is_measured_to_the_midpoint_not_to_the_settled_reading():
    """Sag while limp is real motion and must NOT be folded into the offset.

    The one-key command defines the pose AT THE INSTANT OF THE WRITE as 2048;
    the joint is still where it started then, ~50 ms of limp later. It then sags
    before torque returns, and the post-write reading includes that sag. Using
    `after - present` as the offset would rotate home_tick by the size of the
    sag and put a permanent bias into every angle the joint reports — invisible,
    because the file would still be perfectly self-consistent.

    Real numbers from 2026-08-05: J3 read 4079 (unwrapped -17) before, and 2035
    after. The offset is 2065, not 2052.
    """
    p = rj.Plan(make_cal(), 3, present_tick=WRAPPED_J3)
    settled_after_sag = 2035

    assert p.offset == rj.MIDPOINT_TICK - p.present_unwrapped == 2065
    assert settled_after_sag - p.present_unwrapped == 2052, "the tempting wrong one"


# --- repairing a half-applied re-centre -------------------------------------

def test_apply_offset_shifts_all_three_ticks_and_leaves_angles_alone(tmp_path):
    """The bookkeeping half on its own, for when the EEPROM write landed and the
    file write did not — which is exactly what happened on 2026-08-05."""
    cal = make_cal()
    before = {k: cal["3"][k] for k in ("home_tick", "min_tick", "max_tick")}
    path = tmp_path / "servo_calibration.json"
    save_calibration_file(cal, path)

    loaded = load_calibration_file(path)
    rj.apply_offset(loaded, 3, 2065, path, assume_yes=True,
                    limits_path=tmp_path / "joint_limits_rad.json")

    written = load_calibration_file(path)["3"]
    for key, old in before.items():
        assert written[key] == old + 2065
    assert "recentred" in written

    after_cal = make_cal(home=written["home_tick"], lo=written["min_tick"],
                         hi=written["max_tick"])
    for tick in (63, 551, 1096):
        assert ticks_to_rad(after_cal, 3, tick + 2065) == pytest.approx(
            ticks_to_rad(cal, 3, tick), abs=1e-12)


def test_apply_offset_touches_no_servo(tmp_path, monkeypatch):
    """It must be usable when the arm is in a state nothing should command.

    Importing or constructing a ServoBus would open a serial port; the repair
    path is deliberately file-only, because the joint it repairs is one whose
    every reported angle is currently wrong.
    """
    def explode(*a, **k):
        raise AssertionError("apply_offset must not touch the bus")

    monkeypatch.setattr(rj.ServoBus, "__init__", explode)
    cal = make_cal()
    path = tmp_path / "servo_calibration.json"
    save_calibration_file(cal, path)
    rj.apply_offset(load_calibration_file(path), 3, 2065, path, assume_yes=True,
                    limits_path=tmp_path / "joint_limits_rad.json")


def test_apply_offset_refuses_an_offset_that_falls_off_the_encoder(tmp_path):
    """A whole-turn error is the likely mistake here, and it is catchable: it
    pushes a stored tick outside 0..4095, where it can never be commanded."""
    cal = make_cal()
    path = tmp_path / "servo_calibration.json"
    save_calibration_file(cal, path)
    with pytest.raises(SystemExit, match="outside"):
        rj.apply_offset(load_calibration_file(path), 3, 2065 - 4096, path,
                        assume_yes=True,
                        limits_path=tmp_path / "joint_limits_rad.json")


def test_here_can_be_asked_for_without_a_wrap():
    """An operator may not want to move a joint at all, wrapped or not."""
    p = rj.Plan(make_cal(), 3, present_tick=551, here=True)
    assert p.here and not p.wrapped
    assert p.anchor == 551
    assert p.in_range_after, "551 is inside the range, so it stays inside"


# --- the two limit files staying in sync ------------------------------------

def test_angle_limits_follow_the_tick_limits():
    cal = make_cal()
    limits = angle_limits(cal)
    assert set(limits) == {"3"}, "only joints with measured ticks appear"
    lo = joint_angle_rad(cal, 3, 63)
    hi = joint_angle_rad(cal, 3, 1096)
    assert limits["3"]["min_rad"] == pytest.approx(min(lo, hi))
    assert limits["3"]["max_rad"] == pytest.approx(max(lo, hi))


def test_angle_limits_sorted_despite_a_negative_dir_sign():
    """dir_sign -1 swaps which tick end is the larger angle; min_rad must still
    be the smaller number or MATLAB's PositionLimits are inverted."""
    cal = make_cal(dir_sign=-1)
    limits = angle_limits(cal)
    assert limits["3"]["min_rad"] < limits["3"]["max_rad"]


def test_angle_limits_are_identical_before_and_after_recentring():
    """The two files must agree, and re-centring must not disturb the angles.

    A mismatch here is the nastiest failure mode available: MATLAB would solve
    against one range while ServoBus enforced another, so IK returns a solution
    the bus refuses and the run dies mid-move with nothing obviously wrong.
    """
    cal = make_cal()
    before = angle_limits(cal)

    p = rj.Plan(cal, 3, present_tick=551)
    after = angle_limits(make_cal(home=p.new["home_tick"], lo=p.new["min_tick"],
                                  hi=p.new["max_tick"]))

    assert after["3"]["min_rad"] == pytest.approx(before["3"]["min_rad"])
    assert after["3"]["max_rad"] == pytest.approx(before["3"]["max_rad"])


def test_write_angle_limits_round_trips(tmp_path):
    cal = make_cal()
    out = tmp_path / "joint_limits_rad.json"
    path, limits = write_angle_limits(cal, out)
    assert path == out
    assert json.loads(out.read_text(encoding="utf-8")) == limits


# --- the encoding bug that bit twice ----------------------------------------

def test_calibration_round_trip_preserves_non_ascii_notes(tmp_path):
    """Windows' default encoding is cp1252, and these files carry prose notes.

    A default-encoding read-modify-write mangles every em-dash into mojibake and
    bakes it in. It happened twice on 2026-08-05, once to J1's dir_sign_basis.
    """
    cal = make_cal()
    cal["3"]["limit_basis"] = "ground collision — not mechanical; re-measure ±5°"
    path = tmp_path / "servo_calibration.json"

    save_calibration_file(cal, path)
    reloaded = load_calibration_file(path)
    assert reloaded["3"]["limit_basis"] == cal["3"]["limit_basis"]


def test_saved_calibration_keeps_notes_readable_not_escaped(tmp_path):
    cal = make_cal()
    cal["3"]["limit_basis"] = "re-measure — see notes"
    path = tmp_path / "servo_calibration.json"
    save_calibration_file(cal, path)
    assert "—" in path.read_text(encoding="utf-8")

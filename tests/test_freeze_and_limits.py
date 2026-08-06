"""ServoBus freeze retries and the travel-limit margin.

Both changes came out of one live run on 2026-08-05:

  * the run hit a bus error and the freeze that followed reported FREEZE FAILED
    on two of five joints -- the soft e-stop not working at the moment it was
    called for, while those joints were still travelling to their old goals;
  * a re-centre was refused for taking J2 twelve ticks past a limit that is
    recorded as GROUND-DERIVED, i.e. measured where the claw met the table at
    one particular elbow angle, not where the joint runs out of travel.

The freeze tests matter most. Every other safety mechanism in this driver
refuses to do something; freeze is the only one that has to successfully DO
something, on a bus that is by then already misbehaving.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vision_pipeline import config  # noqa: E402
from vision_pipeline.robot_interface.servo_calibration import (  # noqa: E402
    unwrap_tick,
    wrapped_past_seam,
)
from vision_pipeline.robot_interface.servo_driver import (  # noqa: E402
    ServoBus,
    ServoSafetyError,
)


class FlakyBus(ServoBus):
    """A ServoBus with no serial port, whose reads fail a scripted number of times."""

    def __init__(self, positions, fail_reads=None, limit_margin_ticks=0):
        self._calibration = {}
        self.serial = None
        self.limit_margin_ticks = max(0, int(limit_margin_ticks))
        self._load_calibration(str(ROOT / "data" / "servo_calibration.json"))
        self.positions = dict(positions)
        self.fail_reads = dict(fail_reads or {})   # servo_id -> failures remaining
        self.goals = {}
        self.read_attempts = {sid: 0 for sid in positions}

    def read_position(self, servo_id):
        self.read_attempts[servo_id] = self.read_attempts.get(servo_id, 0) + 1
        if self.fail_reads.get(servo_id, 0) > 0:
            self.fail_reads[servo_id] -= 1
            raise RuntimeError(f"No response reading position from servo {servo_id}")
        return self.positions[servo_id]

    def _write_register(self, servo_id, addr, data):
        if addr == ServoBus.ADDR_GOAL_POSITION:
            self.goals[servo_id] = data[0] | (data[1] << 8)
        return True


# --- freeze -----------------------------------------------------------------

def test_freeze_holds_every_joint_on_a_clean_bus():
    bus = FlakyBus({1: 2000, 2: 2900, 3: 600})
    held = bus.freeze([1, 2, 3])
    assert held == {1: 2000, 2: 2900, 3: 600}
    assert bus.goals == {1: 2000, 2: 2900, 3: 600}


def test_freeze_recovers_from_a_transient_read_failure():
    """The observed failure: one dropped read used to lose the joint entirely."""
    bus = FlakyBus({1: 2000, 2: 2900, 4: 1700}, fail_reads={2: 1, 4: 1})
    held = bus.freeze([1, 2, 4])
    assert set(held) == {1, 2, 4}, "a retry must recover a transient failure"
    assert bus.goals[2] == 2900 and bus.goals[4] == 1700


def test_freeze_pins_the_goal_to_the_present_position():
    """Freezing means overwriting the goal with where the joint already is.

    Writing anything else would command a move, which is the opposite of what
    the caller asked for at the one moment they least want a surprise.
    """
    bus = FlakyBus({3: 640})
    bus.freeze([3])
    assert bus.goals[3] == 640


def test_freeze_still_stops_the_others_when_one_joint_is_dead():
    """A servo that cannot be frozen must not prevent the rest from freezing."""
    bus = FlakyBus({1: 2000, 2: 2900, 5: 2500}, fail_reads={2: 999})
    held = bus.freeze([1, 2, 5])
    assert set(held) == {1, 5}
    assert 2 not in bus.goals


def test_freeze_retries_in_passes_not_per_joint():
    """Ordering property: everything gets one attempt before anything gets two.

    A joint being retried in place would make one unresponsive servo's timeouts
    delay the freeze of every joint after it -- and those joints are still
    moving while that happens.
    """
    bus = FlakyBus({1: 2000, 2: 2900, 3: 600}, fail_reads={1: 1})
    bus.freeze([1, 2, 3])
    # J1 failed once and was retried; J2 and J3 succeeded first time and were
    # never read again, which is only true if the retry was a second PASS.
    assert bus.read_attempts[1] == 2
    assert bus.read_attempts[2] == 1
    assert bus.read_attempts[3] == 1


def test_freeze_gives_up_after_the_configured_attempts():
    bus = FlakyBus({1: 2000}, fail_reads={1: 999})
    assert bus.freeze([1]) == {}
    assert bus.read_attempts[1] == config.SERVO_FREEZE_ATTEMPTS


# --- read retry backoff -----------------------------------------------------

def test_read_retry_waits_between_attempts(monkeypatch):
    """The retry must PAUSE, or it cannot do the job it exists for.

    Reads fail because motor current puts noise on the shared line, in bursts
    lasting longer than a few immediate retries take. A tight loop spends every
    attempt inside one burst and reports a dead servo -- which is exactly what
    happened on 2026-08-05 to a J2 that was holding torque and answered every
    read-only check before and after.
    """
    slept = []
    monkeypatch.setattr("vision_pipeline.robot_interface.servo_driver.time.sleep",
                        slept.append)

    bus = FlakyBus({2: 2900}, fail_reads={2: 2})
    assert bus.read_position_retrying(2) == 2900
    assert len(slept) == 2, "one pause before each retry"
    assert all(s > 0 for s in slept)


def test_read_retry_backoff_grows(monkeypatch):
    """Doubling covers a short burst quickly and a long one eventually, without
    guessing in advance how long the noise lasts."""
    slept = []
    monkeypatch.setattr("vision_pipeline.robot_interface.servo_driver.time.sleep",
                        slept.append)

    bus = FlakyBus({2: 2900}, fail_reads={2: 3})
    bus.read_position_retrying(2)
    assert slept == sorted(slept)
    assert slept[-1] > slept[0]


def test_read_retry_backoff_is_capped(monkeypatch):
    """Doubling must not run away. At 15 attempts an uncapped backoff would
    wait 20 ms * 2^14 on the last one — over five minutes for a single read."""
    slept = []
    monkeypatch.setattr("vision_pipeline.robot_interface.servo_driver.time.sleep",
                        slept.append)

    bus = FlakyBus({2: 2900}, fail_reads={2: config.SERVO_MOVE_READ_RETRIES - 1})
    bus.read_position_retrying(2)
    assert max(slept) <= config.SERVO_READ_RETRY_MAX_S
    assert sum(slept) < 5.0, "the whole retry sequence must fit inside a move"


def test_read_retry_does_not_sleep_when_the_first_read_works():
    """No cost on the overwhelmingly common path."""
    slept = []
    bus = FlakyBus({2: 2900})
    import vision_pipeline.robot_interface.servo_driver as sd
    real_sleep, sd.time.sleep = sd.time.sleep, slept.append
    try:
        assert bus.read_position_retrying(2) == 2900
    finally:
        sd.time.sleep = real_sleep
    assert slept == []


def test_plain_read_position_is_not_retried():
    """ping and scan_ids depend on 'no answer' meaning 'not there' — retrying
    inside read_position would make an absent servo look merely slow."""
    bus = FlakyBus({2: 2900}, fail_reads={2: 1})
    with pytest.raises(RuntimeError):
        bus.read_position(2)


# --- travel-limit margin ----------------------------------------------------

def test_margin_widens_limits_at_both_ends():
    tight = FlakyBus({2: 2900})
    loose = FlakyBus({2: 2900}, limit_margin_ticks=60)
    lo, hi = tight.travel_limits(2)
    mlo, mhi = loose.travel_limits(2)
    assert (mlo, mhi) == (lo - 60, hi + 60)


def test_margin_admits_a_move_just_outside_the_range():
    """The shape of the 2026-08-05 refusal: a target a few ticks past min_tick.

    Derived from whatever J2's limits currently are, not written as literals.
    They have already moved twice (2883 -> 2600 when the ground-derived range
    was widened), and a hardcoded tick silently stops testing anything the next
    time a limit is re-measured.
    """
    lo, _hi = FlakyBus({2: 3000}).travel_limits(2)
    start, target = lo + 23, lo - 12          # inside -> just outside

    with pytest.raises(ServoSafetyError):
        FlakyBus({2: start})._check_travel_limits(2, start, target)

    loose = FlakyBus({2: start}, limit_margin_ticks=60)
    loose._check_travel_limits(2, start, target)   # must not raise


def _below_range(bus, joint, by=10):
    """A tick that is definitely outside `joint`'s range, whatever it is now.

    Derived rather than hardcoded: these tests are about the RULE, not about
    today's numbers, and J3's limit has already moved once (200 -> 63 when the
    seam margin was cut from 18 deg to 5 deg on 2026-08-05). A literal tick
    would silently stop testing anything the next time a limit is re-measured.
    """
    lo, _hi = bus.travel_limits(joint)
    return lo - by


def test_a_joint_outside_its_range_may_always_hold_still():
    """The rescue case, found on hardware 2026-08-05.

    J3 had sagged just below its limit. hold_pose asked it to stay exactly where
    it was and the check refused, because equal violations are not a STRICT
    improvement. With no goal written the joint kept falling, and each further
    attempt to catch it was refused for being further out — the limit was
    preventing the rescue of the joint it had trapped. Zero commanded motion
    cannot violate a travel limit, so it is always allowed.
    """
    bus = FlakyBus({3: 600})
    out = _below_range(bus, 3)
    bus._check_travel_limits(3, out, out)     # must not raise


def test_holding_still_inside_the_range_is_allowed_too():
    bus = FlakyBus({3: 600})
    bus._check_travel_limits(3, 600, 600)


def test_a_joint_outside_its_range_may_move_back_toward_it():
    bus = FlakyBus({3: 600})
    out = _below_range(bus, 3)
    bus._check_travel_limits(3, out, out + 5)


def test_a_joint_outside_its_range_may_not_go_further_out():
    bus = FlakyBus({3: 600})
    out = _below_range(bus, 3)
    with pytest.raises(ServoSafetyError):
        bus._check_travel_limits(3, out, out - 10)


def test_margin_does_not_invent_limits_for_an_unmeasured_joint():
    """A margin must not conjure a range — an invented one reads as protection
    while permitting the moves it appears to forbid.

    Uses whichever joint genuinely has no recorded limits rather than naming
    one: J1 had none until 2026-08-05, when a working envelope was added to it
    after two runaways, and this test silently stopped testing anything.
    """
    bus = FlakyBus({j: 2000 for j in range(1, 7)}, limit_margin_ticks=200)
    unmeasured = [j for j in range(1, 7) if bus.travel_limits(j) is None]
    assert unmeasured, "no unmeasured joint left to test the rule with"
    for j in unmeasured:
        assert bus.travel_limits(j) is None


def test_margin_is_still_a_limit():
    """Widened is not removed: far outside the widened range is still refused."""
    loose = FlakyBus({2: 2900}, limit_margin_ticks=60)
    with pytest.raises(ServoSafetyError):
        loose._check_travel_limits(2, 2900, 2000)


def test_zero_margin_is_the_default():
    """Nobody gets a wider range by accident — it has to be asked for.

    Asserts the RULE (limits pass through untouched), not today's tick values.
    """
    assert config.SERVO_LIMIT_MARGIN_TICKS == 0
    bus = FlakyBus({2: 2900})
    entry = bus._cal(2)
    assert bus.travel_limits(2) == (entry["min_tick"], entry["max_tick"])


# --- the encoder seam -------------------------------------------------------
#
# On 2026-08-05 J3 crossed the 0/4095 seam during a visual-servo run and came
# back reading 4079. Every check above then misread the situation: raw
# arithmetic put the joint ~3000 ticks PAST its maximum when it was really 80
# ticks BELOW its minimum, so the direction that recovers it looked like the
# direction that makes it worse, and the error message told the operator to
# re-measure a range that was perfectly correct.

def test_unwrap_reads_a_seam_crossing_as_the_small_negative_it_is():
    assert unwrap_tick(4079, 579) == -17
    assert unwrap_tick(20, 4000) == 4116
    assert unwrap_tick(600, 579) == 600        # nothing to undo


def test_unwrap_is_the_identity_well_away_from_the_seam():
    for tick in range(0, 4096, 97):
        assert unwrap_tick(tick, tick) == tick


def test_wrapped_past_seam_distinguishes_wrapped_from_merely_out_of_range():
    assert wrapped_past_seam(4079, 63, 1096)       # the real J3 case
    assert not wrapped_past_seam(53, 63, 1096)     # 10 ticks under, walkable
    assert not wrapped_past_seam(1150, 63, 1096)   # over the top, walkable


def test_a_wrapped_joint_is_refused_with_the_seam_named():
    """The refusal must say WHAT is wrong, because the two remedies are opposite.

    A joint merely out of range gets walked back. A wrapped one cannot be —
    the servo drives goals linearly and would take the long way round, through
    every hard stop between here and there. The old message blamed the travel
    range and invited re-measuring it, which would have been the wrong repair
    applied to a correct number.
    """
    bus = FlakyBus({3: 4079})
    with pytest.raises(ServoSafetyError) as exc:
        bus._check_travel_limits(3, 4079, 4095)
    message = str(exc.value)
    assert "WRAPPED" in message
    assert "-17" in message                     # where it really is
    assert "recentre_joint" in message          # the remedy that works
    assert "do not re-measure" in message       # the remedy that does not


def test_a_wrapped_joint_may_still_hold_still():
    """The seam refusal must not re-create the trap it sits next to.

    Holding writes the joint's own present position as its goal: no motion, so
    nothing for a travel limit or a seam to protect against, and it is precisely
    what freeze and hold_pose do when an operator is trying to catch a falling
    arm. Refusing that would repeat the sag bug one line above, in a case where
    the joint is in worse shape.
    """
    bus = FlakyBus({3: 4079})
    bus._check_travel_limits(3, 4079, 4079)      # must not raise


def test_a_wrapped_joint_is_refused_in_both_directions():
    """Neither tick direction is 'back toward range' across a seam."""
    bus = FlakyBus({3: 4079})
    for target in (4090, 4060):
        with pytest.raises(ServoSafetyError, match="WRAPPED"):
            bus._check_travel_limits(3, 4079, target)

"""What the arm did after it stopped being able to see, and the log of it.

THE MEASUREMENT THIS RESTS ON. The camera sits above and behind the claw, so the
brick leaves the bottom of the frame at a height that depends on where its top
surface actually is. That moment is therefore a measurement of the brick's
height -- and the only thing it needs from FK is DIFFERENCES, which is the half
of FK this arm is good at. Every property below is a difference from the tip at
loss of sight; none of them is an absolute height. A test that starts asserting
absolute z here has lost the plot.

Pure: no arm, no camera, no file except a tmp_path.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vision_pipeline.planning import blind_travel as bt  # noqa: E402


def journey(lost_z=0.100, steps=(), outcome=bt.GRIPPED, flat=False,
            grip=None, lost_xy=(0.150, 0.020)):
    j = bt.BlindJourney(lost_tip=(lost_xy[0], lost_xy[1], lost_z),
                        flat_on_board=flat)
    for z in steps:
        j.step((lost_xy[0], lost_xy[1], z))
    j.finish(outcome, grip)
    return j


# --- what one journey measured -----------------------------------------------

def test_the_drop_is_measured_from_loss_of_sight_not_from_the_table():
    j = journey(lost_z=0.100, steps=(0.092, 0.084, 0.076))
    assert j.drop_mm == pytest.approx(24.0)


def test_the_drop_ends_where_the_claw_CLOSED_when_there_was_a_grip():
    """The grip is the answer; the last blind step is only where the arm got
    to before it. They differ whenever the claw settled short."""
    j = journey(lost_z=0.100, steps=(0.092, 0.084),
                grip=(0.150, 0.020, 0.081))
    assert j.drop_mm == pytest.approx(19.0)


def test_a_journey_with_no_steps_measured_nothing():
    j = journey(lost_z=0.100, steps=())
    assert j.drop_mm == pytest.approx(0.0)
    assert j.travel_mm == pytest.approx(0.0)


def test_sideways_wander_is_reported_separately_from_the_drop():
    """A blind descent asks for the same x and y with a lower z, so travel much
    larger than drop means the arm went sideways while it believed it was going
    straight down. Worth seeing, hence two numbers rather than one."""
    j = bt.BlindJourney(lost_tip=(0.150, 0.020, 0.100))
    j.step((0.153, 0.024, 0.090))
    j.finish(bt.GRIPPED)

    assert j.drop_mm == pytest.approx(10.0)
    assert j.sideways_mm == pytest.approx(5.0)
    assert j.travel_mm == pytest.approx((10 ** 2 + 5 ** 2) ** 0.5)


def test_describe_says_what_happened():
    j = journey(lost_z=0.100, steps=(0.090,), outcome=bt.AIR)
    text = j.describe()
    assert "lost sight at z +100.0 mm" in text
    assert "10.0 mm" in text
    assert "shut on air" in text


# --- the suggestion ----------------------------------------------------------

def test_only_journeys_that_GRIPPED_inform_the_suggestion():
    """One that shut on air measured where the brick ISN'T. Useful to a human
    reading the log, actively misleading as an input to this."""
    js = [journey(steps=(0.090,), outcome=bt.AIR),          # 10 mm, missed
          journey(steps=(0.070,), outcome=bt.GRIPPED),      # 30 mm, worked
          journey(steps=(0.068,), outcome=bt.GRIPPED)]      # 32 mm, worked
    assert bt.suggest_drop_mm(js) == pytest.approx(31.0)


def test_the_median_not_the_mean():
    """A run whose blind descent was refused by the floor guard contributes a
    short drop and one where the brick had been knocked over a long one. Both
    are real records worth keeping and neither should drag the estimate."""
    js = [journey(steps=(0.099,)),      # 1 mm  -- refused early
          journey(steps=(0.075,)),      # 25 mm
          journey(steps=(0.074,)),      # 26 mm
          journey(steps=(0.000,))]      # 100 mm -- something went wrong
    assert bt.suggest_drop_mm(js) == pytest.approx(25.5)


def test_no_history_returns_None_not_a_default():
    """Deliberately. A caller that has to handle None will say out loud that it
    is guessing; one handed a number will not."""
    assert bt.suggest_drop_mm([]) is None
    assert bt.suggest_drop_mm([journey(outcome=bt.AIR)]) is None


def test_flat_and_raised_journeys_are_kept_apart():
    """They are answers to different questions -- a flat brick's loss-of-sight
    height and a raised one's are not the same measurement."""
    js = [journey(steps=(0.090,), flat=True),      # 10 mm
          journey(steps=(0.060,), flat=False)]     # 40 mm
    assert bt.suggest_drop_mm(js, flat_on_board=True) == pytest.approx(10.0)
    assert bt.suggest_drop_mm(js, flat_on_board=False) == pytest.approx(40.0)
    assert bt.suggest_drop_mm(js) == pytest.approx(25.0)


# --- the file ----------------------------------------------------------------

def test_a_journey_survives_a_round_trip(tmp_path):
    path = tmp_path / "blind_travel.json"
    bt.append(path, journey(lost_z=0.100, steps=(0.090, 0.080)))
    bt.append(path, journey(lost_z=0.120, steps=(0.100,)))

    back = bt.load(path)
    assert len(back) == 2
    assert back[0].drop_mm == pytest.approx(20.0)
    assert back[1].drop_mm == pytest.approx(20.0)


def test_a_missing_file_is_no_history_not_an_error(tmp_path):
    assert bt.load(tmp_path / "nothing.json") == []


def test_a_CORRUPT_file_is_no_history_not_an_error(tmp_path):
    """This is an advisory log. Refusing to run a pick because a JSON file is
    malformed would be the tail wagging the dog."""
    path = tmp_path / "blind_travel.json"
    path.write_text("{not json at all")
    assert bt.load(path) == []


def test_one_unreadable_entry_does_not_lose_the_others(tmp_path):
    path = tmp_path / "blind_travel.json"
    good = journey(lost_z=0.100, steps=(0.090,))
    path.write_text(json.dumps({"journeys": [
        {"nonsense": True},
        {k: v for k, v in vars(good).items()},
    ]}))
    back = bt.load(path)
    assert len(back) == 1
    assert back[0].drop_mm == pytest.approx(10.0)


def test_appending_creates_the_directory(tmp_path):
    path = tmp_path / "made" / "up" / "blind_travel.json"
    bt.append(path, journey())
    assert path.exists()


def test_the_summary_says_when_there_is_nothing_to_go_on():
    assert "no blind-travel history" in bt.summarise([])[0]


def test_the_summary_separates_flat_from_raised():
    js = [journey(steps=(0.090,), flat=True), journey(steps=(0.060,), flat=False)]
    text = "\n".join(bt.summarise(js))
    assert "flat on the board" in text and "raised" in text
    assert "10.0 mm" in text and "40.0 mm" in text

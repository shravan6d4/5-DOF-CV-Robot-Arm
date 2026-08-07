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
            grip=None, lost_xy=(0.150, 0.020), lost_sight=True):
    j = bt.BlindJourney(lost_tip=(lost_xy[0], lost_xy[1], lost_z),
                        flat_on_board=flat, lost_sight=lost_sight)
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


# --- the height model: apparent size as depth --------------------------------
#
# THE OBVIOUS CANDIDATE IS WRONG AND THAT IS WHY THIS EXISTS. DescentModel's `a`
# (px per mm of descent) looks like inverse depth. Measured on hardware
# 2026-08-07 it FELL from 4.72 to 3.08 px/mm as the claw came down on the brick;
# parallax would have made it grow. It is dominated by the camera rotating as
# the wrist swings, which is depth-independent. Apparent AREA has none of that:
# rotation-invariant, and sqrt(area) ~ 1/Z.

def seen(j, step, z, area, px=(320.0, 240.0)):
    j.see(step, (0.15, 0.02, z), [[1, 0, 0, 0.15], [0, 1, 0, 0.02],
                                  [0, 0, 1, z + 0.06], [0, 0, 0, 1]], px, area)


def training_journey(grip_z=0.000, sightings=(), flat=True, outcome=bt.GRIPPED):
    j = bt.BlindJourney(lost_tip=(0.15, 0.02, 0.100), flat_on_board=flat,
                        lost_sight=True)
    for i, (z, area) in enumerate(sightings):
        seen(j, i + 1, z, area)
    j.finish(outcome, (0.15, 0.02, grip_z))
    return j


def synthetic(n=12, c=900.0, d=5.0, grip_z=0.0):
    """Sightings obeying remaining = c/sqrt(area) - d exactly."""
    out = []
    for i in range(n):
        remaining = 5.0 + i * 6.0                     # 5..71 mm
        area = (c / (remaining + d)) ** 2
        out.append((grip_z + remaining / 1000.0, area))
    return out


def test_the_model_recovers_the_relationship_it_was_trained_on():
    j = training_journey(sightings=synthetic())
    m = bt.fit_height_model([j])

    assert m.ready
    assert m.c == pytest.approx(900.0, rel=1e-3)
    assert m.d == pytest.approx(5.0, abs=0.5)
    assert m.rms_mm < 0.5


def test_the_model_predicts_the_drop_still_to_come():
    m = bt.fit_height_model([training_journey(sightings=synthetic())])
    area_at_20mm = (900.0 / 25.0) ** 2
    assert m.remaining_mm(area_at_20mm) == pytest.approx(20.0, abs=0.5)


def test_an_unfitted_model_says_nothing_rather_than_guessing():
    m = bt.HeightModel()
    assert not m.ready
    assert m.remaining_mm(1000.0) is None
    assert "not fitted" in m.describe()


def test_too_few_pairs_leaves_the_model_unfitted():
    j = training_journey(sightings=synthetic(n=bt.HeightModel.MIN_PAIRS - 1))
    assert not bt.fit_height_model([j]).ready


def test_sightings_that_all_look_the_SAME_SIZE_do_not_fit():
    """The slope is unconstrained there, and lstsq would return one anyway. A
    fit over a flat spread is noise wearing a confident face -- the same failure
    as the hand-eye capture whose rotations were all too small."""
    j = training_journey(sightings=[(0.05, 4000.0)] * 20)
    m = bt.fit_height_model([j])
    assert not m.ready
    assert m.n == 20, "the pairs are still counted, just not fitted"


def test_only_journeys_that_gripped_train_the_model():
    """A run that shut on air has no ground truth for where the brick was."""
    j = training_journey(sightings=synthetic(), outcome=bt.AIR)
    assert not bt.fit_height_model([j]).ready
    assert bt.training_pairs([j]) == []


def test_flat_and_raised_runs_BOTH_train_the_model():
    """Unlike the drop suggestion. How big the brick looks versus how far there
    is to go is a property of the camera and the brick, not of which answer the
    operator gave -- and the flat runs are the ones that work, so excluding them
    would starve the model that the raised path depends on."""
    flat = training_journey(sightings=synthetic(n=6), flat=True)
    raised = training_journey(sightings=synthetic(n=6), flat=False)
    assert bt.fit_height_model([flat, raised]).ready


def test_a_sighting_survives_the_round_trip(tmp_path):
    path = tmp_path / "blind_travel.json"
    bt.append(path, training_journey(sightings=synthetic(n=3)))
    back = bt.load(path)
    assert len(back[0].sightings) == 3
    assert isinstance(back[0].sightings[0], bt.Sighting)
    assert len(back[0].sightings[0].wrist) == 4


# --- triangulation as a cross-check, never as control ------------------------

class FakeCalibrator:
    """Answers with a fixed point, so the pairing logic is what is tested."""

    def __init__(self, z=0.010, parallax=7.0, residual=0.002):
        import numpy as np
        self.np = np
        self.z, self.parallax, self.residual = z, parallax, residual
        self.calls = []

    def triangulate_pixels(self, views):
        self.calls.append(views)

        class R:
            pass
        r = R()
        r.point_base = self.np.array([0.15, 0.02, self.z])
        r.parallax_deg = self.parallax
        r.residual_m = self.residual
        return r


def test_triangulation_only_pairs_steps_far_enough_apart():
    """Adjacent 8 mm steps give 2.3 deg of parallax at 200 mm, under the gate.
    Three steps apart is 24 mm and 6.8 deg, which passes -- so the pairing is
    by step gap, not by consecutive frames."""
    j = training_journey(sightings=synthetic(n=5))
    cal = FakeCalibrator()
    out = bt.triangulate_sightings(j, cal, min_step_gap=3)

    assert len(out) == 3, [e["steps"] for e in out]
    for e in out:
        assert e["steps"][1] - e["steps"][0] >= 3


def test_triangulation_is_scored_against_the_grip():
    j = training_journey(grip_z=0.0, sightings=synthetic(n=5))
    out = bt.triangulate_sightings(j, FakeCalibrator(z=0.010), min_step_gap=3)
    assert all(e["vs_grip_mm"] == pytest.approx(10.0) for e in out)


def test_a_weak_pair_is_recorded_but_not_accepted():
    """Recorded, because a rejected pair is evidence about the capture. Not
    accepted, because the depth from near-parallel rays is noise."""
    j = training_journey(sightings=synthetic(n=5))
    out = bt.triangulate_sightings(j, FakeCalibrator(parallax=1.0),
                                   min_step_gap=3)
    assert out and not any(e["accepted"] for e in out)


def test_a_calibrator_that_explodes_does_not_take_the_run_with_it():
    class Broken:
        def triangulate_pixels(self, views):
            raise RuntimeError("no intrinsics")

    j = training_journey(sightings=synthetic(n=5))
    assert bt.triangulate_sightings(j, Broken(), min_step_gap=3) == []


def test_the_verdict_names_the_disagreement_when_there_is_one():
    j = training_journey(grip_z=0.0, sightings=synthetic(n=5))
    j.triangulated = bt.triangulate_sightings(j, FakeCalibrator(z=0.052),
                                              min_step_gap=3)
    text = "\n".join(bt.triangulation_verdict([j]))
    assert "DISAGREES" in text
    assert "held-out" in text


def test_the_verdict_says_so_when_triangulation_agrees():
    j = training_journey(grip_z=0.0, sightings=synthetic(n=5))
    j.triangulated = bt.triangulate_sightings(j, FakeCalibrator(z=0.002),
                                              min_step_gap=3)
    text = "\n".join(bt.triangulation_verdict([j]))
    assert "agrees with the grip" in text


def test_the_verdict_is_honest_about_having_nothing():
    assert "no accepted pairs" in bt.triangulation_verdict([])[0]


def test_the_gates_come_from_config_not_from_hardcoded_twins():
    """The cross-check and PickPipeline.locate_brick_two_view must not drift
    into disagreeing about what 'trustworthy' means -- they end at the same
    triangulate_pixels call and should accept the same pairs."""
    from vision_pipeline import config

    j = training_journey(sightings=synthetic(n=5))
    just_under = bt.triangulate_sightings(
        j, FakeCalibrator(parallax=config.TWO_VIEW_MIN_PARALLAX_DEG - 0.1),
        min_step_gap=3)
    just_over = bt.triangulate_sightings(
        j, FakeCalibrator(parallax=config.TWO_VIEW_MIN_PARALLAX_DEG + 0.1),
        min_step_gap=3)

    assert not any(e["accepted"] for e in just_under)
    assert all(e["accepted"] for e in just_over)

    too_far = bt.triangulate_sightings(
        j, FakeCalibrator(residual=config.TWO_VIEW_MAX_RESIDUAL_M * 2),
        min_step_gap=3)
    assert not any(e["accepted"] for e in too_far)


def test_both_triangulation_paths_end_at_the_same_call():
    """One engine, two callers. If this ever stops being true, the cross-check
    is checking something other than what the pick would use."""
    import inspect
    from vision_pipeline import pipeline

    assert "triangulate_pixels" in inspect.getsource(
        pipeline.PickPipeline.locate_brick_two_view)
    assert "triangulate_pixels" in inspect.getsource(bt.triangulate_sightings)

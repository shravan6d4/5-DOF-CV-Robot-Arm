"""Tests for the closed-loop visual servo control law.

Fully synthetic — no camera, no serial port, no arm. That is the point of
keeping the control law as pure functions: this is the first code in the repo
that moves the arm repeatedly on its own judgement between frames, so its
behaviour under a WRONG measurement matters as much as under a right one, and
only a test can exercise the wrong cases safely.

The load-bearing tests here are the refusal ones. A proportional loop that
converges when everything is correct is easy; what makes this safe to point at
real hardware is that it stops when the sign is wrong, when the joint is stuck,
and when the probe measured nothing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline.planning.visual_servo import (  # noqa: E402
    ProgressMonitor,
    ServoAbort,
    clamp_predicted_shift,
    effective_deadband,
    enforce_minimum_step,
    estimate_axis,
    pixel_error,
    resolution_px,
    step_ticks,
)
from vision_pipeline.planning.visual_servo import DescentModel  # noqa: E402


# --- the descent model ------------------------------------------------------
#
# The fix for the 2026-08-05 run where descending and re-aiming fought over the
# same degree of freedom. Descending swings the camera (~7 px/mm); correcting
# the swing by jogging the elbow raised the tip by more than the step had
# gained. Four steps produced 4 mm of net descent and then lost the brick.
#
# Numbers below are taken from that log: a 20 mm descent moved the brick about
# 140 px UP the frame. Image y grows downward, so that is dpx = -140 for a
# dz of -20 mm, giving a = +7 px/mm. Both signs negative, so the coefficient is
# positive — worth stating, because the intuition "descending pushes the brick
# up, so the gain is negative" is wrong and cost a round of failing tests.


def fit_from(a, b, moves):
    """A model fitted from a known ground truth, the way the arm would supply it."""
    model = DescentModel()
    for dz, dr in moves:
        model.observe(dz, dr, a * dz + b * dr)
    return model


def test_two_independent_moves_recover_both_coefficients():
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    assert model.ready
    assert model.a == pytest.approx(7.0)
    assert model.b == pytest.approx(2.5)


def test_two_pure_descents_cannot_fit_the_reach_term():
    """The trap. Both samples lie on the same line through the origin, so no
    amount of descending says anything about reaching out. lstsq would happily
    return SOME b; the rank check must refuse instead, because a fabricated b
    goes straight into a Cartesian target."""
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-8.0, 0.0)])
    assert not model.ready
    with pytest.raises(ServoAbort, match="not been fitted"):
        model.radial_for(error_px=-100.0, dz_mm=-20.0)


def test_the_correction_anticipates_the_swing_the_descent_will_cause():
    """The whole point: aim at where the brick WILL be, not where it is.

    With the brick 140 px above the aim point and a 20 mm descent about to push
    it a further 140 px up, the reach must cancel BOTH — not just the error
    that is visible right now.
    """
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    dr = model.radial_for(error_px=-140.0, dz_mm=-20.0)

    residual = -140.0 + model.a * -20.0 + model.b * dr
    assert residual == pytest.approx(0.0, abs=1e-9)
    assert dr == pytest.approx((140.0 + 140.0) / 2.5)


def test_ignoring_the_descent_term_would_undercorrect_by_half():
    """Guards the failure the old two-loop version had, in one number.

    A correction that only cancels the visible error leaves the descent's own
    contribution uncancelled — which is exactly the disturbance the next
    iteration then had to chase, and did not catch.
    """
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    anticipating = model.radial_for(error_px=-140.0, dz_mm=-20.0)
    naive = 140.0 / 2.5           # cancels only what is currently visible
    assert anticipating == pytest.approx(2 * naive)


def test_the_fit_tracks_a_gain_that_changes_as_the_arm_unfolds():
    """The coupling weakens as reach grows, so the model must not freeze after
    its first pair — a gain fitted at 60 mm of reach is wrong at 160 mm."""
    model = DescentModel()
    for dz, dr in [(-20.0, 0.0), (-20.0, 8.0)]:
        model.observe(dz, dr, 7.0 * dz + 2.5 * dr)
    steep = model.a
    for dz, dr in [(-20.0, 0.0), (-20.0, 8.0), (-20.0, -8.0)]:
        model.observe(dz, dr, 2.0 * dz + 2.5 * dr)
    assert abs(model.a) < abs(steep), "must move toward the newer, gentler gain"


def test_a_reach_that_does_not_move_the_image_is_refused_not_inverted():
    """b ~ 0 means reaching out does nothing to the vertical error, so there is
    no correction to compute. Dividing by it would manufacture an enormous
    Cartesian target out of noise."""
    model = fit_from(7.0, 0.0, [(-20.0, 0.0), (-20.0, 8.0)])
    assert not model.ready


# --- plan_step: descend only as fast as the aim can be held ------------------

def test_a_step_too_big_to_stay_aimed_is_shrunk_to_nothing():
    """The rate limit, in the numbers that exposed it.

    a=7, b=2.5: a 20 mm descent injects 140 px of error, and 25 mm of reach can
    only remove 62 px. Taking that step ends further from the target than it
    began — do it repeatedly and the brick leaves the frame, which is what
    happened. With the brick already 140 px off, the honest answer is to descend
    zero this step and spend it on the aim.
    """
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    dz, dr = model.plan_step(error_px=-140.0, desired_dz_mm=-20.0,
                             max_reach_mm=25.0)
    assert dz == pytest.approx(0.0)
    assert dr == pytest.approx(25.0)


def test_descent_resumes_once_the_error_is_small_enough_to_carry():
    """Holding height is temporary, not a deadlock — the point of shrinking the
    step is that progress restarts by itself."""
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    dz, dr = model.plan_step(error_px=-15.0, desired_dz_mm=-20.0,
                             max_reach_mm=25.0)
    assert dz < -1.0, "a small error must permit real descent"
    assert abs(-15.0 + model.a * dz + model.b * dr) == pytest.approx(0.0, abs=1e-6)


def test_a_well_aimed_arm_takes_the_full_step():
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    dz, _dr = model.plan_step(error_px=0.0, desired_dz_mm=-5.0, max_reach_mm=25.0)
    assert dz == pytest.approx(-5.0)


def test_the_plan_never_climbs_to_improve_the_aim():
    """Ascending would fix the image beautifully and defeat the entire purpose."""
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    for error in (-400.0, -140.0, 0.0, 140.0, 400.0):
        dz, _dr = model.plan_step(error, desired_dz_mm=-20.0, max_reach_mm=25.0)
        assert dz <= 0.0, f"climbed on error {error}"


def test_the_plan_never_descends_further_than_asked():
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    for error in (-400.0, -140.0, 0.0, 140.0, 400.0):
        dz, _dr = model.plan_step(error, desired_dz_mm=-20.0, max_reach_mm=25.0)
        assert dz >= -20.0 - 1e-9, f"overshot on error {error}"


def test_the_reach_is_always_within_its_cap():
    model = fit_from(7.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    for error in (-1000.0, -140.0, 0.0, 140.0, 1000.0):
        _dz, dr = model.plan_step(error, desired_dz_mm=-20.0, max_reach_mm=25.0)
        assert abs(dr) <= 25.0 + 1e-9


def test_a_descent_that_does_not_disturb_the_view_is_not_rate_limited():
    """If a is ~0 the two axes are decoupled and there is nothing to trade off —
    the step should be taken in full."""
    model = fit_from(0.0, 2.5, [(-20.0, 0.0), (-20.0, 8.0)])
    dz, _dr = model.plan_step(error_px=-140.0, desired_dz_mm=-20.0,
                              max_reach_mm=25.0)
    assert dz == pytest.approx(-20.0)


def test_noisy_samples_still_fit_and_a_third_sample_helps():
    """Least squares, not exact solve — real observations carry detection noise."""
    model = DescentModel()
    for dz, dr, noise in [(-20.0, 0.0, 3.0), (-20.0, 8.0, -3.0), (-12.0, -6.0, 2.0)]:
        model.observe(dz, dr, 7.0 * dz + 2.5 * dr + noise)
    assert model.a == pytest.approx(7.0, abs=0.6)
    assert model.b == pytest.approx(2.5, abs=0.6)


# --- the stiction floor -----------------------------------------------------

def test_resolution_is_the_pixel_size_of_the_smallest_real_step():
    """0.78 ticks/px and a 25-tick floor means ~32 px is the finest achievable."""
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=-51.2)
    assert resolution_px(est, min_step=25) == pytest.approx(32.0, rel=0.02)


def test_deadband_never_finer_than_half_a_step():
    """Aiming inside the smallest executable step asks for a move the joint
    ignores — which is how both stalled runs on 2026-08-05 ended."""
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=-51.2)
    assert effective_deadband(est, deadband_px=12.0, min_step=25) == pytest.approx(16.0, rel=0.02)


def test_a_generous_deadband_is_left_alone():
    """The floor raises a too-tight deadband; it must not lower a loose one."""
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=-51.2)
    assert effective_deadband(est, deadband_px=100.0, min_step=25) == 100.0


def test_small_corrections_are_rounded_up_to_something_that_moves():
    """Overshooting slightly and coming back beats commanding nothing."""
    assert enforce_minimum_step(17.0, min_step=25) == 25
    assert enforce_minimum_step(-17.0, min_step=25) == -25


def test_a_large_correction_is_untouched():
    assert enforce_minimum_step(-35.0, min_step=25) == -35.0


def test_zero_stays_zero():
    """Zero is the deadband's decision — the minimum-step rule must not
    resurrect a correction the deadband deliberately suppressed."""
    assert enforce_minimum_step(0.0, min_step=25) == 0.0


# --- clamp_predicted_shift: the clamp that binds ---------------------------

def test_a_step_within_the_frame_is_left_alone():
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=20.0)
    # 2 ticks per px, so 40 ticks moves 20 px — nowhere near a third of 480.
    assert clamp_predicted_shift(40, est, (480, 640)) == 40


def test_a_step_that_would_leave_the_frame_is_scaled_down():
    """The 2026-08-05 failure: a step inside its unit clamp still flung the
    brick out of view, and the loop cannot correct what it cannot see."""
    est = estimate_axis(probe_ticks=8, error_before_px=0.0, error_after_px=400.0)
    out = clamp_predicted_shift(12.0, est, (480, 640), fraction=0.30)
    assert abs(out) < 12.0
    predicted_px = abs(out / est.ticks_per_px)
    assert predicted_px == pytest.approx(0.30 * 480, rel=1e-6)


def test_the_clamp_preserves_direction():
    """Scaling must never flip the sign — that would drive away from the target."""
    est = estimate_axis(probe_ticks=8, error_before_px=0.0, error_after_px=400.0)
    assert clamp_predicted_shift(-12.0, est, (480, 640)) < 0
    assert clamp_predicted_shift(12.0, est, (480, 640)) > 0


def test_the_clamp_uses_the_shorter_frame_side():
    """A brick leaves a 480-tall frame sooner than a 640-wide one."""
    est = estimate_axis(probe_ticks=8, error_before_px=0.0, error_after_px=400.0)
    out = clamp_predicted_shift(50.0, est, (480, 640), fraction=0.5)
    assert abs(out / est.ticks_per_px) == pytest.approx(240.0, rel=1e-6)


# --- pixel_error -----------------------------------------------------------

def test_pixel_error_is_zero_at_frame_centre():
    assert pixel_error((320.0, 240.0), (480, 640, 3)) == (0.0, 0.0)


def test_pixel_error_signs_are_image_convention():
    """+x is right of centre, +y is BELOW it (image coordinates, not world)."""
    dx, dy = pixel_error((420.0, 340.0), (480, 640, 3))
    assert dx == 100.0
    assert dy == 100.0


def test_pixel_error_accepts_a_custom_aim_point():
    """The grasp point need not be the optical centre; it is measurable by hand."""
    dx, dy = pixel_error((320.0, 240.0), (480, 640), aim_px=(300.0, 200.0))
    assert (dx, dy) == (20.0, 40.0)


# --- estimate_axis: the direction measurement ------------------------------

def test_estimate_recovers_a_positive_relationship():
    """+40 ticks moved the brick +20 px, so 2 ticks buy one pixel."""
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=20.0)
    assert est.ticks_per_px == pytest.approx(2.0)
    assert est.direction == 1


def test_estimate_recovers_an_inverted_relationship():
    """The whole reason the loop probes: an inverted arm is just a negative gain.

    This is the case a wrong dir_sign produces. Nothing special happens — the
    estimate comes back negative and every later correction is computed through
    it, so the loop converges exactly as well as in the positive case.
    """
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=-20.0)
    assert est.ticks_per_px == pytest.approx(-2.0)
    assert est.direction == -1


def test_estimate_refuses_a_probe_that_produced_no_response():
    """A dead axis must not become an enormous gain built out of noise."""
    with pytest.raises(ServoAbort, match="does not control this axis"):
        estimate_axis(probe_ticks=40, error_before_px=100.0, error_after_px=100.5)


def test_estimate_refuses_a_zero_probe():
    with pytest.raises(ServoAbort):
        estimate_axis(probe_ticks=0, error_before_px=0.0, error_after_px=5.0)


# --- step_ticks: the correction --------------------------------------------

def test_step_moves_against_the_error():
    """A brick right of centre must be corrected leftward, i.e. opposite sign."""
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=20.0)
    assert step_ticks(100.0, est, gain=1.0, max_step_ticks=1000) == -200


def test_step_follows_an_inverted_estimate():
    """Same error, inverted arm: the correction flips with it, automatically."""
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=-20.0)
    assert step_ticks(100.0, est, gain=1.0, max_step_ticks=1000) == 200


def test_step_is_clamped():
    """The clamp bounds a bad estimate to one small wrong move."""
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=20.0)
    assert step_ticks(10000.0, est, gain=1.0, max_step_ticks=35) == -35


def test_step_is_zero_inside_the_deadband():
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=20.0)
    assert step_ticks(5.0, est, deadband_px=12.0) == 0


def test_gain_under_one_undershoots_on_purpose():
    """Take most of the gap and re-measure, rather than trusting one probe fully."""
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=20.0)
    assert step_ticks(100.0, est, gain=0.6, max_step_ticks=1000) == -120


# --- ProgressMonitor: the runaway guard ------------------------------------

def test_monitor_allows_steady_improvement():
    m = ProgressMonitor(patience=3)
    for err in (100.0, 70.0, 40.0, 20.0, 5.0):
        m.update(err)
    assert m.best_error == 5.0
    assert not m.diverging


def test_monitor_aborts_when_the_error_grows():
    """The signature of a wrong direction: every step looks fine, the trend does not."""
    m = ProgressMonitor(patience=3)
    m.update(50.0)
    with pytest.raises(ServoAbort, match="not improved"):
        for err in (70.0, 95.0, 130.0):
            m.update(err)


def test_monitor_aborts_on_a_stall():
    """A joint against a hard stop improves by nothing, forever."""
    m = ProgressMonitor(patience=3, min_improvement_px=2.0)
    m.update(40.0)
    with pytest.raises(ServoAbort):
        for _ in range(3):
            m.update(39.9)


def test_monitor_tolerates_noise_within_patience():
    """One bad frame must not stop a run that is otherwise converging."""
    m = ProgressMonitor(patience=3)
    for err in (100.0, 60.0, 62.0, 30.0, 10.0):
        m.update(err)
    assert m.best_error == 10.0


def test_monitor_reports_divergence_against_the_start():
    m = ProgressMonitor(patience=99)
    for err in (20.0, 40.0):
        m.update(err)
    assert m.diverging


# --- the loop as a whole, simulated ----------------------------------------

def _simulate(true_ticks_per_px: float, start_error: float, iterations: int = 30):
    """Run the real control law against a linear arm model. Returns error history."""
    est = estimate_axis(
        probe_ticks=40,
        error_before_px=0.0,
        error_after_px=40.0 / true_ticks_per_px,
    )
    monitor = ProgressMonitor()
    err = start_error
    history = [err]
    for _ in range(iterations):
        delta = step_ticks(err, est)
        if delta == 0:
            break
        monitor.update(abs(err))
        err += delta / true_ticks_per_px   # the arm's actual response
        history.append(err)
    return history


def test_loop_converges_when_the_arm_behaves_as_probed():
    history = _simulate(true_ticks_per_px=2.0, start_error=200.0)
    assert abs(history[-1]) <= 12.0
    assert len(history) < 30


def test_loop_converges_just_as_well_on_an_inverted_arm():
    """The point of measuring instead of assuming: inversion is a non-event."""
    history = _simulate(true_ticks_per_px=-2.0, start_error=200.0)
    assert abs(history[-1]) <= 12.0


def test_loop_stops_itself_when_the_arm_reverses_after_the_probe():
    """The probe was right, then reality changed — a stop, not a runaway.

    Stands in for a joint hitting a limit, the detector switching objects, or
    the brick being nudged: whatever the cause, the correction now makes things
    worse, and the guard must end the run rather than push harder.
    """
    est = estimate_axis(probe_ticks=40, error_before_px=0.0, error_after_px=20.0)
    monitor = ProgressMonitor()
    err = 200.0
    with pytest.raises(ServoAbort):
        for _ in range(20):
            delta = step_ticks(err, est)
            monitor.update(abs(err))
            err -= delta / 2.0    # sign reversed relative to the probe
    assert abs(err) < 1e6, "guard must fire long before the error explodes"

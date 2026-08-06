"""How a jog is DESCRIBED to the operator, pinned to real measured geometry.

The prediction text is the whole interface of scripts/jog_joint.py: a dir_sign
is confirmed or rejected by an operator comparing what they saw against this
sentence. A description that names a direction the joint cannot physically move
in is unfalsifiable, and on 2026-08-06 two of them in a row sent the operator
looking for sideways motion that no pitch joint can produce.

Numbers below are the model's own, read out by scripts/audit_model_axes.py at
the home pose and confirmed against the arm with a ruler.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from jog_joint import describe_motion  # noqa: E402

# Base yaw axis and claw tip at the all-zero home pose, base-frame mm. The two
# are only 25 mm apart, and the yaw axis misses the model origin by 81 mm --
# both facts break naive "radial from the origin" reasoning.
YAW_XY = np.array([77.8, 23.6])
TIP = np.array([70.0, -0.1, -67.7])

PITCH_AXIS = np.array([-1.0, -0.010, 0.0])   # J2, J3 and J4 all share this
YAW_AXIS = np.array([0.0, 0.0, 1.0])         # J1

# Measured tip displacements for +10 deg on each joint.
J3_STEP = np.array([+0.2, -15.6, +23.2])
J2_STEP = np.array([+0.3, -26.0, +8.6])
J1_STEP = np.array([+4.2, -1.0, +0.0])


def test_a_pitch_joint_is_never_described_as_moving_sideways():
    """The bug the operator caught twice: a clean pitch reported as 'left'."""
    text = describe_motion(J3_STEP, TIP, YAW_XY, PITCH_AXIS)
    assert "left" not in text and "right" not in text
    assert "out (away from the base column)" in text
    assert "up" in text


def test_pitch_motion_is_reported_as_out_and_up_with_the_right_sizes():
    text = describe_motion(J3_STEP, TIP, YAW_XY, PITCH_AXIS)
    assert "15.6 mm out" in text
    assert "23.2 mm up" in text


def test_measuring_from_the_yaw_axis_alone_still_invents_lateral_motion():
    """Why the axis argument exists at all.

    Radially about the yaw axis, this same clean pitch splits into 14.8 mm out
    AND 5.1 mm lateral -- real arithmetic about the wrong centre, because the
    arm's plane is offset from that axis by the 12.7 mm J1->J2 link. Dropping
    the axis argument reproduces the old behaviour, so this test fails the day
    someone "simplifies" it away.
    """
    radial = (TIP[:2] - YAW_XY) / np.linalg.norm(TIP[:2] - YAW_XY)
    lateral = float(np.dot(J3_STEP[:2], np.array([-radial[1], radial[0]])))
    assert abs(lateral) > 5.0                       # the phantom sideways motion
    # ...against zero on the joint's own axis. The tolerance is 0.05 rather
    # than 0 only because the constants above are quoted to 0.1 mm; the audit
    # script reports this component as -0.00 mm at full precision.
    assert abs(float(np.dot(J3_STEP, PITCH_AXIS))) < 0.05


def test_a_yaw_joint_IS_described_as_moving_sideways():
    """The opposite error would be just as bad: J1 genuinely swings the claw."""
    text = describe_motion(J1_STEP, TIP, YAW_XY, YAW_AXIS)
    assert "left" in text or "right" in text
    assert "up" not in text and "down" not in text


def test_out_and_in_are_signed_away_from_the_base_column():
    """Sign convention, checked both ways round so a flip cannot pass."""
    out = describe_motion(J2_STEP, TIP, YAW_XY, PITCH_AXIS)
    back = describe_motion(-J2_STEP, TIP, YAW_XY, PITCH_AXIS)
    assert "out (away from the base column)" in out
    assert "in (toward the base column)" in back


def test_the_axis_sign_does_not_change_the_description():
    """screw_axis signs its result by the sense of rotation, so the same joint
    comes back as +axis or -axis depending on jog direction. That must not
    change which way 'out' points."""
    a = describe_motion(J3_STEP, TIP, YAW_XY, PITCH_AXIS)
    b = describe_motion(J3_STEP, TIP, YAW_XY, -PITCH_AXIS)
    assert a == b


def test_a_tilted_axis_falls_back_to_plain_base_frame_terms():
    """No clean two-term story exists, so it must not invent one."""
    tilted = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    text = describe_motion(np.array([5.0, 0.0, 0.0]), TIP, YAW_XY, tilted)
    assert "along +X" in text
    assert "out" not in text and "left" not in text


def test_negligible_motion_says_so():
    assert describe_motion(np.array([0.05, -0.05, 0.01]), TIP, YAW_XY,
                           PITCH_AXIS) == "no appreciable movement"


# --- how a ROLL is described -------------------------------------------------
# J5 spins the claw about the forearm. Describing that against the base frame
# called it "tilting back/up", because the base frame is rotated ~89 deg from
# the arm's forward -- so the roll axis landed nearest base +Y, which the
# naming table reads as a pitch. Spin sense about the claw's own direction is
# immune to that, and is what the operator can actually watch.

def _spin(axis, angle_deg):
    """Rotation of `angle_deg` about `axis`, as a (R0, R1) pair with R0 = I."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    a = np.deg2rad(angle_deg)
    return np.eye(3), np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def test_a_roll_is_described_as_a_spin_not_a_tilt():
    from jog_joint import describe_rotation
    claw = np.array([0.0, 1.0, 0.0])
    text, deg = describe_rotation(*_spin(claw, 17.6), claw_dir=claw)
    assert "SPINS" in text
    assert "tilting" not in text
    assert deg == pytest.approx(17.6, abs=1e-6)


def test_spin_sense_flips_with_the_direction_of_rotation():
    from jog_joint import describe_rotation
    claw = np.array([0.0, 1.0, 0.0])
    cw, _ = describe_rotation(*_spin(claw, 17.6), claw_dir=claw)
    acw, _ = describe_rotation(*_spin(claw, -17.6), claw_dir=claw)
    assert "CLOCKWISE" in cw and "ANTICLOCKWISE" not in cw
    assert "ANTICLOCKWISE" in acw


def test_spin_sense_is_stated_from_a_named_viewpoint():
    """'Clockwise' is meaningless without saying from where -- the operator can
    stand on either side of the arm and see opposite senses."""
    from jog_joint import describe_rotation
    claw = np.array([0.0, 1.0, 0.0])
    text, _ = describe_rotation(*_spin(claw, 17.6), claw_dir=claw)
    assert "viewed from the wrist looking out along the claw" in text


def test_a_pitch_is_not_called_a_spin():
    """The roll branch must not swallow joints that rotate ACROSS the claw."""
    from jog_joint import describe_rotation
    claw = np.array([0.0, 1.0, 0.0])
    text, _ = describe_rotation(*_spin([1.0, 0.0, 0.0], 20.0), claw_dir=claw)
    assert "SPINS" not in text


def test_base_frame_naming_is_marked_as_such():
    """It is ~89 deg off on this arm; the words must not read as authoritative."""
    from jog_joint import describe_rotation
    text, _ = describe_rotation(*_spin([1.0, 0.0, 0.0], 20.0),
                                claw_dir=np.array([0.0, 1.0, 0.0]))
    assert "base-frame naming" in text


def test_a_roll_is_watched_by_its_spin_even_when_it_translates_visibly():
    """J5 moves the tip 7.5 mm for a 17.6 deg turn -- over MIN_VISIBLE_MM, so a
    travel-only rule would have the operator judging a sign from a 7 mm wobble
    whose direction has no clean description at that pose."""
    from jog_joint import MIN_VISIBLE_MM, is_roll
    claw = np.array([0.0, 1.0, 0.0])
    assert 7.5 > MIN_VISIBLE_MM
    assert is_roll(claw, claw)


def test_a_pitch_is_not_treated_as_a_roll():
    from jog_joint import is_roll
    assert not is_roll(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))


def test_is_roll_ignores_the_axis_sign_and_a_missing_claw_direction():
    from jog_joint import is_roll
    claw = np.array([0.0, 1.0, 0.0])
    assert is_roll(-claw, claw)
    assert not is_roll(claw, None)
    assert not is_roll(claw, np.zeros(3))

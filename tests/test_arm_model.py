"""The ruler-survey kinematics, checked against data it was not built from.

The load-bearing test here is test_held_out_touches_land_on_the_table. Everything
else is arithmetic; that one is the reason to believe the model at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from vision_pipeline.kinematics import arm_model
from vision_pipeline.robot_interface import servo_calibration as sc

IK_JOINTS = (1, 2, 3, 4, 5)

# Seven touches of the tabletop, recorded 2026-08-07 by scripts/measure_table_plane.py
# BEFORE the survey in arm_model existed and used nowhere in building it. Every one
# is the same flat plane, so every one must come back at the table height. That is
# a property of the KINEMATICS, not of the table -- it holds whatever the table
# height turns out to be, which is what makes these a fair held-out test.
TOUCHES = [
    {1: 1995, 2: 3602, 3: 2590, 4: 1499, 5: 2745},
    {1: 2359, 2: 3759, 3: 2344, 4: 1500, 5: 2744},
    {1: 1883, 2: 3759, 3: 2340, 4: 1567, 5: 2746},
    {1: 2071, 2: 3759, 3: 2343, 4: 1567, 5: 2743},
    {1: 2073, 2: 3295, 3: 2906, 4: 1462, 5: 2746},
    {1: 2321, 2: 3295, 3: 2912, 4: 1464, 5: 2745},
    {1: 1794, 2: 3294, 3: 2907, 4: 1462, 5: 2747},
]

# What the IMPORTED MATLAB model returns for the same seven, for contrast.
MATLAB_SPREAD_MM = 73.1


def angles(ticks):
    cal = sc.load_calibration()
    return [sc.ticks_to_rad(cal, j, ticks[j]) for j in IK_JOINTS]


def test_home_reproduces_the_survey_exactly():
    """Home built the model, so this is a consistency check on the arithmetic
    rather than evidence -- but a failure here means the chain was assembled
    wrong and every other number is meaningless."""
    cal = sc.load_calibration()
    home = {j: cal[str(j)]["home_tick"] for j in IK_JOINTS}
    a = angles(home)
    forward, up = arm_model.tip_in_plane(a[1], a[2], a[3])
    assert forward == pytest.approx(arm_model.SURVEY_MM["tip"][0], abs=0.05)
    assert up == pytest.approx(arm_model.SURVEY_MM["tip"][1], abs=0.05)


def test_link_lengths_match_the_independently_measured_ones():
    """The survey gives (forward, up) per joint; the link lengths were ALSO
    measured directly along the links. Two routes, and they must agree -- this
    is what rules out a transcription error in the survey."""
    lengths = [L for L, _angle in arm_model.segments()]
    assert lengths[0] == pytest.approx(102.7, abs=3.0)   # shoulder -> elbow
    assert lengths[1] == pytest.approx(136.2, abs=3.5)   # elbow -> wrist pitch
    assert lengths[3] == pytest.approx(70.1, abs=2.0)    # wrist -> tip (CLAW_LEN)


def test_held_out_touches_land_on_the_table():
    """THE TEST THAT EARNS THE MODEL ITS TRUST.

    Seven touches of one flat plane, taken before the survey existed. The
    imported CAD model spreads them over 73.1 mm; this one must do far better,
    and must put them at the table rather than merely agreeing with itself -- a
    model can be self-consistent and still 100 mm off, which is exactly the
    failure this replaces."""
    heights = np.array([arm_model.tip_in_plane(*angles(t)[1:4])[1] for t in TOUCHES])

    spread = float(heights.max() - heights.min())
    assert spread < 10.0, f"touches of one flat plane spread {spread:.1f} mm"
    assert spread < MATLAB_SPREAD_MM / 5, "must beat the imported model decisively"
    assert abs(float(heights.mean())) < 5.0, \
        "the plane must sit at the TABLE, not merely be flat"


def test_j1_cannot_change_the_tips_height():
    """Base yaw rotates about a vertical axis, so it moves the tool sideways and
    never up. Worth pinning because it is the invariant that let the touches be
    trusted: within each posture cluster J1 swung ~50 deg while the measured
    height held to ~1 mm."""
    a = angles(TOUCHES[0])
    z = [arm_model.tip_position(t1, a[1], a[2], a[3])[2]
         for t1 in np.deg2rad([-40.0, 0.0, 40.0])]
    assert z == pytest.approx([z[0]] * 3, abs=1e-12)


def test_the_lateral_offset_is_declared_unmeasured():
    """It biases y and never z, so nothing above is affected -- but it must stay
    visibly zero rather than inheriting the imported model's ~27 mm, which comes
    from the source this module exists to distrust."""
    assert arm_model.LATERAL_OFFSET_MM == 0.0

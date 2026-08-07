"""Forward kinematics built from a RULER SURVEY of the physical arm.

WHY THIS EXISTS RATHER THAN USING THE IMPORTED MODEL. `importrobot` on
`Robomainassemjoints.slx` produces an arm that is the wrong shape. Measured
2026-08-07 at the home pose, heights above the tabletop:

        shoulder    model 158.2   ruler  90     off by -68 mm
        elbow       model 106.4   ruler 146     off by +40
        wrist pitch model 108.0   ruler 152     off by +44
        wrist       model  71.3   ruler  71.3   exact
        claw tip    model   5.5   ruler   0

The shoulder settles it: that shaft is bolted to the base column, so no joint
angle, `dir_sign`, `ticks_per_rad` or backlash can move it, and it reads 158.2 in
the model at every pose. The model puts the elbow 51.8 mm BELOW the shoulder
where the arm has it 56 mm ABOVE -- the upper arm points ~30 deg down in the
model and ~33 deg up in reality. That single shape error is why a J2 jog swept
the claw 30.0 mm where the model predicted 13.8.

WHAT MAKES THIS TRUSTWORTHY, in the order the evidence should be weighed:

1. **It is not fitted.** Every constant below is a ruler reading. Nothing was
   tuned to make anything agree.
2. **Two independent measurement routes cross-validate.** The link lengths implied
   by the (forward, up) survey match lengths measured directly along the links:
   100.96 vs 102.7, 139.13 vs 136.2, 71.31 vs 70.1 -- all within 3 mm.
3. **It is validated against HELD-OUT data.** Seven touches of the tabletop, taken
   before this survey existed and used nowhere in building it, must all return
   z = 0 because they are all the same flat plane. They come back with a 6.0 mm
   spread about -1.4 mm; the imported model gives 73.1 mm about +20.6. Six of the
   seven land within 1 mm. `tests/test_arm_model.py` pins this.

THE GEOMETRY IS A PLANAR CHAIN, and that is a property of the arm rather than a
simplification: J2, J3 and J4 are parallel to 0.0 deg
(`scripts/audit_model_axes.py`), so they move the tool in one vertical plane whose
bearing J1 alone sets, and J5 is a roll that does not pitch the claw. So the whole
arm is base-yaw + a planar 3-link chain, and a survey of five points in that plane
determines it outright.

LIMITS, stated because this will be trusted:
  * The survey is 2-D. Any LATERAL offset of the claw from the arm's plane is not
    measured -- CLAUDE.md estimates ~27 mm from the model, which is the source
    this module exists to distrust. `LATERAL_OFFSET_MM` is therefore 0.0 and
    flagged; it biases y, never z, so every validation above is unaffected.
  * Height is what the touches validate. The forward coordinate is only as good as
    the survey, and nothing independent has checked it yet.
  * Servo backlash is not modelled. The touches were taken by hand-pressing a limp
    arm down, which loads each joint against several degrees of gear lash; that is
    the most likely source of the residual 6 mm.
"""

from __future__ import annotations

import numpy as np

# --- the survey ------------------------------------------------------------
#
# Operator ruler measurements at the HOME pose (all five joint angles zero),
# 2026-08-07. Each point is (forward, up) in millimetres: forward from servo 1's
# shaft, up from the tabletop. Servo 1's shaft is taken as the base column.
#
# Note the elbow sits BEHIND the base column at home (-72) while the wrist pitch
# is well in front (+67): at home this arm is folded back over itself, upper arm
# up-and-back, forearm forward. That is the posture the imported model gets
# inside-out.
SURVEY_MM = {
    "shoulder":    (12.0, 90.0),     # servo 2 shaft
    "elbow":       (-72.0, 146.0),   # servo 3 shaft
    "wrist_pitch": (67.0, 152.0),    # servo 4 shaft
    "wrist":       (74.0, 71.3),     # servo 5 shaft -- the hand-eye frame
    "tip":         (75.0, 0.0),      # claw tip, ON the table at home
}

CHAIN = ("shoulder", "elbow", "wrist_pitch", "wrist", "tip")

# Which pitch joints carry each segment. Segment i rotates with every pitch joint
# at or above its base: the upper arm with J2, the forearm with J2+J3, and
# everything past the wrist pitch with J2+J3+J4. J5 is a ROLL and is absent by
# design -- it spins the claw about its own axis and does not pitch it.
CARRIED_BY = ((2,), (2, 3), (2, 3, 4), (2, 3, 4))

# UNMEASURED. The claw may hang to one side of the arm's plane; the imported model
# says ~27 mm but that model is wrong about the geometry this module replaces, so
# it is not inherited. Affects y only, never height. Measure and set it before
# trusting a lateral target to better than this.
LATERAL_OFFSET_MM = 0.0


def segments() -> list[tuple[float, float]]:
    """(length_mm, absolute angle at home) for each link, from the survey.

    Derived rather than stored so the survey stays the single source of truth: a
    corrected measurement changes one number above and everything follows.
    """
    out = []
    for a, b in zip(CHAIN, CHAIN[1:]):
        dx = SURVEY_MM[b][0] - SURVEY_MM[a][0]
        dz = SURVEY_MM[b][1] - SURVEY_MM[a][1]
        out.append((float(np.hypot(dx, dz)), float(np.arctan2(dz, dx))))
    return out


def tip_in_plane(theta2: float, theta3: float, theta4: float) -> tuple[float, float]:
    """Claw tip as (forward_mm, height_above_table_mm) in the arm's own plane.

    Args:
        theta2, theta3, theta4: pitch joint angles in radians, ZERO AT HOME. Get
            them from servo_calibration.ticks_to_rad, whose sign convention this
            matches -- validated against the touches with the stored dir_sign
            values unchanged, which is consistent with all five having been
            confirmed by physical jog.

    Returns:
        (forward, up) in mm. Height is the coordinate the held-out touches
        validate to 6 mm; forward carries only the survey's own accuracy.
    """
    th = {2: theta2, 3: theta3, 4: theta4}
    x, z = SURVEY_MM["shoulder"]
    for (length, angle), carriers in zip(segments(), CARRIED_BY):
        a = angle + sum(th[j] for j in carriers)
        x += length * np.cos(a)
        z += length * np.sin(a)
    return float(x), float(z)


def tip_position(theta1: float, theta2: float, theta3: float,
                 theta4: float) -> np.ndarray:
    """Claw tip in the base frame, METRES, +Z up and the table below the origin.

    The planar chain above swung about the base yaw axis. J5 is omitted: it is a
    roll, so to the tip it contributes only through LATERAL_OFFSET_MM, which is
    unmeasured and currently zero.

    THE FRAME IS NOT THE IMPORTED MODEL'S BASE FRAME, and the difference is
    deliberate. x and y are measured from the BASE YAW AXIS -- the column the arm
    actually turns about -- because that is what the survey was taken from and
    what physically exists. The imported model's origin is a CAD artefact sitting
    81 mm away from that axis (`scripts/audit_model_axes.py`), which is exactly
    the sort of number this module exists to stop inheriting. z IS shared with the
    pipeline: it is measured from the base origin, so `table_z()` places the table
    in it and the two agree vertically.

    A consumer mixing this with `MatlabIKClient` poses must therefore offset x and
    y by the yaw axis position (`MatlabIKClient.base_yaw_axis_xy()`), or work in
    radial/tangential terms, which need no origin at all.

    Returns:
        (3,) array (x, y, z) in metres. Orientation is the physical convention:
        +Z up, +X the arm's forward at yaw zero, +Y left.
    """
    forward, up = tip_in_plane(theta2, theta3, theta4)
    x = forward * np.cos(theta1) - LATERAL_OFFSET_MM * np.sin(theta1)
    y = forward * np.sin(theta1) + LATERAL_OFFSET_MM * np.cos(theta1)
    return np.array([x, y, up + table_z() * 1000.0]) / 1000.0


def table_z() -> float:
    """Where the tabletop sits in the base frame, metres. Negative: below it.

    Read from config rather than hardcoded so there is one table height in the
    project. The survey measures heights ABOVE the table, so this is the only
    number needed to place them in the base frame.
    """
    from vision_pipeline import config
    return float(config.TABLE_Z_IN_BASE)

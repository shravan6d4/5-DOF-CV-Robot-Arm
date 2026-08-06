"""Named joint-space poses, and the one safe way to drive to them.

WHY THIS EXISTS. Every closed-loop routine here starts by measuring where the
brick is in the image, because the camera is eye-in-hand and there is no other
source of that information. So a run that begins from an arbitrary pose begins
with the operator hand-positioning the arm until the brick appears -- which is
not reproducible, and quietly changes the geometry the probe gains are measured
against. Two runs from two different starting postures are not comparable, and
on 2026-08-05 that was part of why a gain measured before a descent no longer
described the arm during it.

Driving to a known hover first fixes both: the brick is in view, and every run
starts from the same place.

JOINT SPACE, NOT CARTESIAN, and deliberately. These poses are recorded as raw
ticks and commanded as raw ticks. Going through IK would make the move depend on
the hand-eye transform being right (it is not, see CLAUDE.md), on the solver
picking the same branch twice, and on the arm not being somewhere the solver
dislikes. A tick is a tick. The whole point of a recovery pose is that it works
when the clever paths do not.
"""

from __future__ import annotations

import logging

import numpy as np

from vision_pipeline import config

logger = logging.getLogger(__name__)

HOME = dict(config.SERVO_HOME_TICKS)
HOVER = dict(config.SERVO_HOVER_TICKS)

POSES = {"home": HOME, "hover": HOVER}


def describe_move(bus, targets: dict) -> list[str]:
    """Per-joint preview of a move, in ticks and degrees. Reads only.

    Printed before anything is commanded, because the degree column is the one
    an operator can sanity-check against the arm in front of them, and a joint
    about to swing further than SERVO_WATCH_POWER_MOVE_DEG is the moment to be
    standing by the power cut.
    """
    lines = []
    biggest = 0.0
    for joint in sorted(targets):
        present = bus.read_position_retrying(joint)
        delta = int(targets[joint]) - present
        tpr = bus._cal(joint)["ticks_per_rad"]
        deg = abs(delta) / tpr * 180.0 / np.pi
        biggest = max(biggest, deg)
        lines.append(f"    J{joint}: {present:>5} -> {int(targets[joint]):>5}  "
                     f"({delta:+5d} ticks, {deg:5.1f} deg)")
    if biggest > config.SERVO_WATCH_POWER_MOVE_DEG:
        lines.append(f"    *** {biggest:.0f} deg of travel on one joint — over the "
                     f"{config.SERVO_WATCH_POWER_MOVE_DEG:.0f} deg watch threshold.")
        lines.append(f"        Stand by the power cut. Ctrl-C freezes without dropping.")
    return lines


def goto(bus, pose, label: str = "", progress=None) -> dict:
    """Drive the arm to a named pose (or an explicit tick dict). MOVES THE ARM.

    Steps every joint together in sub-cap increments via move_joints_stepped, so
    no single command exceeds SERVO_MAX_MOVE_DELTA_TICKS and the travel limits
    are checked per hop rather than once at the end.

    Args:
        pose: "home", "hover", or a {joint: tick} dict.
        label: what to call this move in log output.

    Returns:
        {joint: tick} actually read back afterwards.

    Raises:
        KeyError: unknown pose name.
        ServoSafetyError: the pose is outside a joint's travel limits. That is
            worth surfacing rather than clamping -- a named pose that no longer
            fits the recorded limits means one of the two is wrong, and guessing
            which would move the arm somewhere nobody chose.
    """
    targets = dict(POSES[pose] if isinstance(pose, str) else pose)
    logger.info(f"Driving to {label or pose}: {targets}")

    bus.move_joints_stepped(
        targets,
        step_ticks=config.PICK_STEP_TICKS,
        pause_s=config.PICK_STEP_PAUSE_S,
        progress=progress,
    )
    return {j: bus.read_position_retrying(j) for j in sorted(targets)}


def at_pose(bus, pose, tolerance_ticks: int = 40) -> bool:
    """Is the arm already at this pose? Reads only, commands nothing.

    Lets a script skip the move when it is already there, which matters because
    the alternative -- commanding a zero-length move -- still walks the whole
    stepped-move machinery and costs a second or two per run.
    """
    targets = POSES[pose] if isinstance(pose, str) else pose
    return all(abs(bus.read_position_retrying(j) - int(t)) <= tolerance_ticks
               for j, t in targets.items())

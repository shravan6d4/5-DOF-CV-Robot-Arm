"""
The vision -> robot handoff: what to pick, and the motion to pick it.

`PickTarget` is the single object this whole pipeline exists to produce: a place
on the table for the arm to grasp a brick, expressed the way a 5-DOF top-down
pick needs it — position (x, y, z) plus a yaw to line the gripper up with the
brick. Roll and pitch are NOT in the target because a 5-DOF arm keeps the tool
pointing straight down; those fixed angles come from config when we expand a
PickTarget into a full robot Pose.

`plan_pick_sequence` turns one PickTarget into the ordered list of Poses +
gripper actions that actually lift the brick (hover, descend, close, lift). It's
kept here, backend-agnostic, so any RobotInterface implementation — sim or real
arm — executes the same well-tested sequence and only has to provide the
primitive "go to this pose" / "set the gripper" moves.
"""

from __future__ import annotations

from dataclasses import dataclass

from vision_pipeline import config
from vision_pipeline.robot_interface.base import Pose


@dataclass
class PickTarget:
    """Where and how to grasp one brick, in the robot base frame.

    Attributes:
        x, y, z: grasp point in meters (z is the height the gripper closes at).
        yaw_deg: rotation of the gripper about the vertical axis to align with
            the brick's long axis.
        num_studs: carried through from detection, purely informational (how
            confident we were that this was a real brick).
    """

    x: float
    y: float
    z: float
    yaw_deg: float
    num_studs: int = 0

    def to_pose(self) -> Pose:
        """Expand into a full end-effector Pose using the fixed top-down angles."""
        return Pose(
            x=self.x,
            y=self.y,
            z=self.z,
            roll_deg=config.PICK_ROLL_DEG,
            pitch_deg=config.PICK_PITCH_DEG,
            yaw_deg=self.yaw_deg,
        )


@dataclass
class PickStep:
    """One step of a pick sequence: move to a pose, optionally then set gripper.

    gripper_closed is None when the step is a pure move (don't touch the
    gripper), True to close after arriving, False to open after arriving.
    """

    pose: Pose
    gripper_closed: bool | None = None
    label: str = ""


def plan_pick_sequence(
    target: PickTarget,
    approach_height: float = config.APPROACH_HEIGHT,
) -> list[PickStep]:
    """Break a PickTarget into an ordered, backend-agnostic grasp sequence.

    The sequence approaches straight down so the gripper doesn't sweep sideways
    into the brick or neighbours:

        1. open the gripper while hovering above the target
        2. descend to the grasp height
        3. close the gripper on the brick
        4. lift back up to the hover height

    Args:
        target: the grasp to execute.
        approach_height: how far above the grasp point to hover/lift, meters.

    Returns:
        A list of PickSteps for a RobotInterface to run in order.
    """
    grasp = target.to_pose()
    hover = Pose(
        x=grasp.x,
        y=grasp.y,
        z=grasp.z + approach_height,
        roll_deg=grasp.roll_deg,
        pitch_deg=grasp.pitch_deg,
        yaw_deg=grasp.yaw_deg,
    )

    return [
        PickStep(hover, gripper_closed=False, label="hover + open gripper"),
        PickStep(grasp, gripper_closed=None, label="descend to grasp"),
        PickStep(grasp, gripper_closed=True, label="close gripper"),
        PickStep(hover, gripper_closed=None, label="lift"),
    ]

"""
A simulated robot backend: implements RobotInterface without any hardware.

Purpose: let the entire pick pipeline (detect -> world coordinate -> plan ->
execute) run and be tested end-to-end before the real 5-DOF arm code exists.
It doesn't move anything — it just reports a fixed "current" gripper pose and
records every command it receives, so tests and the demo script can assert the
pipeline commanded the right poses and gripper actions.

Merge path: when the real arm arrives, write a sibling backend (e.g. an
`ArmRobot`) that implements the same three methods against your hardware, and
swap it in wherever SimRobot is constructed. Nothing else in the pipeline
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vision_pipeline.robot_interface.base import Pose, RobotInterface


@dataclass
class CommandLog:
    """Record of everything the pipeline asked the (simulated) arm to do."""

    poses: list[Pose] = field(default_factory=list)
    gripper_states: list[bool] = field(default_factory=list)


class SimRobot(RobotInterface):
    """In-memory RobotInterface for tests and dry runs.

    Args:
        ee_pose: the gripper pose this fake arm reports from
            get_end_effector_pose. For an eye-in-hand rig this stands in for the
            arm's forward kinematics — set it to wherever you're pretending the
            camera-carrying gripper is hovering while it looks at the table.
        verbose: if True, print each command as it arrives (handy in the demo).
    """

    def __init__(self, ee_pose: Pose | None = None, verbose: bool = False) -> None:
        # A sensible default "looking down at the table" pose: 0.3 m above the
        # base origin, tool pointing down. Override for your real geometry.
        self.ee_pose = ee_pose if ee_pose is not None else Pose(x=0.0, y=0.0, z=0.3, roll_deg=180.0)
        self.verbose = verbose
        self.log = CommandLog()
        self.gripper_closed = False

    def get_end_effector_pose(self) -> Pose:
        return self.ee_pose

    def send_target_pose(self, pose: Pose) -> None:
        self.log.poses.append(pose)
        if self.verbose:
            print(
                f"[SimRobot] move -> x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f} "
                f"yaw={pose.yaw_deg:.1f}"
            )

    def set_gripper(self, closed: bool) -> None:
        self.gripper_closed = closed
        self.log.gripper_states.append(closed)
        if self.verbose:
            print(f"[SimRobot] gripper -> {'CLOSED' if closed else 'OPEN'}")

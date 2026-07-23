"""Real robot backend: MATLAB IK/FK server + Feetech servo bus.

Implements RobotInterface contract with live FK from MATLAB and servo read-back
verification. This is the merge-ready real hardware backend.
"""

import logging

from vision_pipeline import config
from vision_pipeline.calibration import geometry
from vision_pipeline.robot_interface.base import Pose, RobotInterface
from vision_pipeline.robot_interface.matlab_client import IKUnreachableError, MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

logger = logging.getLogger(__name__)


class HardwareRobot(RobotInterface):
    """Real robot backend combining MATLAB IK/FK with servo hardware.

    On init, reads back actual servo positions to seed the state. Every
    get_end_effector_pose() calls live FK from MATLAB. Every send_target_pose()
    solves IK fresh, commands servos, and updates state from read-back positions.
    """

    def __init__(
        self,
        matlab_host: str = config.MATLAB_SERVER_HOST,
        matlab_port: int = config.MATLAB_SERVER_PORT,
        servo_port: str = config.SERVO_PORT,
        servo_baud: int = config.SERVO_BAUD,
    ):
        """Initialize real hardware: MATLAB client + servo bus.

        Args:
            matlab_host, matlab_port: connection to ik_fk_server.m.
            servo_port, servo_baud: serial connection to Waveshare adapter.

        Raises:
            Any exception from MatlabIKClient or ServoBus if connections fail.
        """
        self.matlab_client = MatlabIKClient(matlab_host, matlab_port)
        self.servo_bus = ServoBus(servo_port, servo_baud)

        # Seed state by reading back current servo positions (J1..J5 only)
        self._last_angles_rad = self._read_servo_state()
        logger.info(f"HardwareRobot initialized. Current angles (rad): {self._last_angles_rad}")

    def _read_servo_state(self) -> list[float]:
        """Read present positions from J1..J5 servos and convert to radians.

        Returns:
            List of 5 angles in radians.
        """
        angles_rad = []
        for j_id in range(1, 6):  # J1..J5 only
            try:
                present_tick = self.servo_bus.read_position(j_id)
                angle_rad = self.servo_bus.ticks_to_rad(j_id, present_tick)
                angles_rad.append(angle_rad)
                logger.debug(f"J{j_id}: read {present_tick} ticks -> {angle_rad:.3f} rad")
            except Exception as e:
                logger.error(f"Failed to read J{j_id} position: {e}")
                raise
        return angles_rad

    def get_end_effector_pose(self) -> Pose:
        """Return current wrist pose via live FK from MATLAB.

        Uses the last verified servo positions read after send_target_pose(),
        so state never drifts from physical reality (read-back is ground truth).

        Returns:
            Pose in the robot base frame (from MATLAB FK of Body08).
        """
        try:
            T_wrist = self.matlab_client.request_fk(self._last_angles_rad)
            x, y, z, roll_deg, pitch_deg, yaw_deg = geometry.transform_to_pose(T_wrist)
            pose = Pose(x=x, y=y, z=z, roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg)
            logger.debug(f"FK: {pose}")
            return pose
        except Exception as e:
            logger.error(f"FK failed: {e}")
            raise

    def send_target_pose(self, pose: Pose) -> bool:
        """Solve IK for a target pose and command servos J1..J5.

        Position (x, y, z) is sent to the MATLAB IK solver; roll/pitch/yaw
        are logged and dropped (5-DOF arm, position-only IK). After servo
        moves complete, reads back actual positions and updates internal state.

        Args:
            pose: target pose in the robot base frame.

        Returns:
            True if successful, False otherwise.
        """
        # Log dropped orientation (documented 5-DOF limitation)
        if pose.roll_deg != config.PICK_ROLL_DEG or pose.pitch_deg != config.PICK_PITCH_DEG:
            logger.debug(
                f"Orientation requested: roll={pose.roll_deg}° pitch={pose.pitch_deg}° "
                f"yaw={pose.yaw_deg}°, but arm is 5-DOF position-only. "
                f"Only (x,y,z) will be used; orientation is fixed to "
                f"({config.PICK_ROLL_DEG}°, {config.PICK_PITCH_DEG}°)."
            )

        # Solve IK for position only, seeded from the arm's current angles so
        # the solver returns the NEAREST solution. Unseeded it can return a
        # valid posture ~180 deg away, which ServoBus would then refuse move
        # by move (see MatlabIKClient.request_ik).
        try:
            angles_rad, err_mm = self.matlab_client.request_ik(
                pose.x, pose.y, pose.z, seed_rad=self._last_angles_rad
            )
            logger.info(f"IK solved: {err_mm:.1f} mm error. Angles: {[f'{a:.3f}' for a in angles_rad]}")
        except IKUnreachableError as e:
            logger.error(f"IK failed for target ({pose.x}, {pose.y}, {pose.z}): {e}")
            return False
        except Exception as e:
            logger.error(f"IK communication error: {e}")
            return False

        # Command servos J1..J5, read back actual positions
        try:
            updated_angles_rad = []
            for j_id, angle_rad in enumerate(angles_rad, start=1):
                target_tick = self.servo_bus.rad_to_ticks(j_id, angle_rad)
                actual_tick = self.servo_bus.move_and_verify(j_id, target_tick)
                actual_angle_rad = self.servo_bus.ticks_to_rad(j_id, actual_tick)
                updated_angles_rad.append(actual_angle_rad)

            # Update state from verified read-back
            self._last_angles_rad = updated_angles_rad
            logger.info(f"Servos moved and verified. New state: {[f'{a:.3f}' for a in self._last_angles_rad]}")
            return True

        except ServoSafetyError as e:
            logger.error(f"Servo move refused as unsafe: {e}")
            self._resync_state_after_partial_move()
            return False

        except Exception as e:
            logger.error(f"Servo move failed: {e}")
            self._resync_state_after_partial_move()
            return False

    def _resync_state_after_partial_move(self) -> None:
        """Re-read every joint after a move aborts partway through.

        The joints are commanded one at a time, so a refusal or failure on J3
        leaves J1-J2 already moved while `_last_angles_rad` still describes the
        pre-move pose. Left stale, the next get_end_effector_pose() would report
        a camera pose the arm is not actually in — and the vision pipeline would
        back-project through it without any indication anything was wrong.
        Re-reading costs one bus round-trip per joint and keeps state honest.
        """
        try:
            self._last_angles_rad = self._read_servo_state()
            logger.warning(
                f"Re-synced joint state after aborted move: "
                f"{[f'{a:.3f}' for a in self._last_angles_rad]}"
            )
        except Exception as e:
            # Read-back is itself failing (bus down, servo unpowered). State is
            # now untrustworthy and we cannot repair it here, so say so loudly
            # rather than leaving a plausible-looking stale pose in place.
            logger.critical(
                f"Could not re-read servo state after an aborted move: {e}. "
                f"Joint state is STALE and end-effector poses derived from it "
                f"are unreliable until the bus recovers."
            )

    def set_gripper(self, closed: bool) -> bool:
        """Open or close the gripper (J6 servo).

        Args:
            closed: True to close, False to open.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # J6 open/closed positions come from config (angle offset from home),
            # converted to ticks through the same per-servo calibration ServoBus loaded.
            if closed:
                angle_rad = config.SERVO_GRIPPER_CLOSE_RAD
                state = "close"
            else:
                angle_rad = config.SERVO_GRIPPER_OPEN_RAD
                state = "open"

            target_tick = self.servo_bus.rad_to_ticks(6, angle_rad)
            actual_tick = self.servo_bus.move_and_verify(6, target_tick)
            logger.info(f"Gripper {state}. Commanded {target_tick} ticks, read {actual_tick} ticks.")
            return True

        except Exception as e:
            logger.error(f"Gripper command failed: {e}")
            return False

    def close(self):
        """Close connections to MATLAB and servo bus."""
        if self.matlab_client:
            self.matlab_client.close()
        if self.servo_bus:
            self.servo_bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

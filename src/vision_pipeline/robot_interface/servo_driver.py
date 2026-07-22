"""Feetech STS3215 servo bus driver over a Waveshare bus-servo adapter.

Supports write-and-verify: command a position, then read back to confirm it moved.
Per-servo calibration (home tick, direction, ticks-per-radian) is stored in JSON
and follows the camera_intrinsics/hand_eye placeholder pattern.

Protocol note: the Feetech STS/SMS series speaks a Dynamixel-1.0-compatible half-
duplex serial protocol. A packet is:

    0xFF 0xFF  ID  LENGTH  INSTRUCTION  PARAM_1 ... PARAM_N  CHECKSUM

where LENGTH = N + 2 (params + instruction + checksum) and
CHECKSUM = ~(ID + LENGTH + INSTRUCTION + PARAMs) & 0xFF. A status packet echoes the
same framing with an ERROR byte in the INSTRUCTION slot. Word registers on the
STS3215 are little-endian.

This driver is validated structurally against that spec but has NOT been exercised
against physical hardware yet — that is Stage 2 of the integration plan (sweep one
servo, read back, confirm within tolerance) and must be done on the real bus.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import serial

from vision_pipeline import config

logger = logging.getLogger(__name__)


class ServoCalibrationError(Exception):
    """Raised when servo calibration fails or is incomplete."""
    pass


class ServoBus:
    """Interface to a Feetech STS3215 servo bus over a serial port.

    The Waveshare driver board is a transparent USB-to-half-duplex-serial bridge;
    this class speaks the Feetech packet protocol directly over it, with position
    commands and read-back verification.
    """

    HEADER = 0xFF

    # Instructions (Dynamixel 1.0 / Feetech SCS compatible)
    INST_PING = 0x01
    INST_READ = 0x02
    INST_WRITE = 0x03

    # STS3215 register addresses
    ADDR_GOAL_POSITION = 0x2A       # 42, 2 bytes little-endian
    ADDR_PRESENT_POSITION = 0x38    # 56, 2 bytes little-endian

    TICK_MIN = 0
    TICK_MAX = 4095                 # STS3215 is 12-bit (0..4095)

    def __init__(self, port: str, baud: int = 1000000, calibration_path: Optional[str] = None):
        """Initialize the servo bus.

        Args:
            port: serial port name (e.g., 'COM3', '/dev/ttyUSB0').
            baud: baud rate (default 1000000 for STS3215).
            calibration_path: path to servo_calibration.json, or None to use config default.

        Raises:
            ServoCalibrationError: if calibration is missing or invalid.
            serial.SerialException: if the port cannot be opened.
        """
        self.port = port
        self.baud = baud
        self.serial = None
        self._calibration = {}

        # Load calibration (JSON file, fallback to config placeholder)
        self._load_calibration(calibration_path or config.SERVO_CALIBRATION_PATH)

        # Open serial port
        try:
            self.serial = serial.Serial(port, baud, timeout=0.5)
            logger.info(f"Servo bus opened on {port} @ {baud} baud")
        except serial.SerialException as e:
            logger.error(f"Failed to open {port}: {e}")
            raise

    # --- calibration ------------------------------------------------------

    def _load_calibration(self, path: str):
        """Load per-servo calibration from JSON, fallback to config.

        JSON format (keys are strings, matching config.SERVO_CALIBRATION_FALLBACK):
            {
                "1": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1},
                ...
            }
        """
        calibration_path = Path(path)
        if calibration_path.exists():
            try:
                with open(calibration_path) as f:
                    self._calibration = json.load(f)
                logger.info(f"Loaded servo calibration from {path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load {path}, using fallback: {e}")
                self._calibration = config.SERVO_CALIBRATION_FALLBACK
        else:
            logger.warning(f"Calibration file {path} not found, using fallback")
            self._calibration = config.SERVO_CALIBRATION_FALLBACK

        # Validate: all J1..J6 present with the required keys
        for j in range(1, 7):
            if str(j) not in self._calibration:
                raise ServoCalibrationError(f"Servo J{j} missing from calibration")
            cal = self._calibration[str(j)]
            for key in ("home_tick", "ticks_per_rad", "dir_sign"):
                if key not in cal:
                    raise ServoCalibrationError(f"Servo J{j} missing key '{key}'")

    def _cal(self, servo_id: int) -> dict:
        """Return the calibration dict for a servo, or raise if uncalibrated."""
        if str(servo_id) not in self._calibration:
            raise ServoCalibrationError(f"Servo {servo_id} not calibrated")
        return self._calibration[str(servo_id)]

    def rad_to_ticks(self, servo_id: int, angle_rad: float) -> int:
        """Convert a joint angle (radians, relative to home) to a raw tick target."""
        cal = self._cal(servo_id)
        return int(round(cal["home_tick"] + cal["dir_sign"] * angle_rad * cal["ticks_per_rad"]))

    def ticks_to_rad(self, servo_id: int, ticks: int) -> float:
        """Convert a raw present-position tick reading to a joint angle (radians)."""
        cal = self._cal(servo_id)
        return cal["dir_sign"] * (ticks - cal["home_tick"]) / cal["ticks_per_rad"]

    # --- low-level packet protocol ---------------------------------------

    def _checksum(self, body: bytes) -> int:
        """Feetech checksum: ~(ID + LENGTH + INSTRUCTION + PARAMs) & 0xFF."""
        return (~sum(body)) & 0xFF

    def _make_packet(self, servo_id: int, instruction: int, params: bytes) -> bytes:
        """Build a full instruction packet: FF FF ID LEN INST PARAMS CHECKSUM."""
        length = len(params) + 2  # spec: N params + instruction byte + checksum byte
        body = bytes([servo_id, length, instruction]) + params
        checksum = self._checksum(body)
        return bytes([self.HEADER, self.HEADER]) + body + bytes([checksum])

    def _send(self, packet: bytes) -> bool:
        """Write a packet to the bus, flushing any stale input first."""
        try:
            self.serial.reset_input_buffer()
            self.serial.write(packet)
            return True
        except serial.SerialException as e:
            logger.error(f"Serial write failed: {e}")
            return False

    def _read_status(self, servo_id: int) -> Optional[bytes]:
        """Read a status packet and return its parameter bytes (after the ERROR byte).

        Status framing: FF FF ID LENGTH ERROR PARAM_1..N CHECKSUM
        Returns the PARAM bytes, or None on timeout / framing / checksum error.
        """
        try:
            # Sync to the 0xFF 0xFF header (tolerate one stray leading byte).
            header = self.serial.read(2)
            if len(header) < 2:
                return None
            if header != b"\xFF\xFF":
                nxt = self.serial.read(1)
                if not nxt or header[1:2] + nxt != b"\xFF\xFF":
                    return None

            sid_b = self.serial.read(1)
            if not sid_b or sid_b[0] != servo_id:
                return None
            servo_response_id = sid_b[0]

            length_b = self.serial.read(1)
            if not length_b:
                return None
            length = length_b[0]

            rest = self.serial.read(length)  # ERROR + PARAMS + CHECKSUM
            if len(rest) < length or length < 2:
                return None

            error = rest[0]
            params = rest[1:-1]
            received_checksum = rest[-1]
            expected_checksum = self._checksum(
                bytes([servo_response_id, length, error]) + params
            )
            if received_checksum != expected_checksum:
                logger.warning(f"Checksum mismatch from servo {servo_id}")
                return None
            if error != 0:
                logger.warning(f"Servo {servo_id} reported error byte 0x{error:02X}")

            return params

        except (serial.SerialException, IndexError):
            return None

    def _write_register(self, servo_id: int, address: int, data: bytes) -> bool:
        """WRITE_DATA instruction: write `data` bytes starting at `address`."""
        params = bytes([address]) + data
        return self._send(self._make_packet(servo_id, self.INST_WRITE, params))

    def _read_register(self, servo_id: int, address: int, count: int) -> Optional[bytes]:
        """READ_DATA instruction: request `count` bytes from `address`, return them."""
        params = bytes([address, count])
        if not self._send(self._make_packet(servo_id, self.INST_READ, params)):
            return None
        data = self._read_status(servo_id)
        if data is None or len(data) < count:
            return None
        return data[:count]

    # --- public API -------------------------------------------------------

    def read_position(self, servo_id: int) -> int:
        """Read present position from a servo without commanding it.

        Args:
            servo_id: servo ID (1-6).

        Returns:
            Present position in ticks (0..4095).

        Raises:
            RuntimeError: if the read fails or the servo does not respond.
        """
        data = self._read_register(servo_id, self.ADDR_PRESENT_POSITION, 2)
        if data is None:
            raise RuntimeError(f"No response reading position from servo {servo_id}")
        return data[0] | (data[1] << 8)  # little-endian

    def move_and_verify(self, servo_id: int, target_ticks: int, tolerance_ticks: int = None) -> int:
        """Command a servo to a position and verify it actually moved there.

        Args:
            servo_id: servo ID (1-6).
            target_ticks: goal position in raw ticks (clamped to 0..4095).
            tolerance_ticks: max acceptable error in ticks (default from config).

        Returns:
            Actual present position (ticks) read back after the move.

        Raises:
            ServoCalibrationError: if servo_id is not calibrated.
            RuntimeError: if the command or read-back fails.
        """
        if tolerance_ticks is None:
            tolerance_ticks = config.SERVO_READ_VERIFY_TOLERANCE_TICKS

        self._cal(servo_id)  # raises ServoCalibrationError if uncalibrated

        target_ticks = max(self.TICK_MIN, min(self.TICK_MAX, int(target_ticks)))

        goal_lo = target_ticks & 0xFF
        goal_hi = (target_ticks >> 8) & 0xFF
        if not self._write_register(servo_id, self.ADDR_GOAL_POSITION, bytes([goal_lo, goal_hi])):
            raise RuntimeError(f"Failed to command servo {servo_id}")

        present_ticks = self.read_position(servo_id)

        error_ticks = abs(present_ticks - target_ticks)
        if error_ticks > tolerance_ticks:
            logger.warning(
                f"Servo {servo_id} position verify failed: "
                f"commanded {target_ticks}, read {present_ticks}, "
                f"error {error_ticks} > tolerance {tolerance_ticks}"
            )
        else:
            logger.debug(
                f"Servo {servo_id}: cmd {target_ticks} -> read {present_ticks} "
                f"(err {error_ticks}/{tolerance_ticks} ticks)"
            )
        return present_ticks

    def close(self):
        """Close the serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Servo bus closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

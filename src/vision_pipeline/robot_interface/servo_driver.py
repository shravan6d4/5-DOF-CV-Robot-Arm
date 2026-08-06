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
import time
from pathlib import Path
from typing import Optional

import serial

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_calibration import (
    unwrap_tick,
    wrapped_past_seam,
)

logger = logging.getLogger(__name__)


class ServoCalibrationError(Exception):
    """Raised when servo calibration fails or is incomplete."""
    pass


class ServoSafetyError(Exception):
    """Raised when a commanded move is refused as unsafe before being sent.

    Distinct from a comms failure: nothing was written to the bus, the arm has
    not moved, and retrying the identical command will fail identically. The
    caller must decide (re-plan, or have an operator reposition the arm).
    """
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
    ADDR_ID = 0x05                  # 5, 1 byte, servo bus ID (0..253), EEPROM
    ADDR_LOCK = 0x37                # 55, 1 byte, EEPROM write-protect (0=unlocked, 1=locked)
    # Both SRAM (the SRAM block starts at 40 = Torque Enable), so these are
    # volatile: they reset to the servo's defaults on every power cycle and must
    # be re-applied after one. Writing them is NOT an EEPROM operation.
    ADDR_ACCELERATION = 0x29        # 41, 1 byte, units of 100 ticks/s^2
    ADDR_GOAL_SPEED = 0x2E          # 46, 2 bytes little-endian, ticks/s (0 = max)
    # Health registers. A servo that has tripped its own overload or thermal
    # protection still answers the bus but refuses to hold position, which looks
    # identical to "no power" from the outside -- these tell the two apart.
    ADDR_TORQUE_ENABLE = 0x28       # 40, 1 byte, 0 = torque off (limp)
    ADDR_PRESENT_LOAD = 0x3C        # 60, 2 bytes, signed-magnitude
    ADDR_PRESENT_VOLTAGE = 0x3E     # 62, 1 byte, units of 0.1 V
    ADDR_PRESENT_TEMPERATURE = 0x3F  # 63, 1 byte, degrees C
    ADDR_STATUS = 0x41              # 65, 1 byte, error bitfield

    TICK_MIN = 0
    TICK_MAX = 4095                 # STS3215 is 12-bit (0..4095)
    EEPROM_SETTLE_S = 0.02          # let an EEPROM write commit before the next command

    def __init__(self, port: str, baud: int = 1000000, calibration_path: Optional[str] = None,
                 limit_margin_ticks: int = config.SERVO_LIMIT_MARGIN_TICKS):
        """Initialize the servo bus.

        Args:
            port: serial port name (e.g., 'COM3', '/dev/ttyUSB0').
            baud: baud rate (default 1000000 for STS3215).
            calibration_path: path to servo_calibration.json, or None to use config default.
            limit_margin_ticks: widen every measured travel range by this many
                ticks at BOTH ends. Zero by default, and it should stay zero
                unless a specific range is known to be too tight. The recorded
                limits carry a `limit_basis` field saying how they were found:
                J2's and J3's are GROUND-DERIVED, meaning the joint stopped
                because the claw met the table at the elbow angle used when
                measuring, not because the joint ran out of travel. Fold the
                elbow differently and the same angle is safe, which is exactly
                when a margin is justified. A margin on a MECHANICAL limit is
                not justified and drives the joint into a hard stop.

        Raises:
            ServoCalibrationError: if calibration is missing or invalid.
            serial.SerialException: if the port cannot be opened.
        """
        self.port = port
        self.baud = baud
        self.serial = None
        self._calibration = {}
        self.limit_margin_ticks = max(0, int(limit_margin_ticks))

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
                "1": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1,
                      "home_angle_rad": 1.629019},
                ...
            }

        'home_angle_rad' is required rather than defaulted: a file written before
        it existed would otherwise silently fall back to 0 and reintroduce the
        MATLAB-absolute vs servo-relative zero mismatch, which produces confident
        wrong poses instead of an error.
        """
        calibration_path = Path(path)
        if calibration_path.exists():
            try:
                # UTF-8 EXPLICITLY. open() with no encoding uses the platform
                # default, which is cp1252 on Windows, while this file is
                # written as UTF-8 (servo_calibration.save_calibration_file) and
                # carries prose notes full of em-dashes. The mismatch is
                # invisible until a note contains a byte cp1252 has no character
                # for, at which point the loader raises UnicodeDecodeError from
                # inside a codec and the arm looks like a driver failure.
                #
                # It stayed hidden this long because the two scripts that
                # rewrite the file used json.dumps' default ensure_ascii=True,
                # quietly re-encoding every non-ASCII character as \\uXXXX and
                # scrubbing the file clean on the way past. The first write that
                # preserved the characters broke every test that opens a bus.
                with open(calibration_path, encoding="utf-8") as f:
                    self._calibration = json.load(f)
                logger.info(f"Loaded servo calibration from {path}")
            except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
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
            for key in ("home_tick", "ticks_per_rad", "dir_sign", "home_angle_rad"):
                if key not in cal:
                    raise ServoCalibrationError(
                        f"Servo J{j} missing key '{key}'"
                        + (
                            " — calibration files written before the MATLAB home"
                            " offset was introduced lack it; add it from"
                            " config.MATLAB_HOME_DEG (J6 uses 0.0)."
                            if key == "home_angle_rad"
                            else ""
                        )
                    )

    @property
    def calibration(self) -> dict:
        """The loaded calibration dict, keyed by string joint ID.

        Read-only accessor so callers can inspect provenance (dir_sign_basis and
        friends) without reaching into _calibration or re-reading the file behind
        the bus's back — a second read could pick up a different file if this bus
        was constructed with a non-default calibration_path.
        """
        return self._calibration

    def _cal(self, servo_id: int) -> dict:
        """Return the calibration dict for a servo, or raise if uncalibrated."""
        if str(servo_id) not in self._calibration:
            raise ServoCalibrationError(f"Servo {servo_id} not calibrated")
        return self._calibration[str(servo_id)]

    def rad_to_ticks(self, servo_id: int, angle_rad: float) -> int:
        """Convert a MATLAB joint angle (radians, absolute) to a raw tick target.

        `angle_rad` is in the MATLAB model's frame — the same absolute convention
        ik_fk_server.m solves and reports in, where a joint at home reads its
        `home_angle_rad` rather than 0. Subtracting that offset is what puts the
        two halves of the system on a common zero.
        """
        cal = self._cal(servo_id)
        offset_rad = angle_rad - cal["home_angle_rad"]
        return int(round(cal["home_tick"] + cal["dir_sign"] * offset_rad * cal["ticks_per_rad"]))

    def ticks_to_rad(self, servo_id: int, ticks: int) -> float:
        """Convert a raw present-position tick reading to a MATLAB joint angle.

        Inverse of rad_to_ticks: returns the ABSOLUTE angle in the MATLAB model's
        frame, so the result can be handed straight to request_fk/request_ik.
        """
        cal = self._cal(servo_id)
        offset_rad = cal["dir_sign"] * (ticks - cal["home_tick"]) / cal["ticks_per_rad"]
        return cal["home_angle_rad"] + offset_rad

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

    def set_speed(self, servo_id: int, ticks_per_sec: int) -> bool:
        """Cap how fast this servo travels toward its goal position.

        Without this the servo slews to every Goal Position at its default
        speed, which is fast enough that a wrong move is over before an operator
        can react -- capping the SIZE of a step (SERVO_MAX_MOVE_DELTA_TICKS)
        bounds where it ends up, not how violently it gets there. During
        bring-up, where a move going the wrong way is a live possibility, slow
        is the difference between "watch it and cut power" and "hear a bang".

        Args:
            servo_id: servo ID (1-6).
            ticks_per_sec: speed limit, 0 for the servo's maximum. 4096 ticks is
                a full revolution, so 200 ticks/s is roughly 18 deg/s.

        Returns:
            True if the write was acknowledged.

        Note:
            SRAM, so it does NOT survive a power cycle. Re-apply after one.
        """
        value = max(0, min(int(ticks_per_sec), 0xFFFF))
        ok = self._write_register(
            servo_id, self.ADDR_GOAL_SPEED, bytes([value & 0xFF, (value >> 8) & 0xFF])
        )
        if not ok:
            logger.warning(f"Servo {servo_id}: speed limit write not acknowledged")
        return ok

    def set_acceleration(self, servo_id: int, accel: int) -> bool:
        """Ramp rate toward the speed limit, in units of 100 ticks/s^2 (0 = max).

        Pairs with set_speed: a low speed with maximum acceleration still starts
        with a jerk, which on a loaded arm shows up as the whole assembly
        rocking. Keeping both low is what makes the motion look deliberate.
        """
        value = max(0, min(int(accel), 0xFF))
        ok = self._write_register(servo_id, self.ADDR_ACCELERATION, bytes([value]))
        if not ok:
            logger.warning(f"Servo {servo_id}: acceleration write not acknowledged")
        return ok

    def set_motion_profile(self, servo_ids, ticks_per_sec: int, accel: int) -> None:
        """Apply the same speed and acceleration limits to several servos."""
        for servo_id in servo_ids:
            self.set_speed(servo_id, ticks_per_sec)
            self.set_acceleration(servo_id, accel)

    def move_joints_stepped(
        self,
        targets: dict,
        step_ticks: Optional[int] = None,
        pause_s: Optional[float] = None,
        progress=None,
    ) -> dict:
        """Move several joints to their targets together, in small paced steps.

        The motion primitive for anything near the table. A servo commanded
        straight to a distant goal slews there at whatever speed it can manage
        and stops hard at the end; broken into short hops with a pause between
        them, the same move becomes something an operator can watch and stop.
        `config.PICK_STEP_TICKS` / `PICK_STEP_PAUSE_S` set the pace.

        All joints advance TOGETHER, a fraction of their travel per round,
        rather than each being driven to its target in turn. Sequential motion
        sends the arm through poses nobody planned -- swinging the base through
        its full arc while the elbow is still folded back, for instance -- and
        those intermediate poses are where the claw hits things.

        This is separate from the speed cap in set_motion_profile and both
        matter: the speed cap governs how fast a single hop executes, this
        governs how far each hop goes and how long the arm rests between them.

        Args:
            targets: {servo_id: goal_tick}.
            step_ticks: maximum ticks any joint moves per round.
            pause_s: seconds to rest between rounds.
            progress: optional callback(round, total) for UI.

        Returns:
            {servo_id: final_position} read back after the last step.

        Raises:
            ServoSafetyError: propagated from move_and_verify. The arm stops
                part-way; joints already moved stay where they are.
        """
        if step_ticks is None:
            step_ticks = config.PICK_STEP_TICKS
        if pause_s is None:
            pause_s = config.PICK_STEP_PAUSE_S
        step_ticks = max(1, min(int(step_ticks), config.SERVO_MAX_MOVE_DELTA_TICKS))

        starts = {sid: self.read_position(sid) for sid in targets}
        deltas = {sid: targets[sid] - starts[sid] for sid in targets}
        biggest = max((abs(d) for d in deltas.values()), default=0)
        if biggest == 0:
            return starts
        rounds = int(-(-biggest // step_ticks))    # ceil

        last = dict(starts)
        for k in range(1, rounds + 1):
            for sid, goal in targets.items():
                want = goal if k == rounds else starts[sid] + int(round(deltas[sid] * k / rounds))
                if want == last[sid]:
                    continue                      # no-op, and no bus traffic for it
                self.move_and_verify(sid, want)
                last[sid] = want
            if progress:
                progress(k, rounds)
            if pause_s and k < rounds:
                time.sleep(pause_s)

        return {sid: self.read_position(sid) for sid in targets}

    def freeze(self, servo_ids) -> dict:
        """Stop every listed joint where it stands, WITHOUT dropping the arm.

        The software e-stop, and better than the power cut for almost every
        failure. Once a Goal Position has been written the servo will travel to
        it whether or not anything is still talking to it -- killing the script
        does not stop the arm. The only ways to stop it are to cut power, which
        drops holding torque on every joint simultaneously and lets the arm
        fall (this is how J3 was overloaded on 2026-08-04, falling face-first
        after an operator hit the power cut mid-move), or to overwrite the goal
        with where the joint already is. This does the latter.

        Deliberately does no verification and no settling -- it is meant to run
        in milliseconds. Failures are collected and returned rather than raised,
        because a servo that cannot be frozen must not prevent the others from
        being frozen.

        RETRIED IN PASSES, not per joint. Every joint gets one fast attempt
        first, and only the ones that failed are retried. Retrying a joint
        in place would make a single unresponsive servo's timeouts delay the
        freeze of every joint after it -- and during a freeze those joints are
        still travelling toward their old goals. Observed 2026-08-05: a run hit
        a bus error and the freeze that followed silently failed on two of five
        joints, which is the soft e-stop not working at the moment it was
        called for. A dropped byte on a shared serial chain is common; giving up
        on the first one is not acceptable for this particular function.

        Returns:
            {servo_id: position_held} for each joint successfully frozen.
        """
        held = {}
        pending = list(servo_ids)
        errors: dict = {}

        for _attempt in range(max(1, config.SERVO_FREEZE_ATTEMPTS)):
            failed = []
            for servo_id in pending:
                try:
                    present = self.read_position(servo_id)
                    self._write_register(
                        servo_id, self.ADDR_GOAL_POSITION,
                        bytes([present & 0xFF, (present >> 8) & 0xFF]),
                    )
                    held[servo_id] = present
                except Exception as e:
                    errors[servo_id] = e
                    failed.append(servo_id)
            pending = failed
            if not pending:
                break
            # Same reasoning as read_position_retrying: the reads fail because
            # of a noise burst, and asking again inside the same burst fails
            # again. Kept short — the joints still pending are still moving.
            time.sleep(config.SERVO_READ_RETRY_BACKOFF_S)

        for servo_id in pending:
            logger.error(
                f"Servo {servo_id}: FREEZE FAILED after "
                f"{config.SERVO_FREEZE_ATTEMPTS} attempts — {errors[servo_id]}. "
                f"This joint is still travelling to its last goal. If it does not "
                f"stop, cut power."
            )
        return held

    def enable_torque(self, servo_id: int) -> int:
        """Re-enable a servo's torque WITHOUT it lurching to a stale goal.

        A servo that has tripped its overload protection sits limp, answering
        the bus normally, while gravity moves the joint somewhere else entirely
        (J3 sagged 54 deg this way on 2026-08-04). Its Goal Position register
        still holds whatever it was last commanded to. Enabling torque with that
        stale goal in place makes the servo snap back to it at full speed, under
        no supervision, from a pose nobody chose -- the arm's most dangerous
        single instruction.

        So: read where the joint actually IS, make that the goal, and only then
        enable torque. The servo wakes up holding its current position.

        Returns:
            The position it is now holding.
        """
        present = self.read_position(servo_id)
        if not self._write_register(
            servo_id, self.ADDR_GOAL_POSITION,
            bytes([present & 0xFF, (present >> 8) & 0xFF]),
        ):
            raise RuntimeError(f"Servo {servo_id}: could not set holding goal")
        time.sleep(0.02)
        if not self._write_register(servo_id, self.ADDR_TORQUE_ENABLE, bytes([1])):
            raise RuntimeError(f"Servo {servo_id}: could not enable torque")
        logger.info(f"Servo {servo_id}: torque enabled, holding {present}")
        return present

    def disable_torque(self, servo_id: int) -> bool:
        """Go limp. The joint will then be moved by gravity — support it first."""
        return self._write_register(servo_id, self.ADDR_TORQUE_ENABLE, bytes([0]))

    def read_diagnostics(self, servo_id: int) -> dict:
        """Read a servo's health registers: voltage, temperature, load, faults.

        Returns whatever it can rather than raising, because the whole point is
        to characterise a servo that is already misbehaving -- a partial answer
        ("answers the bus, reports 6.2 V") is far more diagnostic than an
        exception. Keys are absent when that register did not come back.

        `status` is the raw fault bitfield; `faults` is a best-effort decode of
        it. Trust the raw byte over the decode -- the bit meanings vary across
        Feetech firmware revisions and are not something this driver can verify.
        """
        out: dict = {}
        readers = (
            ("voltage_v", self.ADDR_PRESENT_VOLTAGE, 1, lambda b: b[0] / 10.0),
            ("temperature_c", self.ADDR_PRESENT_TEMPERATURE, 1, lambda b: b[0]),
            ("torque_enabled", self.ADDR_TORQUE_ENABLE, 1, lambda b: bool(b[0])),
            ("status", self.ADDR_STATUS, 1, lambda b: b[0]),
            ("load", self.ADDR_PRESENT_LOAD, 2, lambda b: b[0] | (b[1] << 8)),
        )
        for name, addr, count, decode in readers:
            try:
                raw = self._read_register(servo_id, addr, count)
                if raw is not None and len(raw) >= count:
                    out[name] = decode(raw)
            except Exception:
                pass

        if "status" in out:
            bits = {0x01: "voltage", 0x02: "angle sensor", 0x04: "overheat",
                    0x08: "current", 0x10: "angle limit", 0x20: "overload"}
            out["faults"] = [n for b, n in bits.items() if out["status"] & b]
        return out

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

    def ping(self, servo_id: int) -> bool:
        """Return True if a servo answers at this ID (reads its ID register).

        Unlike read_position/move_and_verify this ignores calibration, so it works
        on any raw ID — including a factory-default servo you haven't set up yet.
        """
        data = self._read_register(servo_id, self.ADDR_ID, 1)
        return data is not None and len(data) >= 1

    def scan_ids(self, id_range=range(0, 21)) -> list[int]:
        """Return the IDs currently answering on the bus (pings each in id_range).

        Handy for finding a servo's current ID before reassigning it. Absent IDs
        each cost one serial timeout, so a wide range is slow — the default 0..20
        covers factory defaults plus our J1..J6 with margin.
        """
        return [i for i in id_range if self.ping(i)]

    def write_servo_id(self, old_id: int, new_id: int, verify: bool = True) -> bool:
        """Permanently change a servo's bus ID (an EEPROM write).

        !!! ONLY ONE SERVO MAY BE ON THE BUS when you call this. Every servo
        currently at `old_id` gets reprogrammed, and factory servos usually all
        ship at the SAME default ID — so set IDs one servo at a time, before
        wiring the whole chain together, or they'll collide.

        Feetech EEPROM sequence: unlock (LOCK=0) -> write ID -> re-lock (LOCK=1).
        The re-lock is addressed to the NEW id, because the servo starts answering
        to it the instant the ID register is written.

        Args:
            old_id: the servo's current ID (use scan_ids/ping to find it).
            new_id: the desired ID (0..253; this project uses 1..6 for J1..J6).
            verify: if True, ping the new ID afterwards and only return True if it answers.

        Returns:
            True on success (verified if verify=True), False if any step failed.

        Raises:
            ValueError: if old_id or new_id is outside 0..253.
        """
        for label, value in (("old_id", old_id), ("new_id", new_id)):
            if not 0 <= value <= 253:
                raise ValueError(f"{label} must be 0..253, got {value}")

        # Unlock EEPROM on the current ID.
        if not self._write_register(old_id, self.ADDR_LOCK, bytes([0])):
            logger.error(f"Failed to unlock EEPROM on servo {old_id}")
            return False
        time.sleep(self.EEPROM_SETTLE_S)

        # Write the new ID; the servo answers to new_id from here on.
        if not self._write_register(old_id, self.ADDR_ID, bytes([new_id])):
            logger.error(f"Failed to write new ID {new_id} to servo {old_id}")
            return False
        time.sleep(self.EEPROM_SETTLE_S)

        # Re-lock EEPROM, now addressing the NEW id.
        if not self._write_register(new_id, self.ADDR_LOCK, bytes([1])):
            logger.error(f"Wrote ID {new_id} but failed to re-lock EEPROM")
            return False
        time.sleep(self.EEPROM_SETTLE_S)

        if verify and not self.ping(new_id):
            logger.error(f"Servo did not answer at new ID {new_id} after the write")
            return False

        logger.info(f"Servo ID changed {old_id} -> {new_id}"
                    f"{' (verified)' if verify else ''}.")
        return True

    def move_and_verify(
        self,
        servo_id: int,
        target_ticks: int,
        tolerance_ticks: int = None,
        max_delta_ticks: int = None,
    ) -> int:
        """Command a servo to a position and verify it actually arrived there.

        Reads the current position first and refuses the move outright if the
        travel exceeds `max_delta_ticks`, then waits for the servo to physically
        settle before reading back — a read taken immediately after the write
        catches the servo mid-travel and reports a stale position, which callers
        would store as the arm's true state.

        Args:
            servo_id: servo ID (1-6).
            target_ticks: goal position in raw ticks (clamped to 0..4095).
            tolerance_ticks: max acceptable error in ticks (default from config).
            max_delta_ticks: refuse moves travelling further than this from the
                current position (default from config). Pass a larger value to
                deliberately override for a known-safe long move.

        Returns:
            Actual present position (ticks) read back after the servo settled.

        Raises:
            ServoCalibrationError: if servo_id is not calibrated.
            ServoSafetyError: if the move exceeds max_delta_ticks (nothing sent).
            RuntimeError: if the command or read-back fails.
        """
        if tolerance_ticks is None:
            tolerance_ticks = config.SERVO_READ_VERIFY_TOLERANCE_TICKS
        if max_delta_ticks is None:
            max_delta_ticks = config.SERVO_MAX_MOVE_DELTA_TICKS

        self._cal(servo_id)  # raises ServoCalibrationError if uncalibrated

        target_ticks = max(self.TICK_MIN, min(self.TICK_MAX, int(target_ticks)))

        # --- safety gate: refuse before writing anything to the bus ---------
        # Retrying here is free: nothing has been committed yet, so a transient
        # read glitch shouldn't abort an otherwise-fine move attempt.
        start_ticks = self._read_position_retrying(servo_id)
        delta = abs(target_ticks - start_ticks)
        if delta > max_delta_ticks:
            raise ServoSafetyError(
                f"Refusing to move servo {servo_id}: {start_ticks} -> {target_ticks} "
                f"is {delta} ticks (cap {max_delta_ticks}). Nothing was commanded. "
                f"A delta this large is usually an encoder wrap-seam artifact or a bad "
                f"solve, not a real target — verify the servo's position by hand before "
                f"overriding with max_delta_ticks."
            )
        self._check_travel_limits(servo_id, start_ticks, target_ticks)

        goal_lo = target_ticks & 0xFF
        goal_hi = (target_ticks >> 8) & 0xFF
        if not self._write_register(servo_id, self.ADDR_GOAL_POSITION, bytes([goal_lo, goal_hi])):
            raise RuntimeError(f"Failed to command servo {servo_id}")

        present_ticks = self._wait_for_settle(servo_id, target_ticks, tolerance_ticks)

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

    def travel_limits(self, servo_id: int) -> Optional[tuple[int, int]]:
        """This servo's (min_tick, max_tick), or None if it has not been measured.

        Absent by design rather than defaulted: an invented range is worse than
        none, because it reads as protection while permitting the very moves it
        appears to forbid. Measure with scripts/find_joint_limits.py.
        """
        cal = self._cal(servo_id)
        lo, hi = cal.get("min_tick"), cal.get("max_tick")
        if lo is None or hi is None:
            return None
        return int(lo) - self.limit_margin_ticks, int(hi) + self.limit_margin_ticks

    def _check_travel_limits(self, servo_id: int, start_ticks: int, target_ticks: int) -> None:
        """Refuse a move that would leave this joint's measured travel range.

        The existing max_delta gate only bounds how far ONE command travels, so
        a joint can be walked into a hard stop in small legal steps — which is
        how both jams on 2026-08-04 happened. This bounds WHERE the joint may
        go, not just how far it moves at once.

        A joint already outside its range is not trapped: moves that reduce the
        violation are allowed, so an arm parked out of bounds can always be
        driven back in. Only moves that go further out are refused.

        Raises:
            ServoSafetyError: target is out of range and not an improvement.
        """
        limits = self.travel_limits(servo_id)
        if limits is None:
            return
        lo, hi = limits
        if lo <= target_ticks <= hi:
            return

        def violation(t: int) -> int:
            return max(lo - t, t - hi, 0)

        # Holding still is ALWAYS allowed, in range or out of it. Commanding a
        # joint to the position it already occupies moves it nowhere, so there
        # is nothing for a travel limit to protect against -- and refusing it
        # breaks the one thing that helps most when a joint is out of range.
        #
        # Found 2026-08-05: J3 had sagged to tick 190, just below its limit of
        # 200. hold_pose.py asked it to hold at 190 and this check refused,
        # because `violation(target) < violation(start)` is false when the two
        # are equal. With no goal written the joint kept falling -- 190, then
        # 155 on the next attempt -- and every further attempt to catch it was
        # refused for being even further out. The limit was actively preventing
        # the rescue of the joint it had trapped.
        if target_ticks == start_ticks:
            if violation(start_ticks) > 0:
                logger.warning(
                    f"Servo {servo_id} is outside its travel range [{lo}, {hi}] "
                    f"at {start_ticks}; allowing it to HOLD there (zero motion). "
                    f"Drive it back into range before commanding anything else."
                )
            return

        # Everything past this point compares tick numbers, and that arithmetic
        # is only meaningful while the reading and the limits share a numbering.
        # Once a joint crosses the 0/4095 seam they do not: J3 read 4079 on
        # 2026-08-05 while really sitting 80 ticks BELOW a minimum of 63, so
        # `violation` scored it ~3000 ticks past its MAXIMUM and the direction
        # that recovers it looked like the direction that makes it worse. The
        # one useful move was the one refused, under a message blaming a travel
        # range that was entirely correct.
        #
        # Checked after the hold case above, not before: holding a wrapped joint
        # writes its own present position as the goal and moves it nowhere, and
        # that is exactly the rescue an operator reaches for. Only MOTION is
        # refused.
        if wrapped_past_seam(start_ticks, lo, hi):
            unwrapped = unwrap_tick(start_ticks, (lo + hi) // 2)
            raise ServoSafetyError(
                f"Servo {servo_id}'s position reading has WRAPPED past the 0/4095 "
                f"encoder seam: it reads {start_ticks}, which is really {unwrapped} "
                f"relative to its range [{lo}, {hi}]. Nothing was commanded, and no "
                f"goal position can fix this — the servo drives goals linearly and "
                f"would take the long way round the circle, through every hard stop "
                f"between here and there. The range is NOT wrong; do not re-measure "
                f"it. Move the seam instead, which needs no motion at all: "
                f"python scripts/recentre_joint.py --joint {servo_id} --here"
            )

        if violation(target_ticks) < violation(start_ticks):
            logger.warning(
                f"Servo {servo_id} is outside its travel range [{lo}, {hi}] at "
                f"{start_ticks}; allowing {target_ticks} because it moves back toward range."
            )
            return

        raise ServoSafetyError(
            f"Refusing to move servo {servo_id} to {target_ticks}: outside its "
            f"measured travel range [{lo}, {hi}] (currently at {start_ticks}). "
            f"Nothing was commanded. Small in-range steps can still walk a joint "
            f"into a hard stop, which is what this prevents — re-measure with "
            f"scripts/find_joint_limits.py if the range itself is wrong."
        )

    def read_position_retrying(self, servo_id: int) -> int:
        """read_position, absorbing transient serial failures with a backoff.

        Use this for any read taken while the arm is moving or has just moved.
        Plain read_position stays un-retried because ping/scan_ids depend on
        "no answer" meaning "not there".

        THE BACKOFF IS THE POINT. This originally retried in a tight loop, which
        cannot help with the failure it exists for: reads fail because motor
        current puts noise on the shared serial line, and noise comes in bursts
        lasting longer than three immediate retries take. All three attempts
        landed inside the same burst and the loop reported a dead servo --
        observed 2026-08-05, where J2 answered perfectly at rest, before and
        after, and the bus was healthy on every read-only check. Waiting a few
        milliseconds and asking again is what actually crosses a burst.
        """
        last_error = None
        attempts = max(1, config.SERVO_MOVE_READ_RETRIES)
        for attempt in range(attempts):
            try:
                return self.read_position(servo_id)
            except RuntimeError as e:
                last_error = e
                if attempt < attempts - 1:
                    # Growing pause: a short burst clears quickly, a long one
                    # needs more room, and doubling covers both without a fixed
                    # guess at how long the noise lasts. Capped, because doubling
                    # over a dozen attempts would otherwise reach minutes.
                    time.sleep(min(config.SERVO_READ_RETRY_BACKOFF_S * (2 ** attempt),
                                   config.SERVO_READ_RETRY_MAX_S))
        raise RuntimeError(
            f"Servo {servo_id} did not respond after {attempts} attempts "
            f"spanning {self._backoff_total():.2f}s: {last_error}. "
            f"If the joint is visibly holding torque and answers "
            f"scripts/check_servo_health.py at rest, this is line noise under "
            f"motor current, not a failed servo."
        )

    # Kept for the internal call sites that predate the public name.
    _read_position_retrying = read_position_retrying

    @staticmethod
    def _backoff_total() -> float:
        """Total time the retry backoff spans, for error messages."""
        attempts = max(1, config.SERVO_MOVE_READ_RETRIES)
        return sum(min(config.SERVO_READ_RETRY_BACKOFF_S * (2 ** i),
                       config.SERVO_READ_RETRY_MAX_S)
                   for i in range(attempts - 1))

    def _wait_for_settle(self, servo_id: int, target_ticks: int, tolerance_ticks: int) -> int:
        """Poll present position until the servo arrives, stalls, or times out.

        Polling rather than a fixed sleep: a 5-tick nudge and a 300-tick sweep take
        very different times, and a blind delay is either wastefully slow or too
        short. Returns as soon as the servo is within tolerance of the target.

        Stall detection matters as much as arrival — a servo blocked by the table
        (see the J3 near-miss during bring-up) will never reach tolerance, and
        without this it would hold against the obstruction for the full timeout.
        Returning early lets the tolerance check above report the mismatch.

        Stall-counting only starts after SERVO_MOVE_STALL_GRACE_S has elapsed —
        a real servo has a command-processing / acceleration ramp-up before it
        visibly starts moving, and without this grace period the stall check
        false-triggers on that ramp instead of a genuine obstruction (caught on
        hardware: an 80-tick move reported "stalled" near its start position,
        but had actually fully arrived by the time a later read checked it).

        The goal position was already written to the servo before this is called,
        so the servo drives toward it via its own onboard control regardless of
        whether OUR polling succeeds — a single dropped byte on the read side
        should not abort verification of an otherwise-successful move. Each poll
        retries a bounded number of times before treating the servo as actually
        unresponsive.

        Returns:
            The last present position read (ticks).

        Raises:
            RuntimeError: if a read fails SERVO_MOVE_READ_RETRIES times in a row
                (real communication loss, not a one-off glitch).
        """
        start = time.monotonic()
        deadline = start + config.SERVO_MOVE_SETTLE_TIMEOUT_S
        present_ticks = self._read_position_retrying(servo_id)
        stalled_polls = 0

        while time.monotonic() < deadline:
            if abs(present_ticks - target_ticks) <= tolerance_ticks:
                return present_ticks

            time.sleep(config.SERVO_MOVE_POLL_INTERVAL_S)
            previous = present_ticks
            present_ticks = self._read_position_retrying(servo_id)

            past_grace = (time.monotonic() - start) >= config.SERVO_MOVE_STALL_GRACE_S
            if not past_grace:
                continue  # still in the acceleration ramp-up window; don't judge yet

            if abs(present_ticks - previous) <= config.SERVO_MOVE_STALL_EPSILON_TICKS:
                stalled_polls += 1
                if stalled_polls >= config.SERVO_MOVE_STALL_POLLS:
                    logger.warning(
                        f"Servo {servo_id} stopped moving at {present_ticks} while "
                        f"travelling to {target_ticks} — obstructed, torque-limited, "
                        f"or past its travel limit."
                    )
                    return present_ticks
            else:
                stalled_polls = 0

        logger.warning(
            f"Servo {servo_id} did not settle within "
            f"{config.SERVO_MOVE_SETTLE_TIMEOUT_S}s (last read {present_ticks}, "
            f"target {target_ticks})."
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

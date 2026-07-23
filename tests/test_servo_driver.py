"""
Protocol-level tests for the Feetech servo driver — no hardware needed.

A FakeServoSerial stands in for pyserial and emulates ONE STS3215 servo: it decodes
the packets the driver writes, applies WRITE_DATA to a tiny register model, and
answers READ_DATA with correctly-framed status packets (right checksum, right
little-endian layout). That lets us verify the exact wire bytes and the multi-step
ID-change sequence against the Feetech spec, which is the part we can't sanity-check
on real hardware from here.
"""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vision_pipeline.robot_interface.servo_driver as sd
from vision_pipeline import config
from vision_pipeline.robot_interface.servo_driver import ServoBus

NO_CAL_FILE = "__nonexistent_servo_cal__.json"  # forces config fallback calibration


class FakeServoSerial:
    """Emulates a single Feetech servo over the serial line."""

    def __init__(self, servo_id=1, present_ticks=2048):
        self.is_open = True
        self.servo_id = servo_id
        self.present_ticks = present_ticks
        self.written = bytearray()   # every byte the host wrote (for assertions)
        self._in = bytearray()       # bytes queued for the host to read back

    # pyserial surface the driver uses:
    def reset_input_buffer(self):
        self._in.clear()

    def write(self, data):
        self.written += data
        self._handle(bytes(data))
        return len(data)

    def read(self, n):
        chunk = bytes(self._in[:n])
        del self._in[: len(chunk)]
        return chunk

    def close(self):
        self.is_open = False

    # servo emulation:
    def _handle(self, pkt):
        if len(pkt) < 6 or pkt[0] != 0xFF or pkt[1] != 0xFF:
            return
        pid, length, inst = pkt[2], pkt[3], pkt[4]
        params = pkt[5 : 5 + (length - 2)]
        if pid != self.servo_id and pid != 0xFE:  # not us, and not broadcast
            return
        if inst == ServoBus.INST_READ:
            addr, count = params[0], params[1]
            self._enqueue_status(self._reg_bytes(addr, count))
        elif inst == ServoBus.INST_WRITE:
            addr, values = params[0], params[1:]
            if addr == ServoBus.ADDR_ID and values:
                self.servo_id = values[0]  # servo answers to the new ID immediately
            # LOCK / GOAL_POSITION writes are accepted but need no model here.

    def _reg_bytes(self, addr, count):
        if addr == ServoBus.ADDR_ID:
            return bytes([self.servo_id])[:count].ljust(count, b"\x00")
        if addr == ServoBus.ADDR_PRESENT_POSITION:
            return bytes([self.present_ticks & 0xFF, (self.present_ticks >> 8) & 0xFF])[:count]
        return bytes(count)

    def _enqueue_status(self, data):
        body = bytes([self.servo_id, len(data) + 2, 0x00]) + data  # id, len, error, data
        checksum = (~sum(body)) & 0xFF
        self._in += bytes([0xFF, 0xFF]) + body + bytes([checksum])


class MovingFakeServo(FakeServoSerial):
    """Fake servo that actually travels toward its goal, a step per position read.

    The base fake holds `present_ticks` fixed, so a servo is always trivially
    "already arrived" and the settle logic never runs. This one models travel:
    each PRESENT_POSITION read advances it `step_ticks` toward the goal, so a
    move takes a realistic number of polls to complete. `step_ticks=0` models a
    servo that is stuck (obstructed, unpowered, or past a travel limit).
    """

    def __init__(self, servo_id=1, present_ticks=2048, step_ticks=50):
        super().__init__(servo_id=servo_id, present_ticks=present_ticks)
        self.step_ticks = step_ticks
        self.goal_ticks = present_ticks

    def _handle(self, pkt):
        super()._handle(pkt)
        if len(pkt) < 6:
            return
        pid, length, inst = pkt[2], pkt[3], pkt[4]
        if pid != self.servo_id and pid != 0xFE:
            return
        params = pkt[5 : 5 + (length - 2)]
        if inst == ServoBus.INST_WRITE and params and params[0] == ServoBus.ADDR_GOAL_POSITION:
            values = params[1:]
            if len(values) >= 2:
                self.goal_ticks = values[0] | (values[1] << 8)

    def _reg_bytes(self, addr, count):
        data = super()._reg_bytes(addr, count)
        if addr == ServoBus.ADDR_PRESENT_POSITION:
            # Advance AFTER serving this read, so the first read is the position
            # at the moment the goal was issued.
            remaining = self.goal_ticks - self.present_ticks
            if remaining:
                move = min(abs(remaining), self.step_ticks)
                self.present_ticks += move if remaining > 0 else -move
        return data


class DelayedMovingFakeServo(MovingFakeServo):
    """A servo that sits motionless for its first `delay_reads` position reads
    before starting to move — models a real STS3215's command-processing /
    acceleration ramp-up, where the servo hasn't started visibly moving yet
    even though it received the goal write.
    """

    def __init__(self, servo_id=1, present_ticks=2048, step_ticks=50, delay_reads=0):
        super().__init__(servo_id=servo_id, present_ticks=present_ticks, step_ticks=step_ticks)
        self.delay_reads = delay_reads
        self.position_reads = 0

    def _reg_bytes(self, addr, count):
        if addr == ServoBus.ADDR_PRESENT_POSITION:
            self.position_reads += 1
            if self.position_reads <= self.delay_reads:
                return FakeServoSerial._reg_bytes(self, addr, count)  # no advance yet
        return super()._reg_bytes(addr, count)


class FlakyReadFakeServo(FakeServoSerial):
    """A servo whose first `fail_reads` status responses time out.

    Simulates a dropped/corrupted byte on the read side: the header read
    returns empty (what pyserial gives back on a real timeout), which makes
    _read_status bail out after a single read() call — matching how one
    flaky attempt actually behaves on the wire, not just "read() raises".
    """

    def __init__(self, servo_id=1, present_ticks=2048, fail_reads=0):
        super().__init__(servo_id=servo_id, present_ticks=present_ticks)
        self.fail_reads = fail_reads
        self.read_calls = 0

    def read(self, n):
        self.read_calls += 1
        if self.read_calls <= self.fail_reads:
            return b""
        return super().read(n)


def _bus(monkeypatch, fake) -> ServoBus:
    monkeypatch.setattr(sd.serial, "Serial", lambda *a, **k: fake)
    return ServoBus("COM_FAKE", calibration_path=NO_CAL_FILE)


def _parse_packets(buf: bytes):
    """Split a raw host->servo byte stream into decoded packets."""
    packets = []
    i = 0
    while i + 4 <= len(buf):
        if buf[i] != 0xFF or buf[i + 1] != 0xFF:
            i += 1
            continue
        sid, length = buf[i + 2], buf[i + 3]
        total = 4 + length
        pkt = buf[i : i + total]
        inst = pkt[4]
        params = pkt[5 : 5 + (length - 2)]
        packets.append({"id": sid, "inst": inst,
                        "addr": params[0] if params else None,
                        "data": list(params[1:])})
        i += total
    return packets


# --- packet framing / checksum ------------------------------------------

def test_make_packet_matches_feetech_spec(monkeypatch):
    bus = _bus(monkeypatch, FakeServoSerial())
    # WRITE goal position 2048 (0x0800) to servo 1: FF FF 01 05 03 2A 00 08 C4
    pkt = bus._make_packet(1, ServoBus.INST_WRITE, bytes([ServoBus.ADDR_GOAL_POSITION, 0x00, 0x08]))
    assert pkt == bytes([0xFF, 0xFF, 0x01, 0x05, 0x03, 0x2A, 0x00, 0x08, 0xC4])


# --- read / ping / move -------------------------------------------------

def test_read_position_little_endian(monkeypatch):
    bus = _bus(monkeypatch, FakeServoSerial(servo_id=1, present_ticks=0x0123))
    assert bus.read_position(1) == 0x0123


def test_ping_present_and_absent(monkeypatch):
    bus = _bus(monkeypatch, FakeServoSerial(servo_id=1))
    assert bus.ping(1) is True
    assert bus.ping(4) is False  # nothing answers at ID 4


def test_scan_ids_finds_the_one_servo(monkeypatch):
    bus = _bus(monkeypatch, FakeServoSerial(servo_id=3))
    assert bus.scan_ids(range(0, 7)) == [3]


def test_move_and_verify_reads_back_present(monkeypatch):
    bus = _bus(monkeypatch, FakeServoSerial(servo_id=1, present_ticks=2048))
    # Fake doesn't simulate motion, so present stays 2048; commanding 2048 verifies clean.
    assert bus.move_and_verify(1, 2048) == 2048


# --- ID change sequence -------------------------------------------------

def test_write_servo_id_sequence_and_addressing(monkeypatch):
    fake = FakeServoSerial(servo_id=1)
    bus = _bus(monkeypatch, fake)

    assert bus.write_servo_id(1, 5) is True
    assert fake.servo_id == 5  # servo now answers to the new ID

    writes = [p for p in _parse_packets(bytes(fake.written)) if p["inst"] == ServoBus.INST_WRITE]
    # Exactly: unlock@old, write-id@old, re-lock@new — in that order.
    assert writes[0] == {"id": 1, "inst": 0x03, "addr": ServoBus.ADDR_LOCK, "data": [0]}
    assert writes[1] == {"id": 1, "inst": 0x03, "addr": ServoBus.ADDR_ID, "data": [5]}
    # The re-lock MUST be addressed to the NEW id (5), not the old (1) — the whole
    # subtlety of the sequence.
    assert writes[2] == {"id": 5, "inst": 0x03, "addr": ServoBus.ADDR_LOCK, "data": [1]}


def test_write_servo_id_rejects_out_of_range(monkeypatch):
    bus = _bus(monkeypatch, FakeServoSerial())
    with pytest.raises(ValueError):
        bus.write_servo_id(1, 300)
    with pytest.raises(ValueError):
        bus.write_servo_id(-1, 5)


# --- calibration conversions -------------------------------------------

def test_tick_rad_roundtrip(monkeypatch):
    bus = _bus(monkeypatch, FakeServoSerial())
    # J6 fallback: home 2048, 325.95 ticks/rad, dir +1, home_angle_rad 0.0 —
    # the gripper stays on offset-from-home semantics.
    ticks = bus.rad_to_ticks(6, 0.2)
    assert ticks == round(2048 + 0.2 * 325.95)
    assert bus.ticks_to_rad(6, ticks) == pytest.approx(0.2, abs=1e-3)


def test_servo_at_home_maps_to_matlab_home_config(monkeypatch):
    """A servo at home_tick must report the MATLAB angle the arm is actually in.

    MATLAB works in absolute joint angles; a servo at its home tick reads 0.
    Those coincide only because importrobot leaves HomePosition at 0 (it bakes
    the CAD assembly pose into the link transforms instead) — confirmed on
    hardware, where FK of [0,0,0,0,0] reproduced the measured physical pose and
    FK of the smiData Rz.Pos angles did not.

    This pins the mapping in both directions: if someone re-imports the model
    with a non-zero HomePosition, or resurrects the Rz.Pos values as joint
    angles, the reported pose silently stops matching the arm — so assert the
    conversion agrees with whatever home_angle_rad claims.
    """
    bus = _bus(monkeypatch, FakeServoSerial())
    for j in range(1, 6):  # J1..J5, the IK-driven joints
        cal = bus._cal(j)
        assert bus.ticks_to_rad(j, cal["home_tick"]) == pytest.approx(
            cal["home_angle_rad"], abs=1e-9
        ), f"J{j}: servo at home_tick must report exactly home_angle_rad"


def test_rad_ticks_roundtrip_is_exact_across_joints(monkeypatch):
    """rad_to_ticks and ticks_to_rad must invert each other on the IK joints."""
    bus = _bus(monkeypatch, FakeServoSerial())
    for j in range(1, 6):
        home_angle = bus._cal(j)["home_angle_rad"]
        for delta in (-0.4, -0.05, 0.0, 0.05, 0.4):
            angle = home_angle + delta
            recovered = bus.ticks_to_rad(j, bus.rad_to_ticks(j, angle))
            # One tick is ~1/651.89 rad, so rounding bounds the error.
            assert recovered == pytest.approx(angle, abs=2e-3)


def test_j1_dir_sign_is_inverted_reconciled_2026_07_22(monkeypatch):
    """J1's dir_sign must be -1, not the +1 placeholder every other unresolved
    joint still carries.

    Reconciled by comparing the physical bring-up log (README: "+ticks =
    counterclockwise viewed from above") against MATLAB's own convention
    (right-hand rule on J1's FK rotation axis, -Z, gives clockwise from above
    for +angle). Opposite senses -> dir_sign must flip the sign, or a
    positive tick delta gets reported to MATLAB as a negative angle change
    and vice versa -- silently commanding/interpreting the wrong direction.
    Locking this in so it can't drift back to +1 by accident.
    """
    bus = _bus(monkeypatch, FakeServoSerial())
    assert bus._cal(1)["dir_sign"] == -1


def test_j4_dir_sign_stays_positive_reconciled_2026_07_22(monkeypatch):
    """J4's dir_sign is +1 -- confirmed correct, not merely untested.

    J4 and J1 share the same reconciliation method (README bring-up log vs.
    MATLAB's FK rotation axis) but land on opposite conclusions: J4's physical
    "+ticks = ccw from the side" (confirmed vantage point: the left side, +Y)
    matches MATLAB's own +Y-axis convention (also ccw viewed from the left),
    so no flip is needed. Locking in the placeholder value here specifically
    so a future accidental sign flip (e.g. someone "fixing" it to match J1) is
    caught by a test failure, not a wrong-direction move on real hardware.
    """
    bus = _bus(monkeypatch, FakeServoSerial())
    assert bus._cal(4)["dir_sign"] == 1


def test_j2_dir_sign_is_inverted_reconciled_2026_07_22(monkeypatch):
    """J2's dir_sign must be -1.

    J2's physical description ("tilts up") doesn't state a viewing
    convention, so this was reconciled differently from J1/J4: by directly
    comparing which way the WRIST moves for a pure MATLAB +angle delta on J2
    alone (the same observable a human watches during a single-joint jog).
    MATLAB's +angle moves the wrist DOWN (~2.6mm for a 0.05 rad delta from
    true home) -- opposite the physical "tilts up" -- so dir_sign must flip.
    """
    bus = _bus(monkeypatch, FakeServoSerial())
    assert bus._cal(2)["dir_sign"] == -1


def test_j3_dir_sign_stays_positive_reconciled_2026_07_22(monkeypatch):
    """J3's dir_sign is +1 -- confirmed correct via the same wrist-displacement
    method used for J2 (see test_j2_dir_sign_is_inverted_reconciled_2026_07_22).

    MATLAB's +angle moves the wrist DOWN (~6.9mm for a 0.05 rad delta from
    true home), matching the physical "folds down, toward the table" -- same
    sense, no flip needed. Locked in so an accidental "fix" doesn't flip it
    to match J2's sign by mistaken pattern-matching.
    """
    bus = _bus(monkeypatch, FakeServoSerial())
    assert bus._cal(3)["dir_sign"] == 1


# --- move safety + settling ---------------------------------------------

def _goal_writes(fake):
    """Every GOAL_POSITION write the host sent to `fake`, decoded."""
    return [
        p for p in _parse_packets(bytes(fake.written))
        if p["inst"] == ServoBus.INST_WRITE and p["addr"] == ServoBus.ADDR_GOAL_POSITION
    ]


@pytest.fixture
def _fast_polls(monkeypatch):
    """Strip the poll delay and stall grace period so settle tests run at full
    speed. Both are wall-clock-timed (time.monotonic()), so leaving the grace
    period at its real 0.5s would make stall tests either slow or flaky."""
    monkeypatch.setattr(config, "SERVO_MOVE_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(config, "SERVO_MOVE_STALL_GRACE_S", 0.0)


def test_move_waits_for_servo_to_arrive(monkeypatch, _fast_polls):
    """The returned position must be where the servo ENDED, not where it started.

    Reading immediately after the write catches the servo mid-travel; that stale
    value used to flow into HardwareRobot._last_angles_rad and become the camera
    pose the vision pipeline back-projected through.
    """
    fake = MovingFakeServo(servo_id=1, present_ticks=2048, step_ticks=20)
    bus = _bus(monkeypatch, fake)

    target = 2248  # 200 ticks away: ~10 polls at 20 ticks each
    arrived = bus.move_and_verify(1, target, tolerance_ticks=5)

    assert abs(arrived - target) <= 5, "returned a position the servo had not reached"
    assert arrived != 2048, "returned the pre-move position"


def test_move_refuses_travel_beyond_safety_cap(monkeypatch):
    """A too-large move must raise and write NOTHING to the bus.

    This is the J1 wrap-seam runaway guard: a home tick near the 0/4095 seam
    reads as ~17 instead of ~4086 after a power cycle, and a blind return-to-home
    would drive ~358 deg the long way round.
    """
    fake = MovingFakeServo(servo_id=1, present_ticks=17, step_ticks=20)
    bus = _bus(monkeypatch, fake)
    fake.written.clear()

    with pytest.raises(sd.ServoSafetyError, match="Refusing to move"):
        bus.move_and_verify(1, 4086)  # the runaway that actually happened

    assert _goal_writes(fake) == [], "a refused move must not command the servo"
    assert fake.present_ticks == 17, "servo moved despite the refusal"


def test_move_within_cap_is_allowed(monkeypatch, _fast_polls):
    """The same wrap-seam correction is fine once expressed as a short move."""
    fake = MovingFakeServo(servo_id=1, present_ticks=17, step_ticks=10)
    bus = _bus(monkeypatch, fake)
    fake.written.clear()

    arrived = bus.move_and_verify(1, 40, tolerance_ticks=5)  # the ~2 deg it really needed

    assert abs(arrived - 40) <= 5
    # Proves _goal_writes actually detects commands, so the refusal test above
    # is asserting on a working detector rather than passing vacuously.
    assert _goal_writes(fake), "expected a goal-position write for a permitted move"


def test_settle_survives_a_transient_read_glitch(monkeypatch, _fast_polls):
    """A dropped read (gate check or settle poll — both share the same retry
    helper) must not abort a move whose goal has already been written.

    The goal is written to the servo BEFORE polling starts, so it drives
    toward the target via its own onboard control regardless of whether OUR
    verification read succeeds. This is exactly what happened during
    bring-up: move_and_verify crashed on a single flaky read from J2, even
    though the servo had already received the goal and (per a separate
    read-only check afterward) landed almost exactly on target anyway.
    """
    fake = FlakyReadFakeServo(servo_id=1, present_ticks=2048, fail_reads=2)
    bus = _bus(monkeypatch, fake)

    # Under SERVO_MOVE_READ_RETRIES=3 (default), 2 failures then a real
    # response must still resolve correctly.
    arrived = bus.move_and_verify(1, 2048, tolerance_ticks=5)
    assert arrived == 2048


def test_settle_gives_up_after_sustained_read_failure(monkeypatch, _fast_polls):
    """Real communication loss (not one glitch) must still surface as an error."""
    fake = FlakyReadFakeServo(servo_id=1, present_ticks=2048, fail_reads=999)
    bus = _bus(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="did not respond"):
        bus.move_and_verify(1, 2048, tolerance_ticks=5)


def test_stalled_servo_returns_early_instead_of_hanging(monkeypatch, _fast_polls):
    """An obstructed servo must be reported, not held against the obstruction.

    Mirrors the J3/table near-miss: the joint cannot reach its target, so waiting
    the full timeout achieves nothing while the arm strains.
    """
    fake = MovingFakeServo(servo_id=1, present_ticks=2048, step_ticks=0)  # stuck
    bus = _bus(monkeypatch, fake)

    arrived = bus.move_and_verify(1, 2148, tolerance_ticks=5)

    assert arrived == 2048, "should report where the servo actually is"


def test_j5_dir_sign_stays_positive_confirmed_by_jog_2026_07_22(monkeypatch):
    """J5's dir_sign is +1 -- the only joint that needed an actual physical
    jog rather than desk reconciliation (its FK rotation axis is only 73%
    pure at the home pose, unlike the clean 100%-pure axes for J1-J4, so
    neither the viewpoint method used for J1/J4 nor the wrist-displacement
    method used for J2/J3 applied confidently).

    Confirmed via scripts/jog_joint.py: a +80 tick jog predicted the claw
    would rotate ~7 deg counterclockwise seen from above; a second +80 jog
    (total ~14 deg from the starting pose, chosen after the first jog's
    result was inconclusive by eye) was reported by the operator as matching
    the prediction. No flip applied -- this is the last of the six joints'
    dir_sign values to be settled.
    """
    bus = _bus(monkeypatch, FakeServoSerial())
    assert bus._cal(5)["dir_sign"] == 1


def test_settle_grace_period_survives_acceleration_ramp_up(monkeypatch):
    """A servo that hasn't started moving yet must not be misreported as
    stalled during its acceleration ramp-up.

    Confirmed on hardware 2026-07-22: an 80-tick J5 move was reported settled
    at essentially its start position (a false stall), but a later read-only
    check found it had actually travelled the full distance. Without a grace
    period, STALL_POLLS consecutive "no movement yet" reads during a normal
    ramp-up look identical to a genuine obstruction.

    Uses explicit (non-fast) timing rather than the _fast_polls fixture,
    since this specifically tests the grace period's real wall-clock effect:
    delay_reads is sized so a servo that starts moving only after it expires
    would have falsely tripped the OLD (ungated) stall check by roughly
    STALL_POLLS iterations in, but succeeds under the grace-period fix.
    """
    monkeypatch.setattr(config, "SERVO_MOVE_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(config, "SERVO_MOVE_STALL_GRACE_S", 0.08)
    # Movement starts at the 11th position read (~0.10s in) -- after the grace
    # period (0.08s) expires, but well past where 6 consecutive "no movement"
    # reads would already have tripped an ungated stall check (~0.06s in).
    fake = DelayedMovingFakeServo(servo_id=1, present_ticks=2048, step_ticks=50, delay_reads=10)
    bus = _bus(monkeypatch, fake)

    arrived = bus.move_and_verify(1, 2148, tolerance_ticks=5)

    assert abs(arrived - 2148) <= 5, "should have reached the real target, not a false-stall position"


def test_calibration_missing_home_angle_is_rejected(monkeypatch, tmp_path):
    """A pre-offset calibration file must fail loudly, not default to zero."""
    stale = tmp_path / "servo_cal.json"
    stale.write_text(json.dumps({
        str(j): {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1}
        for j in range(1, 7)
    }))
    monkeypatch.setattr(sd.serial, "Serial", lambda *a, **k: FakeServoSerial())
    with pytest.raises(sd.ServoCalibrationError, match="home_angle_rad"):
        ServoBus("COM_FAKE", calibration_path=str(stale))

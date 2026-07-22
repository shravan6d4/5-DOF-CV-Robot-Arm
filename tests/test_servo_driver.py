"""
Protocol-level tests for the Feetech servo driver — no hardware needed.

A FakeServoSerial stands in for pyserial and emulates ONE STS3215 servo: it decodes
the packets the driver writes, applies WRITE_DATA to a tiny register model, and
answers READ_DATA with correctly-framed status packets (right checksum, right
little-endian layout). That lets us verify the exact wire bytes and the multi-step
ID-change sequence against the Feetech spec, which is the part we can't sanity-check
on real hardware from here.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vision_pipeline.robot_interface.servo_driver as sd
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
    # J6 fallback: home 2048, 325.95 ticks/rad, dir +1.
    ticks = bus.rad_to_ticks(6, 0.2)
    assert ticks == round(2048 + 0.2 * 325.95)
    assert bus.ticks_to_rad(6, ticks) == pytest.approx(0.2, abs=1e-3)

"""
Frame-conversion tests for MatlabIKClient — no MATLAB needed.

The imported model's base frame is upside-down relative to the physical robot
(CLAUDE.md "COORDINATE FRAMES"): physical (x, y, z) = model (x, -y, -z), a
180-degree rotation about X. MatlabIKClient is the single seam where that
conversion happens — request_ik converts targets going in, request_fk(_tip)
converts transforms coming out. These tests pin the seam with a stub server so
neither direction can be dropped or applied twice without a failure here.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vision_pipeline.robot_interface.matlab_client as mc
from vision_pipeline.robot_interface.matlab_client import MatlabIKClient


class StubSocket:
    """Captures the request line; replies with a canned JSON response."""

    def __init__(self, response_dict):
        self.response = (json.dumps(response_dict) + "\n").encode()
        self.sent = b""

    def connect(self, addr):
        pass

    def sendall(self, data):
        self.sent += data

    def recv(self, n):
        chunk, self.response = self.response, b""
        return chunk

    def close(self):
        pass


def _client_with(monkeypatch, response_dict):
    stub = StubSocket(response_dict)
    monkeypatch.setattr(mc.socket, "socket", lambda *a, **k: stub)
    return MatlabIKClient(), stub


def test_flip_matrix_is_self_inverse():
    F = mc._F_PHYS_FROM_MODEL
    assert np.allclose(F @ F, np.eye(4))


def test_request_ik_converts_physical_target_to_model(monkeypatch):
    """A physical target must reach the wire with y and z negated.

    Physical (0.15, 0.02, -0.063) — 63mm BELOW the origin, i.e. near the real
    table — must be sent as model (0.15, -0.02, +0.063), the model's
    "reaching upward" half-space. Without the conversion the solver would be
    asked for a point ~136mm away from the intended one.
    """
    client, stub = _client_with(
        monkeypatch, {"ok": True, "angles_rad": [0.0] * 5, "err_mm": 0.0}
    )
    with client:
        client.request_ik(0.15, 0.02, -0.063)

    sent = json.loads(stub.sent.decode().strip())
    assert sent["x"] == pytest.approx(0.15)
    assert sent["y"] == pytest.approx(-0.02)
    assert sent["z"] == pytest.approx(0.063)


def test_request_fk_converts_model_transform_to_physical(monkeypatch):
    """The model-frame home wrist (~[91, 12, 2]mm) must come back physical.

    Under the flip that's [91, -12, -2]: essentially at origin height, y
    mirrored. Uses the real home numbers so the test doubles as documentation
    of what "correct" looks like.
    """
    T_model = np.eye(4)
    T_model[:3, 3] = [0.091, 0.012, 0.002]
    client, _ = _client_with(
        monkeypatch, {"ok": True, "T": list(T_model.flatten())}
    )
    with client:
        T_phys = client.request_fk([0.0] * 5)

    assert T_phys[:3, 3] == pytest.approx([0.091, -0.012, -0.002])


def test_request_fk_tip_physical_tip_hangs_below_wrist(monkeypatch):
    """The invariant that catches a dropped conversion anywhere upstream.

    Model frame reports the tip ABOVE the wrist (model z: tip +67.7 vs wrist
    +1.9 at home) because the model is upside-down. Physically the claw hangs
    DOWN: converted, the tip must land BELOW the wrist. If this ever fails,
    someone dropped (or doubled) the frame conversion.
    """
    T_wrist_model = np.eye(4)
    T_wrist_model[:3, 3] = [0.091, 0.012, 0.002]
    T_tip_model = np.eye(4)
    T_tip_model[:3, 3] = [0.070, 0.000, 0.068]
    client, _ = _client_with(
        monkeypatch,
        {
            "ok": True,
            "T": list(T_wrist_model.flatten()),
            "T_tip": list(T_tip_model.flatten()),
        },
    )
    with client:
        T_wrist, T_tip = client.request_fk_tip([0.0] * 5)

    assert T_tip[2, 3] < T_wrist[2, 3], "physical tip must be below the wrist"
    assert T_tip[2, 3] == pytest.approx(-0.068)
    # Rotation part converts too, not just the origin: the returned pose's
    # z-column must reflect the flip (F @ R), or downstream hand-eye math
    # would mix frames.
    assert T_wrist[:3, :3] == pytest.approx(np.diag([1.0, -1.0, -1.0]))


def test_request_fk_tip_missing_field_names_the_fix(monkeypatch):
    client, _ = _client_with(
        monkeypatch, {"ok": True, "T": list(np.eye(4).flatten())}
    )
    with client, pytest.raises(RuntimeError, match="ik_fk_server"):
        client.request_fk_tip([0.0] * 5)

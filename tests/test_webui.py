"""Tests for the Flask arm-observation dashboard (webui/app.py).

Uses Flask's test_client() against an app wired to MockJointController, so
these run with no camera and no hardware. /video_feed (the MJPEG stream) is
deliberately not exercised here — it's an infinite generator meant for a
browser <img> tag, not a single-response test assertion.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.calibration import geometry
from vision_pipeline.robot_interface.joint_controller import GRIPPER_JOINT_ID, NUM_JOINTS, MockJointController
from vision_pipeline.robot_interface.servo_driver import ServoSafetyError
from vision_pipeline.webui.app import create_app

NO_CAL_FILE = "__nonexistent_servo_cal__.json"  # forces config fallback calibration


@pytest.fixture
def client():
    controller = MockJointController(calibration_path=NO_CAL_FILE)
    # camera_index=None: skip opening a real camera, dashboard still functions.
    app = create_app(controller, camera_index=None, enable_overlay=False)
    app.testing = True
    with app.test_client() as c:
        yield c


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Arm Observation" in resp.data


def test_get_joints_returns_all_six_with_both_units(client):
    resp = client.get("/api/joints")
    assert resp.status_code == 200
    data = resp.get_json()
    joints = data["joints"]
    assert len(joints) == NUM_JOINTS
    ids = [j["joint_id"] for j in joints]
    assert ids == list(range(1, NUM_JOINTS + 1))
    for j in joints:
        assert "ticks" in j
        assert "degrees" in j


def test_jog_joint_changes_position_by_delta(client):
    before = {j["joint_id"]: j["ticks"] for j in client.get("/api/joints").get_json()["joints"]}

    resp = client.post("/api/joints/3/jog", json={"delta_ticks": 50})
    assert resp.status_code == 200
    state = resp.get_json()
    assert state["joint_id"] == 3
    assert state["ticks"] == before[3] + 50


def test_jog_joint_rejects_out_of_range_joint_id(client):
    resp = client.post("/api/joints/7/jog", json={"delta_ticks": 10})
    assert resp.status_code == 400


def test_jog_joint_rejects_missing_delta_ticks(client):
    resp = client.post("/api/joints/1/jog", json={})
    assert resp.status_code == 400


def test_gripper_close_and_open(client):
    cal = config.SERVO_CALIBRATION_FALLBACK[str(GRIPPER_JOINT_ID)]

    resp = client.post("/api/gripper", json={"closed": True})
    assert resp.status_code == 200
    state = resp.get_json()
    assert state["joint_id"] == GRIPPER_JOINT_ID
    expected_closed_ticks = round(
        cal["home_tick"] + cal["dir_sign"] * config.SERVO_GRIPPER_CLOSE_RAD * cal["ticks_per_rad"]
    )
    assert state["ticks"] == expected_closed_ticks

    resp = client.post("/api/gripper", json={"closed": False})
    assert resp.status_code == 200
    expected_open_ticks = round(
        cal["home_tick"] + cal["dir_sign"] * config.SERVO_GRIPPER_OPEN_RAD * cal["ticks_per_rad"]
    )
    assert resp.get_json()["ticks"] == expected_open_ticks


def test_gripper_rejects_non_boolean_closed(client):
    resp = client.post("/api/gripper", json={"closed": "yes"})
    assert resp.status_code == 400


class _UnsafeMoveController(MockJointController):
    """A controller that always refuses, standing in for a real ServoBus
    whose move_and_verify raised ServoSafetyError (see
    config.SERVO_MAX_MOVE_DELTA_TICKS) -- the case being tested is app.py's
    error mapping, not the refusal logic itself, so a plain override is enough.
    """

    def jog(self, joint_id, delta_ticks):
        raise ServoSafetyError("refusing to move: delta exceeds cap")

    def set_gripper(self, closed):
        raise ServoSafetyError("refusing to move: delta exceeds cap")


@pytest.fixture
def unsafe_client():
    controller = _UnsafeMoveController(calibration_path=NO_CAL_FILE)
    app = create_app(controller, camera_index=None, enable_overlay=False)
    app.testing = True
    with app.test_client() as c:
        yield c


def test_jog_joint_reports_safety_refusal_as_400_not_500(unsafe_client):
    resp = unsafe_client.post("/api/joints/1/jog", json={"delta_ticks": 1000})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_gripper_reports_safety_refusal_as_400_not_500(unsafe_client):
    resp = unsafe_client.post("/api/gripper", json={"closed": True})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


# --------------------------------------------------------------------------
# Hand-eye calibration routes (/api/calib/*) — only enabled when create_app
# gets an ik_client (see scripts/run_arm_ui.py --calibrate).
# --------------------------------------------------------------------------

def test_calib_routes_501_without_ik_client(client):
    """client (module fixture) has no ik_client — every /api/calib/* route
    must say so explicitly (501) rather than 404, since the routes DO exist."""
    assert client.get("/api/calib/detect").status_code == 501
    assert client.post("/api/calib/sample").status_code == 501
    assert client.get("/api/calib/samples").status_code == 501
    assert client.post("/api/calib/solve").status_code == 501


def test_index_page_omits_calib_panel_without_ik_client(client):
    resp = client.get("/")
    assert b"Hand-eye calibration" not in resp.data


class _FakeIKClient:
    """Stands in for MatlabIKClient.request_fk: returns a scripted sequence
    of T_base_gripper transforms, ignoring the angles it's given. Isolates
    the ROUTE wiring (detect -> read joints -> FK -> accumulator) from
    whether MockJointController's ticks correspond to a real FK solve, which
    the route doesn't need for these tests — calibration/hand_eye.py already
    covers the math end-to-end against real angle->pose relationships.
    """

    def __init__(self, transforms):
        self._transforms = list(transforms)
        self._i = 0

    def request_fk(self, angles_rad):
        t = self._transforms[self._i]
        self._i += 1
        return t


@pytest.fixture
def calib_client(monkeypatch, tmp_path):
    # Real gripper-pose diversity (rotation AND translation) so the eventual
    # calibrateHandEye solve is well-conditioned, mirroring test_hand_eye.py's
    # synthetic fixture rather than testing route wiring with degenerate data.
    gripper_poses = [
        geometry.make_transform(0.15, -0.05, 0.10, roll_deg=175, pitch_deg=-5, yaw_deg=-30),
        geometry.make_transform(0.18, -0.03, 0.12, roll_deg=180, pitch_deg=0, yaw_deg=-20),
        geometry.make_transform(0.20, 0.00, 0.14, roll_deg=185, pitch_deg=5, yaw_deg=-10),
        geometry.make_transform(0.22, 0.02, 0.16, roll_deg=180, pitch_deg=10, yaw_deg=0),
        geometry.make_transform(0.19, 0.04, 0.13, roll_deg=170, pitch_deg=-10, yaw_deg=10),
        geometry.make_transform(0.17, -0.02, 0.11, roll_deg=190, pitch_deg=0, yaw_deg=20),
        geometry.make_transform(0.21, 0.01, 0.15, roll_deg=180, pitch_deg=3, yaw_deg=30),
        geometry.make_transform(0.16, -0.04, 0.10, roll_deg=178, pitch_deg=-3, yaw_deg=-25),
        geometry.make_transform(0.23, 0.03, 0.17, roll_deg=182, pitch_deg=8, yaw_deg=15),
        geometry.make_transform(0.19, -0.01, 0.12, roll_deg=180, pitch_deg=-8, yaw_deg=5),
        geometry.make_transform(0.20, 0.05, 0.14, roll_deg=175, pitch_deg=0, yaw_deg=-15),
        geometry.make_transform(0.18, -0.05, 0.16, roll_deg=185, pitch_deg=4, yaw_deg=25),
    ]
    t_gripper_camera_true = geometry.make_transform(0.02, -0.01, 0.03, yaw_deg=15.0)
    t_base_board = geometry.make_transform(0.20, 0.05, 0.0)

    board_pose_sequence = []
    for t_base_gripper in gripper_poses:
        t_base_camera = t_base_gripper @ t_gripper_camera_true
        t_cam_board = geometry.invert_transform(t_base_camera) @ t_base_board
        board_pose_sequence.append({0: t_cam_board})

    calls = {"i": 0}

    def fake_detect(gray, intrinsics, detectors):
        i = calls["i"]
        calls["i"] += 1
        return board_pose_sequence[i] if i < len(board_pose_sequence) else {}

    monkeypatch.setattr("vision_pipeline.webui.app.detect_board_poses", fake_detect)
    monkeypatch.setattr(config, "HAND_EYE_PATH", str(tmp_path / "hand_eye.json"))

    controller = MockJointController(calibration_path=NO_CAL_FILE)
    ik_client = _FakeIKClient(gripper_poses)
    app = create_app(controller, camera_index=None, ik_client=ik_client)
    app.testing = True
    with app.test_client() as c:
        yield c, len(gripper_poses), t_gripper_camera_true


def test_index_page_includes_calib_panel_with_ik_client(calib_client):
    c, _n, _true = calib_client
    resp = c.get("/")
    assert resp.status_code == 200
    assert b"Hand-eye calibration" in resp.data


def test_calib_sample_buckets_by_board_and_solve_recovers_transform(calib_client):
    c, n_samples, t_gripper_camera_true = calib_client

    for i in range(n_samples):
        resp = c.post("/api/calib/sample")
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["recorded_boards"] == [0]
        assert data["counts"] == {"0": i + 1}

    resp = c.get("/api/calib/samples")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["counts"] == {"0": n_samples}
    assert data["solvable_boards"] == [0]

    resp = c.post("/api/calib/solve")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["selected_board"] == 0
    assert data["cross_board_agreement_mm"] is None  # only one board sampled
    recovered_mm = np.array(data["boards"]["0"]["t_gripper_camera_mm"])
    expected_mm = t_gripper_camera_true[:3, 3] * 1000.0
    assert recovered_mm == pytest.approx(expected_mm, abs=1e-3)

    # save_hand_eye actually wrote the file (path patched to tmp_path).
    assert Path(config.HAND_EYE_PATH).exists()


def test_calib_sample_reports_400_when_no_board_visible(calib_client):
    c, n_samples, _true = calib_client
    for _ in range(n_samples):
        c.post("/api/calib/sample")  # drain the scripted sequence

    resp = c.post("/api/calib/sample")  # sequence exhausted -> fake_detect returns {}
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_calib_solve_reports_400_before_min_samples(calib_client):
    c, _n, _true = calib_client
    c.post("/api/calib/sample")  # only one sample recorded

    resp = c.post("/api/calib/solve")
    assert resp.status_code == 400
    assert "error" in resp.get_json()

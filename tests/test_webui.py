"""Tests for the Flask arm-observation dashboard (webui/app.py).

Uses Flask's test_client() against an app wired to MockJointController, so
these run with no camera and no hardware. /video_feed (the MJPEG stream) is
deliberately not exercised here — it's an infinite generator meant for a
browser <img> tag, not a single-response test assertion.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
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

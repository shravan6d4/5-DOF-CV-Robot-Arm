"""Flask app: live camera view + per-joint jog controls.

create_app(controller, camera_index, enable_overlay=False) builds the app; see
scripts/run_arm_ui.py for how a JointController (mock or real ServoBus-backed)
gets constructed and passed in.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request

from vision_pipeline import config
from vision_pipeline.robot_interface.joint_controller import (
    GRIPPER_JOINT_ID,
    NUM_JOINTS,
    JointController,
    JointState,
)
from vision_pipeline.robot_interface.servo_calibration import ServoCalibrationError
from vision_pipeline.webui.camera_stream import CameraStreamer

logger = logging.getLogger(__name__)

# Errors a controller call can raise when talking to real hardware (bad servo
# read, uncalibrated joint). Reported as JSON 502s instead of a 500 stack trace,
# so a flaky read shows up in the dashboard instead of killing the request.
_CONTROLLER_ERRORS = (RuntimeError, ServoCalibrationError)


def _joint_state_json(state: JointState) -> dict:
    return {"joint_id": state.joint_id, "ticks": state.ticks, "degrees": round(state.degrees, 2)}


def create_app(
    controller: JointController,
    camera_index: Optional[int] = config.CAMERA_INDEX,
    enable_overlay: bool = False,
) -> Flask:
    """Build the Flask app wired to one JointController and one camera.

    A single lock serializes every controller call (read/jog/gripper): ServoBus
    talks over one serial port and can't handle concurrent commands, and Flask's
    dev server (threaded=True) will otherwise call in from multiple requests.

    camera_index=None skips opening a camera; the feed shows a placeholder.
    """
    app = Flask(__name__)
    controller_lock = threading.Lock()
    streamer = CameraStreamer(camera_index=camera_index, enable_overlay=enable_overlay)

    # Kept on app.config mainly so tests / a REPL can reach in if needed.
    app.config["JOINT_CONTROLLER"] = controller
    app.config["CAMERA_STREAMER"] = streamer

    @app.route("/")
    def index():
        # Calibration is fixed for the controller's lifetime, so compute the
        # tick->degree rate once per page load rather than on every /api/joints
        # poll; the JS multiplies it by the slider's tick value live.
        degrees_per_tick = {j: controller.degrees_per_tick(j) for j in range(1, NUM_JOINTS + 1)}
        return render_template(
            "index.html",
            num_joints=NUM_JOINTS,
            gripper_joint_id=GRIPPER_JOINT_ID,
            jog_default_step=config.JOG_DEFAULT_STEP_TICKS,
            jog_max_step=config.JOG_MAX_STEP_TICKS,
            degrees_per_tick=degrees_per_tick,
        )

    @app.route("/video_feed")
    def video_feed():
        return Response(
            streamer.mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/joints")
    def get_joints():
        try:
            with controller_lock:
                states = controller.read_all()
            return jsonify({"joints": [_joint_state_json(s) for s in states]})
        except _CONTROLLER_ERRORS as e:
            logger.error(f"read_all failed: {e}")
            return jsonify({"error": str(e)}), 502

    @app.route("/api/joints/<int:joint_id>/jog", methods=["POST"])
    def jog_joint(joint_id):
        if not 1 <= joint_id <= NUM_JOINTS:
            return jsonify({"error": f"joint_id must be 1..{NUM_JOINTS}"}), 400

        body = request.get_json(silent=True) or {}
        try:
            delta_ticks = int(body["delta_ticks"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "body must include integer 'delta_ticks'"}), 400

        try:
            with controller_lock:
                state = controller.jog(joint_id, delta_ticks)
            return jsonify(_joint_state_json(state))
        except _CONTROLLER_ERRORS as e:
            logger.error(f"jog(J{joint_id}, {delta_ticks}) failed: {e}")
            return jsonify({"error": str(e)}), 502

    @app.route("/api/gripper", methods=["POST"])
    def set_gripper():
        body = request.get_json(silent=True) or {}
        closed = body.get("closed")
        if not isinstance(closed, bool):
            return jsonify({"error": "body must include boolean 'closed'"}), 400

        try:
            with controller_lock:
                state = controller.set_gripper(closed)
            return jsonify(_joint_state_json(state))
        except _CONTROLLER_ERRORS as e:
            logger.error(f"set_gripper({closed}) failed: {e}")
            return jsonify({"error": str(e)}), 502

    return app

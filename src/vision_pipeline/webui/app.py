"""Flask app: live camera view + per-joint jog controls (+ optional hand-eye
calibration capture panel).

create_app(controller, camera_index, enable_overlay=False) builds the app; see
scripts/run_arm_ui.py for how a JointController (mock or real ServoBus-backed)
gets constructed and passed in. Pass ik_client to also enable the /api/calib/*
routes (see scripts/run_arm_ui.py --calibrate).
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request

from vision_pipeline import config
from vision_pipeline.calibration import charuco
from vision_pipeline.calibration.camera_model import CameraIntrinsics, load_intrinsics
from vision_pipeline.calibration.hand_eye import (
    HandEyeAccumulator,
    cross_board_agreement_mm,
    detect_board_poses,
    select_best,
)
from vision_pipeline.calibration.pixel_to_world import save_hand_eye
from vision_pipeline.robot_interface.joint_controller import (
    GRIPPER_JOINT_ID,
    NUM_JOINTS,
    JointController,
    JointState,
)
from vision_pipeline.robot_interface.matlab_client import IKUnreachableError, MatlabIKClient
from vision_pipeline.robot_interface.servo_calibration import ServoCalibrationError
from vision_pipeline.robot_interface.servo_driver import ServoSafetyError
from vision_pipeline.webui.camera_stream import CameraStreamer

logger = logging.getLogger(__name__)

# Errors a controller call can raise when talking to real hardware (bad servo
# read, uncalibrated joint). Reported as JSON 502s instead of a 500 stack trace,
# so a flaky read shows up in the dashboard instead of killing the request.
_CONTROLLER_ERRORS = (RuntimeError, ServoCalibrationError)

# J1..J5 are IK-driven and feed hand-eye's FK; J6 (GRIPPER_JOINT_ID) does not.
_IK_JOINT_IDS = range(1, GRIPPER_JOINT_ID)


def _joint_state_json(state: JointState) -> dict:
    return {"joint_id": state.joint_id, "ticks": state.ticks, "degrees": round(state.degrees, 2)}


def _board_result_json(r) -> dict:
    return {
        "board_index": r.board_index,
        "n_samples": r.n_samples,
        "t_gripper_camera_mm": (r.t_gripper_camera_tsai[:3, 3] * 1000.0).tolist(),
        "tsai_park_disagreement_mm": round(r.tsai_park_disagreement_mm, 2),
        "board_spread_mm": round(r.board_spread_mm, 2),
        "mean_board_origin_base_m": r.mean_board_origin_base.tolist(),
    }


def _charuco_overlay(detectors):
    """Draw every board's detected corners on the live feed (calibration mode)."""
    def _apply(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        for _idx, _board, det in detectors:
            corners, ids = charuco.detect(det, gray)
            if corners is not None and len(corners) > 0:
                try:
                    cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
                except cv2.error:
                    pass
        return display
    return _apply


def create_app(
    controller: JointController,
    camera_index: Optional[int] = config.CAMERA_INDEX,
    enable_overlay: bool = False,
    ik_client: Optional[MatlabIKClient] = None,
    intrinsics: Optional[CameraIntrinsics] = None,
) -> Flask:
    """Build the Flask app wired to one JointController and one camera.

    A single lock serializes every controller call (read/jog/gripper): ServoBus
    talks over one serial port and can't handle concurrent commands, and Flask's
    dev server (threaded=True) will otherwise call in from multiple requests.

    camera_index=None skips opening a camera; the feed shows a placeholder.

    ik_client: pass a connected MatlabIKClient to also enable the hand-eye
    calibration capture routes (/api/calib/*) — without it they 501, so the
    default dashboard keeps needing no MATLAB server. When given, the ChArUco
    overlay replaces the brick-detection overlay (enable_overlay is ignored)
    since the two are mutually exclusive uses of the same feed.
    """
    app = Flask(__name__)
    controller_lock = threading.Lock()

    calibrating = ik_client is not None
    detectors = charuco.build_detectors() if calibrating else None
    resolved_intrinsics = (intrinsics or load_intrinsics()) if calibrating else None
    accumulator = HandEyeAccumulator() if calibrating else None

    if calibrating:
        streamer = CameraStreamer(camera_index=camera_index, overlay=_charuco_overlay(detectors))
    else:
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
            calibrating=calibrating,
            min_hand_eye_samples=config.CALIB_HAND_EYE_MIN_SAMPLES,
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
        except ServoSafetyError as e:
            # A deliberate refusal, not a hardware fault: ServoBus.move_and_verify
            # caps how far a single move may travel from the servo's current
            # position (see config.SERVO_MAX_MOVE_DELTA_TICKS) and nothing was
            # sent to the bus. 400, not 502 -- the request itself is what's
            # rejected, distinct from a flaky read/write.
            logger.warning(f"jog(J{joint_id}, {delta_ticks}) refused: {e}")
            return jsonify({"error": str(e)}), 400
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
        except ServoSafetyError as e:
            logger.warning(f"set_gripper({closed}) refused: {e}")
            return jsonify({"error": str(e)}), 400
        except _CONTROLLER_ERRORS as e:
            logger.error(f"set_gripper({closed}) failed: {e}")
            return jsonify({"error": str(e)}), 502

    # --- Hand-eye calibration capture (see calibration/hand_eye.py) ----------
    # 501 (not 404) when ik_client was not supplied: the ROUTE exists, the
    # dashboard just wasn't started in --calibrate mode, which is a more
    # useful signal than "not found" for a client that expected these to work.
    def _not_calibrating():
        return jsonify({"error": "dashboard was not started with --calibrate (no ik_client)"}), 501

    def _current_angles_rad() -> list[float]:
        """J1..J5 angles in radians, read straight from the controller — no Pose
        round trip (see calibration/hand_eye.py's module docstring for why that
        matters near the top-down tool orientation's gimbal-lock singularity)."""
        with controller_lock:
            states = [controller.read_joint(j) for j in _IK_JOINT_IDS]
        return [math.radians(s.degrees) for s in states]

    @app.route("/api/calib/detect")
    def calib_detect():
        if not calibrating:
            return _not_calibrating()
        gray = cv2.cvtColor(streamer.latest_frame(), cv2.COLOR_BGR2GRAY)
        counts = {}
        for idx, _board, det in detectors:
            corners, _ids = charuco.detect(det, gray)
            counts[idx] = 0 if corners is None else len(corners)
        return jsonify({"corner_counts": counts, "min_corners": config.CALIB_CHARUCO_MIN_CORNERS})

    @app.route("/api/calib/sample", methods=["POST"])
    def calib_sample():
        if not calibrating:
            return _not_calibrating()
        gray = cv2.cvtColor(streamer.latest_frame(), cv2.COLOR_BGR2GRAY)
        board_poses = detect_board_poses(gray, resolved_intrinsics, detectors)
        if not board_poses:
            return jsonify({"error": "no board visible with enough corners in the current frame"}), 400

        try:
            angles_rad = _current_angles_rad()
        except _CONTROLLER_ERRORS as e:
            logger.error(f"calib_sample: joint read failed: {e}")
            return jsonify({"error": str(e)}), 502

        try:
            t_base_gripper = ik_client.request_fk(angles_rad)
        except (IKUnreachableError, OSError) as e:
            logger.error(f"calib_sample: FK request failed: {e}")
            return jsonify({"error": f"FK request failed: {e}"}), 502

        counts = accumulator.add(board_poses, t_base_gripper)
        return jsonify({"recorded_boards": sorted(board_poses), "counts": counts})

    @app.route("/api/calib/samples")
    def calib_samples():
        if not calibrating:
            return _not_calibrating()
        return jsonify({
            "counts": accumulator.counts(),
            "min_samples": accumulator.min_samples,
            "solvable_boards": accumulator.solvable_boards(),
        })

    @app.route("/api/calib/solve", methods=["POST"])
    def calib_solve():
        if not calibrating:
            return _not_calibrating()
        if not accumulator.solvable_boards():
            return jsonify({
                "error": f"no board has reached {accumulator.min_samples} samples yet",
                "counts": accumulator.counts(),
            }), 400

        results = accumulator.solve_all()
        best = select_best(results, accumulator)
        agreement_mm = cross_board_agreement_mm(results)
        save_hand_eye(best.t_gripper_camera_tsai, config.HAND_EYE_PATH)

        return jsonify({
            "boards": {idx: _board_result_json(r) for idx, r in results.items()},
            "selected_board": best.board_index,
            "cross_board_agreement_mm": agreement_mm,
            "saved_to": config.HAND_EYE_PATH,
        })

    return app

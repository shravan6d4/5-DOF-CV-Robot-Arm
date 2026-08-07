"""Flask app: live camera feed + a real interactive shell, side by side.

Built so hardware bring-up scripts (visual_servo.py, jog_joint.py,
hand_eye_report.py, ...) can be watched and driven from one browser tab
instead of alt-tabbing between a console window and the arm's OpenCV --view
window. The shell is a real ConPTY session (scripts/run_dashboard.py spawns
it), so it behaves like any other PowerShell window: arrow keys, tab
completion, Ctrl-C all work.

The camera panel is independent of whatever runs in the shell -- it opens its
own cv2.VideoCapture. A script started in the embedded shell that also opens
the camera (visual_servo.py, run_live_detection.py, ...) will contend with
this dashboard for the same device; see scripts/run_dashboard.py's --no-camera
flag.
"""

from __future__ import annotations

import base64
import logging
import queue

from flask import Flask, Response, jsonify, render_template, request

from vision_pipeline.webui.camera_stream import CameraStreamer
from vision_pipeline.webui.terminal_session import TerminalSession

logger = logging.getLogger(__name__)

# How long an SSE listener waits for the next chunk before sending a keepalive
# comment. Purely to (a) let a closed browser tab's dead socket surface as a
# write error so the subscriber gets cleaned up, and (b) keep intermediary
# proxies/timeouts from deciding the connection is idle.
_SSE_HEARTBEAT_S = 15.0


def create_dashboard_app(
    camera_index: "int | None",
    shell: "list[str]",
    cwd: "str | None" = None,
    enable_overlay: bool = False,
) -> Flask:
    """Build the Flask app wired to one camera and one shared shell session.

    camera_index=None skips opening a camera; the feed shows a placeholder
    (useful when a script running in the embedded shell needs the device).
    """
    app = Flask(__name__)

    streamer = CameraStreamer(camera_index=camera_index, enable_overlay=enable_overlay)
    terminal = TerminalSession(shell=shell, cwd=cwd)

    app.config["CAMERA_STREAMER"] = streamer
    app.config["TERMINAL_SESSION"] = terminal

    @app.route("/")
    def index():
        return render_template("dashboard.html", shell=" ".join(shell))

    @app.route("/video_feed")
    def video_feed():
        return Response(
            streamer.mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/terminal/stream")
    def terminal_stream():
        def generate():
            q = terminal.subscribe()
            try:
                while True:
                    try:
                        chunk = q.get(timeout=_SSE_HEARTBEAT_S)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    payload = base64.b64encode(chunk.encode("utf-8", errors="replace")).decode("ascii")
                    yield f"data: {payload}\n\n"
            finally:
                terminal.unsubscribe(q)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/terminal/input", methods=["POST"])
    def terminal_input():
        body = request.get_json(silent=True) or {}
        data = body.get("data")
        if not isinstance(data, str):
            return jsonify({"error": "body must include string 'data'"}), 400
        terminal.write(data)
        return jsonify({"ok": True})

    @app.route("/terminal/resize", methods=["POST"])
    def terminal_resize():
        body = request.get_json(silent=True) or {}
        try:
            rows = int(body["rows"])
            cols = int(body["cols"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "body must include integer 'rows' and 'cols'"}), 400
        if rows <= 0 or cols <= 0:
            return jsonify({"error": "rows and cols must be positive"}), 400
        terminal.resize(rows, cols)
        return jsonify({"ok": True})

    return app

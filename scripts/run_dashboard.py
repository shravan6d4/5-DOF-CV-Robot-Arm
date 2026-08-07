"""Launch the camera + terminal dashboard.

A Flask page with the live camera feed next to a REAL interactive shell (a
ConPTY session via pywinpty, streamed over SSE) -- typed commands, arrow keys,
Ctrl-C, tab completion all behave like any other PowerShell window. Built so
hardware bring-up scripts (visual_servo.py, jog_joint.py, hand_eye_report.py,
...) can be watched and driven from one browser tab instead of alt-tabbing
between a console window and the arm's OpenCV --view window.

    python scripts/run_dashboard.py                  # PowerShell + webcam
    python scripts/run_dashboard.py --no-camera       # free the camera for a
                                                       # script you'll run in
                                                       # the embedded shell
    python scripts/run_dashboard.py --shell cmd.exe
    python scripts/run_dashboard.py --overlay         # brick-detection overlay

CAMERA CONTENTION: the camera panel opens its own cv2.VideoCapture,
independent of the embedded shell. A script started in that shell which also
opens the camera (visual_servo.py, run_live_detection.py, ...) will contend
with this dashboard for the same device -- most webcams only allow one open
handle. Use --no-camera when you're about to run one of those.

Then open http://127.0.0.1:5000 (or --host/--port) in a browser.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.webui.dashboard_app import create_dashboard_app

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--camera-index", type=int, default=config.CAMERA_INDEX,
        help="Webcam index for the live feed (default: %(default)s).",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Skip opening a camera; the feed panel shows a placeholder. Use "
             "this when a script run in the embedded shell needs the device.",
    )
    parser.add_argument("--overlay", action="store_true", help="Draw red-brick detection overlay on the camera feed.")
    parser.add_argument(
        "--shell", default="powershell.exe",
        help="Shell command to spawn in the terminal panel (default: %(default)s).",
    )
    parser.add_argument(
        "--cwd", default=str(REPO_ROOT),
        help="Working directory for the shell (default: repo root, so scripts/ commands work as documented).",
    )
    parser.add_argument("--host", default=config.WEBUI_HOST, help="Host to bind (default: %(default)s).")
    parser.add_argument("--port", type=int, default=config.WEBUI_PORT, help="Port to bind (default: %(default)s).")
    args = parser.parse_args()

    camera_index = None if args.no_camera else args.camera_index

    app = create_dashboard_app(
        camera_index=camera_index,
        shell=[args.shell],
        cwd=args.cwd,
        enable_overlay=args.overlay,
    )

    print(f"Dashboard at http://{args.host}:{args.port} (Ctrl+C to stop)")
    print(f"Shell: {args.shell}  cwd: {args.cwd}")
    if camera_index is None:
        print("Camera: disabled (--no-camera)")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

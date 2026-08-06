"""Live camera window showing brick detection AND every ChArUco board at once.

    Read-only. Opens the camera, draws on frames, commands no motion ever.

WHAT IT IS FOR. Aiming. Before any script that moves the arm, this answers the
questions that otherwise get answered by a failed run: is the brick actually in
frame, does the detector agree that it is a brick, are the boards visible and
which ones, and is the focus/exposure good enough to trust. run_live_detection.py
already shows bricks; this adds the boards, the aim point, and a status line,
which is what makes it usable for lining up a calibration or a servo run.

  green circle    detected brick (largest-first, the one scripts would act on)
  red crosshair   frame centre — where the visual servo drives the brick
  amber line      the pixel error between them, the quantity being minimised
  blue dots       ChArUco corners, labelled with which board they belong to

THE ONE THING TO KNOW. A webcam can be opened by ONE process at a time. While
this viewer is running, visual_servo.py and every other camera script will fail
to open the camera, and vice versa. That is an OS-level constraint, not
something a flag can work around. To watch the servo loop as it runs, use its
own live window instead:

    python scripts/visual_servo.py --view

Keys:  q or Esc  quit        s  save the current frame to data/
       b         toggle ChArUco detection (it is the slow part)
       m         toggle the RED MASK view — what the detector actually sees

The mask view is the one to reach for when detection flickers. Glare on a glossy
stud can dip below the saturation floor and cut the silhouette in two, and on
the colour view that is invisible: the brick looks fine to a human eye while the
detector sees two fragments too small to be a brick. On the mask it is obvious.

Usage (from the repo root):
    python scripts/run_live_view.py
    python scripts/run_live_view.py --no-charuco       # bricks only, fastest
    python scripts/run_live_view.py --boards 2         # only check boards 1-2
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from vision_pipeline import config
from vision_pipeline.calibration import charuco
from vision_pipeline.capture.camera import Camera
from vision_pipeline.detection.lego_detector import LegoBrickDetector
from vision_pipeline.overlay import draw_aim, draw_boards, draw_hud

WINDOW = "Live view — q quits, s saves, b toggles boards"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards", type=int, default=config.CALIB_BOARD_COUNT,
                    help=f"how many boards to look for (default "
                         f"{config.CALIB_BOARD_COUNT})")
    ap.add_argument("--no-charuco", action="store_true",
                    help="skip board detection entirely")
    ap.add_argument("--charuco-every", type=int, default=3,
                    help="run board detection every Nth frame (default 3). "
                         "Detecting several boards per frame is the expensive "
                         "part; the last result is redrawn in between.")
    ap.add_argument("--deadband", type=float, default=config.SERVO_VISUAL_DEADBAND_PX,
                    help="draw the servo deadband ring at this radius, px")
    ap.add_argument("--mask", action="store_true",
                    help="start in red-mask view (toggle with m)")
    args = ap.parse_args()

    detector = LegoBrickDetector()
    detectors = [] if args.no_charuco else charuco.build_detectors(args.boards)
    show_boards = not args.no_charuco
    show_mask = args.mask

    print("Live view. q or Esc quits, s saves, b toggles boards, m toggles mask.")
    print("NOTE: this holds the camera — no other camera script can run "
          "while it is open.")

    try:
        camera = Camera()
    except RuntimeError as e:
        print(f"\nNo camera: {e}")
        sys.exit(1)

    boards_seen, frame_i = 0, 0
    with camera:
        if not camera.autofocus_disabled:
            print("\nNOTE: autofocus could not be disabled. Fine for aiming, but")
            print("any calibration captured this way will drift as the lens hunts.")

        for frame in camera.frames():
            frame_i += 1
            detections = detector.detect(frame)

            if show_mask:
                # Rebuild the mask the detector just used and show it as the
                # base image, so the overlays below land in the same places and
                # the two views can be flipped between without re-aiming.
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = detector.color_detector._build_mask(hsv)
                view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                for d in detections:
                    cv2.drawContours(view, [d.contour], -1, (0, 255, 0), 2)
            else:
                view = detector.draw_debug_overlay(frame, detections)

            if show_boards and detectors:
                # Board detection is the slow part, so it runs on a subset of
                # frames; the count from the last real detection is what gets
                # reported in between rather than a flickering zero.
                if frame_i % max(1, args.charuco_every) == 0:
                    boards_seen = draw_boards(view, detectors)
                else:
                    draw_boards(view, detectors[:1])

            centroid = detections[0].centroid_px if detections else None
            draw_aim(view, centroid, deadband_px=args.deadband)

            h, w = view.shape[:2]
            status = [f"{w}x{h}{'  [MASK]' if show_mask else ''}   brick: "
                      + (f"{len(detections)} found" if detections else "NONE")]
            if detections:
                best = detections[0]
                status.append(f"studs {best.num_studs}  shape {best.shape_score:.2f}  "
                              f"conf {best.confidence:.2f}")
                dx = best.centroid_px[0] - w / 2
                dy = best.centroid_px[1] - h / 2
                status.append(f"error {dx:+.0f}, {dy:+.0f} px")
            if show_boards and detectors:
                status.append(f"charuco: {boards_seen}/{len(detectors)} boards")
            draw_hud(view, status)

            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                show_boards = not show_boards and bool(detectors)
                print(f"  ChArUco detection {'on' if show_boards else 'off'}")
            if key == ord("m"):
                show_mask = not show_mask
                print(f"  mask view {'on' if show_mask else 'off'}")
            if key == ord("s"):
                out = Path("data") / f"view_{time.strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(str(out), view)
                print(f"  saved {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

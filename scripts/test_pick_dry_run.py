"""Locate a red brick with the REAL arm's pose — and command NOTHING.

    This script does not move the arm. It only reads.

It is the last checkpoint before a real pick. The pick path has three links that
have never been exercised together on hardware:

    FK (validated on the vertical axis)
      -> hand-eye T_gripper_camera (NOT validated; see below)
        -> table-plane back-projection (TABLE_Z_IN_BASE, good to ~2 mm)

A wrong hand-eye transform does not fail loudly. It produces a confident,
well-formed PickTarget pointing at the wrong place, and `run_once` would drive
the arm straight to it. So: run this, put a ruler on the brick, and compare.

What it prints, in the order the errors would compound:

  1. where FK says the wrist is
  2. where the hand-eye transform therefore puts the CAMERA, and which way it
     looks. The optical axis MUST point downward (base -Z) if the camera can see
     a table. If it points up, the hand-eye solve is wrong and everything after
     this line is meaningless.
  3. the detected brick, and the base-frame point its pixel back-projects to
  4. whether that point is reachable and above the floor guard — asked of the
     MATLAB solver as pure math, no motion

    python scripts/test_pick_dry_run.py
    python scripts/test_pick_dry_run.py --image path/to/frame.jpg   # no camera
    python scripts/test_pick_dry_run.py --save out.png              # annotated
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.calibration import geometry
from vision_pipeline.calibration.pixel_to_world import PixelToWorldCalibrator, load_hand_eye
from vision_pipeline.capture.camera import Camera
from vision_pipeline.detection.lego_detector import LegoBrickDetector
from vision_pipeline.pipeline import PickPipeline
from vision_pipeline.planning.pick import plan_pick_sequence
from vision_pipeline.robot_interface.base import Pose
from vision_pipeline.robot_interface.matlab_client import IKUnreachableError, MatlabIKClient
from vision_pipeline.robot_interface.servo_driver import ServoBus

IK_JOINTS = (1, 2, 3, 4, 5)
# The first frames off a freshly-opened capture are stale/auto-exposing.
CAMERA_WARMUP_FRAMES = 8


def read_joint_angles(bus: ServoBus) -> tuple[list[float], list[int]]:
    """Current J1-J5 angles in radians, plus the raw ticks they came from."""
    ticks = [bus.read_position(j) for j in IK_JOINTS]
    rads = [bus.ticks_to_rad(j, t) for j, t in zip(IK_JOINTS, ticks)]
    return rads, ticks


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image", help="use a static image instead of the live camera")
    ap.add_argument("--save", help="write the annotated frame here")
    ap.add_argument("--host", default=config.MATLAB_SERVER_HOST)
    ap.add_argument("--port", type=int, default=config.MATLAB_SERVER_PORT)
    ap.add_argument("--serial", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    print("=" * 68)
    print("PICK DRY RUN — reads only, commands no motion")
    print("=" * 68)

    # ---- 1. where the arm actually is -------------------------------------
    bus = ServoBus(args.serial, args.baud)
    ik = MatlabIKClient(args.host, args.port)

    with bus:
        angles_rad, ticks = read_joint_angles(bus)
        print("\n[1] ARM POSE")
        for j, t, a in zip(IK_JOINTS, ticks, angles_rad):
            print(f"    J{j}: {t:5d} ticks = {np.degrees(a):+7.2f} deg")

        # One call returns both frames; they are ~70 mm apart and NOT
        # interchangeable — hand-eye is solved against the wrist, IK targets
        # the tip.
        t_base_wrist, t_base_tip = ik.request_fk_tip(angles_rad)
        wx, wy, wz = t_base_wrist[:3, 3]
        tx, ty, tz = t_base_tip[:3, 3]
        print(f"    wrist (FK, physical frame): "
              f"x={wx * 1000:+7.1f}  y={wy * 1000:+7.1f}  z={wz * 1000:+7.1f} mm")
        print(f"    claw tip:                   "
              f"x={tx * 1000:+7.1f}  y={ty * 1000:+7.1f}  z={tz * 1000:+7.1f} mm")

        # Frame-flip invariant from CLAUDE.md: the tip hangs BELOW the wrist in
        # any sane pose. If this trips, a conversion has been dropped and every
        # number below is in the wrong frame.
        if tz >= wz:
            print("    *** WARNING: claw tip is ABOVE the wrist. The model/physical")
            print("        frame conversion is broken — stop and fix that first. ***")
        print(f"    tip is {(tz - config.TABLE_Z_IN_BASE) * 1000:+.0f} mm above the table "
              f"(table z = {config.TABLE_Z_IN_BASE * 1000:.0f} mm)")

        # The pipeline hands the calibrator a Pose, not the raw 4x4. That round
        # trip goes through an Euler decomposition which is ill-conditioned near
        # pitch = +/-90 deg -- the same singularity that poisoned the hand-eye
        # solve. Check it rather than assume it.
        pose = Pose(*geometry.transform_to_pose(t_base_wrist))
        roundtrip_err = float(np.max(np.abs(pose.to_matrix() - t_base_wrist)))
        print(f"    Pose round-trip error: {roundtrip_err:.2e} "
              f"(pitch = {pose.pitch_deg:+.1f} deg)")
        if roundtrip_err > 1e-6:
            print("    *** WARNING: converting FK through Pose is LOSING rotation.")
            print("        Near gimbal lock. The pipeline uses this path. ***")

    # ---- 2. where that puts the camera ------------------------------------
    t_gripper_camera = load_hand_eye()
    hand_eye_path = Path(config.HAND_EYE_PATH)
    t_base_camera = t_base_wrist @ t_gripper_camera

    print("\n[2] CAMERA (via hand-eye)")
    if not hand_eye_path.exists():
        print(f"    {hand_eye_path} MISSING — using identity (camera == wrist).")
        print("    Back-projection below is plumbing only, not a real location.")
    else:
        print(f"    {hand_eye_path}")

    cam_xyz = t_base_camera[:3, 3]
    offset_mm = np.linalg.norm(t_gripper_camera[:3, 3]) * 1000
    optical_axis = t_base_camera[:3, 2]  # camera +Z looks forward
    claw_dir = t_base_tip[:3, 3] - t_base_wrist[:3, 3]
    claw_dir = claw_dir / max(np.linalg.norm(claw_dir), 1e-9)
    axis_vs_claw = np.degrees(np.arccos(np.clip(float(optical_axis @ claw_dir), -1, 1)))

    print(f"    camera sits {offset_mm:.0f} mm from the wrist, at "
          f"x={cam_xyz[0] * 1000:+7.1f}  y={cam_xyz[1] * 1000:+7.1f}  "
          f"z={cam_xyz[2] * 1000:+7.1f} mm")
    print(f"    optical axis: [{optical_axis[0]:+.3f} {optical_axis[1]:+.3f} "
          f"{optical_axis[2]:+.3f}]  ({axis_vs_claw:.0f} deg from the claw direction)")

    if optical_axis[2] > 0:
        print("    *** the camera is looking UP, away from the table. A brick on")
        print("        the table cannot be back-projected onto it. The hand-eye")
        print("        transform is wrong. ***")
    if axis_vs_claw > 90:
        print(f"    *** the camera points {axis_vs_claw:.0f} deg away from where the claw")
        print("        points. Physically it looks roughly where the claw does. ***")

    # ---- 3. detect and back-project ---------------------------------------
    print("\n[3] BRICK")
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"    could not read {args.image}")
            sys.exit(1)
        print(f"    from {args.image}  ({frame.shape[1]}x{frame.shape[0]})")
    else:
        with Camera() as camera:
            for _ in range(CAMERA_WARMUP_FRAMES):
                frame = camera.read_frame()
        print(f"    live frame  ({frame.shape[1]}x{frame.shape[0]})")

    detector = LegoBrickDetector()
    bricks = detector.detect(frame)
    if not bricks:
        print("    NO BRICK DETECTED. Nothing to locate.")
        if args.save:
            cv2.imwrite(args.save, frame)
            print(f"    frame saved to {args.save} — check lighting/HSV.")
        sys.exit(1)

    brick = bricks[0]
    print(f"    found at pixel {brick.centroid_px}  area={brick.area:.0f}px  "
          f"studs={brick.num_studs}  shape={brick.shape_score:.2f}  "
          f"confidence={brick.confidence:.2f}")
    if len(bricks) > 1:
        print(f"    ({len(bricks)} candidates; using the largest)")

    calibrator = PixelToWorldCalibrator()
    pipeline = PickPipeline(detector=detector, calibrator=calibrator)
    target = pipeline.locate_brick(frame, pose)

    if target is None:
        print("    back-projection MISSED the table plane — no PickTarget.")
        print("    (the camera ray never reaches z = TABLE_Z_IN_BASE)")
        sys.exit(1)

    print("\n    PickTarget (robot base frame, physical):")
    print(f"      x = {target.x * 1000:+8.1f} mm")
    print(f"      y = {target.y * 1000:+8.1f} mm   (+y is LEFT)")
    print(f"      z = {target.z * 1000:+8.1f} mm   (table {config.TABLE_Z_IN_BASE * 1000:.0f} "
          f"+ {config.PICK_Z_OFFSET * 1000:.0f} offset)")
    print(f"      yaw = {target.yaw_deg:+.1f} deg  (computed, then DROPPED — 5-DOF arm)")

    reach_mm = float(np.hypot(target.x, target.y)) * 1000
    print(f"      horizontal distance from the base axis: {reach_mm:.0f} mm")

    # ---- 4. could the arm actually go there? ------------------------------
    print("\n[4] REACHABILITY (solver math only — still no motion)")
    steps = plan_pick_sequence(target)
    floor_z = config.TABLE_Z_IN_BASE + config.MIN_CLAW_HEIGHT_M

    for step in steps:
        p = step.pose
        tag = f"    {step.label:<22}"
        if p.z < floor_z:
            print(f"{tag} z={p.z * 1000:+7.1f} mm  REFUSED by floor guard "
                  f"(< {floor_z * 1000:.1f})")
            continue
        try:
            sol, err_mm = ik.request_ik(p.x, p.y, p.z, seed_rad=angles_rad)
            deltas = [abs(np.degrees(s - a)) for s, a in zip(sol, angles_rad)]
            worst = max(deltas)
            flag = "  <-- OVER 45 deg, STAND BY THE POWER CUT" if worst > config.SERVO_WATCH_POWER_MOVE_DEG else ""
            print(f"{tag} z={p.z * 1000:+7.1f} mm  ok, IK err {err_mm:5.1f} mm, "
                  f"largest joint move {worst:5.1f} deg{flag}")
        except IKUnreachableError as e:
            print(f"{tag} z={p.z * 1000:+7.1f} mm  UNREACHABLE ({e})")
        except Exception as e:
            print(f"{tag} z={p.z * 1000:+7.1f} mm  solver error: {e}")

    ik.close()

    # ---- annotated frame ---------------------------------------------------
    if args.save:
        annotated = frame.copy()
        cv2.drawContours(annotated, [brick.contour], -1, (0, 255, 0), 2)
        cx, cy = int(brick.centroid_px[0]), int(brick.centroid_px[1])
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(annotated,
                    f"({target.x * 1000:+.0f}, {target.y * 1000:+.0f}, {target.z * 1000:+.0f}) mm",
                    (max(cx - 90, 5), max(cy - 15, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(args.save, annotated)
        print(f"\n    annotated frame -> {args.save}")

    print("\n" + "=" * 68)
    print("Now measure. Put a ruler on the brick and compare against the")
    print("PickTarget above: +x is forward from the base, +y is to the LEFT.")
    print("If those two numbers do not agree, DO NOT run the pick.")
    print("=" * 68)


if __name__ == "__main__":
    main()

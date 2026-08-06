"""Centre the brick in the camera, then descend onto it — closed loop, both axes.

    !!! THIS DRIVES THE REAL ARM, AND UNLIKE EVERY OTHER SCRIPT HERE IT     !!!
    !!! MOVES REPEATEDLY WITHOUT ASKING BETWEEN STEPS. Ctrl-C freezes it,   !!!
    !!! and so does q in the --view window.                                 !!!

WHAT THIS IS FOR. goto_point.py commands one open-loop move to a point measured
with a ruler; if the arm lands somewhere else, it has no way to know. This
closes that loop with the camera at every stage:

    centre the brick  ->  descend a little  ->  re-centre  ->  descend a little

Descending is where an open-loop pick fails worst, because the eye-in-hand
camera's view shifts as the arm comes down, so any residual aim error grows on
the way in. Re-centring between every descent step means the error is corrected
at the height it appears at, and never accumulates. Overshoot -- the thing that
just happened on J1 -- stops being a failure and becomes the next correction.

WHAT IT DELIBERATELY DOES NOT USE. For CENTRING: no hand-eye transform, no
camera intrinsics, no TABLE_Z_IN_BASE, no assumption about the base origin.
Errors are in pixels, corrections in ticks, and the conversion between them is
MEASURED at runtime by a probe move (see planning/visual_servo.py). That is what
makes it usable while data/hand_eye.json is wrong by 52 mm -- the broken number
is never consulted, so it cannot contribute an error.

DESCENT IS DIFFERENT, AND HONESTLY SO. Lowering the claw is a Cartesian move, so
it does go through MATLAB IK and it does depend on the joint calibration --
including J5's dir_sign, which is currently an UNCONFIRMED HYPOTHESIS. The
descent is therefore staged in small steps, floor-guarded, and pauses after the
FIRST step for you to put a ruler under the claw. That first measurement is the
whole J5 question, answered before the arm is anywhere near the table:

    gap shrank by about the step size  ->  the calibration is describing reality
    gap barely moved, or moved wrong   ->  stop; J5 is inverted

TWO AXES, TWO PROBES. Horizontal image error is driven by J1 (base yaw), vertical
by J2 (shoulder, which changes reach and so slides the view forward/back). Each
axis measures its own ticks-per-pixel with its own probe. Cross-coupling between
them is ignored on purpose -- a diagonal approximation of the image Jacobian is
enough when the loop re-measures every iteration, and it keeps each axis's
direction independently witnessed rather than entangled in a matrix.

IT AIMS ABOVE THE BRICK, AT A REGION. The camera does not look down the claw's
axis -- it sits above and behind it -- so "brick centred in frame" is NOT "claw
over the brick". The aim point is the frame centre pushed DOWN by
--aim-offset-y, which leaves the crosshair sitting ABOVE the brick when the claw
is where it should be. And the target is a BOX (--tolerance-x/-y), not a pixel:
the claw is over the brick for a whole region of image positions, the descent
re-centres at every step so the approach corrects itself on the way in, and the
joints cannot resolve better than ~30 px anyway (a loaded joint will not execute
a step under ~25 ticks at all, which is what stalled two earlier runs).

Measure the offset once: put the claw over the brick by hand, open the live
view, and read off how far below the crosshair the brick sits.

THE DESCENT IS ONE SOLVE PER STEP, NOT TWO LOOPS. Lowering the claw and
correcting the brick's vertical position in the image are the SAME degree of
freedom: both ride the shoulder/elbow chain, and the camera is on the wrist, so
every millimetre of descent swings the view (~7 px/mm, measured 2026-08-05).
Treating them separately made the arm fight itself -- IK lowered the tip 20 mm
and threw the brick 140 px up the frame, a J3 jog dragged it back and raised the
tip 25 mm, and the next step spent itself undoing that. Four steps produced 4 mm
of net descent and then lost the brick.

So each step asks IK for ONE target that descends AND reaches out by exactly the
amount that cancels the swing that descent is about to cause. IK spreads it over
J2/J3/J4 however it likes -- which is the whole reason there is an IK solver in
this loop. Both coefficients are measured from the descent's own motion (step 1
goes straight down, step 2 adds a radial offset; both descend in full, so
neither is a wasted probe).

The step SIZE is derived, not fixed. Reach is capped per step, so when the
descent would inject more error than the reach can absorb, the descent shrinks
-- to zero if it must, which is simply a re-aim at constant height, the one
correction that cannot undo a descent. Progress resumes on its own once the
error is small enough to carry. Sideways error still goes through J1, which is
base yaw and cannot change height, so it costs the descent nothing.

IT WAITS FOR YOU FIRST. On startup nothing moves and nothing is judged: a live
window opens showing what the detector sees, and the run begins only when you
press g. Positioning the arm is the reason you started the script, so a missing
brick at that moment is the normal state of things, not an error. Only after
your go-ahead does 'no brick detected' become a reason to stop.

EVERY RUN STARTS AT THE HOVER POSE, and that is not a convenience -- it is why
the numbers mean anything. The camera is eye-in-hand, so a run beginning from an
arbitrary posture begins with the operator hand-positioning the arm until the
brick appears, and the probe then measures its gains against whatever geometry
that happened to be. Two runs from two postures are not comparable, which is
part of why a gain measured before a descent stopped describing the arm during
it on 2026-08-05. The move is built in: no goto_pose.py beforehand, and
--no-hover to opt out.

Keys in the --view window (click it first — keys go to the focused window):
    g / Enter / Space   give the go-ahead and start
    q / Esc             quit; during a run this also FREEZES the arm

Usage (from the repo root):
    python scripts/visual_servo.py --view --dry-run          # look, move nothing
    python scripts/visual_servo.py --view                    # centre on x+y only
    python scripts/visual_servo.py --view --descend          # THE WHOLE RUN:
                                                             #   hover -> probe ->
                                                             #   centre -> descend
    python scripts/visual_servo.py --view --descend --axes x # single-axis centring
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from vision_pipeline import config
from vision_pipeline.capture.camera import Camera
from vision_pipeline.detection.lego_detector import LegoBrickDetector
from vision_pipeline.overlay import draw_aim, draw_boards, draw_hud
from vision_pipeline.planning.visual_servo import (
    DescentModel,
    ProgressMonitor,
    ServoAbort,
    clamp_predicted_shift,
    effective_deadband,
    enforce_minimum_step,
    estimate_axis,
    pixel_error,
    radial_tangential,
    reach_from_axis,
    resolution_px,
    step_command,
    step_ticks,
)
from vision_pipeline.robot_interface import poses
from vision_pipeline.robot_interface.matlab_client import IKUnreachableError, MatlabIKClient
from vision_pipeline.robot_interface.servo_calibration import dir_sign_report
from vision_pipeline.robot_interface.servo_driver import ServoBus, ServoSafetyError

IK_JOINTS = (1, 2, 3, 4, 5)

# Which joint drives which image axis. J1 is base yaw -> horizontal. J2 is the
# shoulder, which extends and retracts reach -> the view slides forward/back,
# which reads as vertical in the image. Both are only STARTING GUESSES: the
# probe measures the real response and refuses the axis if there isn't one.
AXIS_JOINT = {"x": 1, "y": 2}

# Detection must be settled before the arm moves on it. A single frame straight
# after a move can catch the camera mid-exposure or the arm still ringing.
SETTLE_FRAMES = 3

# How many frames to give detection before calling the brick lost. Glossy studs
# make detection flicker: one unlucky frame is noise, a dozen is a real loss.
DETECT_ATTEMPTS = 6

# (The old MAX_DROP_MULTIPLE lived here. It bounded a "ladder" that let a
# descent step ask for extra drop to recover height a re-centre had given back.
# There is nothing to recover now: the descent no longer has a separate
# re-centring phase that can raise the tip, so a step never needs to be larger
# than one step. See DescentModel.)

# Consecutive corrections that disturb the OTHER axis more than they improve
# their own before the loop calls that axis unusable. Two is noise; three in a
# row is the geometry telling you something.
UNPROFITABLE_PATIENCE = 3

WINDOW = "Visual servo — q or Esc stops and freezes the arm"


class ViewAborted(Exception):
    """The operator pressed q in the live window. Treated as a stop request."""


# --- live view ---------------------------------------------------------------

def show(view_on, frame, lines, centroid=None, ctx=None, detectors=None):
    """Draw and pump the live window, if --view is on.

    Raises ViewAborted if the operator asked to stop. It raises rather than
    returning a flag because pressing q is a genuine e-stop path -- the caller
    freezes the arm on it -- and a flag is something a caller can forget to check.
    """
    if not view_on or frame is None:
        return
    view = frame.copy()
    if detectors:
        draw_boards(view, detectors)
    draw_aim(view, centroid,
             aim_px=ctx.aim(frame.shape) if ctx else None,
             tolerance=ctx.tolerance if ctx else None)
    draw_hud(view, lines)
    cv2.imshow(WINDOW, view)
    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
        raise ViewAborted("operator pressed q in the live window")


def live_progress(ctx, what):
    """A move progress callback that keeps the live window alive AND responsive.

    THE REASON THIS EXISTS. cv2 only redraws and only reads the keyboard inside
    waitKey, so any blocking call that does not pump it freezes the window --
    and the frozen frame is the LAST one, which looks like a live view of an arm
    that is not moving. On 2026-08-05 the operator watched a stale frame while
    the arm executed a multi-hop move, with q doing nothing, and described it
    exactly right: it kept moving without instruction.

    Pumping between hops fixes both halves. The view stays current, and q raises
    ViewAborted from inside the move, which the caller turns into a freeze. That
    makes the stop key available DURING motion, which is the only time it is
    actually needed.

    Returns None when --view is off, which move_joints_stepped accepts.
    """
    if not ctx.view:
        return None

    def progress(k, n):
        print(f"      hop {k}/{n}", flush=True)
        frame = ctx.camera.read_frame()
        dets = ctx.detector.detect(frame)
        centroid = dets[0].centroid_px if dets else None
        show(True, frame,
             [f"MOVING — {what}", f"hop {k}/{n}",
              "brick: " + ("tracked" if centroid else "not in view"),
              "q stops and freezes"],
             centroid, ctx, ctx.detectors)

    return progress


def wait_watching(seconds, ctx, lines):
    """Rest between moves, keeping the live window alive and responsive.

    A plain time.sleep would freeze the window for the whole settle period --
    exactly the interval where the operator most wants to see what the arm just
    did, and the only interval in which q could stop the next move before it
    happens.
    """
    if not ctx.view:
        time.sleep(seconds)
        return
    end = time.time() + seconds
    while time.time() < end:
        frame = ctx.camera.read_frame()
        dets = ctx.detector.detect(frame)
        centroid = dets[0].centroid_px if dets else None
        show(True, frame, lines + ["settling..."], centroid, ctx, ctx.detectors)


class Context:
    """Everything the phases below share, so signatures stay readable."""

    def __init__(self, bus, camera, detector, args, detectors, ik=None):
        self.bus = bus
        self.camera = camera
        self.detector = detector
        self.args = args
        self.detectors = detectors
        self.ik = ik
        self.view = args.view
        # Superseded by the tolerance box below; kept only so older callers and
        # tests that still pass a scalar deadband keep working.
        self.deadband = getattr(args, "deadband", config.SERVO_VISUAL_DEADBAND_PX)
        # Where the brick should end up, and how close counts. Not the frame
        # centre: the camera sits above and behind the claw, so the claw is over
        # the brick when the brick sits BELOW the crosshair by the camera-to-claw
        # offset. Kept here so every phase and the overlay agree on one answer.
        self.aim_offset_y = getattr(args, "aim_offset_y",
                                    config.SERVO_VISUAL_AIM_OFFSET_Y_PX)
        self.tolerance = (getattr(args, "tolerance_x",
                                  config.SERVO_VISUAL_TOLERANCE_X_PX),
                          getattr(args, "tolerance_y",
                                  config.SERVO_VISUAL_TOLERANCE_Y_PX))

    def aim(self, frame_shape):
        """The pixel the brick is being driven to, for this frame size."""
        h, w = frame_shape[0], frame_shape[1]
        return (w / 2.0, h / 2.0 + self.aim_offset_y)


def report_aim_reachability(args) -> None:
    """Say how much of the acceptance box the camera can actually see.

    A large aim offset pushes the box toward the bottom of the frame and can
    push it off entirely. That failure is silent in every other respect: the
    detector works, the arm moves, the gains are measured correctly, and the
    loop simply never reports success because the brick can never occupy a pixel
    that does not exist. Worth one line at startup rather than fifteen wasted
    iterations and a guess about which subsystem is wrong.
    """
    aim_y = config.FRAME_HEIGHT / 2.0 + args.aim_offset_y
    top, bottom = aim_y - args.tolerance_y, aim_y + args.tolerance_y
    visible = max(0.0, min(bottom, config.FRAME_HEIGHT) - max(top, 0.0))
    fraction = visible / (2 * args.tolerance_y)

    print(f"aim point    y = {aim_y:.0f} of {config.FRAME_HEIGHT} px")
    if fraction >= 0.999:
        return
    if visible <= 0:
        print(f"  *** THE AIM POINT IS OFF THE BOTTOM OF THE FRAME. The brick")
        print(f"      cannot reach it and this loop CANNOT converge. Reduce")
        print(f"      --aim-offset-y below "
              f"{config.FRAME_HEIGHT / 2.0 + args.tolerance_y:.0f}.")
        return
    print(f"  NOTE: only {fraction * 100:.0f}% of the target box is on screen "
          f"(rows {max(top, 0):.0f}-{min(bottom, config.FRAME_HEIGHT):.0f}).")
    print(f"      The brick must finish in the bottom {visible:.0f} rows, where it")
    print(f"      may also be clipped by the frame edge — a clipped blob's")
    print(f"      centroid is biased UPWARD, away from the aim point. If the")
    print(f"      loop stalls just short, that is the reason. Largest offset")
    print(f"      keeping the whole box visible: "
          f"{config.FRAME_HEIGHT / 2.0 - args.tolerance_y:.0f}.")


# --- vision ------------------------------------------------------------------

def detect_centroid(ctx, attempts=DETECT_ATTEMPTS):
    """Grab a settled frame and return (centroid_px, frame), or (None, frame).

    Reads several frames and keeps the last: webcams buffer, so the first frame
    after a move is frequently a stale one from before it. Acting on a stale
    frame in a control loop is worse than acting on no frame -- it reads as the
    arm having not responded, which is what the runaway guard is watching for.

    Then RETRIES on a miss. Detection of a glossy brick is not perfectly stable
    frame to frame -- a highlight that shifts by a few pixels can drop the
    confidence below threshold for one frame and restore it on the next. A
    single-frame miss must not end a run: 'Lost the brick' should mean the brick
    is genuinely gone, not that one frame was unlucky. Only if every attempt
    fails is it treated as a real loss.
    """
    frame = None
    for attempt in range(max(1, attempts)):
        for _ in range(SETTLE_FRAMES if attempt == 0 else 1):
            frame = ctx.camera.read_frame()
        detections = ctx.detector.detect(frame)
        if detections:
            return detections[0].centroid_px, frame
    return None, frame


def describe(err_x, err_y):
    """Plain-language rendering of a pixel error, for the operator."""
    parts = []
    if abs(err_x) >= 1:
        parts.append(f"{abs(err_x):.0f}px {'right' if err_x > 0 else 'left'}")
    if abs(err_y) >= 1:
        parts.append(f"{abs(err_y):.0f}px {'below' if err_y > 0 else 'above'}")
    return ", ".join(parts) if parts else "centred"


def axis_error(axis, ex, ey):
    return ex if axis == "x" else ey


# --- actuators ---------------------------------------------------------------
#
# An actuator is anything that can move the brick across the image by a
# commanded amount. The control law never learns which kind it is driving: the
# probe measures "command units per pixel" and the same arithmetic follows.

class JointActuator:
    """Jogs one joint by raw ticks. Simple, direct, and limited by that joint.

    When the joint reaches its travel limit the correction stops, however much
    the rest of the arm could still contribute. That ceiling is the reason
    CartesianActuator exists.
    """

    kind = "joint"

    def __init__(self, joint, probe_amount=None, max_step=None):
        self.joint = joint
        self.probe_amount = (config.SERVO_VISUAL_PROBE_TICKS
                             if probe_amount is None else probe_amount)
        self.max_step = (config.SERVO_VISUAL_MAX_STEP_TICKS
                         if max_step is None else max_step)
        self.unit = "ticks"

    def label(self):
        return f"J{self.joint}"

    def apply(self, ctx, amount):
        amount = int(round(amount))
        if amount == 0:
            return 0
        tick = ctx.bus.read_position_retrying(self.joint)
        ctx.bus.move_and_verify(self.joint, tick + amount)
        return amount

    def reverse(self, ctx, amount):
        """Undo a move, ignoring failures — this runs on the recovery path."""
        try:
            self.apply(ctx, -amount)
        except Exception as e:
            print(f"      could not reverse the last move: {e}")

    def state(self, ctx):
        return f"@{ctx.bus.read_position_retrying(self.joint)}"


class CartesianActuator:
    """Nudges the TOOL along a direction, letting IK pick the joints.

    This is the joint-limit workaround, and it is the whole reason to involve
    MATLAB in re-centring at all. A joint jog can only ever use its one joint.
    A Cartesian request states the goal -- move the tool this way by this much
    -- and the IK solver satisfies it with whatever posture is legal, because
    init_arm.m has already loaded the measured travel limits into the model's
    PositionLimits. A joint at its stop simply stops contributing while the
    others take up the motion.

    Directions are RADIAL and TANGENTIAL rather than fixed base axes, so they
    keep meaning the same thing to the image as J1 swings the camera around.
    """

    kind = "cartesian"

    def __init__(self, direction, probe_amount=None, max_step=None):
        self.direction = direction        # "radial" or "tangential"
        self.probe_amount = (config.SERVO_VISUAL_PROBE_MM
                             if probe_amount is None else probe_amount)
        self.max_step = (config.SERVO_VISUAL_MAX_STEP_MM
                         if max_step is None else max_step)
        self.unit = "mm"

    def label(self):
        return f"{self.direction} nudge"

    def _unit_vector(self, ctx):
        _angles, tip = tip_position(ctx)
        radial, tangential = radial_tangential((tip[0], tip[1]), yaw_axis_xy(ctx))
        vec = radial if self.direction == "radial" else tangential
        return np.array([vec[0], vec[1], 0.0]), tip

    def apply(self, ctx, amount_mm):
        if abs(amount_mm) < config.SERVO_VISUAL_MIN_STEP_MM:
            return 0.0
        vec, tip = self._unit_vector(ctx)
        target = tip + vec * (amount_mm / 1000.0)

        floor_z = config.TABLE_Z_IN_BASE + config.MIN_CLAW_HEIGHT_M
        if target[2] < floor_z:
            raise ServoAbort(
                f"A {amount_mm:+.1f} mm {self.direction} nudge would put the tool "
                f"below the floor guard. Refusing."
            )

        angles = [ctx.bus.ticks_to_rad(j, ctx.bus.read_position_retrying(j))
                  for j in IK_JOINTS]

        # Shrink until the solution is one the CAMERA can live with. A solve can
        # be geometrically perfect and still useless: near the base axis the
        # solver pays for a few millimetres of tip motion with several degrees
        # of base yaw, and since the camera rides on the wrist that yaw sweeps
        # the entire image. The loop would then be reacting to its own motion.
        # No iteration cap: the loop either accepts a solution or shrinks until
        # the nudge is below the minimum useful size and then refuses. A bounded
        # retry count would let a bad solution escape by falling out of the loop,
        # which is the one outcome this guard exists to prevent.
        attempt_mm = amount_mm
        while True:
            try:
                solution, err_mm = ctx.ik.request_ik(*(tip + vec * (attempt_mm / 1000.0)),
                                                     seed_rad=angles)
            except IKUnreachableError as e:
                raise ServoAbort(
                    f"IK cannot reach a {attempt_mm:+.1f} mm {self.direction} "
                    f"nudge from here: {e}. The arm is at the edge of its "
                    f"workspace, not merely at one joint's limit."
                )

            targets = {j: ctx.bus.rad_to_ticks(j, a)
                       for j, a in zip(IK_JOINTS, solution)}
            moved = {j: targets[j] - ctx.bus.rad_to_ticks(j, a)
                     for j, a in zip(IK_JOINTS, angles)}
            pan_deg = abs(moved[1]) / 651.89 * 180 / np.pi
            biggest = max(abs(d) for d in moved.values())

            if (pan_deg <= config.SERVO_VISUAL_MAX_PAN_DEG
                    and biggest <= config.SERVO_VISUAL_MAX_SOLVE_TICKS):
                break
            if abs(attempt_mm) / 2 < config.SERVO_VISUAL_MIN_STEP_MM:
                raise ServoAbort(
                    f"Cartesian control is too poorly conditioned here to use. A "
                    f"{attempt_mm:+.1f} mm {self.direction} nudge needs "
                    f"{pan_deg:.1f} deg of base yaw (limit "
                    f"{config.SERVO_VISUAL_MAX_PAN_DEG}), which pans the camera "
                    f"further than the correction is worth.\n"
                    f"      The tip is {reach_from_axis((tip[0], tip[1]), yaw_axis_xy(ctx)) * 1000:.0f} mm "
                    f"from the base axis; the claw hangs ~27 mm off the arm's "
                    f"plane, so close in the tip's bearing is hypersensitive to "
                    f"J1. Working past ~130 mm fixes it (0.8 deg at 160 mm).\n"
                    f"      Move the brick further out, or re-run with "
                    f"--recentre joint --joint-y 3."
                )
            attempt_mm /= 2
            print(f"      solve needed {pan_deg:.1f} deg of camera pan; "
                  f"halving the nudge to {attempt_mm:+.1f} mm")

        busy = ", ".join(f"J{j}{d:+d}" for j, d in moved.items() if abs(d) >= 2)
        print(f"      IK residual {err_mm:.1f} mm; joints {busy or 'none moved'}"
              f"   (camera pan {pan_deg:.2f} deg)")
        amount_mm = attempt_mm

        ctx.bus.move_joints_stepped(
            targets, step_ticks=config.PICK_STEP_TICKS,
            pause_s=config.PICK_STEP_PAUSE_S,
            progress=live_progress(ctx, f"{self.direction} {amount_mm:+.1f} mm"),
        )
        return amount_mm

    def reverse(self, ctx, amount_mm):
        """Undo a nudge, ignoring failures — this runs on the recovery path."""
        try:
            self.apply(ctx, -amount_mm)
        except Exception as e:
            print(f"      could not reverse the last nudge: {e}")

    def state(self, ctx):
        _angles, tip = tip_position(ctx)
        return f"tip ({tip[0] * 1000:+.0f},{tip[1] * 1000:+.0f},{tip[2] * 1000:+.0f})"


def wait_for_go(ctx):
    """Hold before touching anything, so the operator can position the arm.

    Deliberately BEFORE the detection check, not after. Checking first and
    exiting on 'no brick' punishes the operator for the camera not yet pointing
    at the table -- which is the normal state of things when the script starts,
    since positioning is the reason they ran it. So this phase reports what the
    detector currently sees, live and continuously, and does not judge: the
    operator decides when the view is right, and only then does a missing brick
    become an error worth stopping for.

    With --view the go-ahead is a keypress in the window (where the operator is
    already looking); without it, a typed confirmation in the terminal. Raises
    ViewAborted if they quit instead.
    """
    if ctx.args.no_wait:
        return

    if not ctx.view:
        centroid, _ = detect_centroid(ctx)
        print("\n--- POSITION THE ARM ---")
        print(f"  detector currently sees: "
              f"{'a brick at %.0f, %.0f px' % centroid if centroid else 'NO BRICK'}")
        print("  Move the arm/brick until the camera is looking at it. Re-check")
        print("  any time with:  python scripts/run_live_view.py  (close it again")
        print("  before continuing — only one process can hold the camera).")
        if input("\n  Type 'go' when ready: ").strip().lower() not in ("go", "g", "y", "yes"):
            raise ViewAborted("operator did not confirm")
        return

    print("\n--- POSITION THE ARM ---")
    print("  Live window is open. Move the arm/brick until the brick is outlined")
    print("  in green, then click the window and press g to start.")
    print("  q or Esc quits without commanding anything.")
    while True:
        frame = ctx.camera.read_frame()
        dets = ctx.detector.detect(frame)
        centroid = dets[0].centroid_px if dets else None
        h, w = frame.shape[:2]
        status = ["POSITION THE ARM — nothing has moved yet",
                  "brick: " + (f"found, conf {dets[0].confidence:.2f}"
                               if dets else "NOT DETECTED"),
                  "press g to START,  q to quit"]
        if centroid:
            ax, ay = ctx.aim(frame.shape)
            status.insert(2, f"error {centroid[0] - ax:+.0f}, "
                             f"{centroid[1] - ay:+.0f} px from the aim point")
        view = frame.copy()
        if ctx.detectors:
            draw_boards(view, ctx.detectors)
        draw_aim(view, centroid, aim_px=ctx.aim(frame.shape),
                 tolerance=ctx.tolerance)
        draw_hud(view, status)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise ViewAborted("operator quit from the positioning window")
        if key in (ord("g"), 13, 32):   # g, Enter, Space
            print("  Go-ahead received.")
            return


# --- phase 1: probe ----------------------------------------------------------

def probe_axis(ctx, axis, actuator):
    """Measure how `actuator` moves the brick along image `axis`.

    Identical for a joint jog and a Cartesian nudge -- the probe measures the
    composite response of whatever it drove, in whatever unit that actuator
    commands, so the control law downstream never needs to know which it was.
    """
    centroid, frame = detect_centroid(ctx)
    if centroid is None:
        raise ServoAbort(f"Lost the brick before probing {axis}.")
    before = axis_error(axis, *pixel_error(centroid, frame.shape,
                                           ctx.aim(frame.shape)))

    amount = actuator.probe_amount
    print(f"\n--- PROBE {axis}: {actuator.label()} {amount:+.4g} "
          f"{actuator.unit} ---")
    print("  Measuring which way the brick moves. Watch the arm.")
    show(ctx.view, frame, [f"PROBE {axis}: {actuator.label()} "
                           f"{amount:+.4g} {actuator.unit}",
                           f"error {before:+.0f} px"],
         centroid, ctx, ctx.detectors)

    actuator.apply(ctx, amount)
    wait_watching(ctx.args.settle, ctx, [f"PROBE {axis}: applied {amount:+.4g}"])

    centroid, frame = detect_centroid(ctx)
    if centroid is None:
        raise ServoAbort(f"Lost the brick during the {axis} probe — it may have "
                         f"left the frame. Re-aim and restart.")
    after = axis_error(axis, *pixel_error(centroid, frame.shape,
                                          ctx.aim(frame.shape)))

    est = estimate_axis(amount, before, after)
    print(f"  brick moved {est.response_px:+.1f} px on {axis}")
    print(f"  MEASURED {est.ticks_per_px:+.3g} {actuator.unit}/px "
          f"(direction {est.direction:+d})")
    if actuator.kind == "joint":
        cal_sign = ctx.bus.calibration[str(actuator.joint)]["dir_sign"]
        print(f"  calibration says J{actuator.joint} dir_sign {cal_sign:+d} — this "
              f"loop is unaffected either way.")
    return est


# --- phase 2: centre ---------------------------------------------------------

def centre(ctx, estimates, label="CENTRING"):
    """Close the loop until the brick sits inside the target box.

    Corrects the WORST axis each iteration rather than both at once: the two
    joints couple (moving J2 also shifts the brick horizontally), and applying
    both corrections from one frame double-counts that coupling into an
    overshoot. Fixing the larger error and re-measuring converges without
    needing to model the coupling at all.
    """
    monitor = ProgressMonitor()
    print(f"\n--- {label} (max {ctx.args.max_iterations} iterations) ---")

    last = None          # (actuator, amount) of the previous applied move
    prev = None          # (axis, actuator, ex, ey) to score the last correction
    unprofitable = {}    # axis -> consecutive corrections that cost more than they gained
    for i in range(1, ctx.args.max_iterations + 1):
        centroid, frame = detect_centroid(ctx)

        # Losing the brick is usually SELF-INFLICTED: the move we just made
        # pushed it out of view. Backing that move out restores a view we know
        # was good a moment ago, which beats ending the run and leaving the arm
        # somewhere the camera can see nothing.
        if centroid is None and last is not None:
            act, amount = last
            print(f"  brick out of view — reversing the last "
                  f"{amount:+.4g} {act.unit} to get it back")
            act.reverse(ctx, amount)
            wait_watching(ctx.args.settle, ctx, ["recovering the view..."])
            centroid, frame = detect_centroid(ctx)
            last = None
        if centroid is None:
            raise ServoAbort(
                "Lost the brick and could not recover it by backing out the "
                "last move. Re-aim and restart."
            )
        ex, ey = pixel_error(centroid, frame.shape, ctx.aim(frame.shape))

        # SCORE THE PREVIOUS CORRECTION. Correcting one axis always disturbs the
        # other a little; that is fine, and the alternation absorbs it. What is
        # not fine is a correction that disturbs the other axis MORE than it
        # improves its own — that is a losing trade, and repeating it walks the
        # arm sideways forever while the loop looks busy.
        #
        # Seen 2026-08-05 close to the base axis: each radial nudge bought ~4 px
        # of vertical and cost ~13 px of horizontal, so seven iterations moved
        # the brick from 24 px right to 63 px left while barely touching the
        # error it was aiming at. The progress monitor eventually stops it, but
        # only after a lot of pointless motion and with a message that does not
        # name the cause.
        if prev is not None:
            p_axis, p_act, p_ex, p_ey = prev
            gained = abs(axis_error(p_axis, p_ex, p_ey)) - abs(axis_error(p_axis, ex, ey))
            other = "y" if p_axis == "x" else "x"
            cost = abs(axis_error(other, ex, ey)) - abs(axis_error(other, p_ex, p_ey))
            if cost > max(gained, 0.0):
                unprofitable[p_axis] = unprofitable.get(p_axis, 0) + 1
                print(f"       ({p_axis} correction gained {gained:.0f} px but cost "
                      f"{cost:.0f} px on {other})")
                if unprofitable[p_axis] >= UNPROFITABLE_PATIENCE:
                    raise ServoAbort(
                        f"The {p_axis} axis is costing more than it gains: "
                        f"{unprofitable[p_axis]} corrections in a row disturbed "
                        f"{other} more than they improved {p_axis}. Driving it "
                        f"harder will not help — the geometry is wrong, not the "
                        f"gain.\n"
                        f"      Close to the base axis a radial nudge needs base "
                        f"yaw, and yaw pans the camera far more than the nudge "
                        f"moves the view. Either work further out (past ~130 mm "
                        f"of reach), or drive this axis in joint space where no "
                        f"IK is involved and J1 never moves:\n"
                        f"          --recentre joint --joint-y 3"
                    )
            else:
                unprofitable[p_axis] = 0

        # ALREADY GOOD ENOUGH? The target is a BOX around the aim point, not a
        # point: the camera sits above and behind the claw, so the claw is over
        # the brick for a whole region of image positions, not one pixel. The
        # box is generous on purpose — the descent re-centres at every step, so
        # the approach corrects itself on the way in, and the joints cannot
        # resolve better than ~30 px anyway.
        tx, ty = ctx.tolerance
        if abs(ex) <= tx and abs(ey) <= ty:
            print(f"\n  IN THE TARGET REGION — {describe(ex, ey)} of the aim "
                  f"point, inside the {tx:.0f} x {ty:.0f} px box.")
            return True

        # Pick the axis that is furthest OUT OF THE BOX, measured as a fraction
        # of its own tolerance so the two axes are compared fairly — 40 px of x
        # error against a 45 px tolerance is not worse than 50 px of y against
        # 55, and picking by raw pixels would keep servicing the looser axis.
        def excess(pair):
            a = pair[0][0]
            return abs(axis_error(a, ex, ey)) / (tx if a == "x" else ty)

        worst = max(estimates, key=excess)
        (axis, actuator), est = worst[0], worst[1]
        err = axis_error(axis, ex, ey)
        band = tx if axis == "x" else ty

        # A joint-space actuator cannot execute an arbitrarily small step: under
        # load it does not move at all below ~25 ticks, so a correction smaller
        # than that is rounded up to one that will actually move.
        # Read the floor at call time, not from a default argument: Python binds
        # `= config.X` defaults once at import, so a default would ignore any
        # later change to config and silently pin the value the module happened
        # to load with.
        min_step = config.SERVO_VISUAL_MIN_STEP_TICKS
        if actuator.kind == "joint":
            band = max(band, effective_deadband(est, band, min_step))
        delta = step_command(err, est, max_step=actuator.max_step,
                             deadband_px=band)
        if actuator.kind == "joint":
            delta = enforce_minimum_step(delta, min_step)
        # The binding clamp: never command a step predicted to move the brick
        # further than a fraction of the frame, whatever the unit clamp allows.
        clamped = clamp_predicted_shift(delta, est, frame.shape)
        if abs(clamped) < abs(delta):
            print(f"       (step held back {delta:+.4g} -> {clamped:+.4g} "
                  f"{actuator.unit} to keep the brick in frame)")
        delta = clamped

        print(f"  [{i:2d}] {describe(ex, ey):24s} -> {axis} via "
              f"{actuator.label()} {delta:+.4g} {actuator.unit}")
        show(ctx.view, frame,
             [f"{label} {i}/{ctx.args.max_iterations}",
              f"error {ex:+.0f}, {ey:+.0f} px",
              f"{axis} via {actuator.label()}  next {delta:+.4g} {actuator.unit}",
              "q stops and freezes"],
             centroid, ctx, ctx.detectors)

        if delta == 0:
            print(f"\n  AS CLOSE AS THIS AXIS RESOLVES — {describe(ex, ey)}, "
                  f"and {actuator.label()} cannot execute a step smaller than "
                  f"{min_step} ticks under load ({resolution_px(est, min_step):.0f} "
                  f"px here). Aiming tighter would only command moves it ignores.")
            return True

        monitor.update(abs(err))
        try:
            actuator.apply(ctx, delta)
            last = (actuator, delta)
            prev = (axis, actuator, ex, ey)
        except ServoSafetyError as e:
            # A joint actuator has nowhere else to go; a Cartesian one already
            # asked IK for the best posture available, so if the bus still
            # refuses it, the arm genuinely cannot make this motion.
            if actuator.kind == "joint":
                raise ServoAbort(
                    f"{actuator.label()} is at its travel limit and cannot "
                    f"correct {axis} any further ({e}). Re-run with "
                    f"--recentre cartesian so IK can use the other joints."
                )
            raise ServoAbort(f"Even with IK redistributing, the bus refused: {e}")
        wait_watching(ctx.args.settle, ctx,
                      [f"{label} {i}: {actuator.label()} {delta:+.4g}"])

    print(f"\n  Hit the {ctx.args.max_iterations}-iteration cap without centring "
          f"(best {monitor.best_error:.1f} px). Bounded on purpose.")
    return False


def confirm_descend(ctx):
    """Hold after centring until the operator says to go down.

    Deliberately a hard stop rather than a timeout or a flag. Centring is
    reversible and happens well clear of the table; descending is the part that
    can put the claw through the brick or into the tabletop, and it depends on
    the joint calibration in a way centring does not — J5's dir_sign is still an
    unconfirmed hypothesis. The operator has the arm in front of them and can
    see whether the claw is genuinely over the brick, which is a judgement no
    pixel count substitutes for.
    """
    print("\n" + "-" * 68)
    print("  CENTRED. Look at the arm: is the claw over the brick?")
    print("  Nothing will descend until you say so.")
    print("-" * 68)
    if ctx.view:
        try:
            frame = ctx.camera.read_frame()
            dets = ctx.detector.detect(frame)
            show(True, frame,
                 ["CENTRED — waiting for your go-ahead",
                  "answer in the TERMINAL, not here"],
                 dets[0].centroid_px if dets else None, ctx, ctx.detectors)
        except ViewAborted:
            return False
    try:
        answer = input("  Descend onto the brick? [y/N] ").strip().lower()
    except EOFError:
        return False
    if answer in ("y", "yes"):
        return True
    print("  Not descending. The arm is holding where it is.")
    return False


# --- phase 3: descend --------------------------------------------------------

def yaw_axis_xy(ctx):
    """The base yaw axis in base-frame XY metres, measured from FK and cached.

    Not the model origin -- it misses by 81 mm, which is more than the claw's
    own distance from the axis at the home pose. See
    MatlabIKClient.base_yaw_axis_xy.
    """
    if ctx.ik is None:
        raise ValueError(
            "radial/tangential directions need the MATLAB FK server: the base "
            "yaw axis is measured from FK, and assuming it sits at the origin "
            "is wrong by 81 mm on this arm.")
    return tuple(ctx.ik.base_yaw_axis_xy())


def report_starting_error(ctx, centroid, frame):
    """Print where the brick sits relative to the aim point. Returns the aim px.

    Shared so the dry-run path and a real run print the same thing, and so a
    real run can print it AFTER reaching the hover pose rather than before --
    the numbers are only meaningful for the pose the run actually starts from.
    """
    ax, ay = ctx.aim(frame.shape)
    ex, ey = pixel_error(centroid, frame.shape, (ax, ay))
    h, w = frame.shape[:2]
    print(f"\nframe        {w}x{h}, centre ({w // 2}, {h // 2})")
    print(f"aim point    ({ax:.0f}, {ay:.0f}) px")
    print(f"brick at     ({centroid[0]:.0f}, {centroid[1]:.0f}) px")
    print(f"error        {describe(ex, ey)} of the aim point")
    if abs(ex) <= ctx.args.tolerance_x and abs(ey) <= ctx.args.tolerance_y:
        print("             (already inside the target box)")
    return ax, ay


def tip_position(ctx):
    """Current claw-tip position in the physical base frame, metres."""
    angles = [ctx.bus.ticks_to_rad(j, ctx.bus.read_position_retrying(j)) for j in IK_JOINTS]
    _, T_tip = ctx.ik.request_fk_tip(angles)
    return angles, T_tip[:3, 3]


def descend(ctx, estimates):
    """Lower the claw in small steps, re-centring between each.

    This is the only part that consults the joint calibration and the base
    frame, because 'down' is a Cartesian direction and the camera cannot measure
    it. Everything that can be verified from outside that chain is:

      * the floor guard refuses any commanded z below the table plus clearance;
      * the first step stops for a ruler, which is the one measurement that can
        catch an inverted J5 before the claw is near the table;
      * re-centring after each step means aim error never accumulates on the way
        in, even though the view shifts as the arm comes down.
    """
    floor_z = config.TABLE_Z_IN_BASE + config.MIN_CLAW_HEIGHT_M
    target_z = ctx.args.target_z / 1000.0 if ctx.args.target_z is not None else \
        config.TABLE_Z_IN_BASE + config.PICK_Z_OFFSET

    if target_z < floor_z:
        print(f"\n  Target z {target_z * 1000:+.1f} mm is below the floor guard "
              f"({floor_z * 1000:+.1f} mm). Refusing.")
        return False

    angles, tip = tip_position(ctx)
    print(f"\n--- DESCENT ---")
    print(f"  claw tip now  ({tip[0] * 1000:+.1f}, {tip[1] * 1000:+.1f}, "
          f"{tip[2] * 1000:+.1f}) mm")
    print(f"  target z      {target_z * 1000:+.1f} mm "
          f"({(tip[2] - target_z) * 1000:.1f} mm to go)")
    print(f"  step          {ctx.args.descend_step:.0f} mm, aim corrected inside "
          f"the same solve")
    print(f"  floor guard   {floor_z * 1000:+.1f} mm")

    if tip[2] <= target_z + 0.001:
        print("  Already at or below the target height. Nothing to descend.")
        return True

    # ONE SOLVE PER STEP. Descending and correcting the brick's vertical
    # position in the image are not independent -- both ride on the shoulder /
    # elbow chain, and the camera is on the wrist, so descending swings the view
    # (~7 px per mm, measured 2026-08-05). Running them as two loops made the
    # arm fight itself: IK lowered the tip 20 mm and threw the brick 140 px up
    # the frame; a J3 jog dragged it back down and raised the tip 25 mm; the
    # next step spent itself undoing that. Four steps, 4 mm of net descent, then
    # the brick left the frame entirely.
    #
    # So each step now asks for ONE Cartesian target that descends AND reaches
    # out by exactly the amount that cancels the swing the descent is about to
    # cause. IK distributes that across J2/J3/J4 as it sees fit -- which is the
    # entire reason for having an IK solver in the loop. DescentModel supplies
    # the two coefficients, both measured from the descent's own motion.
    model = DescentModel()
    # estimates is [((axis, ACTUATOR), estimate)] -- an actuator object, not a
    # joint number. Reading it as a joint number and wrapping it in a fresh
    # JointActuator produced JointActuator(JointActuator(...)), which reached
    # the packet builder as a servo ID. Same shape of bug as the one this
    # script's loop tests were originally written for; use what the caller
    # supplied rather than reconstructing it.
    x_axis = next((pair for pair in estimates if pair[0][0] == "x"), None)
    x_actuator = x_axis[0][1] if x_axis else None
    x_estimate = x_axis[1] if x_axis else None
    sideways_monitor = ProgressMonitor()
    sideways_used = 0.0

    if (x_actuator is not None and not ctx.args.no_sideways
            and x_actuator.kind == "joint"
            and ctx.bus.travel_limits(x_actuator.joint) is None):
        print(f"\n  NOTE: {x_actuator.label()} has no measured travel limits, so "
              f"the servo bus cannot refuse a bad sideways move.")
        print(f"    The only bounds are this script's: a "
              f"{ctx.args.sideways_budget:.0f}-tick budget for the whole descent, "
              f"and a stop the first time a correction makes the error worse.")
        print(f"    Measure them and the bus can help: "
              f"python scripts/find_joint_limits.py --joint {x_actuator.joint}")
    step_n = 0
    probe_mm = ctx.args.descend_probe_mm

    while True:
        angles, tip = tip_position(ctx)
        if tip[2] - target_z <= 0.001:
            print(f"\n  AT TARGET HEIGHT — tip z {tip[2] * 1000:+.1f} mm.")
            print("  The claw is over the brick at grasp height. Nothing has been")
            print("  grasped: closing the gripper is still a separate act.")
            return True

        centroid, frame = detect_centroid(ctx)
        if centroid is None:
            print("\n  STOPPED: lost the brick between steps. The arm is holding.")
            return False
        ex, ey = pixel_error(centroid, frame.shape, ctx.aim(frame.shape))

        step_n += 1
        wanted_z = max(tip[2] - ctx.args.descend_step / 1000.0, target_z, floor_z)
        dz_mm = (wanted_z - tip[2]) * 1000.0

        # How far to descend, and how far to reach out, in one decision.
        #   step 1  straight down, so the descent's own image response can be
        #           seen in isolation;
        #   step 2  the same descent plus a deliberate radial offset, which is
        #           what makes the two samples independent enough to fit;
        #   3+      the model plans both together, shrinking the descent to
        #           whatever the reach can keep aimed.
        # Steps 1 and 2 are not wasted probe moves -- both descend in full.
        if model.ready:
            dz_mm, dr_mm = model.plan_step(ey, dz_mm,
                                           ctx.args.descend_max_reach_mm)
            note = ("re-aiming at constant height" if abs(dz_mm) < 0.05
                    else "model")
        elif step_n == 1:
            dr_mm, note = 0.0, "straight down, learning the descent's own swing"
        else:
            dr_mm = probe_mm if ey < 0 else -probe_mm
            dr_mm = float(np.clip(dr_mm, -ctx.args.descend_max_reach_mm,
                                  ctx.args.descend_max_reach_mm))
            note = "descent + a radial offset, to separate the two responses"

        next_z = tip[2] + dz_mm / 1000.0
        if abs(dz_mm) < 0.05 and abs(dr_mm) < config.SERVO_VISUAL_MIN_STEP_MM:
            print(f"\n  STOPPED at step {step_n}: neither descending nor reaching "
                  f"would help.")
            print(f"    {model.describe()}")
            print(f"    brick {describe(ex, ey)} of the aim point. The arm is holding.")
            return False

        try:
            radial, _tangential = radial_tangential((tip[0], tip[1]), yaw_axis_xy(ctx))
        except ValueError as e:
            print(f"\n  STOPPED: {e}")
            return False
        target_xyz = (tip[0] + radial[0] * dr_mm / 1000.0,
                      tip[1] + radial[1] * dr_mm / 1000.0,
                      next_z)

        print(f"\n  step {step_n}: z {tip[2] * 1000:+.1f} -> {next_z * 1000:+.1f} mm "
              f"({dz_mm:+.1f}), reach {dr_mm:+.1f} mm   [{note}]")
        print(f"    brick {describe(ex, ey)} of the aim point")
        if model.ready:
            print(f"    {model.describe()}")

        # LOCK J1. A descent changes height and reach; the base's azimuth is
        # not part of the request, and measurement says it does not need to be:
        # 5, 10 and 20 mm descents from the same pose each needed 1.2 ticks of
        # J1, while a 40 mm one came back wanting 213. That larger number is the
        # solver spending redundancy it was never asked to spend -- five joints
        # against a 3-DOF target leaves a 2-D null space with nothing preferring
        # one point in it. Since the camera is on the wrist, that pans the whole
        # image and feeds straight back into the loop trying to correct it.
        #
        # Sideways error is corrected by a J1 jog that is bounded, monitored and
        # visible in the log. Locking here does not give that up; it stops the
        # base ALSO being moved silently by every descent solve.
        try:
            targets_rad, err_mm = ctx.ik.request_ik(*target_xyz, seed_rad=angles,
                                                    lock=[1] if ctx.args.lock_base
                                                    else None)
        except IKUnreachableError as e:
            print(f"    UNREACHABLE: {e}")
            print("    Stopping the descent here; the arm is holding.")
            return False
        print(f"    IK residual {err_mm:.1f} mm")

        targets = {j: ctx.bus.rad_to_ticks(j, a) for j, a in zip(IK_JOINTS, targets_rad)}
        blocked = [j for j in IK_JOINTS
                   if (lim := ctx.bus.travel_limits(j)) and not (lim[0] <= targets[j] <= lim[1])]
        if blocked:
            print(f"    REFUSED: {', '.join(f'J{j}' for j in blocked)} would leave "
                  f"measured travel. Stopping.")
            return False
        busy = ", ".join(f"J{j}{targets[j] - ctx.bus.rad_to_ticks(j, a):+d}"
                         for j, a in zip(IK_JOINTS, angles)
                         if abs(targets[j] - ctx.bus.rad_to_ticks(j, a)) >= 2)
        print(f"    joints {busy or 'none moved'}")

        show(ctx.view, frame,
             [f"DESCENT step {step_n}",
              f"z {tip[2] * 1000:+.0f} -> {next_z * 1000:+.0f} mm",
              f"reach {dr_mm:+.0f} mm", "q stops and freezes"],
             centroid, ctx, ctx.detectors)

        ctx.bus.move_joints_stepped(
            targets, step_ticks=config.PICK_STEP_TICKS, pause_s=config.PICK_STEP_PAUSE_S,
            progress=live_progress(ctx, f"descent step {step_n}")
            or (lambda k, n: print(f"      hop {k}/{n}", flush=True)),
        )
        wait_watching(ctx.args.settle, ctx, [f"DESCENT step {step_n} done"])

        # Feed the model with what the arm ACTUALLY did, from FK -- this arm
        # lands a millimetre or two off and the servos settle short under load,
        # so the commanded move is the wrong thing to fit against.
        _, tip_after = tip_position(ctx)
        actual_dz = (tip_after[2] - tip[2]) * 1000.0
        axis_xy = yaw_axis_xy(ctx)
        actual_dr = (reach_from_axis((tip_after[0], tip_after[1]), axis_xy)
                     - reach_from_axis((tip[0], tip[1]), axis_xy)) * 1000.0
        print(f"    FK says the tip fell {abs(actual_dz):.1f} mm "
              f"(commanded {abs(dz_mm):.1f}) and reached {actual_dr:+.1f} mm")

        after, frame_after = detect_centroid(ctx)
        if after is None:
            print("    lost the brick after the move; nothing to learn from this step.")
        else:
            _ex2, ey2 = pixel_error(after, frame_after.shape,
                                    ctx.aim(frame_after.shape))
            model.observe(actual_dz, actual_dr, ey2 - ey)
            print(f"    brick moved {ey2 - ey:+.0f} px vertically -> "
                  f"{model.describe()}")

        # The ruler check, after the first step only. This is the J5 question,
        # asked while the claw is still high enough for the answer to be safe.
        if step_n == 1 and not ctx.args.no_confirm:
            print("\n    *** MEASURE THE GAP UNDER THE CLAW NOW. ***")
            print(f"    It should have shrunk by about {abs(dz_mm):.0f} mm.")
            print("    If it barely changed, or moved the wrong way, the joint")
            print("    calibration is not describing reality — most likely J5,")
            print("    whose dir_sign is still an unconfirmed hypothesis. Say no.")
            if input("    Did the gap shrink as expected? [y/N] ").strip().lower() \
                    not in ("y", "yes"):
                print("\n    STOPPING. The arm is holding where it is.")
                print("    Revert J5 to +1 in data/servo_calibration.json (the old")
                print("    value is in servo_calibration.json.bak-20260804-preJ5flip)")
                print("    and re-run, or settle it with scripts/jog_joint.py --joint 5.")
                return False

        # SIDEWAYS. J1 is base yaw: it cannot change the tip's height, so it
        # cannot undo a descent step and has no business in the combined solve.
        # That reasoning is sound and it is also how this loop ran J1 away on
        # 2026-08-05, badly enough that the operator cut power.
        #
        # The gain came from a probe taken before the descent began, at a
        # completely different arm posture. If it is wrong-signed HERE, each
        # correction enlarges the error and the next one is bigger: textbook
        # positive feedback. centre() has carried a ProgressMonitor against
        # exactly this since it was written. This branch was added without one,
        # and J1 has no measured travel limits, so the servo bus could not
        # refuse it either. Nothing in the system was watching.
        #
        # Three independent bounds now, because a runaway must be stopped by
        # whichever notices first, not by the operator:
        #   1. the first correction must actually reduce the error, or stop;
        #   2. a ProgressMonitor across the whole descent;
        #   3. a hard cumulative travel budget, which holds even if the error
        #      readings themselves are lying.
        if x_actuator is None or after is None or ctx.args.no_sideways:
            continue
        ex2, _ = pixel_error(after, frame_after.shape, ctx.aim(frame_after.shape))
        if abs(ex2) <= ctx.tolerance[0]:
            continue

        remaining = ctx.args.sideways_budget - sideways_used
        if remaining <= 0:
            print(f"\n  STOPPED: the sideways budget of "
                  f"{ctx.args.sideways_budget:.0f} {x_actuator.unit} is spent and "
                  f"the brick is still {ex2:+.0f} px off sideways.")
            print(f"    That is a correction that never converged, not a near "
                  f"miss. The arm is holding.")
            return False

        # Unit-agnostic, exactly as centre() does it: step_command returns
        # whatever unit the probe measured (ticks for a joint, mm for a
        # Cartesian nudge) and the actuator's own max_step bounds it.
        delta = step_command(ex2, x_estimate, max_step=x_actuator.max_step,
                             deadband_px=ctx.tolerance[0])
        if x_actuator.kind == "joint":
            delta = enforce_minimum_step(delta, config.SERVO_VISUAL_MIN_STEP_TICKS)
        if not delta:
            continue
        delta = float(np.clip(delta, -remaining, remaining))

        print(f"    sideways: {x_actuator.label()} {delta:+.4g} "
              f"{x_actuator.unit} ({ex2:+.0f} px, "
              f"{remaining:.0f} {x_actuator.unit} of budget left)")
        x_actuator.apply(ctx, delta)
        sideways_used += abs(delta)
        wait_watching(ctx.args.settle, ctx, ["sideways correction"])

        # Did it help? This is the check whose absence caused the runaway.
        confirm, confirm_frame = detect_centroid(ctx)
        if confirm is None:
            print("    lost the brick after the sideways move; stopping.")
            return False
        ex3, _ = pixel_error(confirm, confirm_frame.shape,
                             ctx.aim(confirm_frame.shape))
        print(f"    sideways error {ex2:+.0f} -> {ex3:+.0f} px")

        if abs(ex3) > abs(ex2) + config.SERVO_VISUAL_MIN_IMPROVEMENT_PX:
            print(f"\n  STOPPED: that correction made the sideways error WORSE "
                  f"({abs(ex2):.0f} -> {abs(ex3):.0f} px).")
            print(f"    The gain for {x_actuator.label()} was measured before the "
                  f"descent, at a different posture, and no longer describes this")
            print(f"    one. Continuing would enlarge the error every step — which")
            print(f"    is how J1 ran away on 2026-08-05. The arm is holding.")
            print(f"    Re-run and let centring re-probe, or pass --no-sideways.")
            return False
        try:
            sideways_monitor.update(abs(ex3))
        except ServoAbort as e:
            print(f"\n  STOPPED (sideways): {e}")
            print("    The arm is holding.")
            return False


# --- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default="xy", choices=["x", "y", "xy"],
                    help="which image axes to close the loop on (default xy)")
    ap.add_argument("--joint-x", type=int, default=AXIS_JOINT["x"], choices=list(IK_JOINTS),
                    help=f"joint driving horizontal error in JOINT mode "
                         f"(default J{AXIS_JOINT['x']})")
    ap.add_argument("--joint-y", type=int, default=AXIS_JOINT["y"], choices=list(IK_JOINTS),
                    help=f"joint driving vertical error in JOINT mode "
                         f"(default J{AXIS_JOINT['y']})")
    ap.add_argument("--recentre", choices=["cartesian", "joint", "auto"], default="auto",
                    help="how to correct pixel error. 'cartesian' nudges the TOOL "
                         "and lets MATLAB IK pick the joints, so a joint at its "
                         "travel limit stops contributing while the others take "
                         "over. 'joint' jogs one joint per axis and stops dead "
                         "when that joint does. 'auto' (default) uses cartesian "
                         "whenever a MATLAB connection exists.")
    ap.add_argument("--descend", action="store_true",
                    help="the whole run: drive to HOVER, probe, centre, then "
                         "descend onto the brick in steps, correcting the aim "
                         "within the same move. The hover is automatic (skipped "
                         "only by --no-hover, or when already within 40 ticks of "
                         "it) -- there is no need to run goto_pose.py first")
    ap.add_argument("--descend-step", type=float, default=8.0,
                    help="millimetres of descent per step (default 8)")
    ap.add_argument("--descend-probe-mm", type=float,
                    default=config.SERVO_VISUAL_DESCEND_PROBE_MM,
                    help="radial offset added to the SECOND descent step so the "
                         "image's response to reaching out can be told apart "
                         "from its response to descending. The step still "
                         f"descends in full (default {config.SERVO_VISUAL_DESCEND_PROBE_MM:.0f})")
    ap.add_argument("--no-lock-base", dest="lock_base", action="store_false",
                    help="let the IK solver move J1 during descent steps. Off "
                         "by default: a descent does not ask for base yaw, and "
                         "the camera is on the wrist, so any yaw the solver "
                         "spends pans the image the loop is reading. Needs a "
                         "server with the 'lock' field (restart MATLAB after "
                         "pulling).")
    ap.add_argument("--no-hover", action="store_true",
                    help="do not drive to the hover pose first. The run then "
                         "starts from wherever the arm happens to be, which is "
                         "not reproducible and makes probe gains from different "
                         "runs incomparable.")
    ap.add_argument("--no-sideways", action="store_true",
                    help="do not correct sideways error during the descent. The "
                         "descent's own IK solve handles height and reach; this "
                         "turns off the separate base-yaw jog that runs "
                         "alongside it.")
    ap.add_argument("--sideways-budget", type=float,
                    default=config.SERVO_VISUAL_SIDEWAYS_BUDGET_TICKS,
                    help="total sideways travel allowed across one whole "
                         "descent, in the actuator's own units. A hard ceiling "
                         "that holds even if the pixel measurements are wrong "
                         f"(default {config.SERVO_VISUAL_SIDEWAYS_BUDGET_TICKS:.0f})")
    ap.add_argument("--descend-max-reach-mm", type=float,
                    default=config.SERVO_VISUAL_DESCEND_MAX_REACH_MM,
                    help="cap on the radial correction carried by one descent "
                         "step, mm. Bounds what a badly fitted model can ask "
                         f"for (default {config.SERVO_VISUAL_DESCEND_MAX_REACH_MM:.0f})")
    ap.add_argument("--target-z", type=float, default=None,
                    help="stop the descent at this z in mm (default table + "
                         f"{config.PICK_Z_OFFSET * 1000:.0f} mm grasp offset)")
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip the ruler check after the first descent step "
                         "(NOT recommended while J5 is unconfirmed)")
    ap.add_argument("--probe-ticks", type=int, default=config.SERVO_VISUAL_PROBE_TICKS,
                    help="size of the direction-measuring probe in JOINT mode")
    ap.add_argument("--probe-mm", type=float, default=config.SERVO_VISUAL_PROBE_MM,
                    help="size of the direction-measuring probe in CARTESIAN mode. "
                         "Shrink it if the probe itself pushes the brick out of "
                         "frame — the gain is steeper the closer the camera is.")
    ap.add_argument("--max-iterations", type=int,
                    default=config.SERVO_VISUAL_MAX_ITERATIONS)
    ap.add_argument("--settle", type=float, default=config.SERVO_VISUAL_SETTLE_S,
                    help="seconds to rest after each move before looking")
    ap.add_argument("--aim-offset-y", type=float,
                    default=config.SERVO_VISUAL_AIM_OFFSET_Y_PX,
                    help="how far BELOW the crosshair the brick should end up, "
                         "in pixels. The camera sits above and behind the claw, "
                         "so the claw is over the brick when the brick is below "
                         "the frame centre by this much. Measure it once: put "
                         f"the claw over the brick by hand and read it off the "
                         f"live view (default {config.SERVO_VISUAL_AIM_OFFSET_Y_PX:.0f})")
    ap.add_argument("--tolerance-x", type=float,
                    default=config.SERVO_VISUAL_TOLERANCE_X_PX,
                    help="half-width of the acceptance box, px "
                         f"(default {config.SERVO_VISUAL_TOLERANCE_X_PX:.0f})")
    ap.add_argument("--tolerance-y", type=float,
                    default=config.SERVO_VISUAL_TOLERANCE_Y_PX,
                    help="half-height of the acceptance box, px "
                         f"(default {config.SERVO_VISUAL_TOLERANCE_Y_PX:.0f})")
    # NOTE: there is deliberately no --deadband any more. The target is the
    # --tolerance-x/--tolerance-y box around the aim point; a second, rounder
    # notion of "close enough" alongside it only invited the two to disagree.
    ap.add_argument("--view", action="store_true",
                    help="live camera window. q or Esc stops and freezes the arm. "
                         "Use this rather than run_live_view.py in another "
                         "terminal — only one process can hold the camera.")
    ap.add_argument("--view-boards", action="store_true",
                    help="also draw ChArUco boards in the --view window (slower)")
    ap.add_argument("--limit-margin", type=int, default=config.SERVO_LIMIT_MARGIN_TICKS,
                    help="allow joints this many ticks outside their MEASURED "
                         "travel range (default 0). Justified for J2/J3, whose "
                         "limits are ground-derived at one elbow angle rather "
                         "than mechanical; not justified for a hard stop.")
    ap.add_argument("--no-wait", action="store_true",
                    help="skip the positioning phase and start immediately")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect and report only; command no motion at all")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    args = ap.parse_args()

    # Cartesian re-centring needs MATLAB. 'auto' takes it when it is available,
    # because the joint-space alternative is the one that keeps running out of
    # travel; it silently falls back rather than refusing to run without IK.
    want_ik = args.descend or args.recentre in ("cartesian", "auto")

    print("=" * 70)
    print("VISUAL SERVO — closed loop. The arm moves repeatedly without asking.")
    print("Ctrl-C freezes it in place; so does q in the --view window.")
    print("=" * 70)
    print(f"descend      {'yes, ' + str(args.descend_step) + ' mm per step, on your say-so' if args.descend else 'no'}")
    print(f"aim          crosshair {args.aim_offset_y:.0f} px ABOVE the brick "
          f"(camera sits above the claw)")
    print(f"target box   {args.tolerance_x:.0f} x {args.tolerance_y:.0f} px "
          f"around the aim point")
    report_aim_reachability(args)

    detector = LegoBrickDetector()
    detectors = None
    if args.view and args.view_boards:
        from vision_pipeline.calibration import charuco
        detectors = charuco.build_detectors()

    try:
        camera = Camera()
    except RuntimeError as e:
        print(f"\nNo camera: {e}")
        sys.exit(1)

    with camera:
        ctx = Context(None, camera, detector, args, detectors)
        if not camera.autofocus_disabled:
            print("\n  NOTE: autofocus could not be disabled. Harmless for centring")
            print("  (no intrinsics are used), but it would matter for calibration.")

        # --dry-run touches no hardware, so it keeps the original order: approve
        # the view as it stands, judge it, stop. A REAL run must reach the hover
        # pose before asking for the go-ahead -- see the comment at that point.
        if args.dry_run:
            try:
                wait_for_go(ctx)
            except ViewAborted as e:
                print(f"\n  Quit before anything moved ({e}).")
                cv2.destroyAllWindows()
                return
            centroid, frame = detect_centroid(ctx)
            if centroid is None:
                print("\n  NO BRICK DETECTED. Nothing to servo toward.")
                cv2.destroyAllWindows()
                sys.exit(1)
            ax, ay = report_starting_error(ctx, centroid, frame)
            print(f"\n--dry-run: nothing commanded. Axes to be probed: {args.axes}.")
            cv2.destroyAllWindows()
            if args.save:
                draw_aim(frame, centroid, aim_px=(ax, ay),
                         tolerance=(args.tolerance_x, args.tolerance_y))
                cv2.imwrite(args.save, frame)
                print(f"  wrote {args.save}")
            return

        # --- the arm --------------------------------------------------------
        try:
            bus = ServoBus(args.port, args.baud, limit_margin_ticks=args.limit_margin)
            if args.limit_margin:
                print(f"\n  *** TRAVEL LIMITS WIDENED BY {args.limit_margin} TICKS "
                      f"({np.degrees(args.limit_margin / 651.89):.1f} deg) AT BOTH ENDS ***")
                print("      Stand by the arm. Freeze (Ctrl-C / q) first, power cut second.")
        except Exception as e:
            print(f"\nCould not open the servo bus on {args.port}: {e}")
            sys.exit(1)

        ik = None
        if want_ik:
            try:
                ik = MatlabIKClient(config.MATLAB_SERVER_HOST, config.MATLAB_SERVER_PORT)
            except (ConnectionRefusedError, OSError):
                if args.descend or args.recentre == "cartesian":
                    print(f"\nNo MATLAB server on {config.MATLAB_SERVER_HOST}:"
                          f"{config.MATLAB_SERVER_PORT}. This run needs IK.")
                    print("Start it in MATLAB (matlab/ folder):  >> ik_fk_server")
                    sys.exit(1)
                print("\n  No MATLAB server — falling back to JOINT re-centring.")
                print("  Joint mode stops when a joint reaches its travel limit;")
                print("  start ik_fk_server for cartesian mode, which does not.")

        mode = "cartesian" if (ik is not None and args.recentre in ("cartesian", "auto")) \
            else "joint"
        if mode == "cartesian":
            actuators = {"x": CartesianActuator("tangential", args.probe_mm),
                         "y": CartesianActuator("radial", args.probe_mm)}
        else:
            actuators = {"x": JointActuator(args.joint_x, args.probe_ticks),
                         "y": JointActuator(args.joint_y, args.probe_ticks)}
        axes = [(a, actuators[a]) for a in args.axes]
        print("re-centre   " + mode
              + ("  (tool nudges; IK chooses the joints, so one at its limit "
                 "stops contributing and the rest take over)" if mode == "cartesian"
                 else "  (one joint per axis; stops when that joint does)"))
        print("axes        " + ", ".join(f"{a} via {act.label()}" for a, act in axes))

        with bus:
            ctx.bus, ctx.ik = bus, ik
            active = IK_JOINTS if (args.descend or mode == "cartesian") else \
                sorted({args.joint_x, args.joint_y})
            bus.set_motion_profile(active, config.SERVO_MOVE_SPEED_TICKS_S,
                                   config.SERVO_MOVE_ACCEL)
            print()
            for line in dir_sign_report(bus.calibration, active):
                print(line)
            print("\n  Centring does not trust the signs above — it measures the arm's")
            print("  real response. The DESCENT does depend on them.")

            diag = {j: bus.read_diagnostics(j).get("torque_enabled") for j in active}
            if [j for j, t in diag.items() if t is None]:
                print(f"\n  {[f'J{j}' for j, t in diag.items() if t is None]} DID NOT "
                      f"ANSWER. Run check_servo_health.py.")
                return
            if [j for j, t in diag.items() if t is False]:
                print(f"\n  {[f'J{j}' for j, t in diag.items() if t is False]} HAS NO "
                      f"TORQUE. Run: python scripts/servo_torque.py --enable all")
                return

            try:
                # START FROM A KNOWN POSE. The camera is eye-in-hand, so a run
                # that begins from an arbitrary posture begins with the operator
                # hand-positioning the arm until the brick appears -- and the
                # probe then measures its gains against whatever geometry that
                # happened to be. Two runs from two postures are not comparable,
                # which is part of why a gain measured before a descent stopped
                # describing the arm during it on 2026-08-05.
                if not args.no_hover:
                    if poses.at_pose(bus, "hover"):
                        print("\nAlready at the hover pose.")
                    else:
                        print("\n--- TO HOVER ---")
                        for line in poses.describe_move(bus, poses.HOVER):
                            print(line)
                        print("  This is where runs start: brick in view, same")
                        print("  geometry every time. --no-hover skips it.")
                        poses.goto(bus, "hover", label="hover",
                                   progress=live_progress(ctx, "to hover")
                                   or (lambda k, n: print(f"    hop {k}/{n}",
                                                          flush=True)))
                        wait_watching(max(ctx.args.settle, 0.4), ctx, ["at hover"])

                # Conditioning warning, judged AT THE HOVER POSE. It used to run
                # before the hover, where it measured whatever posture the last
                # run happened to leave the arm in -- a number about a pose this
                # run never visits. Cartesian re-centring degrades badly close to
                # the base axis, and the operator can still reposition the brick
                # or switch modes at the go-ahead below, which is the whole
                # reason for warning before asking.
                if mode == "cartesian" and ik is not None:
                    try:
                        _a, tip0 = tip_position(ctx)
                        radius = reach_from_axis((tip0[0], tip0[1]), yaw_axis_xy(ctx))
                        print(f"\nreach       claw is {radius * 1000:.0f} mm from the "
                              f"base axis at the hover pose")
                        if radius < config.SERVO_VISUAL_MIN_RADIUS_M:
                            print(f"\n  *** WORKING TOO CLOSE IN for good Cartesian "
                                  f"control. ***")
                            print(f"      The claw hangs ~27 mm off the arm's plane, "
                                  f"so at {radius * 1000:.0f} mm its bearing is")
                            print(f"      hypersensitive to J1 — a 5 mm nudge can cost "
                                  f"several degrees of base")
                            print(f"      yaw, and the camera rides on the wrist, so "
                                  f"that pans the whole image.")
                            print(f"      Measured: 6.6 deg of pan at 78 mm, 1.5 at "
                                  f"130 mm, 0.8 at 160 mm.")
                            print(f"      Centring still works (J1 PANS the view well "
                                  f"here even though it barely")
                            print(f"      translates the claw) — it is the GRASP that "
                                  f"suffers, since the brick has")
                            print(f"      to be somewhere the claw can actually reach. "
                                  f"Consider a hover pose")
                            print(f"      further out, or --recentre joint "
                                  f"--joint-y 3.")
                    except Exception as e:
                        print(f"  (could not check reach conditioning: {e})")

                # THE GO-AHEAD COMES AFTER THE HOVER, not before. Asking first
                # made the operator approve a view that the very next move threw
                # away: the arm started wherever the previous run left it, so the
                # brick had to be hand-framed from that posture, and then the
                # hover swung the camera somewhere else entirely. Approving the
                # pose the run will ACTUALLY start from is the only version of
                # this that means anything -- and it is what makes probe gains
                # comparable across runs, since they are then all measured from
                # one geometry.
                wait_for_go(ctx)
                centroid, frame = detect_centroid(ctx)
                if centroid is None:
                    print("\n  NO BRICK DETECTED at the go-ahead. Nothing to servo")
                    print("  toward. The arm is at the hover pose and holding, so")
                    print("  move the brick into view and re-run — or check the")
                    print("  detector with: python scripts/run_live_view.py")
                    return
                report_starting_error(ctx, centroid, frame)

                estimates = [((a, act), probe_axis(ctx, a, act)) for a, act in axes]
                centre(ctx, estimates)
                if args.descend and confirm_descend(ctx):
                    descend(ctx, estimates)
            except (KeyboardInterrupt, ViewAborted) as e:
                held = bus.freeze(IK_JOINTS)
                how = "q in the live window" if isinstance(e, ViewAborted) else "Ctrl-C"
                print(f"\n\n  *** STOPPED ({how}) — frozen at {held} ***")
                print("  Torque is still on; the arm is holding, not falling.")
            except ServoAbort as e:
                bus.freeze(IK_JOINTS)
                print(f"\n  STOPPED: {e}")
                print("  The arm is frozen where it stands, torque on.")
            except ServoSafetyError as e:
                bus.freeze(IK_JOINTS)
                print(f"\n  REFUSED by the servo bus: {e}")

        if ik is not None:
            ik.close()
        if args.view:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


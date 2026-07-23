"""
Central place for all tunable settings used across the vision pipeline.

Why this file exists: hardcoding numbers like HSV thresholds directly inside
detection logic makes them hard to find and tune. Keeping them here means you
can retune the pipeline (e.g. for a new brick color or new lighting) without
touching any detection code.
"""

# --- Camera settings -------------------------------------------------------

# Which camera OpenCV should open. 0 is usually the default/built-in webcam.
# If you have multiple cameras plugged in, try 1, 2, etc.
CAMERA_INDEX = 0

# Requested capture resolution. The camera may ignore this and use its own
# default if the exact size isn't supported.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# --- HSV color thresholds ---------------------------------------------------
#
# OpenCV images are normally stored in BGR (Blue-Green-Red) color order, but
# BGR is a poor space for "find all the red things" style thresholding
# because a color's BGR values change a lot with lighting/shadow.
#
# HSV (Hue, Saturation, Value) separates *what color it is* (Hue) from
# *how vivid* (Saturation) and *how bright* (Value) it is. That makes it much
# easier to write a threshold that finds "red" regardless of lighting changes.
#
# OpenCV's HSV ranges are: H in [0, 179], S in [0, 255], V in [0, 255]
# (note H only goes to 179, not 359, because it's stored in a single byte).
#
# These values target vivid, saturated "Lego red" plastic. Red wraps around
# 0/179 in hue, so we use two ranges (a low-end and a high-end) — the
# detector code handles combining them. Use scripts/tune_hsv.py to dial these
# in for your actual brick and lighting.
#
# Note the fairly HIGH minimum saturation (150) and value (90): these
# deliberately reject dull, desaturated reds like human skin tones. Skin is a
# low-saturation orange-red, so a high saturation floor filters it out — this
# is what stops the detector from firing on a hand in the frame. If your real
# brick isn't being picked up in dimmer light, lower the value floor (the 90s)
# first; if background skin/wood sneaks in, raise the saturation floor (150s).

HSV_LOWER_1 = (0, 150, 90)      # low-hue red range, lower bound
HSV_UPPER_1 = (8, 255, 255)     # low-hue red range, upper bound

HSV_LOWER_2 = (172, 150, 90)    # high-hue red range, lower bound (red wraps around)
HSV_UPPER_2 = (179, 255, 255)   # high-hue red range, upper bound


# --- Contour / detection filtering ------------------------------------------

# Minimum contour area (in pixels) to count as a real detection, not noise.
# Raise this if you're picking up small speckles; lower it if a real brick
# far from the camera is being filtered out. Must stay well above single-
# digit-pixel JPEG/lighting noise specks: those tiny blobs are pixel-grid-
# quantized near-rectangles almost by accident, so they can score deceptively
# high on shape_detector.score_shape below if allowed through as candidates.
# 350 was picked against real photos: it's comfortably above the ~30-300px
# noise speckles found in real sample images, and comfortably below a real
# brick's area even at the farthest distance tested (~740px).
MIN_CONTOUR_AREA = 350

# Kernel size (pixels) for morphological cleanup of the color mask. Larger
# values remove more noise but can erode small/thin brick features. Must be
# >1 to do anything at all: a 1x1 structuring element is a no-op for both
# MORPH_OPEN and MORPH_CLOSE, so a specular-highlight hole punched into the
# middle of the red mask (see SPECULAR_* below) never gets closed back up.
MORPH_KERNEL_SIZE = 5


# --- Lego stud (bump) detection ---------------------------------------------
#
# Color alone can't tell a red Lego brick apart from, say, a red plastic cup —
# both are vivid red. What makes a Lego a Lego is the grid of raised circular
# "studs" (bumps) on it. So after finding a red region, we look for those
# studs as circles using OpenCV's Hough Circle Transform, and only accept the
# region as a brick if it contains enough of them.
#
# These parameters were tuned against real photos of red bricks vs. a red cup.
# See stud_detector.py for what each one means.

# Hough Circle Transform parameters (see cv2.HoughCircles docs):
STUD_HOUGH_DP = 1.2          # inverse accumulator resolution; ~1-2 is typical
STUD_HOUGH_PARAM1 = 100      # upper Canny edge threshold used internally
STUD_HOUGH_PARAM2 = 20       # accumulator threshold: LOWER = more (but falser) circles

# A more permissive accumulator threshold used ONLY for the same small/far
# ROIs that trigger the upscale below (see STUD_ROI_REFERENCE_PX) — a
# genuinely far/small brick's stud pattern is already degraded by the time it
# reaches Hough, so it gets the benefit of the doubt there. Deliberately NOT
# a global relaxation: normal-sized candidates (a hand, a cup) keep the
# strict default above and don't inherit this leniency. Confirmed against
# real photos: loosening this globally recovers far-away studs but also
# fabricates several false "studs" on a hand at native resolution; scoping it
# to only the upscaled path avoids that regression.
STUD_HOUGH_PARAM2_UPSCALED = 12
# Stud radius and spacing are expressed as a fraction of the brick's smaller
# bounding-box dimension, so detection scales with how big the brick appears.
STUD_MIN_RADIUS_FRAC = 0.05
STUD_MAX_RADIUS_FRAC = 0.28
STUD_MIN_DIST_FRAC = 0.15

# The fractions above are clamped to a hardcoded floor inside count_studs
# (3px radius, 8px spacing) so they don't collapse to nothing on a tiny ROI —
# but that floor only makes sense at a "normal" ROI scale. Once the brick is
# far from the camera, its bounding box shrinks below that scale and the
# fixed medianBlur(5) + floors crush the stud pattern before Hough ever runs.
# Fix: if the ROI's smaller side is below this reference, upscale it (see
# stud_detector.count_studs) so the frac math + floors operate at a
# consistent effective resolution regardless of true distance.
#
# Deliberately narrow: upscaling isn't free. cv2.resize's cubic interpolation
# smooths whatever texture is in the ROI, and on an organic, non-brick
# texture (skin, cloth) that smoothing can manufacture false circular edges
# that Hough then miscounts as studs — confirmed against the real hand/cup
# regression photo, where a value of 220 upscaled its ~120px-wide hand region
# and fabricated several spurious "studs" that weren't there at native
# resolution. 90 only kicks in for ROIs meaningfully smaller than that.
STUD_ROI_REFERENCE_PX = 90     # ROI smaller-side the frac math/floors/blur assume
STUD_ROI_MAX_UPSCALE = 6.0     # cap — beyond this, interpolation can't invent
                                # stud detail the camera never captured; that's
                                # what the shape-confidence fallback is for

# color_detector.close_contour_gaps kernel sizing: a specular highlight on a
# glossy stud can dip below the HSV saturation floor, punching a hole through
# the middle of an otherwise-solid red contour, which corrupts both stud
# counting and shape scoring downstream (confirmed against real photos — a
# brick pinched between two fingers showed a visibly fragmented, notched
# contour where highlights broke up the mask). Sized as a fraction of the
# contour's own smaller dimension, so it scales with distance like the stud
# fractions above, and clamped so it can't grow large enough to merge in a
# genuinely separate blob.
STUD_REGION_CLOSE_FRAC = 0.3
STUD_REGION_CLOSE_MIN_PX = 3
STUD_REGION_CLOSE_MAX_PX = 41

# num_studs at/above which the weighted-confidence stud_score below saturates
# to 1.0. A standard 2x2 brick has 4 studs; a 2x4 has 8. Deliberately lower
# than "the whole brick's stud count" so a partial read (some studs
# glare-corrupted, or only half the brick's studs resolvable at distance)
# still earns full stud credit.
STUD_FULL_CREDIT_COUNT = 4

# --- Specular highlight suppression -----------------------------------------
#
# The raised, convex studs are glossy plastic and catch a near-white glare
# under typical lighting when facing the camera/light. That glare (a) gets
# excluded by the HSV mask's high-saturation floor above, punching a hole in
# the red blob, and (b) breaks the clean circular edge cv2.HoughCircles needs
# to find a stud, causing missed detections specifically in the orientation
# where the real studs are actually visible. Detected as "very bright, barely
# saturated" and repaired with cv2.inpaint before stud detection runs.
SPECULAR_V_MIN = 245           # HSV V floor for "blown-out glare," not just "well lit"
SPECULAR_S_MAX = 60            # HSV S ceiling for glare
SPECULAR_INPAINT_RADIUS = 3    # cv2.inpaint neighborhood, px — small, local fill


# --- Shape/geometry confidence (independent of studs) -----------------------
#
# Studs can be legitimately unresolvable (too far away, or glare-corrupted
# even after suppression above). This is a second, independent signal —
# "is this red blob shaped like a rectangular brick?" — that lets a
# detection be confirmed by shape evidence when stud evidence is weak.

# contourArea / minAreaRect area, i.e. how completely the contour fills its
# own oriented bounding rectangle. Any circle/ellipse is mathematically
# capped at pi/4 (~0.785) no matter its size or elongation — a real
# photographed rectangular brick realistically reaches ~0.85-1.0 — so this
# range is a deliberate, justified boxy-vs-round separation, not a guess.
SHAPE_RECT_SCORE_LOW = 0.85    # at/below this fill ratio -> rectangularity score 0.0
SHAPE_RECT_SCORE_HIGH = 0.95   # at/above this fill ratio -> rectangularity score 1.0

# Plausible long:short side ratio (from minAreaRect) for a Lego brick
# footprint: 1.0 covers square bricks (2x2, 4x4...), up to a generous ceiling
# covering an elongated 1x6/1x8 brick. Score decays linearly outside this band
# rather than a hard cutoff.
SHAPE_ASPECT_MIN = 1.0
SHAPE_ASPECT_MAX = 6.0

# score_shape() multiplies rectangularity by aspect-ratio plausibility rather
# than averaging/weighting them: a hand or arm silhouette can easily land
# inside a "plausible brick" aspect-ratio band by chance (confirmed against
# the real hand/cup regression photo), so aspect ratio must never contribute
# credit on its own — it can only narrow down a candidate that rectangularity
# has already judged to actually be boxy in the first place.


# --- Weighted brick confidence (studs + shape) ------------------------------
#
# Replaces the old hard "num_studs >= MIN_STUDS" gate. Studs remain the
# stronger, more specific signal when actually resolved; shape is
# corroborating evidence, especially valuable exactly when studs aren't
# resolvable (far away / glare). A candidate is accepted if the blended score
# clears DETECTION_CONFIDENCE_THRESHOLD.
STUD_WEIGHT = 0.6
SHAPE_WEIGHT = 0.4
DETECTION_CONFIDENCE_THRESHOLD = 0.30    # minimum center-to-center spacing between studs


# --- Camera intrinsics (Phase 2) -------------------------------------------
#
# Intrinsics describe the camera's own optics: focal length (fx, fy) and the
# optical center (cx, cy), all in pixels, plus lens distortion. They convert a
# pixel into a 3D ray leaving the lens. They depend ONLY on the camera + lens,
# not on where the camera is mounted, so you calibrate them ONCE (chessboard +
# cv2.calibrateCamera) and reuse forever.
#
# The values below are a rough placeholder for a 640x480 webcam with a ~60 deg
# horizontal field of view (fx ~= width / (2*tan(hfov/2))). REPLACE them with
# your real calibration before trusting world coordinates — everything
# downstream inherits this error. Path is relative to the repo root; if the
# file exists it overrides the inline numbers (see camera_model.load_intrinsics).
CAMERA_INTRINSICS_PATH = "data/camera_intrinsics.json"

# fx, fy, cx, cy in pixels. cx/cy default to the image center.
CAMERA_FX = 550.0
CAMERA_FY = 550.0
CAMERA_CX = FRAME_WIDTH / 2.0   # 320.0
CAMERA_CY = FRAME_HEIGHT / 2.0  # 240.0

# Radial/tangential lens distortion (k1, k2, p1, p2, k3), OpenCV's order.
# Zeros = "assume a perfect pinhole." Fill in from calibration if your lens
# noticeably bends straight lines near the frame edges.
CAMERA_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)


# --- Hand-eye calibration (Phase 2, eye-in-hand) ---------------------------
#
# The camera is mounted ON the arm, so a pixel only becomes a world coordinate
# once you know where the camera was when the frame was taken. That splits into
# two transforms:
#
#   T_gripper_camera : the FIXED rigid offset from the gripper (end-effector)
#                      frame to the camera frame. This is what "hand-eye
#                      calibration" solves (cv2.calibrateHandEye), ONCE, after
#                      the arm exists. Store it here / in the file below.
#   T_base_gripper   : where the gripper is in the robot's base frame at capture
#                      time. This comes from the ARM's forward kinematics and is
#                      supplied at runtime by the robot code (RobotInterface.
#                      get_end_effector_pose) — it is NOT a config value.
#
# Then T_base_camera = T_base_gripper @ T_gripper_camera, and the brick pixel is
# back-projected through that into the base frame.
#
# The placeholder below says "camera sits at the gripper origin, looking down
# the gripper's +Z with no rotation." It is almost certainly wrong for your real
# mount — measure or calibrate it. Stored as a 4x4 row-major homogeneous matrix
# in the file; if the file is absent this identity is used.
HAND_EYE_PATH = "data/hand_eye.json"


# --- Table / pick geometry (Phase 2 -> 3) ----------------------------------
#
# All in the robot BASE frame, meters. These are the numbers you tune once the
# arm is bolted down and you know where the table is relative to the base.

# Height of the table surface in the base frame. The brick sits on this plane,
# so the pixel back-projection is intersected with z = TABLE_Z_IN_BASE to place
# the brick in 3D from a single camera.
#
# ESTIMATE, not a measurement — good to maybe +/-5mm, replace it with a real one.
# Derived 2026-07-22: with the arm at home, FK put the claw tip at z = -74.2mm
# and the operator eyeballed ~10mm of clearance under it, so the tabletop is
# around -84mm. The previous 0.0 placeholder was wrong by ~84mm, which put every
# single-view back-projection far enough off that a pick would have closed on
# empty air well above the brick.
#
# To measure it properly (no camera, no calibration needed): hand-position the
# claw so it just touches the tabletop, read the servos, and run those angles
# through FK's T_tip -- the tip's z at that moment IS this value. That path
# depends on ticks_to_rad, so it is only trustworthy now that every joint's
# dir_sign is settled.
TABLE_Z_IN_BASE = -0.084

# Where the gripper should end up to grasp, relative to the table surface.
# Slightly above the table so the fingers close around the brick body rather
# than driving into the tabletop. Tune to your gripper + brick height.
PICK_Z_OFFSET = 0.010          # 10 mm above the table

# How high above the pick point to hover before descending and after lifting,
# so the arm approaches straight down and clears the table on retreat.
APPROACH_HEIGHT = 0.060        # 60 mm

# Fixed tool orientation for a top-down pick. A 5-DOF arm can't hit arbitrary
# orientations, so we point the gripper straight down (roll/pitch fixed) and let
# only yaw vary to line up with the brick. Adjust the convention to match your
# arm's kinematics — this is a merge-time agreement with the robot code.
PICK_ROLL_DEG = 180.0          # flip so the tool points down at the table
PICK_PITCH_DEG = 0.0


# --- MATLAB IK/FK server (real hardware backend) ----------------------------
#
# Connection settings for the persistent ik_fk_server.m that hosts the
# validated position-only IK solver and provides live FK. Filled in after the
# server runs (scripts/start_ik_server.m or MATLAB GUI).

MATLAB_SERVER_HOST = "localhost"
MATLAB_SERVER_PORT = 9999


# --- Feetech servo bus driver (real hardware backend) -----------------------
#
# Connection and calibration settings for the Waveshare STS3215 servo adapter
# and Feetech bus servos. Per-servo calibration (home tick, ticks-per-radian,
# motor direction) loads from the JSON file below, with fallback to these
# config values if the file is missing (following the camera_intrinsics /
# hand_eye pattern).

SERVO_PORT = "COM3"            # Windows serial port name (COMx) or /dev/ttyUSBx on Linux
SERVO_BAUD = 1000000           # Feetech STS3215 default baud rate

SERVO_CALIBRATION_PATH = "data/servo_calibration.json"

# Fallback calibration (used if the JSON file is absent). Each servo (J1..J6)
# needs a dict with keys: 'home_tick' (present position at home), 'ticks_per_rad',
# 'dir_sign' (+1 or -1 to flip direction if physical servo is inverted), and
# 'home_angle_rad' (the MATLAB joint angle the arm is in when the servo sits at
# home_tick).
# Keys are STRINGS ("1".."6") to match the JSON file format that ServoBus loads
# (JSON object keys are always strings) — so the same str(servo_id) lookup works
# whether calibration came from the file or from this fallback.
#
# WHY home_angle_rad IS ZERO, and when it would not be:
# MATLAB (ik_fk_server.m) works in ABSOLUTE joint angles, while a servo at its
# calibrated home_tick naturally reads 0. Those two zeros only coincide if the
# imported robot's HomePosition is itself 0 — and it is: importrobot bakes the
# CAD assembly pose (smiData.RevoluteJoint(n).Rz.Pos in Robomainassem_DataFile.m)
# into the LINK TRANSFORMS, not into HomePosition. Do not mistake those Rz.Pos
# values for the home configuration; feeding them in as joint angles produces a
# pose the arm is never in.
# Verified on hardware 2026-07-22: with all servos at their home ticks, FK of
# [0,0,0,0,0] puts the wrist at [91, 12, 2] mm and the claw ~90 mm in front of
# the base, matching a physical measurement. FK of the Rz.Pos angles instead
# claims [-6, -31, -187] mm — the arm hanging below its own mount.
# If the model is ever re-imported with non-zero HomePosition, set these to the
# new home angles; the mechanism is here so that stays a data change, not a
# code change.
# dir_sign for J1-J4 is NOT a placeholder — all four were reconciled
# 2026-07-22 by comparing the physical +ticks direction recorded during
# bring-up (README "Hardware bring-up log") against MATLAB's own
# positive-rotation convention:
#   J1: physical +ticks is CCW from above; MATLAB's +angle rotates about -Z
#       (CW from above, right-hand rule on the FK rotation axis) -> opposite
#       senses -> dir_sign -1.
#   J4: physical +ticks is CCW from the left side (confirmed vantage point);
#       MATLAB's +angle about +Y is also CCW from the left -> same sense ->
#       +1, unchanged.
#   J2/J3: physical descriptions ("tilts up" / "folds down toward the table")
#       don't state a viewing convention, so instead of the axis/viewpoint
#       method, these were resolved by directly comparing which way the
#       WRIST moves for a pure +angle delta (same observable a human watches
#       during a single-joint jog): MATLAB's +angle on J2 moves the wrist
#       DOWN, opposite "tilts up" -> dir_sign -1. MATLAB's +angle on J3 also
#       moves the wrist DOWN, matching "folds down toward the table" -> +1,
#       unchanged.
#   J5: the only joint needing an actual physical jog. Its FK rotation axis is
#       only 73% pure at the home pose (wrist roll's axis depends on upstream
#       joint angles, so it isn't a clean single-axis rotation in base-frame
#       terms), meaning neither method above applied confidently. Confirmed by
#       jog 2026-07-22: predicted ~7 deg CCW seen from above, operator observed
#       a match -> +1, unchanged.
# All six dir_sign values are now reconciled; none remain placeholders.
# home_tick VALUES BELOW ARE STILL PLACEHOLDERS — measure your arm and fill in
# real numbers (dir_sign and home_tick are independent facts; don't conflate
# "dir_sign is known" with "this arm's home_tick is known").
SERVO_CALIBRATION_FALLBACK = {
    "1": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": -1,
          "home_angle_rad": 0.0},                                          # J1
    "2": {"home_tick": 1365, "ticks_per_rad": 651.89, "dir_sign": -1,
          "home_angle_rad": 0.0},                                          # J2
    "3": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1,
          "home_angle_rad": 0.0},                                          # J3
    "4": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1,
          "home_angle_rad": 0.0},                                          # J4
    "5": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1,
          "home_angle_rad": 0.0},                                          # J5
    "6": {"home_tick": 2048, "ticks_per_rad": 325.95, "dir_sign": 1,
          "home_angle_rad": 0.0},                                          # J6 (gripper)
}

SERVO_READ_VERIFY_TOLERANCE_TICKS = 100  # Tolerance for read-back verification (100 ticks ~ ±2.8 deg @ J1..J5)

# --- Move safety + settling -------------------------------------------------
# A servo does NOT arrive instantly. Reading present position immediately after
# writing a goal returns the position it was still travelling through, so
# move_and_verify polls until the servo settles rather than reading once.
SERVO_MOVE_SETTLE_TIMEOUT_S = 3.0    # give up waiting for arrival after this long
SERVO_MOVE_POLL_INTERVAL_S = 0.05    # how often to re-read present position while waiting
SERVO_MOVE_STALL_POLLS = 6           # consecutive ~unchanged reads that mean "stopped moving"
SERVO_MOVE_STALL_EPSILON_TICKS = 3   # movement below this per poll counts as not moving
# Grace period before stall-counting starts. A real STS3215 has a command-
# processing / acceleration ramp-up before it visibly starts moving; without
# this, STALL_POLLS*POLL_INTERVAL_S (0.3s) alone false-triggers on that ramp,
# not a real obstruction -- confirmed on hardware 2026-07-22: an 80-tick J5
# move was reported "stalled" near its start position, but a later read-only
# check found it had fully arrived (within 1 tick) all along.
SERVO_MOVE_STALL_GRACE_S = 0.5
SERVO_MOVE_READ_RETRIES = 3          # transient serial read failures to absorb per poll

# Hard cap on how far a single commanded move may travel from the servo's CURRENT
# position. Exists because of the J1 encoder wrap-seam runaway during bring-up: a
# home tick parked near the 0/4095 seam read as 17 instead of 4086 after a power
# cycle, and a blind "return to home" tried to travel ~358 deg the long way round
# rather than the ~2 deg it actually needed. In single-turn position mode the servo
# cannot cross the seam, so a large delta is nearly always a wrap artifact or a bad
# solve — never a legitimate request. Refuse instead of executing.
# 400 ticks ~ 35 deg at J1..J5: comfortably more than any pick-sequence step,
# far less than a wrap-around.
SERVO_MAX_MOVE_DELTA_TICKS = 400

# Gripper (J6) open/closed positions, expressed as an angle offset (radians) from
# the servo's calibrated home. Converted to ticks through the same per-servo
# calibration as the arm joints. Tune to your claw's actual open/closed spread.
SERVO_GRIPPER_OPEN_RAD = 0.0    # home position = fully open
SERVO_GRIPPER_CLOSE_RAD = 0.2   # ~11.5 deg of claw rotation to close

# Total servos on the bus: J1..J5 (arm) + J6 (gripper).
NUM_JOINTS = 6


# --- Arm observation / jog web UI -------------------------------------------
#
# Settings for scripts/run_arm_ui.py, the Flask dashboard used to watch the
# live camera feed and jog individual joints during bring-up/testing. Jog
# buttons move a joint by a step size (in raw ticks) that a per-joint slider
# controls, bounded by these two.
JOG_DEFAULT_STEP_TICKS = 20   # ~0.5 deg @ J1..J5 fallback calibration
JOG_MAX_STEP_TICKS = 500

WEBUI_HOST = "127.0.0.1"
WEBUI_PORT = 5000


# --- Calibration target (ChArUco board) geometry ----------------------------
#
# Used by scripts/generate_charuco_board.py (prints it) and by
# scripts/calibrate_camera_intrinsics.py / scripts/calibrate_hand_eye.py (detect it).
# ChArUco = chessboard + a unique ArUco marker in every other square. Unlike a plain
# chessboard, every corner is individually identified by its neighboring marker IDs,
# so detection works from PARTIAL / angled views — important here because the
# eye-in-hand camera's view of the board changes constantly as the arm moves.
#
# ALWAYS regenerate the printable board from generate_charuco_board.py after
# changing anything below — it reads these same values, so "what got printed" and
# "what the detector expects" can never drift apart. Do NOT substitute a board from
# anywhere else (e.g. a random image found online): a mismatched dictionary or
# geometry doesn't just fail to detect, it can silently feed WRONG 3D coordinates
# into solvePnP/calibrateCamera and produce a confidently wrong calibration.
#
#   * SQUARES_X/Y count SQUARES (not corners — different from the old plain-
#     chessboard convention, which counted internal corners).
#   * SQUARE_SIZE_M/MARKER_SIZE_M are the real, ruler-measured edge lengths on the
#     PRINTED board (measure a run of several squares and divide — printers
#     rescale). Print at 100% / "actual size", NEVER "fit to page", then measure
#     and correct these if they don't match the nominal values.
#   * Mount the print FLAT and rigid (tape to cardboard/acrylic) — a wavy board
#     corrupts corner positions.
#   * If you build a bigger board tiled across multiple printed sheets, the sheets
#     must be assembled with sub-mm precision (any seam misalignment silently
#     corrupts the samples that touch it) — for now, one single-sheet board is
#     the simplest and safest, and ChArUco's partial-view tolerance means it
#     doesn't need to be physically large to work across many arm poses.
CALIB_ARUCO_DICT = "DICT_5X5_100"   # cv2.aruco.DICT_5X5_100 — 100 unique markers
CALIB_CHARUCO_SQUARES_X = 10        # squares across (NOT internal corners)
CALIB_CHARUCO_SQUARES_Y = 6         # squares down
CALIB_SQUARE_SIZE_M = 0.0267        # 26.7 mm — MEASURED off the actual printout (nominal was 25mm)
CALIB_MARKER_SIZE_M = 0.0192        # 19.2 mm — scaled by the same print factor (kept at 18/25 of square size)
CALIB_CHARUCO_MIN_CORNERS = 6       # min detected corners in a frame to accept it (>=4 needed for solvePnP)
CALIB_INTRINSICS_MIN_SAMPLES = 15   # min board captures before intrinsics calibration
CALIB_HAND_EYE_MIN_SAMPLES = 12     # min pose/board captures before hand-eye calibration


# --- Two-view (moving-camera stereo) depth ---------------------------------
#
# The eye-in-hand camera moves with the arm, so two shots of the same brick from
# two arm poses form a stereo pair with a known baseline — letting us triangulate
# the brick's true 3D position instead of assuming it lies flat on the table plane
# (TABLE_Z_IN_BASE). See PixelToWorldCalibrator.triangulate_pixels /
# PickPipeline.locate_brick_two_view / scripts/run_two_view_pick.py.

# How far to shift the arm between the two shots. Bigger baseline = stronger
# parallax = more stable depth, but too big risks losing the brick from frame or
# leaving the reachable workspace. Meters, base frame.
TWO_VIEW_BASELINE_M = 0.05       # 50 mm sideways shift between the two captures

# Quality gates applied to a triangulation before it is trusted (see
# TriangulationResult). A pair below the parallax floor has too little baseline for
# reliable depth; a residual above the ceiling means the rays didn't actually meet
# (bad detection / calibration / the two views saw different points).
TWO_VIEW_MIN_PARALLAX_DEG = 5.0  # reject near-parallel view pairs (unstable depth)
TWO_VIEW_MAX_RESIDUAL_M = 0.010  # reject if the rays miss each other by > 10 mm

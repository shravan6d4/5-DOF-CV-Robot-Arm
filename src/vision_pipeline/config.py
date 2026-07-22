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
# far from the camera is being filtered out.
MIN_CONTOUR_AREA = 500

# Kernel size (pixels) for morphological cleanup of the color mask. Larger
# values remove more noise but can erode small/thin brick features.
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

# Minimum number of studs (circles) a red region must contain to count as a
# Lego brick. A standard 2x4 brick has 8 studs; a 2x2 has 4. Requiring 3 gives
# margin above noise while accepting most bricks. Lower to 2 if you use small
# 1x2 bricks; raise it if smooth red objects are being misclassified.
MIN_STUDS = 2

# Hough Circle Transform parameters (see cv2.HoughCircles docs):
STUD_HOUGH_DP = 1.2          # inverse accumulator resolution; ~1-2 is typical
STUD_HOUGH_PARAM1 = 100      # upper Canny edge threshold used internally
STUD_HOUGH_PARAM2 = 20       # accumulator threshold: LOWER = more (but falser) circles
# Stud radius and spacing are expressed as a fraction of the brick's smaller
# bounding-box dimension, so detection scales with how big the brick appears.
STUD_MIN_RADIUS_FRAC = 0.05
STUD_MAX_RADIUS_FRAC = 0.28
STUD_MIN_DIST_FRAC = 0.15    # minimum center-to-center spacing between studs


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
# the brick in 3D from a single camera. Measure base-to-tabletop height.
TABLE_Z_IN_BASE = 0.0

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
# 'dir_sign' (+1 or -1 to flip direction if physical servo is inverted).
# Keys are STRINGS ("1".."6") to match the JSON file format that ServoBus loads
# (JSON object keys are always strings) — so the same str(servo_id) lookup works
# whether calibration came from the file or from this fallback.
# PLACEHOLDER VALUES — measure your arm and fill in real numbers.
SERVO_CALIBRATION_FALLBACK = {
    "1": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1},     # J1
    "2": {"home_tick": 1365, "ticks_per_rad": 651.89, "dir_sign": 1},     # J2
    "3": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1},     # J3
    "4": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1},     # J4
    "5": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1},     # J5
    "6": {"home_tick": 2048, "ticks_per_rad": 325.95, "dir_sign": 1},     # J6 (gripper)
}

SERVO_READ_VERIFY_TOLERANCE_TICKS = 100  # Tolerance for read-back verification (100 ticks ~ ±2.8 deg @ J1..J5)

# Gripper (J6) open/closed positions, expressed as an angle offset (radians) from
# the servo's calibrated home. Converted to ticks through the same per-servo
# calibration as the arm joints. Tune to your claw's actual open/closed spread.
SERVO_GRIPPER_OPEN_RAD = 0.0    # home position = fully open
SERVO_GRIPPER_CLOSE_RAD = 0.2   # ~11.5 deg of claw rotation to close


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

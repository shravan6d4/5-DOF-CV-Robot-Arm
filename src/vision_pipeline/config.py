"""
Central place for all tunable settings used across the vision pipeline.

Why this file exists: hardcoding numbers like HSV thresholds directly inside
detection logic makes them hard to find and tune. Keeping them here means you
can retune the pipeline (e.g. for a new brick color or new lighting) without
touching any detection code.
"""

# --- Camera settings -------------------------------------------------------

# Which camera OpenCV should open. This machine enumerates THREE devices and
# only one of them is the arm's eye-in-hand camera:
#     0 = "EOS Webcam Utility"  (virtual; present even with no Canon attached)
#     1 = "NexiGo N930AF"       <-- the real eye-in-hand camera
#     2 = "OBS Virtual Camera"  (virtual)
# The two virtual devices open successfully and return frames (a branded
# placeholder image), so a wrong index does NOT fail loudly — it silently
# calibrates or detects against a static graphic. If anything here behaves
# strangely, confirm the index first.
#
# Indices can shift if USB devices are re-plugged. To re-identify, open each
# index and look at the frame rather than trusting the number.
CAMERA_INDEX = 1

# Fixed focus position, applied by Camera after switching autofocus OFF.
# Autofocus and photogrammetry are incompatible: refocusing moves the lens and
# so changes the focal length, and fx/fy are precisely what turn pixels into
# millimetres. The lens must sit at ONE position for the calibrated intrinsics
# to mean anything, and at the SAME position when they are later used.
#
# 0 is infinity on the UVC scale; higher values focus nearer. If the boards or
# bricks look soft at the working distance, raise this and RE-RUN the intrinsics
# calibration — changing focus invalidates existing intrinsics.
#
# MEASURED 2026-08-05 by sweeping the lens across its range with the arm at its
# usual working height, scoring each step by Laplacian variance over the brick
# and by what the detector made of it:
#
#     focus    brick sharpness    studs found    confidence
#         0               123              0          0.10   <- old value
#        75               263              0          0.06
#       105               709              3          0.64
#       120              1156              4          0.95   <- chosen
#       135              1229              4          0.89
#       165               553              2          0.48
#       255                63              0          0.02
#
# 0 was pinning the lens at infinity while the camera works ~200 mm from the
# table, so every frame was soft. Studs are small and round and the first thing
# blur destroys, which left detection leaning on shape alone and hovering right
# at DETECTION_CONFIDENCE_THRESHOLD — it fired on about 5% of frames and looked
# like a flickering detector rather than a focus problem. 120 sits mid-plateau
# (105-150 all resolve at least 3 studs) so small changes in working height do
# not fall off it.
CAMERA_FOCUS = 120

# Frames to pull BEFORE writing CAP_PROP_FOCUS. This camera ignores the property
# until the stream is actually running, and get(CAP_PROP_FOCUS) reports 0.0
# regardless, so a set that did nothing is indistinguishable from one that
# worked. Measured 2026-08-05: focus set before the first read left frame
# sharpness at 134, set while streaming it reached 2116.
CAMERA_FOCUS_WARMUP_FRAMES = 5
# Setting CAP_PROP_FOCUS only STARTS the lens moving. These give the motor time
# to arrive and throw away the frames captured while it was still travelling —
# without them the first frames of every run are focused somewhere between the
# old position and the new one, which looks exactly like an unreliable detector.
CAMERA_FOCUS_SETTLE_S = 0.6
CAMERA_FOCUS_SETTLE_FRAMES = 5

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

# --- Glare recovery in the COLOR MASK itself --------------------------------
#
# The SPECULAR_* settings above repair glare for STUD detection, inside a region
# that was already found. This block repairs it one stage earlier, in the red
# mask, and exists because of a failure the later repair structurally cannot
# reach: a highlight bright enough to cut a brick's silhouette clean in two
# leaves fragments that individually fall under MIN_CONTOUR_AREA, so
# ColorDetector emits no candidate at all and close_contour_gaps never runs.
# Observed 2026-08-05 as detection flickering on and off frame to frame on a
# brick with strong stud glare.
#
# The rule: a bright, desaturated region is added to the red mask only if it is
# ENCLOSED by red. That enclosure test is what makes this safe on a workspace
# tiled with white ChArUco board — a white square is just as bright and just as
# desaturated as a stud highlight, and is rejected because what surrounds it is
# board, not brick. Thresholds are deliberately looser than SPECULAR_* (a
# highlight's edge is partially washed out, not fully blown), which only widens
# what is CONSIDERED; the enclosure test still decides.
GLARE_RECOVERY = True
GLARE_V_MIN = 200              # bright enough to be a highlight
GLARE_S_MAX = 120              # below the red mask's S floor (150) by construction
GLARE_MAX_REGION_PX = 2500     # a stud highlight is small; a lit wall is not
GLARE_RING_PX = 5              # how far out to look when asking "what surrounds this?"
GLARE_ENCLOSURE_FRAC = 0.55    # this much of the surroundings must be red
#
# Two refinements were tried here and REJECTED by measurement; both look
# obviously right and are not. (1) A "bridge" rule accepting any highlight that
# touches two separate red regions: it repairs a highlight lying across the
# silhouette, but it also merges the fragments of a red cup and a hand into
# brick-shaped blobs, and both sample-photo false-positive guards failed.
# (2) Clipping candidates to within a few pixels of red, to stop a highlight
# merging with the white board: it splits a wide band into two thin strips, one
# hugging each red fragment, and a strip with red on one side and glare on the
# other lands near 50% enclosure — just under the threshold, so the repair
# stopped firing at all. Neither is worth re-attempting without new evidence.


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
# PHYSICAL frame (+Z up; MatlabIKClient converts to/from the flipped model
# frame — see CLAUDE.md "COORDINATE FRAMES"). "The table" = the surface the
# robot stands on, the plane motor 1 (J1) sits on. The base origin is ~73mm
# above it, so this value is NEGATIVE.
#
# CORROBORATED 2026-07-22 (Stage D) by two clearance readings taken at poses
# 40.7mm apart vertically, which is a much stronger check than either reading
# alone — a wrong vertical scale would make them disagree:
#     pre-lift:   FK tip z = -70.7mm, operator measured ~5mm  -> table -75.7mm
#     post-lift:  FK tip z = -30.0mm, operator measured  44mm -> table -74.0mm
# They agree to 1.7mm. Equivalently: FK said the lift moved the tip 40.7mm, the
# ruler said 39mm. So the model's vertical scale (ticks_per_rad x link lengths,
# through the frame conversion) is good to ~2mm over a 40mm move, and the table
# is at about -74mm. Taking the post-lift figure: the 44mm reading was measured
# deliberately, the 5mm was an eyeball estimate, and -74.0 is the conservative
# choice of the two (it assumes the tabletop is HIGHER, so a commanded pick
# height ends up slightly above the surface rather than slightly into it).
#
# Still an estimate to maybe +/-2mm, since both inputs are ruler-to-eye gaps
# under the claw. A touch-probe (below) would remove that, but the payoff is
# now small. Re-derive whenever the arm has moved: an earlier -73mm figure came
# from a different pose (tip at -62.9mm, ~10mm gap) and the confirm jogs walked
# the claw ~8mm closer to the table in between.
#
# This value has been wrong three times; each failure mode is documented in
# CLAUDE.md's frame section (0.0 placeholder; -0.084 from assuming the tip
# hangs CLAW_LEN below the wrist IN THE MODEL FRAME; +0.053 from reading the
# model-frame tip z as if it were physical). Ask request_fk_tip for the tip —
# it returns physical coordinates — and remember the tip is BELOW the wrist
# physically in any sane pose.
#
# To measure properly (no camera, no calibration): hand-position the claw to
# just touch the tabletop, read the servos, run the angles through
# request_fk_tip — the tip's z at that instant IS this value. Depends on
# ticks_to_rad, i.e. on dir_sign being right; J2's confirm jog should happen
# first (see SERVO_CALIBRATION_FALLBACK notes).
#
# -0.0732 SINCE 2026-08-07, from SEVEN touches of the tabletop across three
# distinct postures (scripts/measure_table_plane.py). It briefly sat at -0.0677
# — the claw's height at home, on the reading that home rests ON the table — and
# that was wrong by the arm's own 6 mm gap. It cost a descent: the run drove past
# the surface until J2 stalled 34 ticks from its limit.
#
# WHAT MAKES THIS ONE DIFFERENT is that it does not depend on the dir_sign
# question the touches opened up (see CLAUDE.md). Both hypotheses land here:
#     J2/J3/J4 flipped   -> the seven touches average -73.2 mm directly
#     signs as stored    -> home's tip is -67.7 mm and sits ~6 mm proud -> -73.7
# and the 2026-07-22 ruler pair, taken 40 mm apart vertically, said -74.0 and
# -75.7. Four routes inside 2.5 mm, so this number is safe to use while the sign
# question is still open.
#
# Under the flipped hypothesis the seven touches spread only 7.6 mm about this
# value; under the stored signs they spread 73.1 mm, which is why the sign
# question matters for everything ELSE the arm computes — but not for this line.
TABLE_Z_IN_BASE = -0.0732

# Where the gripper should end up to grasp, relative to the table surface.
# Slightly above the table so the fingers close around the brick body rather
# than driving into the tabletop. Tune to your gripper + brick height.
# Hard floor for any commanded claw-tip height, as a clearance above
# TABLE_Z_IN_BASE. HardwareRobot refuses targets below it.
#
# This is the constraint that actually bounds this arm. Per-joint travel limits
# cannot express it: what stops the arm is the CLAW REACHING THE TABLE, which
# depends on every joint at once. Measured 2026-08-04, J2's usable range came
# out ~51 deg of its real travel purely because the claw grounded out at the
# elbow angle it was measured at — a different elbow angle makes the same J2
# angle safe. Recording that as a joint limit is conservative but misleading;
# the height is the real rule.
#
# Below PICK_Z_OFFSET on purpose: the grasp legitimately descends to
# PICK_Z_OFFSET, and this must not refuse the pick it exists to protect.
MIN_CLAW_HEIGHT_M = 0.005      # 5 mm above the table

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
# dir_sign reconciliation, REDONE 2026-07-22 after the operator identified that
# the imported model's frame is upside-down relative to the physical robot
# (see CLAUDE.md "COORDINATE FRAMES" — model +Z is physically DOWN, model +Y is
# physically RIGHT). An earlier pass reconciled these while interpreting model
# axes as physical ("viewed from above", etc.), which silently inverted every
# frame-based conclusion. Corrected derivations, physical frame throughout:
#   J1 = +1: CONFIRMED BY JOG 2026-07-22. Derivation agreed: MATLAB +angle
#       rotates about model -Z = physically UP -> CCW seen from above, matching
#       the bring-up log's +ticks = CCW from above. (An earlier -1 came from
#       the frame error, before the flip was understood.)
#   J2 = -1: CONFIRMED BY JOG 2026-07-22. The desk derivation said +1 (MATLAB
#       +angle moves the wrist physically UP, and the bring-up log recorded
#       +ticks = shoulder tilts up), but two independent physical jogs both
#       showed the arm moving DOWN for a predicted-up move. Physical evidence
#       beats the derivation: either the bring-up log's "tilts up" was recorded
#       from a different vantage, or the desk chain has an error we have not
#       isolated. J2 is a shoulder joint, so wrist and claw swing together and
#       the observation is unambiguous.
#   J3 = -1: MATLAB +angle moves the wrist model-down = physically UP. The
#       bring-up table-strike incident is the anchor: commanding +ticks
#       physically drove the claw DOWN into the table. Opposite senses.
#   J4 = -1: CONFIRMED BY JOG 2026-07-22, on the second attempt. The first
#       confirm jog was inconclusive-by-construction: jog_joint.py predicted
#       WRIST motion while the operator watched the CLAW, and J4 is wrist
#       pitch — the wrist sits near its own rotation axis and barely
#       translates while the tip swings ~70mm out on the lever, so the two
#       describe different directions. The script now predicts claw-tip
#       motion; the re-run matched. Derivation agreed: MATLAB +angle rotates
#       about model +Y = a physically RIGHT-pointing axis, which viewed from
#       the operator's vantage (the LEFT side) appears CW, against the
#       bring-up log's +ticks = CCW from the left.
#   J5 = +1 (nominal, direction UNCONFIRMED — MUST be resolved before IK moves).
#       The one jog "confirmation" was against a model-frame description, so
#       its physical sense is contaminated by the frame flip.
#       An earlier note here claimed J5 was position-irrelevant because "the
#       tip sits on its rotation axis". THAT WAS WRONG, and measured wrong:
#       a J5 jog moves the WRIST ~0.0mm, but the tip is CLAW_LEN (70mm) out on
#       the lever, so it swings hard — measured 12.6mm for 30 deg, 38.5mm for
#       104 deg. J5 is a major positional contributor and the IK solver uses it
#       heavily to reach lateral targets. Its sign being wrong would send the
#       claw tens of mm the wrong way.
# home_tick VALUES BELOW ARE STILL PLACEHOLDERS — measure your arm and fill in
# real numbers (dir_sign and home_tick are independent facts).
# home_tick VALUES BELOW ARE STILL PLACEHOLDERS — measure your arm and fill in
# real numbers (dir_sign and home_tick are independent facts; don't conflate
# "dir_sign is known" with "this arm's home_tick is known").
SERVO_CALIBRATION_FALLBACK = {
    "1": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": 1,
          "home_angle_rad": 0.0},                                          # J1
    "2": {"home_tick": 1365, "ticks_per_rad": 651.89, "dir_sign": -1,
          "home_angle_rad": 0.0},                                          # J2
    "3": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": -1,
          "home_angle_rad": 0.0},                                          # J3
    "4": {"home_tick": 2048, "ticks_per_rad": 651.89, "dir_sign": -1,
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
SERVO_MOVE_READ_RETRIES = 15         # transient serial read failures to absorb per poll
# Pause before each retry, DOUBLING each time. Non-zero is what makes the retry
# work at all: reads fail because motor current puts noise on the shared serial
# line, and that noise arrives in bursts. The original tight loop spent all its
# attempts inside a single burst and declared the servo dead — on 2026-08-05
# that took down a run whose J2 was holding torque perfectly and answered every
# read-only check before and after.
SERVO_READ_RETRY_BACKOFF_S = 0.02
# ...but CAPPED, or the doubling runs away. Uncapped, 15 attempts would wait
# 20ms * 2^14 on the last one — over five minutes for a single read. Capped at
# 0.2s the whole sequence spans ~2.5s, which is long enough to outlast any
# plausible noise burst and short enough to sit inside a move.
SERVO_READ_RETRY_MAX_S = 0.20

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

# Any commanded joint move at or above this many DEGREES gets a loud
# stand-by-the-power warning before it executes. Operator standing instruction
# (2026-07-22): watch the power cut whenever a move this large is about to run.
# Note the tick cap above already caps a single move at ~35 deg, so this only
# fires if that cap is deliberately raised via move_and_verify(max_delta_ticks)
# — which is exactly the case that most deserves a human watching.
SERVO_WATCH_POWER_MOVE_DEG = 45.0

# How fast a commanded joint move is allowed to run, in ticks/s (4096 ticks is a
# full turn, so 200 ~= 18 deg/s). The servos default to full speed, which during
# bring-up means a wrong move completes before anyone can react to it. Capping
# the SIZE of a step bounds where the joint stops; this bounds how fast it gets
# there, which is what actually makes an unexpected move watchable.
# SRAM on the servo: reset by every power cycle, so it is re-applied per run.
SERVO_MOVE_SPEED_TICKS_S = 200
# Ramp rate in units of 100 ticks/s^2. A low speed with maximum acceleration
# still starts with a jerk that rocks the whole arm.
SERVO_MOVE_ACCEL = 10

# Motion pacing for anything near the table. Every commanded move in the pick
# path is broken into hops of at most PICK_STEP_TICKS with PICK_STEP_PAUSE_S of
# rest between them, so the arm advances in short, watchable increments instead
# of one continuous slew. Operator-specified after a mid-move power cut dropped
# the arm and overloaded J3 (2026-08-04): the pause is what makes a wrong move
# stoppable by hand. Distinct from SERVO_MOVE_SPEED_TICKS_S, which caps how fast
# a single hop runs -- these cap how far it goes and how long the arm rests.
PICK_STEP_TICKS = 60           # ~5.3 deg per hop on J1-J5
PICK_STEP_PAUSE_S = 0.5

# Gripper (J6) open/closed positions, expressed as an angle offset (radians) from
# the servo's calibrated home. Converted to ticks through the same per-servo
# calibration as the arm joints. Tune to your claw's actual open/closed spread.
SERVO_GRIPPER_OPEN_RAD = 0.0    # home position = fully open
SERVO_GRIPPER_CLOSE_RAD = 0.2   # ~11.5 deg of claw rotation to close

# Total servos on the bus: J1..J5 (arm) + J6 (gripper).
NUM_JOINTS = 6

# How far outside a MEASURED travel range a joint may be commanded, in ticks.
# Zero by default: the limits exist because small in-range steps walked J3 and
# then J4 into hard stops on 2026-08-04, and a margin applied blindly gives that
# back. Raise it only for a range whose `limit_basis` says it is GROUND-DERIVED
# (J2, J3) — those stopped where the claw met the table at one particular elbow
# angle, so with the elbow folded differently the same joint angle is safe, and
# the recorded limit is simply too tight. Never widen a mechanical stop.
# Scripts expose this as --limit-margin so it is a per-run decision, visible in
# the command that made it, rather than a quiet change to the stored limits.
SERVO_LIMIT_MARGIN_TICKS = 0

# How many passes ServoBus.freeze makes over the joints it is stopping. A freeze
# that gives up on one dropped byte is a soft e-stop that does not stop the arm,
# and a dropped byte on a shared serial chain is ordinary. Passes, not per-joint
# retries: one unresponsive servo's timeouts must not delay freezing the joints
# after it, which are still travelling meanwhile.
SERVO_FREEZE_ATTEMPTS = 15

# Angle limits handed to MATLAB's IK solver, regenerated from the tick limits in
# servo_calibration.json. Two files for one fact, because two different systems
# enforce it: ServoBus refuses moves in ticks, matlab/init_arm.m constrains the
# solver in radians. They are always written together (see
# servo_calibration.write_angle_limits) — if they disagree, IK returns solutions
# the bus refuses and a run dies mid-move with no obvious cause.
JOINT_LIMITS_RAD_PATH = "data/joint_limits_rad.json"

# How close a recorded travel limit may sit to the 0/4095 encoder seam before
# find_joint_limits.py calls it out. A WARNING threshold, not the margin that
# gets applied: ~18 deg, several times the servo's settling error and enough
# that unpowered sag cannot walk the joint across the boundary. A limit inside
# this distance is a sign the joint's encoder needs re-centring
# (scripts/recentre_joint.py) rather than a sign the limit should be moved.
SERVO_SEAM_WARN_TICKS = 200


# --- Closed-loop visual servoing (scripts/visual_servo.py) ------------------
#
# Tunables for planning/visual_servo.py, which centres the brick in the frame by
# looking, nudging, and looking again. All of these are in PIXELS and TICKS on
# purpose: the loop deliberately never converts to millimetres, so none of the
# calibration above (intrinsics, hand-eye, TABLE_Z_IN_BASE) can affect it.
#
# The probe move the loop uses to measure which way a joint pushes the brick.
# Large enough to produce an unambiguous pixel shift, small enough to be a
# harmless move if the answer turns out to be surprising.
SERVO_VISUAL_PROBE_TICKS = 40
# Below this pixel response, a probe has measured nothing usable. Raise it if
# detection noise makes the loop confident about a gain it should not trust.
SERVO_MIN_PROBE_RESPONSE_PX = 4.0
# Fraction of the measured error to correct per iteration. Under 1.0 because the
# gain comes from a single probe: take most of the gap, re-measure, repeat.
SERVO_VISUAL_GAIN = 0.6
# Hard clamp per iteration. A safety bound on a bad estimate, not a tuning knob
# -- well under PICK_STEP_TICKS, since this moves without an operator confirming
# each step.
SERVO_VISUAL_MAX_STEP_TICKS = 35
# Close enough. At typical working distance a brick stud is tens of pixels, so
# this is comfortably tighter than the grasp needs.
SERVO_VISUAL_DEADBAND_PX = 12.0
# Runaway guard: iterations of non-improvement tolerated, and the shrink that
# counts as improvement at all.
SERVO_VISUAL_PATIENCE = 3
SERVO_VISUAL_MIN_IMPROVEMENT_PX = 2.0
# Absolute cap on iterations, so a marginal loop terminates on its own.
SERVO_VISUAL_MAX_ITERATIONS = 25
# Rest between iterations, seconds: settle the arm, then take a fresh frame.
SERVO_VISUAL_SETTLE_S = 0.6

# --- Cartesian re-centring (the joint-limit workaround) ---------------------
#
# Re-centring by jogging ONE joint has a hard ceiling: when that joint reaches
# its travel limit the correction simply stops, even though the arm as a whole
# could easily make the motion with a different posture. That is what happened
# on 2026-08-05 -- re-centring drove J2 to its stop twice, and widening the
# limit only bought a few more cycles.
#
# Cartesian mode asks for a small MOVEMENT OF THE TOOL instead, and lets
# MATLAB's IK choose the joints. matlab/init_arm.m already loads the measured
# limits into PositionLimits, so the solver will not propose a joint angle the
# arm cannot reach -- it redistributes onto the joints that still have travel,
# which is precisely "hold the joint that ran out, move the others".
#
# The probe works identically; it simply measures pixels per MILLIMETRE instead
# of pixels per tick, so the loop stays free of any camera calibration.
SERVO_VISUAL_PROBE_MM = 8.0        # test nudge for the Cartesian probe
SERVO_VISUAL_MAX_STEP_MM = 12.0    # per-iteration clamp, millimetres
SERVO_VISUAL_MIN_STEP_MM = 0.5     # below this the move is not worth commanding

# The clamp that actually matters: most of the frame's shorter side that ONE
# step may move the brick across. The tick and millimetre clamps above bound the
# COMMAND, which says nothing about how far the image moves -- that depends on
# the measured gain, and the same 12 mm nudge is gentle from far away and half a
# frame from close up. On 2026-08-05 a 12 mm step, well inside its millimetre
# clamp, put the brick outside the frame and the run died on the next look. A
# loop cannot correct what it cannot see, so the binding limit belongs in pixels.
SERVO_VISUAL_MAX_FRAME_FRACTION = 0.30

# The smallest commanded step a loaded joint will actually execute. Below this
# the servo cannot break stiction and simply does not move, so a loop that keeps
# computing ever-smaller corrections stalls while looking busy. Measured on J3
# (load 56, carrying the forearm) 2026-08-05: -35 ticks moved 28 px, -23 moved
# 9 px, -18 moved 2 px, -17 moved nothing at all. Both stalled runs that day
# ended exactly here, at corrections of 17-19 ticks.
#
# Consequences the loop must respect: a correction below this is rounded UP to
# it (overshooting and coming back beats commanding a move that does nothing),
# and the deadband can never be finer than half of it, because aiming inside
# that asks for a step the joint will ignore.
SERVO_VISUAL_MIN_STEP_TICKS = 25

# --- Where to aim, and how close is close enough -----------------------------
#
# The camera does not look down the claw's axis — it sits above and behind it —
# so "brick at the image centre" is NOT "claw over the brick". The brick should
# come to rest BELOW the crosshair by roughly the camera-to-claw offset, which
# is what these express: the aim point is the frame centre pushed DOWN by
# AIM_OFFSET_Y_PX, so a centred run leaves the crosshair sitting above the brick.
#
# Measure it once: put the claw over the brick by hand, look at the live view,
# and read off how far below the crosshair the brick sits.
#
# Set to 240 on 2026-08-05 (4x the original 60 px placeholder) from the live
# view. In a 480 px frame that puts the aim point on the BOTTOM EDGE, y = 480,
# so only the upper half of the acceptance box is on screen and the brick has to
# finish in the last ~55 rows. That is legal but tight, and it is the reason
# visual_servo.py reports how much of the box is actually visible at startup:
# an aim point the camera cannot see is a loop that can never converge, and
# nothing else about the run would look wrong.
#
# BACK TO 0 on 2026-08-07, at the operator's request, after watching a descent.
# The arm reaches FORWARD as it descends -- that is the coupling DescentModel
# exists to model -- which drives the brick DOWN the frame. Aiming at the bottom
# edge asks the loop to hold it there, so the brick has nowhere to go but out of
# view, and a brick that leaves the frame ends the run whatever else is working.
# An aim point at the centre gives it half a frame of room, and the correction
# it produces is exactly the wanted one: the loop holds the brick at the aim
# point by pulling REACH BACK, which is the "move it back / get the brick higher
# up" behaviour, arrived at through the existing reach term rather than a new
# mechanism.
#
# THE TRADE IS REAL AND THIS IS NOT THE GRASP VALUE. The camera sits above and
# behind the claw, so at offset 0 a centred brick is NOT under the claw -- it is
# short by the camera-to-claw offset, which is what 240 measured. Expect the
# claw to stop behind the brick. Restore the measured value, or pass
# --aim-offset-y, before trying to actually close on one.
SERVO_VISUAL_AIM_OFFSET_Y_PX = 0
#
# The tolerance is a BOX, and it is deliberately generous. Two reasons. The
# descent re-centres at every step, so the approach corrects itself on the way
# in and the hover does not need to be exact. And the joints cannot resolve
# better than ~30 px anyway (see SERVO_VISUAL_MIN_STEP_TICKS), so a tighter
# target only produces corrections the arm ignores.
SERVO_VISUAL_TOLERANCE_X_PX = 45
SERVO_VISUAL_TOLERANCE_Y_PX = 55

# Sideways companion to AIM_OFFSET_Y, and the same kind of quantity: the claw
# does not sit under the pixel the camera calls centre, so the aim point is the
# frame centre pushed by the camera-to-claw offset. Y covers the "camera is
# above and behind" part; this covers the sideways part.
#
# Set 2026-08-07 at the operator's request, from watching runs: THREE box
# lengths LEFT of the detection, walked out one at a time (-90, -180, -270).
# TOLERANCE_X is a HALF-width, so the box is 90 px across and one box length is
# 90 px, negative being left in image coordinates. Written as a plain pixel
# count rather than derived from TOLERANCE_X, so that widening the acceptance
# box later does not silently move the aim point with it -- those are two
# separate decisions and coupling them would hide one inside the other.
#
# THIS IS THE LAST FULL BOX LENGTH AVAILABLE. The box now spans x 5-95 of 640,
# five pixels off the left edge; -275 clips it and report_aim_reachability will
# say so. A fourth step is not available at this tolerance, and shrinking
# TOLERANCE_X to buy room would be the wrong trade -- it tightens the acceptance
# test below what the joints can resolve (~30 px, see SERVO_VISUAL_MIN_STEP_TICKS).
#
# And note what 270 px means: 42% of the frame width, for what is nominally the
# sideways camera-to-claw offset. That is large for a lens sitting next to the
# claw. If a further step is ever wanted, suspect the cause rather than the
# number -- a rotated camera mount, or the sideways axis carrying a scale error
# -- because at some point this stops being an offset and starts being a lever
# arm that no fixed pixel count can describe.
SERVO_VISUAL_AIM_OFFSET_X_PX = -270

# --- Hand-eye: the acceptance test, and why capture geometry decides it ------
#
# MEASURED WITH A RULER: the camera sits about this far from the wrist joint.
# A hand-eye solve that disagrees with this is wrong, however small its own
# residual is. On 2026-08-05 five independent solvers (TSAI, PARK, HORAUD,
# DANIILIDIS, and the separate AX=ZB robot-world formulation) all agreed on
# 80 mm, against 24 mm on the ruler -- mutual agreement between solvers is not
# evidence of correctness when they are all fed the same badly-conditioned data.
HAND_EYE_EXPECTED_OFFSET_MM = 24.0
HAND_EYE_OFFSET_TOLERANCE_MM = 25.0    # generous: the model's wrist-body origin
                                       # need not sit exactly where a ruler is
                                       # naturally placed on the bracket.
#
# MINIMUM ROTATION BETWEEN CAPTURE POSES. The translation part of a hand-eye
# solve is recovered from how the camera swings about the unknown offset, so it
# is only as observable as the rotation is large. The 2026-08-04 session used a
# median of 15.5 deg between poses, with 2-7 mm of gripper translation -- and
# produced a translation that no amount of re-solving could fix.
#
# MVTec's HALCON documentation and the surrounding literature put the working
# figure at "at least 30 degrees, better 60" for articulated arms, with >= 8
# poses and at least two non-parallel rotation axes. Our axis spread was already
# fine (90 deg); the rotation MAGNITUDE was half the minimum.
CALIB_HAND_EYE_MIN_ROTATION_DEG = 30.0
#
# ...and rotation magnitude is only half of it. Rotating repeatedly about the
# SAME axis leaves the camera offset ALONG that axis unobservable no matter how
# large the rotations or how many the samples, because (R - I) n = 0 for a
# rotation about n. hand_eye.translation_conditioning measures that directly as
# the condition number of the stacked (R_a - I); 1 is perfect.
#
# 3.0 is set just under the 3.6 measured on the 2026-08-06 capture, which passed
# every other check (median rotation 35 deg, axis spread 82 deg) while nine of
# its thirteen pose changes shared one axis to within 1 degree -- all J5 wrist
# roll. Rotation solved to 0.8 deg; position split 24 mm vs 50 mm between
# solvers. Note that AXIS SPREAD did not catch it: spread reads the widest gap
# between any two poses, so three unusual poses hide a clustered bulk.
CALIB_HAND_EYE_MAX_CONDITION = 3.0
#
# ...and a condition number is SCALE-INVARIANT, which is its blind spot: a
# capture of uniformly tiny rotations spread evenly over three axes scores a
# perfect 1.0 and determines nothing. O3, the smallest singular value of the
# same matrix, is not scale-invariant and so catches both failures at once --
# ||(R - I)v|| = 2 sin(theta/2) * |v_perp| is small when rotations are SMALL or
# when they SHARE AN AXIS. The robot-calibration literature (Sun & Hollerbach,
# ICRA 2008) settles on it as the best single predictor of pose uncertainty.
#
# 1.5 is what CALIB_HAND_EYE_MIN_SAMPLES worth of good poses produce, so the two
# gates agree instead of contradicting each other. Each pair contributes
# 2 sin(theta/2) to the two directions perpendicular to its own axis, so N pairs
# split evenly over two perpendicular axes give sigma_min ~ sqrt(N/2) *
# 2 sin(theta/2); at N = 12 pairs and theta = 35 deg that is sqrt(6) * 0.60 =
# 1.47. Below this the set is either too small, too timid, or too co-axial --
# and O3 does not care which, which is the point of using it.
#
# The 2026-08-06 capture scored 0.86 on its cleanest board (9 samples, 8 pairs).
CALIB_HAND_EYE_MIN_O3 = 1.5
#
# SCREW CONGRUENCE (Chen 1991) tolerances. AX = XB makes A and B conjugate, so
# every pose pair must agree on rotation ANGLE and on PITCH (how far the motion
# slid along its own axis) whatever X is. Both are testable with no solve, which
# makes them the only checks here that a solver cannot flatter.
#
# Angle 3 deg: the clean 2026-08-06 capture ran a 1.16 deg median with its worst
# good pair at 4.5 deg, against 33 deg and 18 deg for the two pairs touching the
# one bad sample. The gap is wide enough that the exact threshold barely matters.
# Pitch 15 mm: the arm's own open-loop positioning error is several mm and the
# board sits ~470 mm away, so honest pairs carry real millimetres of disagreement.
CALIB_HAND_EYE_CONGRUENCE_ANGLE_DEG = 3.0
CALIB_HAND_EYE_CONGRUENCE_PITCH_MM = 15.0
#
# A sample is condemned when it is inconsistent with more than this fraction of
# ALL the others, not merely with its neighbours in capture order. Congruence
# holds between any two poses, so consecutive ordering carries no meaning and
# testing only neighbours both wastes the evidence and leaves attribution
# ambiguous -- an isolated bad consecutive pair implicates both its endpoints
# equally. Over all pairs a truly bad sample disagrees with nearly everything
# (score near 1.0) while its innocent neighbour disagrees only with it.
#
# 0.5 is the majority rule: a sample the majority cannot reconcile goes. It also
# bounds what this can do -- if more than half the set is bad there is no
# majority to appeal to, which is a recapture, not a cleanup.
CALIB_HAND_EYE_MAX_DISAGREEMENT = 0.5
#
# ACCEPTANCE GATES for a finished solve. The ruler alone is NOT enough: on
# 2026-08-05 a sample file that silently mixed two calibration frames produced
# |t| = 33 mm, which sits inside 24 +/- 25 and "passed" -- while its board
# spread was 183 mm and TSAI disagreed with PARK by 170 degrees. A solve has to
# clear every one of these, because each catches a different kind of wrong:
#   board spread   -- the chain does not place the stationary board consistently
#   method spread  -- the solvers do not agree on an answer
#   TSAI vs PARK   -- rotation is not determined at all
HAND_EYE_MAX_BOARD_SPREAD_MM = 30.0
HAND_EYE_MAX_METHOD_SPREAD_MM = 20.0
HAND_EYE_MAX_TSAI_PARK_ROT_DEG = 5.0

# --- Named poses, in raw ticks ----------------------------------------------
#
# HOME is the pose matlab/init_arm.m calls home: all five joint angles zero.
# These ticks ARE the definition of home_tick in data/servo_calibration.json --
# they are repeated here only so a script can drive back to it, and the two must
# agree. FK at this pose puts the claw tip 70 mm in front of the base, dead
# centre, 6.3 mm above the table.
#
# HOVER is where a run should START. The camera is eye-in-hand, so a brick that
# is not in view cannot be detected, cannot be probed, and cannot be servoed to:
# every loop in this repo begins by measuring the brick's pixel position, and
# beginning from an arbitrary pose means the first thing the operator has to do
# is hand-position the arm until the brick appears. Driving to a known hover
# first makes a run reproducible -- the same starting geometry every time, which
# is also what makes the probe gains comparable between runs.
#
# Re-measured 2026-08-06 by the operator parking the arm where the brick sits
# comfortably in frame and reading hold_pose.py. Supersedes the 2026-08-05
# values {1: 2021, 2: 2873, 3: 2448, 4: 1749, 5: 2744}, which framed the work
# area worse.
#
# J1..J5 only, matching HOME: these are the IK joints. hold_pose also reports J6
# (the gripper, 3094 at this pose) and it is deliberately NOT included -- driving
# to a viewing pose must never open or close the claw, which could drop or crush
# whatever is already held.
SERVO_HOME_TICKS = {1: 2020, 2: 2683, 3: 3298, 4: 1547, 5: 2744}
SERVO_HOVER_TICKS = {1: 2057, 2: 3145, 3: 2205, 4: 1841, 5: 2688}
# J6 is deliberately absent: it is the gripper, not part of positioning, and
# pinning it here would make every goto_pose call quietly open or close the jaw.

# --- Descent: one solve per step, not two loops ------------------------------
#
# Descending and correcting the brick's VERTICAL position in the image are the
# same degree of freedom. Both ride on the shoulder/elbow chain, and the camera
# is on the wrist, so lowering the claw swings the view -- about 7 px per mm,
# measured 2026-08-05, which throws the brick 140 px up the frame on a 20 mm
# step against a 55 px acceptance box.
#
# Running them as separate loops made the arm fight itself: IK lowered the tip,
# a J3 jog dragged the brick back down and raised the tip by more than the step
# had gained, and the next step spent itself undoing that. Four steps produced
# 4 mm of net descent and then lost the brick off the top of the frame. So each
# step now carries its own aim correction as a RADIAL reach in the same IK
# solve, sized to cancel the swing the descent is about to cause
# (planning/visual_servo.DescentModel).
#
# The model is fitted from the descent's own motion, which needs two steps whose
# reach differs -- step 1 goes straight down, step 2 adds this offset. Both
# still descend in full, so neither is a wasted probe move.
SERVO_VISUAL_DESCEND_PROBE_MM = 8.0
#
# Cap on the radial correction one step may carry. The model is refitted every
# step and an early fit can be poor; this bounds what a bad one can ask for.
# Large enough to be useful (the brick is often tens of mm out), small enough
# that a wrong sign is one recoverable step rather than a lunge.
SERVO_VISUAL_DESCEND_MAX_REACH_MM = 25.0
#
# Total sideways (base-yaw) travel permitted across ONE whole descent.
#
# The descent corrects height and reach inside its IK solve, but sideways error
# is left to a J1 jog running alongside it -- J1 cannot change tip height, so it
# cannot undo a descent step. On 2026-08-05 that jog ran J1 away far enough that
# the operator cut power: its gain had been measured before the descent, at a
# different posture, and once wrong-signed each correction enlarged the error
# and the next one was bigger. J1 has no measured travel limits, so the servo
# bus could not refuse it either.
#
# This is the bound that does not depend on the pixel measurements being right.
# ~200 ticks is about 17 deg of base yaw -- comfortably more than any real
# sideways correction needs across a descent, and far short of a swing that
# threatens anything.
SERVO_VISUAL_SIDEWAYS_BUDGET_TICKS = 200.0

# --- Cartesian conditioning guards ------------------------------------------
#
# Close to the base axis, Cartesian control of this arm is badly conditioned.
# The claw hangs ~27 mm off the arm's own plane, so when the tip radius is small
# that offset is a large fraction of the radius and the tip's BEARING becomes
# hypersensitive to J1 — the solver then buys a few millimetres of tip motion
# with several degrees of base yaw. The camera is on the wrist, so that yaw pans
# the whole image and swamps the pixel error the loop is trying to null.
#
# Measured 2026-08-05, cost of a 5 mm radial nudge:
#     radius  78 mm -> 6.59 deg of pan      (unusable)
#     radius 130 mm -> 1.49 deg
#     radius 160 mm -> 0.79 deg             (clean)
# The fix is operational — work further out — but the loop must not quietly
# thrash when it is not.
SERVO_VISUAL_MAX_PAN_DEG = 1.5
#
# The same guard for J5, the WRIST ROLL, which spins the camera about its own
# optical axis and so ROTATES the image: a brick 100 px off centre swings
# 100·sin(θ), about 10 px at this limit, against a 55 px acceptance box.
#
# It is a conditioning guard, not a null-space guard, and the distinction cost a
# run. J5 is not free motion the solver is wasting — the claw tip sits OFF the
# roll axis, so J5 genuinely translates it, and a solve can lean on the wrist to
# reach sideways instead of using the shoulder/elbow chain. Measured 2026-08-07
# from the hover, a 9 mm radial nudge came back wanting 260 ticks (23°) of roll
# and doing 99% of its work with them. Refusing to COMMAND that motion while
# keeping the rest of the solution executes 1% of the request; the only sound
# response is to reject the whole solution and ask for less.
#
# THIS IS THE ONLY CONSTRAINT J5 HAS, so it carries the weight a travel limit
# would elsewhere. J5 spins continuously and has no stop to hit, but its lever
# arm is the SHORTEST on the arm — tip motion per +2° of each joint, measured
# from the hover:
#     J3 4.14 mm    J4 3.61 mm    J2 1.34 mm    J5 0.85 mm
# roughly a quarter of the elbow's effect. So it is a fine-adjustment joint:
# gross motion bought through J5 costs a great deal of rotation for very little
# travel, and the camera bolted to the wrist pays that rotation in full. Express
# this HERE and not as a min_tick/max_tick — the joint really can turn further,
# and faking a travel limit to encode a preference puts a lie in the data.
#
# 6° is set where the probe's own solves land (5.2° at 8 mm, 1.8° at 2 mm) so an
# ordinary nudge passes, while the 23-35° solutions that stalled the centring
# loop are shrunk until they are honest.
SERVO_VISUAL_MAX_ROLL_DEG = 6.0
# Below this tip radius, warn that Cartesian re-centring will be poor.
SERVO_VISUAL_MIN_RADIUS_M = 0.120
# A "nudge" whose solution moves some joint this far is not a nudge: it is the
# solver jumping to a different arm posture that happens to reach the same point.
# Seen at 190 mm reach with J2 pinned at its limit: a 5 mm request came back as
# J2-416, J3+366. Commanding that unsupervised would be a violent move.
SERVO_VISUAL_MAX_SOLVE_TICKS = 120


# --- Arm observation / jog web UI -------------------------------------------
#
# Settings for scripts/run_arm_ui.py, the Flask dashboard used to watch the
# live camera feed and jog individual joints during bring-up/testing. Jog
# buttons move a joint by a step size (in raw ticks) that a per-joint slider
# controls, bounded by these two.
JOG_DEFAULT_STEP_TICKS = 20   # ~0.5 deg @ J1..J5 fallback calibration
# Capped at SERVO_MAX_MOVE_DELTA_TICKS (not above it): ServoBus.move_and_verify
# refuses any single move travelling farther than that from the servo's current
# position (see the encoder-wrap-seam incident note above SERVO_MAX_MOVE_DELTA_TICKS),
# so a slider max above that cap would let the dashboard offer a step size the
# hardware always rejects.
JOG_MAX_STEP_TICKS = SERVO_MAX_MOVE_DELTA_TICKS

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
# DICT_5X5_250, not _100, purely for marker budget: a 10x6 board consumes 30
# markers and CALIB_BOARD_COUNT boards must not share any, so 6 boards need
# 180. OpenCV's predefined dictionaries are nested by prefix — the first 100
# markers of _250 ARE _100 — so board 0 renders byte-identical to a board made
# against the old dictionary, and an existing printout of it stays valid.
CALIB_ARUCO_DICT = "DICT_5X5_250"   # 250 unique markers; see charuco.check_dictionary_capacity
CALIB_CHARUCO_SQUARES_X = 10        # squares across (NOT internal corners)
CALIB_CHARUCO_SQUARES_Y = 6         # squares down

# How many DISTINCT boards exist. Each takes its own slice of the dictionary
# (board i uses IDs i*30..i*30+29) so several can be tiled in view at once and
# still be told apart. Printing one board N times instead is a silent failure
# mode, not a shortcut: duplicate IDs make a marker's board membership
# ambiguous, and the detector answers with mismatched corner/ID arrays or with
# nothing — neither of which says "you printed the wrong thing".
CALIB_BOARD_COUNT = 6

# TWO square sizes, and using the wrong one scales every distance the camera
# reports:
#   * _M is what physically exists — ruler-measured across a run of squares on
#     the actual printout. Detection and pose solving use this.
#   * _NOMINAL_M is what the generator ASKS the printer for. The printer adds
#     its own scale factor (here ~6.8%: 25mm asked, 26.7mm delivered), so
#     rendering at the measured size would apply that factor a SECOND time and
#     the next print would come out at ~28.5mm while config still claimed 26.7.
# Re-measure after any print and update _M; leave _NOMINAL_M alone unless you
# deliberately want a different physical size.
CALIB_SQUARE_SIZE_M = 0.0267        # 26.7 mm — MEASURED off the actual printout
CALIB_SQUARE_SIZE_NOMINAL_M = 0.025  # 25 mm — what the renderer targets
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

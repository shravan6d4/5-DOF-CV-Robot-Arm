"""What the arm did AFTER it stopped being able to see, and what came of it.

THE PROBLEM THIS EXISTS FOR. The single-view pick assumes the brick lies flat on
the table at `config.TABLE_Z_IN_BASE`, and descends to an ABSOLUTE height derived
from it. That assumption is what a brick standing on a book, on another brick, or
on its end breaks -- and there is no way to detect the breakage from one camera
without a hand-eye transform, which this project does not yet have that it can
trust (see CLAUDE.md).

WHAT IS AVAILABLE INSTEAD is the moment sight is lost. The camera sits above and
behind the claw, so the brick slides out of the bottom of the frame at a height
that depends on where the brick's top surface actually is -- a brick sitting
20 mm higher disappears roughly 20 mm earlier. That moment is therefore a
MEASUREMENT of the brick's height, taken by the camera, requiring no calibration
beyond FK's own differential accuracy, which is the part of FK this arm is good
at. Its absolute z is the part not to trust, and nothing here uses it: every
number below is a DIFFERENCE from the tip position at loss of sight.

So one journey records:

    where the tip was when sight was lost   (the anchor)
    every blind step after that              (how far, and which way)
    whether the claw then found anything     (did the guess work?)

and a file of past journeys answers the question the next run needs: from the
moment you lose sight, how much further down did the runs that actually GRIPPED
have to travel? That is `suggest_drop_mm`.

DELIBERATELY NOT A CALIBRATION. It does not fit a model, does not touch
servo_calibration.json, and cannot move the arm. It is a log with a median over
it, and it is honest about having no data -- `suggest_drop_mm` returns None
rather than a default, so a caller has to decide what to do with nothing rather
than being handed a confident number built from nothing.

Pure and file-scoped: no serial port, no camera, no MATLAB.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Outcomes of a journey, from the gripper's own report.
GRIPPED = "gripped"       # the jaws met something: the guess was good
AIR = "air"               # they shut on nothing: too high, or off to one side
UNKNOWN = "unknown"       # the run ended before the claw was closed


@dataclass
class Sighting:
    """One frame in which the brick was seen, and where the arm was for it.

    Everything a depth estimate could want, recorded raw so the estimating can
    be redone later against better calibration than exists today:

        tip / wrist -- FK, both of them. The tip is what the height model uses;
            the WRIST is what triangulation needs, because hand-eye was solved
            against request_fk (Body08) and chaining it onto the tip would be
            wrong by the claw's own 70 mm.
        px, area -- the brick in the image. Area is the one depth cue on this
            arm that survives the camera rotating (see HeightModel).
    """

    step: int
    tip: tuple
    wrist: list                  # 4x4, row-major, JSON-friendly
    px: tuple
    area: float

    def __post_init__(self):
        self.tip = tuple(float(v) for v in self.tip)
        self.px = tuple(float(v) for v in self.px)
        self.area = float(self.area)
        self.wrist = [[float(v) for v in row] for row in self.wrist]


@dataclass
class BlindJourney:
    """One descent's worth of "what happened after the camera stopped helping".

    Positions are metres in the base frame; travel is millimetres. Every travel
    number is a DIFFERENCE from `lost_tip`, never an absolute height -- see the
    module docstring for why that distinction is the whole point.
    """

    lost_tip: tuple                      # FK claw tip when sight was lost
    lost_error_px: Optional[tuple] = None    # aim error at that moment
    flat_on_board: bool = True           # what the operator answered
    # Was sight ACTUALLY lost, or did the descent finish with the brick still
    # visible? Until it is, lost_tip holds the descent's starting pose, so
    # drop_mm means "how far this descent went" rather than "how far past loss
    # of sight" -- two different quantities, and only the second one may inform
    # suggest_drop_mm.
    lost_sight: bool = False
    when: str = ""
    steps: list = field(default_factory=list)   # tip after each blind step
    sightings: list = field(default_factory=list)   # every frame the brick was seen in
    triangulated: list = field(default_factory=list)  # cross-check, never control
    outcome: str = UNKNOWN
    grip_tip: Optional[tuple] = None     # FK tip when the claw closed
    contact_ticks: int = -1
    commanded_past: int = 0
    note: str = ""

    def __post_init__(self):
        if not self.when:
            self.when = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.lost_tip = tuple(float(v) for v in self.lost_tip)
        self.sightings = [s if isinstance(s, Sighting) else Sighting(**s)
                          for s in self.sightings]

    # --- recording ----------------------------------------------------------

    def see(self, step: int, tip, wrist, px, area) -> None:
        """Record one frame in which the brick was visible.

        Called on EVERY descent step, flat or raised. A journey object exists
        from the start of a descent now rather than only from loss of sight, so
        that the flat runs -- which are the ones that work -- contribute their
        data too. Nothing about a flat run's behaviour reads any of this.
        """
        self.sightings.append(Sighting(step, tip, wrist, px, area))

    def step(self, tip) -> None:
        """Note where the tip ended up after one blind step."""
        self.steps.append(tuple(float(v) for v in tip))

    def finish(self, outcome: str, tip=None, contact_ticks: int = -1,
               commanded_past: int = 0, note: str = "") -> None:
        self.outcome = outcome
        if tip is not None:
            self.grip_tip = tuple(float(v) for v in tip)
        self.contact_ticks = int(contact_ticks)
        self.commanded_past = int(commanded_past)
        self.note = note

    # --- what it measured ---------------------------------------------------

    @property
    def end_tip(self):
        """Where the tip finished: at the grip if there was one, else the last
        blind step, else where sight was lost (a journey with no steps)."""
        if self.grip_tip is not None:
            return self.grip_tip
        return self.steps[-1] if self.steps else self.lost_tip

    @property
    def drop_mm(self) -> float:
        """How far DOWN the tip went after sight was lost. Positive is down."""
        return (self.lost_tip[2] - self.end_tip[2]) * 1000.0

    @property
    def travel_mm(self) -> float:
        """Straight-line distance travelled after sight was lost.

        Reported alongside the drop rather than instead of it, because the two
        disagreeing is the interesting case: a blind descent asks for the same
        x and y with a lower z, so travel much larger than drop means the arm
        went sideways while it believed it was going straight down.
        """
        dx = (self.end_tip[0] - self.lost_tip[0]) * 1000.0
        dy = (self.end_tip[1] - self.lost_tip[1]) * 1000.0
        dz = (self.end_tip[2] - self.lost_tip[2]) * 1000.0
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    @property
    def sideways_mm(self) -> float:
        """Horizontal wander during the blind part. Should be ~0 by design."""
        dx = (self.end_tip[0] - self.lost_tip[0]) * 1000.0
        dy = (self.end_tip[1] - self.lost_tip[1]) * 1000.0
        return (dx * dx + dy * dy) ** 0.5

    def describe(self) -> str:
        parts = [f"lost sight at z {self.lost_tip[2] * 1000:+.1f} mm",
                 f"dropped {self.drop_mm:.1f} mm over {len(self.steps)} blind "
                 f"step{'s' if len(self.steps) != 1 else ''}"]
        if self.sideways_mm > 1.0:
            parts.append(f"wandered {self.sideways_mm:.1f} mm sideways")
        parts.append({GRIPPED: "GRIPPED",
                      AIR: "shut on air",
                      UNKNOWN: "outcome unknown"}[self.outcome])
        return "; ".join(parts)


# --- the file ---------------------------------------------------------------

def load(path) -> list:
    """Every journey recorded so far, oldest first. Missing file = no history.

    A corrupt or unreadable file is treated as no history, not as an error: this
    is an advisory log, and refusing to run a pick because a JSON file is
    malformed would be the tail wagging the dog.
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
    except Exception as e:                                        # noqa: BLE001
        logger.warning(f"could not read {p}: {e}; treating as no history")
        return []
    out = []
    for entry in raw.get("journeys", []):
        try:
            out.append(BlindJourney(**entry))
        except Exception as e:                                    # noqa: BLE001
            logger.warning(f"skipping unreadable journey in {p}: {e}")
    return out


def append(path, journey: BlindJourney) -> None:
    """Add one journey to the file, creating it if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load(p)
    existing.append(journey)
    p.write_text(json.dumps(
        {"journeys": [asdict(j) for j in existing]}, indent=2))


def suggest_drop_mm(journeys, flat_on_board: Optional[bool] = None):
    """How far past loss-of-sight the runs that GRIPPED had to travel.

    The median, not the mean: a run that missed and then had its blind descent
    refused by the floor guard contributes a short drop, and a run where the
    brick had been knocked over contributes a long one. Both are real records
    worth keeping and neither should drag the estimate.

    Only GRIPPED journeys count. A journey that shut on air measured where the
    brick ISN'T, which is useful to a human reading the log and actively
    misleading as an input to this.

    Returns None when there is nothing to go on. Deliberately not a default:
    a caller that has to handle None will say out loud that it is guessing,
    whereas one handed a number will not.
    """
    usable = [j for j in journeys if j.outcome == GRIPPED and j.lost_sight]
    if flat_on_board is not None:
        usable = [j for j in usable if j.flat_on_board == flat_on_board]
    if not usable:
        return None
    return statistics.median(j.drop_mm for j in usable)


# --- how far is there left to go? -------------------------------------------

class HeightModel:
    """Remaining drop, estimated from how big the brick looks. No hand-eye.

    WHY APPARENT SIZE AND NOT THE DESCENT MODEL. The obvious candidate is
    planning.visual_servo.DescentModel's `a`, px of image motion per mm of
    descent, which looks like an inverse-depth signal -- closer brick, more
    pixels per mm. It is not. Measured on hardware 2026-08-07, `a` FELL from
    4.72 to 3.08 px/mm as the claw came down on the brick; parallax would have
    made it grow. It is dominated by the camera ROTATING as the wrist swings,
    which is depth-independent, and separating the two terms needs the camera's
    rotation rate -- which is hand-eye, which is the thing being avoided.

    Apparent AREA has none of that trouble. It is invariant to camera rotation
    about any axis, needs no hand-eye and no intrinsics (the focal length folds
    into the fitted constant), and for a target of fixed physical size

        sqrt(area)  ~  1 / Z          =>      Z  ~  c / sqrt(area)

    so the drop still to come is affine in 1/sqrt(area):

        remaining_mm  =  c / sqrt(area)  -  d

    THE GROUND TRUTH IS THE GRIP. Each journey that gripped knows where the claw
    finally closed, so every sighting in it yields a training pair: what the
    brick looked like then, and how much drop actually remained. That pairing
    costs nothing -- the run was happening anyway -- and it is the only
    measurement in this project that has never been wrong about height.

    FIT ACROSS RUNS, not within one. There is no useful fit from a single
    descent's worth of sightings early on, and the relationship is a property of
    the camera and the brick rather than of one run. Refitted every run from the
    whole log, the same way DescentModel refits every step.

    Not trusted blindly: `ready` requires a minimum number of pairs spanning a
    real range of apparent size, because a fit over sightings that all look the
    same size is a fit to noise with a confident-looking slope.
    """

    MIN_PAIRS = 8
    MIN_SPREAD = 0.25       # fractional range of 1/sqrt(area) the pairs must span

    def __init__(self, c: float = None, d: float = None, n: int = 0,
                 rms_mm: float = float("nan")):
        self.c, self.d, self.n, self.rms_mm = c, d, n, rms_mm

    @property
    def ready(self) -> bool:
        return self.c is not None and self.d is not None

    def remaining_mm(self, area: float):
        """Drop still to come, or None if the model cannot say."""
        if not self.ready or area <= 0:
            return None
        return self.c / (area ** 0.5) - self.d

    def describe(self) -> str:
        if not self.ready:
            return "height model: not fitted"
        return (f"height model: remaining = {self.c:.0f}/sqrt(area) - {self.d:.1f} "
                f"mm  ({self.n} pairs, rms {self.rms_mm:.1f} mm)")


def training_pairs(journeys, flat_on_board=None):
    """(1/sqrt(area), remaining_mm) from every sighting of every gripped run."""
    pairs = []
    for j in journeys:
        if j.outcome != GRIPPED or j.grip_tip is None:
            continue
        if flat_on_board is not None and j.flat_on_board != flat_on_board:
            continue
        for s in j.sightings:
            if s.area <= 0:
                continue
            pairs.append((1.0 / (s.area ** 0.5),
                          (s.tip[2] - j.grip_tip[2]) * 1000.0))
    return pairs


def fit_height_model(journeys, flat_on_board=None) -> HeightModel:
    """Least-squares fit of remaining_mm = c*(1/sqrt(area)) - d.

    Returns an unfitted model rather than raising when there is not enough to go
    on -- the caller then falls back to the loss-of-sight drop, which is what it
    did before this existed.
    """
    pairs = training_pairs(journeys, flat_on_board)
    if len(pairs) < HeightModel.MIN_PAIRS:
        return HeightModel(n=len(pairs))

    xs = [p[0] for p in pairs]
    lo, hi = min(xs), max(xs)
    if hi <= 0 or (hi - lo) / hi < HeightModel.MIN_SPREAD:
        # Every sighting looked the same size, so the slope is unconstrained.
        # A fit here would be noise wearing a confident face.
        return HeightModel(n=len(pairs))

    n = len(pairs)
    sx = sum(xs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in pairs)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-18:
        return HeightModel(n=n)
    c = (n * sxy - sx * sy) / denom
    intercept = (sy - c * sx) / n
    rms = (sum((c * x + intercept - y) ** 2 for x, y in pairs) / n) ** 0.5
    return HeightModel(c=c, d=-intercept, n=n, rms_mm=rms)


# --- the cross-check ---------------------------------------------------------

def triangulate_sightings(journey, calibrator, min_step_gap: int = 3,
                          min_parallax_deg: float = None,
                          max_residual_m: float = None) -> list:
    """Two-view depth from the descent's own frames. NEVER feeds control.

    THE SAME ENGINE PickPipeline.locate_brick_two_view USES -- both end at
    PixelToWorldCalibrator.triangulate_pixels -- and the same acceptance gates,
    taken from config so the two cannot drift into disagreeing about what
    "trustworthy" means. What this does NOT do is go through
    locate_brick_two_view itself, for two reasons:

      * it re-runs the DETECTOR on stored frames, and storing frames would put
        megabytes of image into a JSON log. Detection already happened live.
      * IT TAKES `Pose`, NOT A 4x4. Routing an FK matrix through Pose means
        matrix -> RPY -> matrix, and geometry.transform_to_pose forces roll = 0
        near pitch = +-90 deg -- which is exactly where a top-down tool sits.
        That is the gimbal-lock bug that poisoned calibrate_hand_eye.py, and
        corrupting the pose would corrupt the very thing being cross-checked.
        triangulate_pixels takes the 4x4 directly.

    THE DESCENT IS ALREADY A STEREO RIG. It takes a frame at every step and
    knows the arm's pose for each, so a baseline is free -- but only between
    steps far enough apart. At ~200 mm from the brick, adjacent 8 mm steps give
    2.3 deg of parallax, under the 5 deg gate; three steps apart is 24 mm and
    6.8 deg, which passes. Hence min_step_gap rather than consecutive pairs.

    WHY THIS IS A CROSS-CHECK AND NOT THE ANSWER. Triangulation needs the
    camera's base-frame pose, which is FK @ hand-eye, and data/hand_eye.json is
    known wrong -- 52 mm and 91 deg out. So its answer is recorded beside the
    grip height and never acted on. That is deliberately the same arrangement
    that caught the hand-eye failure in the first place: five solvers agreeing
    with each other meant nothing until a ruler disagreed with all of them.

    What makes this different from a ruler is that it accumulates. Every
    successful grip is a ground-truth pixel-to-base-frame correspondence
    generated free by a run that was happening anyway, so the residual below is
    the held-out measurement hand-eye has never had.

    Returns a list of dicts, JSON-safe, one per usable pair.
    """
    from vision_pipeline import config
    if min_parallax_deg is None:
        min_parallax_deg = config.TWO_VIEW_MIN_PARALLAX_DEG
    if max_residual_m is None:
        max_residual_m = config.TWO_VIEW_MAX_RESIDUAL_M

    out = []
    sightings = list(journey.sightings)
    for i in range(len(sightings)):
        for k in range(i + min_step_gap, len(sightings)):
            a, b = sightings[i], sightings[k]
            try:
                import numpy as np
                result = calibrator.triangulate_pixels([
                    (a.px, np.array(a.wrist, dtype=float)),
                    (b.px, np.array(b.wrist, dtype=float)),
                ])
            except Exception as e:                                # noqa: BLE001
                logger.debug(f"triangulation failed for {i}/{k}: {e}")
                continue
            if result is None:
                continue
            entry = {
                "steps": [a.step, b.step],
                "z_mm": float(result.point_base[2]) * 1000.0,
                "parallax_deg": float(result.parallax_deg),
                "residual_mm": float(result.residual_m) * 1000.0,
                "accepted": bool(result.parallax_deg >= min_parallax_deg
                                 and result.residual_m <= max_residual_m),
            }
            if journey.grip_tip is not None:
                entry["vs_grip_mm"] = entry["z_mm"] - journey.grip_tip[2] * 1000.0
            out.append(entry)
    return out


def triangulation_verdict(journeys) -> list:
    """Does triangulation agree with where the claw actually touched?

    The whole point of logging it. Reports over every accepted pair in the log,
    against the one number that cannot be argued with.
    """
    errs = [e["vs_grip_mm"] for j in journeys for e in j.triangulated
            if e.get("accepted") and "vs_grip_mm" in e]
    if not errs:
        return ["triangulation: no accepted pairs with a grip to check against"]
    med = statistics.median(errs)
    spread = max(errs) - min(errs)
    lines = [f"triangulation vs grip: median {med:+.1f} mm over {len(errs)} "
             f"pair{'s' if len(errs) != 1 else ''}, spread {spread:.1f} mm"]
    if abs(med) < 10.0 and spread < 30.0:
        lines.append("  -> agrees with the grip. hand_eye.json may be usable for "
                     "height after all; check with scripts/hand_eye_report.py "
                     "before promoting it.")
    else:
        lines.append("  -> DISAGREES. Consistent with data/hand_eye.json being "
                     "wrong (52 mm / 91 deg). This is the held-out measurement "
                     "the hand-eye solve never had.")
    return lines


def summarise(journeys) -> list:
    """A few lines about the history, for printing at the start of a run."""
    if not journeys:
        return ["no blind-travel history yet"]
    gripped = [j for j in journeys if j.outcome == GRIPPED]
    sightings = sum(len(j.sightings) for j in journeys)
    lines = [f"{len(journeys)} journey{'s' if len(journeys) != 1 else ''} "
             f"recorded, {len(gripped)} of them gripped, {sightings} sightings"]
    for flat, label in ((True, "flat on the board"), (False, "raised")):
        drop = suggest_drop_mm(journeys, flat_on_board=flat)
        n = len([j for j in gripped if j.flat_on_board == flat and j.lost_sight])
        if drop is not None:
            lines.append(f"  {label}: {drop:.1f} mm below loss of sight "
                         f"(median of {n})")
    lines.append("  " + fit_height_model(journeys).describe())
    lines.extend("  " + l for l in triangulation_verdict(journeys))
    return lines

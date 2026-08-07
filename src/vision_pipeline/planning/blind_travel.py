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
class BlindJourney:
    """One descent's worth of "what happened after the camera stopped helping".

    Positions are metres in the base frame; travel is millimetres. Every travel
    number is a DIFFERENCE from `lost_tip`, never an absolute height -- see the
    module docstring for why that distinction is the whole point.
    """

    lost_tip: tuple                      # FK claw tip when sight was lost
    lost_error_px: Optional[tuple] = None    # aim error at that moment
    flat_on_board: bool = True           # what the operator answered
    when: str = ""
    steps: list = field(default_factory=list)   # tip after each blind step
    outcome: str = UNKNOWN
    grip_tip: Optional[tuple] = None     # FK tip when the claw closed
    contact_ticks: int = -1
    commanded_past: int = 0
    note: str = ""

    def __post_init__(self):
        if not self.when:
            self.when = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.lost_tip = tuple(float(v) for v in self.lost_tip)

    # --- recording ----------------------------------------------------------

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
    usable = [j for j in journeys if j.outcome == GRIPPED]
    if flat_on_board is not None:
        usable = [j for j in usable if j.flat_on_board == flat_on_board]
    if not usable:
        return None
    return statistics.median(j.drop_mm for j in usable)


def summarise(journeys) -> list:
    """A few lines about the history, for printing at the start of a run."""
    if not journeys:
        return ["no blind-travel history yet"]
    gripped = [j for j in journeys if j.outcome == GRIPPED]
    lines = [f"{len(journeys)} journey{'s' if len(journeys) != 1 else ''} "
             f"recorded, {len(gripped)} of them gripped"]
    for flat, label in ((True, "flat on the board"), (False, "raised")):
        drop = suggest_drop_mm(journeys, flat_on_board=flat)
        n = len([j for j in gripped if j.flat_on_board == flat])
        if drop is not None:
            lines.append(f"  {label}: {drop:.1f} mm below loss of sight "
                         f"(median of {n})")
    return lines

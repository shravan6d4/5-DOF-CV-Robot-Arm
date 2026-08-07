"""Stop the arm where it stands, right now, without dropping it.

Run this from a SECOND terminal while something else is driving the arm.

The power cut is the only stop that always works, but it drops holding torque on
every joint at once and the arm falls -- which is how J3 was overloaded on
2026-08-04. This is the gentler stop: it overwrites each joint's Goal Position
with where that joint already is, so motion halts while torque stays on.

Use the power cut if this does not visibly stop the arm within a second, or if
the bus is unresponsive. This needs a working serial link; the power cut does not.

    python scripts/freeze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_pipeline import config
from vision_pipeline.robot_interface.servo_driver import ServoBus

def main() -> None:
    bus = ServoBus(config.SERVO_PORT, config.SERVO_BAUD)
    with bus:
        held = bus.freeze(range(1, 7))

    if held:
        print("FROZEN — holding at "
              + ", ".join(f"J{j}={p}" for j, p in sorted(held.items())))
    missing = [j for j in range(1, 7) if j not in held]
    if missing:
        print("COULD NOT FREEZE: " + ", ".join(f"J{j}" for j in missing))
        print("CUT POWER if the arm is still moving.")


# BEHIND A MAIN GUARD, and it did not used to be: this script's body ran at
# module level, so merely IMPORTING it opened the serial port and commanded all
# six joints. Anything that imports scripts -- a test collector, an editor's
# autocomplete, a lint pass, a human checking whether the file parses -- would
# have driven the arm as a side effect. Discovered 2026-08-07 by an import check
# that froze the arm mid-session.
#
# Harmless in THIS script's case, since a freeze writes each joint's present
# position and travels nowhere. That is exactly why it went unnoticed, and why
# the guard belongs here anyway: the next script written to this pattern will
# not be a no-op.
if __name__ == "__main__":
    main()

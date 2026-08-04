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

bus = ServoBus(config.SERVO_PORT, config.SERVO_BAUD)
with bus:
    held = bus.freeze(range(1, 7))

if held:
    print("FROZEN — holding at " + ", ".join(f"J{j}={p}" for j, p in sorted(held.items())))
missing = [j for j in range(1, 7) if j not in held]
if missing:
    print("COULD NOT FREEZE: " + ", ".join(f"J{j}" for j in missing))
    print("CUT POWER if the arm is still moving.")

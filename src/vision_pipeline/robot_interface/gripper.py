"""Closing the claw, one operator-approved jog at a time.

Shared by scripts/close_claw.py (standalone) and scripts/visual_servo.py (the
prompt at the end of a descent), so the two cannot drift apart. The descent is
where this actually gets used -- a run that ends with the claw at grasp height
and nothing holding the brick has done all the work and none of the point -- but
it must also exist standalone, because the recovery case is "the descent ended
badly and I want to open the claw", which by definition cannot go through the
descent.

WHY APPROVAL PER JOG rather than one confirmation up front. The gripper is the
only joint whose job is to stall: it stops when it meets the brick, and where
that happens is not known in advance -- it depends on where the brick actually
is, which is precisely what the vision chain is still bad at. So the operator is
the sensor. Each jog is small enough (config.SERVO_GRIPPER_JOG_TICKS) that the
gap between "not touching" and "gripping" spans several of them.

A STALL WHILE CLOSING IS SUCCESS, and it is the one place this differs from
every other motion primitive in the repo. In goto_tick.py a joint that stops
short is obstructed and the script backs off. Here the obstruction is the brick.

Positions are in config (SERVO_GRIPPER_*_TICKS), measured on hardware
2026-08-07. Ticks DECREASE as the claw closes.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from vision_pipeline import config

GRIPPER_JOINT = 6

# Two reads this close together mean the servo has stopped.
SETTLE_TOL_TICKS = 2

# A jog that advanced less than this did not really move.
STALL_TICKS = 3

# Outcomes. Strings rather than an enum so a caller can print one directly.
REACHED = "reached"       # arrived at the requested position
GRIPPED = "gripped"       # stalled on the way there: the brick is in the jaws
DECLINED = "declined"     # the operator said no to a jog
REFUSED = "refused"       # the bus or this module rejected the move
FAILED = "failed"         # the bus errored


@dataclass
class GripResult:
    outcome: str
    ticks: int              # where J6 ended up
    message: str

    @property
    def holding(self) -> bool:
        """Is something plausibly in the jaws?

        GRIPPED only. REACHED means the claw arrived at the commanded position
        without meeting anything, which for a close is the *miss* case -- the
        jaws shut on air. Distinguishing them is the whole value of watching for
        the stall, so do not let a caller treat "no error" as "got it".
        """
        return self.outcome == GRIPPED


def describe(ticks: int) -> str:
    """Name a J6 position against the three measured landmarks."""
    if ticks >= config.SERVO_GRIPPER_OPEN_TICKS - 10:
        return "fully open"
    if ticks <= config.SERVO_GRIPPER_FULL_CLOSE_TICKS + 10:
        return "FULLY CLOSED (jaws touching)"
    if abs(ticks - config.SERVO_GRIPPER_GRIP_TICKS) <= 10:
        return "at the grip position"
    span = config.SERVO_GRIPPER_OPEN_TICKS - config.SERVO_GRIPPER_FULL_CLOSE_TICKS
    pct = 100.0 * (ticks - config.SERVO_GRIPPER_FULL_CLOSE_TICKS) / span
    return f"{pct:.0f}% open"


def refusal(target: int) -> Optional[str]:
    """Why this target may not be commanded, or None if it may.

    Duplicates J6's travel limits on purpose. The bus enforces them and would
    refuse anyway, but generically and only once the run is already several
    approved jogs in -- and the full-close stop is the reason this module
    exists, so it deserves to be named in its own words, before anything moves.
    """
    if target < config.SERVO_GRIPPER_FULL_CLOSE_TICKS:
        return (f"Target {target} is past the full-close stop at "
                f"{config.SERVO_GRIPPER_FULL_CLOSE_TICKS} -- the jaws shut on "
                f"themselves. With a brick in the way the servo would stall "
                f"against it.")
    if target > config.SERVO_GRIPPER_OPEN_TICKS:
        return (f"Target {target} is past the open end of travel at "
                f"{config.SERVO_GRIPPER_OPEN_TICKS}.")
    return None


def settled(bus, timeout_s: float = 3.0, sleep=None) -> int:
    """Where J6 is once it has actually stopped moving.

    move_and_verify's return value is unreliable mid-travel -- its stall check
    can trip during the acceleration ramp -- and here that matters more than
    usual, because an early read looks exactly like the claw having gripped.
    """
    if sleep is None:
        import time as _time
        sleep, now = _time.sleep, _time.monotonic
    else:
        # Injected clock for tests: they poll a scripted sequence, so the only
        # thing needed from "time" is a bound on the number of reads.
        now = lambda: 0.0                                        # noqa: E731

    previous = None
    deadline = now() + timeout_s
    while True:
        current = bus.read_position(GRIPPER_JOINT)
        if previous is not None and abs(current - previous) <= SETTLE_TOL_TICKS:
            return current
        previous = current
        if now() >= deadline:
            return current
        sleep(0.15)


def close_in_jogs(bus, target: int, step: int, approve: Callable[[str], bool],
                  say: Callable[[str], None], settle=settled) -> GripResult:
    """Walk J6 to `target` in `step`-tick jogs, asking `approve` before each.

    J6 ONLY. Whatever pose the arm is holding, it keeps.

    Args:
        bus: a ServoBus (read_position, move_and_verify, set_motion_profile).
        target: goal tick. Refused outside J6's measured travel.
        step: ticks per jog.
        approve: callable(prompt) -> bool, asked before every jog.
        say: callable(line) -> None for progress.
        settle: injectable for tests.

    Returns:
        GripResult. Check `.holding`, not just the absence of an error: a close
        that REACHED its target met nothing on the way.
    """
    from vision_pipeline.robot_interface.servo_driver import ServoSafetyError

    why = refusal(target)
    if why is not None:
        return GripResult(REFUSED, -1, why)

    step = max(1, min(abs(int(step)), config.SERVO_MAX_MOVE_DELTA_TICKS))

    try:
        current = bus.read_position(GRIPPER_JOINT)
    except Exception as e:                                        # noqa: BLE001
        return GripResult(FAILED, -1, f"could not read J6: {e}")

    # J6 has its own speed profile to set: the arm joints' profile is applied to
    # J1..J5 at the start of a run and the servos default to full speed
    # otherwise, so an unconfigured gripper would snap shut at whatever rate it
    # likes -- on a brick, at the end of a descent.
    try:
        bus.set_motion_profile([GRIPPER_JOINT], config.SERVO_MOVE_SPEED_TICKS_S,
                               config.SERVO_MOVE_ACCEL)
    except Exception:                                             # noqa: BLE001
        say("      (could not set J6's speed profile; it will move at its default)")

    closing = target < current
    if current == target:
        return GripResult(REACHED, current,
                          f"already at {current} ({describe(current)})")

    while current != target:
        remaining = target - current
        nxt = current + max(-step, min(step, remaining))

        if not approve(f"  jog {current} -> {nxt}  ({target - nxt:+d} left)  [Enter/n] "):
            return GripResult(DECLINED, current,
                              f"stopped by the operator at {current} "
                              f"({describe(current)})")

        try:
            bus.move_and_verify(GRIPPER_JOINT, nxt)
        except ServoSafetyError as e:
            return GripResult(REFUSED, current, f"refused by the servo bus: {e}")
        except Exception as e:                                    # noqa: BLE001
            return GripResult(FAILED, current, f"move failed: {e}")

        arrived = settle(bus)
        moved = arrived - current
        current = arrived

        if abs(moved) < STALL_TICKS:
            if closing:
                return GripResult(
                    GRIPPED, current,
                    f"THE CLAW HAS GRIPPED at {current}. It stopped "
                    f"{target - current:+d} ticks short of the target, which "
                    f"means it met the brick first.")
            return GripResult(
                REACHED, current,
                f"stopped at {current} without reaching {target}: that is the "
                f"open end of travel, or something is holding the jaws.")

        say(f"  at {current}  ({describe(current)}, {target - current:+d} to go)")

    return GripResult(REACHED, current,
                      f"at {current} ({describe(current)}), commanded position "
                      f"reached without meeting anything")

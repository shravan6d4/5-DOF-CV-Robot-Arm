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
    contact_ticks: int = -1     # where the jaws first met the object
    commanded_past: int = 0     # goal error accumulated beyond contact

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


def open_fully(bus, say: Callable[[str], None], step: int = None,
               settle=None) -> GripResult:
    """Drive the claw to the fully-open position. No approval, no questions.

    Run at the hover, before a descent: the claw has to be open before it can
    close on anything, and a run that arrives at grasp height with the jaws
    already shut has spent the whole descent unable to do the one thing it came
    for. Doing it at the TOP is also the safe place -- the jaws swing open
    ~490 ticks, and there is nothing near them up there.

    Unconditionally safe to call: if the claw is already open this is a no-op.
    """
    return close_in_jogs(
        bus, config.SERVO_GRIPPER_OPEN_TICKS,
        config.SERVO_GRIPPER_JOG_TICKS if step is None else step,
        lambda _prompt: True, say,
        **({} if settle is None else {"settle": settle}))


def auto_close(bus, say: Callable[[str], None], settle=None) -> "GripResult":
    """Close on whatever is there, detecting contact from the motion itself.

    NO APPROVALS. The claw is at grasp height with the brick between the jaws;
    asking per jog was right while nobody had ever driven J6 and is now just
    friction. What replaces the operator's eye is the servo's own position
    read-back, which is a better sensor for this one question anyway.

    TWO PHASES, and they are asking different things of the same measurement:

      SEARCH -- close in SERVO_GRIPPER_AUTO_CLOSE_TICKS steps. Each step should
        deliver its full travel while the jaws are moving through air. A step
        that delivers less than SERVO_GRIPPER_CONTACT_TICKS has met something.

      FIRM UP -- from contact, squeeze in SERVO_GRIPPER_SQUEEZE_TICKS steps.
        The first squeezes often still move a little as the jaws seat on the
        brick and it settles between them; when SERVO_GRIPPER_FIRM_STEPS in a
        row deliver almost nothing, the grip is loaded rather than merely
        touching.

    THE SEARCH FLOOR IS THE MEASURED GRIP POSITION, not the full-close stop.
    Contact should happen at or before 3003 for the brick this was measured on,
    so reaching it with the jaws still moving freely means there is nothing
    between them -- and closing further on nothing is the case the operator
    named as "it should never be this much". So that is reported as a MISS and
    the claw stops there, rather than continuing to shut on air.

    Returns a GripResult: GRIPPED with contact_ticks set, or REACHED for a miss.
    """
    from vision_pipeline.robot_interface.servo_driver import ServoSafetyError

    if settle is None:
        settle = settled

    try:
        current = bus.read_position(GRIPPER_JOINT)
    except Exception as e:                                        # noqa: BLE001
        return GripResult(FAILED, -1, f"could not read J6: {e}")

    try:
        bus.set_motion_profile([GRIPPER_JOINT], config.SERVO_MOVE_SPEED_TICKS_S,
                               config.SERVO_MOVE_ACCEL)
    except Exception:                                             # noqa: BLE001
        say("    (could not set J6's speed profile; it will move at its default)")

    floor = config.SERVO_GRIPPER_GRIP_TICKS
    step = config.SERVO_GRIPPER_AUTO_CLOSE_TICKS
    contact = None

    say(f"    closing from {current} in {step}-tick steps; contact is a step "
        f"that moves less than {config.SERVO_GRIPPER_CONTACT_TICKS}")

    while contact is None and current > floor:
        target = max(current - step, floor)
        try:
            bus.move_and_verify(GRIPPER_JOINT, target)
        except ServoSafetyError as e:
            return GripResult(REFUSED, current, f"refused by the servo bus: {e}")
        except Exception as e:                                    # noqa: BLE001
            return GripResult(FAILED, current, f"move failed: {e}")

        arrived = settle(bus)
        moved = abs(arrived - current)
        asked = current - target
        current = arrived

        if moved < min(config.SERVO_GRIPPER_CONTACT_TICKS, asked):
            contact = current
            say(f"    {current}: asked {asked}, moved {moved} -- CONTACT")
        else:
            say(f"    {current}: asked {asked}, moved {moved}")

    if contact is None:
        return GripResult(
            REACHED, current,
            f"closed to {current} ({describe(current)}) without meeting "
            f"anything. The jaws are shutting on AIR -- the claw is not where "
            f"the brick is. Not closing further: past here is the full-close "
            f"stop, which is not a position to drive to with nothing in the "
            f"jaws.")

    # FIRM UP.
    past = 0
    quiet = 0
    while quiet < config.SERVO_GRIPPER_FIRM_STEPS:
        squeeze = squeeze_once(bus, contact, past,
                               config.SERVO_GRIPPER_SQUEEZE_TICKS, settle=settle)
        past = squeeze.commanded_past
        if squeeze.refused:
            say(f"    {squeeze.message}")
            break
        say(f"    {squeeze.message}")
        quiet = quiet + 1 if abs(squeeze.moved) < STALL_TICKS else 0

    final = bus.read_position(GRIPPER_JOINT)
    result = GripResult(
        GRIPPED, final,
        f"GRIPPED. Met the brick at {contact}, then loaded onto it for "
        f"{past} ticks; J6 is holding at {final} ({describe(final)}).")
    result.contact_ticks = contact
    result.commanded_past = past
    return result


@dataclass
class SqueezeResult:
    ticks: int              # where J6 ended up
    moved: int              # how far it actually travelled this squeeze
    commanded_past: int     # cumulative ticks commanded beyond the contact point
    message: str
    refused: bool = False


def squeeze_once(bus, contact_ticks: int, commanded_past: int, step: int,
                 settle=None) -> SqueezeResult:
    """One more closing jog from wherever the claw is. Floors at full close.

    A SQUEEZE THAT DOES NOT MOVE IS THE POINT, and that is what makes this a
    different function rather than another call to close_in_jogs. There, no
    motion means the claw met the brick and the loop stops. Here the claw has
    ALREADY met the brick, and commanding further shut is how a Feetech servo is
    asked to hold harder -- position error is what it converts into torque. So
    "moved 0" is a normal outcome and not a stop condition.

    What bounds it is therefore not motion but the accumulated position error,
    tracked as `commanded_past`: how far beyond the contact point the goal has
    been pushed. Past config.SERVO_GRIPPER_MAX_SQUEEZE_TICKS the servo is being
    asked for a lot of torque against a stationary load, which is the condition
    that overloaded J3 on 2026-08-04 and put the checksum storm on the bus
    during the 2026-08-07 descent. The caller is told; the floor is what
    actually refuses.

    Args:
        contact_ticks: where the claw first stalled -- the brick's surface.
        commanded_past: ticks already commanded beyond that, from prior calls.
        step: ticks to squeeze by.

    Returns:
        SqueezeResult. `refused` means nothing was commanded.
    """
    from vision_pipeline.robot_interface.servo_driver import ServoSafetyError

    if settle is None:
        settle = settled

    try:
        current = bus.read_position(GRIPPER_JOINT)
    except Exception as e:                                        # noqa: BLE001
        return SqueezeResult(-1, 0, commanded_past,
                             f"could not read J6: {e}", refused=True)

    floor = config.SERVO_GRIPPER_FULL_CLOSE_TICKS
    target = max(current - abs(int(step)), floor)
    if target >= current:
        return SqueezeResult(
            current, 0, commanded_past,
            f"J6 is at {current} and the full-close stop is {floor}. There is "
            f"nothing left to squeeze -- the jaws are shut on themselves.",
            refused=True)

    try:
        bus.move_and_verify(GRIPPER_JOINT, target)
    except ServoSafetyError as e:
        return SqueezeResult(current, 0, commanded_past,
                             f"refused by the servo bus: {e}", refused=True)
    except Exception as e:                                        # noqa: BLE001
        return SqueezeResult(current, 0, commanded_past,
                             f"move failed: {e}", refused=True)

    arrived = settle(bus)
    moved = arrived - current
    past = commanded_past + (current - target)

    if abs(moved) < STALL_TICKS:
        note = (f"commanded {current - target} ticks further, moved {abs(moved)} "
                f"-- the claw is loading against the brick, not closing on it")
    else:
        note = (f"closed {abs(moved)} more ticks to {arrived} -- it was still "
                f"finding the brick")

    if past > config.SERVO_GRIPPER_MAX_SQUEEZE_TICKS:
        note += (f"\n      *** {past} ticks past first contact, over the "
                 f"{config.SERVO_GRIPPER_MAX_SQUEEZE_TICKS}-tick advisory. The "
                 f"servo is holding a lot of\n          torque against a "
                 f"stationary load; that is what overloaded J3 on 2026-08-04.")

    return SqueezeResult(arrived, moved, past, note)


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

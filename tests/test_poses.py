"""Named poses: the tick tables, the move, and their agreement with calibration.

The load-bearing test here is the FIRST one. HOME is not an arbitrary pose -- it
is the physical pose that matlab/init_arm.m calls home, where all five joint
angles are zero. That makes config.SERVO_HOME_TICKS and each joint's home_tick
in servo_calibration.json two copies of the same measurement, and if they ever
drift apart, every angle the arm reports is silently offset.

That is not hypothetical. The stored home_tick was wrong by up to 268 ticks
(23.6 deg on J2) until 2026-08-05, and nothing caught it: an offset in home_tick
cancels out of any differential measurement, which is what the Stage D lift
validation was, and it does not affect dir_sign or ticks_per_rad at all. What
eventually caught it was that FK put the claw tip 1.5 mm BELOW the table.

No serial port and no arm -- the tick tables and the pose logic are pure.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vision_pipeline import config  # noqa: E402
from vision_pipeline.robot_interface import poses  # noqa: E402
from vision_pipeline.robot_interface.servo_calibration import (  # noqa: E402
    load_calibration_file,
    ticks_to_rad,
)

CAL = load_calibration_file(ROOT / "data" / "servo_calibration.json")
IK_JOINTS = (1, 2, 3, 4, 5)


class PoseBus:
    """Minimal bus: reports positions, records stepped moves. No serial port."""

    def __init__(self, start):
        self.ticks = dict(start)
        self.stepped = []

    def read_position_retrying(self, servo_id):
        return self.ticks[servo_id]

    def _cal(self, servo_id):
        return CAL[str(servo_id)]

    def move_joints_stepped(self, targets, **kw):
        self.stepped.append(dict(targets))
        self.ticks.update({j: int(t) for j, t in targets.items()})


# --- the invariant that was broken --------------------------------------------

def test_home_ticks_are_exactly_the_calibrations_home_tick():
    """Two copies of one measurement. If they disagree, FK is lying."""
    for joint, tick in config.SERVO_HOME_TICKS.items():
        assert int(CAL[str(joint)]["home_tick"]) == tick, f"J{joint}"


def test_the_home_pose_reads_as_zero_on_every_joint():
    """The definition of home, stated as a test.

    MATLAB's init_arm.m sets homeAngles = zeros(1,6) and HomePosition = 0, so
    the physical home pose must convert to 0.0 rad. Any other value means the
    Python and MATLAB frames disagree by exactly that much.
    """
    for joint, tick in config.SERVO_HOME_TICKS.items():
        assert ticks_to_rad(CAL, joint, tick) == pytest.approx(0.0, abs=1e-12)


def test_both_named_poses_are_inside_every_measured_travel_limit():
    """A limit that forbids a named pose is a bug in one of the two.

    J3's max_tick was 3161 while home sits at 3298 -- the arm was not legally
    allowed to return to where it starts.
    """
    for name, pose in poses.POSES.items():
        for joint, tick in pose.items():
            entry = CAL[str(joint)]
            if "min_tick" not in entry:
                continue
            lo, hi = entry["min_tick"], entry["max_tick"]
            assert lo <= tick <= hi, f"{name} J{joint}={tick} outside [{lo}, {hi}]"


def test_hover_is_a_different_pose_from_home():
    assert poses.HOVER != poses.HOME


def test_hover_lifts_the_arm_off_the_table():
    """Home puts the claw ~6 mm above the table; hover has to be clear of it.

    Checked through the elbow, whose angle is what raises the tip: hover must
    fold J3 well away from the home pose rather than merely differ from it.
    """
    home_j3 = ticks_to_rad(CAL, 3, poses.HOME[3])
    hover_j3 = ticks_to_rad(CAL, 3, poses.HOVER[3])
    assert abs(hover_j3 - home_j3) > 1.0, "hover barely moves the elbow"


# --- driving to a pose --------------------------------------------------------

def test_goto_commands_every_joint_in_the_pose():
    bus = PoseBus({j: 2000 for j in IK_JOINTS})
    final = poses.goto(bus, "hover")
    assert final == {j: poses.HOVER[j] for j in sorted(poses.HOVER)}
    assert bus.stepped == [poses.HOVER]


def test_goto_accepts_an_explicit_tick_dict():
    bus = PoseBus({j: 2000 for j in IK_JOINTS})
    poses.goto(bus, {3: 2500})
    assert bus.ticks[3] == 2500
    assert bus.ticks[2] == 2000, "joints not named must not be commanded"


def test_goto_refuses_an_unknown_pose_name():
    bus = PoseBus({j: 2000 for j in IK_JOINTS})
    with pytest.raises(KeyError):
        poses.goto(bus, "somewhere")


def test_at_pose_is_true_only_when_actually_there():
    assert poses.at_pose(PoseBus(dict(poses.HOVER)), "hover")
    assert not poses.at_pose(PoseBus(dict(poses.HOME)), "hover")


def test_at_pose_tolerates_servo_settling_error():
    """Servos land a few ticks short under load; that is arrival, not a miss."""
    near = {j: t + 12 for j, t in poses.HOVER.items()}
    assert poses.at_pose(PoseBus(near), "hover")
    far = {j: t + 200 for j, t in poses.HOVER.items()}
    assert not poses.at_pose(PoseBus(far), "hover")


def test_describe_move_reads_only():
    bus = PoseBus({j: 2000 for j in IK_JOINTS})
    poses.describe_move(bus, poses.HOVER)
    assert bus.stepped == [], "a preview must command nothing"
    assert bus.ticks == {j: 2000 for j in IK_JOINTS}


def test_describe_move_warns_when_a_joint_swings_past_the_watch_threshold():
    bus = PoseBus({j: 2000 for j in IK_JOINTS})
    far = dict(poses.HOVER)
    far[2] = 2000 + int(config.SERVO_WATCH_POWER_MOVE_DEG * 2 * 651.89 * 3.14159 / 180)
    assert any("power cut" in line for line in poses.describe_move(bus, far))


def test_describe_move_stays_quiet_for_a_small_move():
    bus = PoseBus(dict(poses.HOVER))
    nudge = dict(poses.HOVER)
    nudge[3] += 20
    assert not any("power cut" in line for line in poses.describe_move(bus, nudge))


# --- driving the script itself ----------------------------------------------
#
# These call goto_pose.main() end to end against a fake bus. Three scripts this
# session shipped a wrong CALL SIGNATURE to the hardware -- step_ticks(max_step=)
# where it takes max_step_ticks, JointActuator(actuator) where it takes a joint
# number, and set_motion_profile() with no arguments where it takes three.
# Python cannot see any of those until the line runs, and unit-testing the
# helpers around them does not run them. Only invoking main() does.

class ScriptBus(PoseBus):
    """PoseBus plus the rest of the surface a script touches."""

    def __init__(self, start):
        super().__init__(start)
        self.profile = None
        self.frozen = None

    def set_motion_profile(self, servo_ids, ticks_per_sec, accel):
        self.profile = (sorted(servo_ids), ticks_per_sec, accel)

    def freeze(self, servo_ids):
        self.frozen = sorted(servo_ids)
        return {j: self.ticks[j] for j in servo_ids}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_script(monkeypatch, bus, argv):
    """Invoke goto_pose.main() with a fake bus and a fake command line."""
    import goto_pose

    monkeypatch.setattr(goto_pose, "ServoBus", lambda *a, **k: bus)
    monkeypatch.setattr(sys, "argv", ["goto_pose.py", *argv])
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    goto_pose.main()
    return bus


def test_goto_pose_script_runs_end_to_end(monkeypatch):
    """The regression: main() must survive being called."""
    bus = run_script(monkeypatch, ScriptBus({j: 2000 for j in IK_JOINTS}),
                     ["--pose", "hover", "--yes"])
    assert bus.stepped, "the move never happened"
    assert bus.ticks == {j: poses.HOVER[j] for j in IK_JOINTS}


def test_the_script_applies_a_motion_profile_before_moving(monkeypatch):
    """Servo speed limits live in SRAM and reset on every power cycle, so a
    script that forgets them runs the arm at full default speed."""
    bus = run_script(monkeypatch, ScriptBus({j: 2000 for j in IK_JOINTS}),
                     ["--pose", "hover", "--yes"])
    assert bus.profile is not None
    ids, speed, accel = bus.profile
    assert ids == sorted(poses.HOVER)
    assert speed == config.SERVO_MOVE_SPEED_TICKS_S
    assert accel == config.SERVO_MOVE_ACCEL


def test_dry_run_commands_nothing(monkeypatch):
    bus = run_script(monkeypatch, ScriptBus({j: 2000 for j in IK_JOINTS}),
                     ["--pose", "home", "--dry-run"])
    assert bus.stepped == []
    assert bus.ticks == {j: 2000 for j in IK_JOINTS}


def test_the_script_skips_a_move_it_is_already_at(monkeypatch):
    bus = run_script(monkeypatch, ScriptBus(dict(poses.HOVER)),
                     ["--pose", "hover", "--yes"])
    assert bus.stepped == []


def test_declining_the_prompt_commands_nothing(monkeypatch):
    import goto_pose
    bus = ScriptBus({j: 2000 for j in IK_JOINTS})
    monkeypatch.setattr(goto_pose, "ServoBus", lambda *a, **k: bus)
    monkeypatch.setattr(sys, "argv", ["goto_pose.py", "--pose", "hover"])
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    goto_pose.main()
    assert bus.stepped == []


def test_both_named_poses_are_reachable_by_the_script(monkeypatch):
    """Every pose the CLI offers must actually drive, not just 'hover'."""
    for name in sorted(poses.POSES):
        bus = run_script(monkeypatch, ScriptBus({j: 2000 for j in IK_JOINTS}),
                         ["--pose", name, "--yes"])
        assert bus.ticks == {j: poses.POSES[name][j] for j in poses.POSES[name]}


# --- step size is a caller's choice -----------------------------------------
# The move was always broken into hops; nothing said so, and a preview showing
# "92.1 deg" on one joint reads as a single lunge. Smaller hops mean more points
# at which the travel limits are re-checked and more moments at which Ctrl-C can
# freeze the arm partway -- which is what an operator standing over a big
# reconfiguration actually wants control of.

class _RecordingBus:
    def __init__(self, start):
        self.ticks = dict(start)
        self.stepped = []

    def read_position_retrying(self, j):
        return self.ticks[j]

    def move_joints_stepped(self, targets, step_ticks, pause_s, progress=None):
        self.stepped.append({"targets": dict(targets), "step_ticks": step_ticks,
                             "pause_s": pause_s})
        self.ticks.update({j: int(t) for j, t in targets.items()})


def test_goto_defaults_to_the_configured_step_size():
    bus = _RecordingBus({j: 2000 for j in range(1, 6)})
    poses.goto(bus, "home")
    assert bus.stepped[0]["step_ticks"] == config.PICK_STEP_TICKS
    assert bus.stepped[0]["pause_s"] == config.PICK_STEP_PAUSE_S


def test_goto_honours_an_explicit_step_size_and_pause():
    bus = _RecordingBus({j: 2000 for j in range(1, 6)})
    poses.goto(bus, "home", step_ticks=15, pause_s=1.25)
    assert bus.stepped[0]["step_ticks"] == 15
    assert bus.stepped[0]["pause_s"] == 1.25


def test_a_smaller_step_does_not_change_where_the_arm_ends_up():
    """Step size is about how the journey is made, never about the destination."""
    coarse = _RecordingBus({j: 2000 for j in range(1, 6)})
    fine = _RecordingBus({j: 2000 for j in range(1, 6)})
    poses.goto(coarse, "home", step_ticks=60)
    poses.goto(fine, "home", step_ticks=5)
    assert coarse.ticks == fine.ticks == {j: int(t) for j, t in poses.HOME.items()}

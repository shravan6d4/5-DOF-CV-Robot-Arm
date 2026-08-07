"""Extract the imported model's kinematic structure and compare it to the arm.

    python scripts/audit_model_axes.py

Read-only. Talks to the running MATLAB FK server (localhost:9999) and commands
no motion -- it only asks "if joint k moved, where would the wrist go".

WHY THIS EXISTS. Hand-eye calibration failed for two weeks with one signature:
FK and the camera agreed on rotation MAGNITUDE (0.4 deg) but not on rotation
AXIS. Joint angles and ticks_per_rad were repeatedly confirmed correct. That
leaves the model's geometry, which nothing in the repo had ever measured.

HOW. Rotating joint k by theta moves the wrist by

    M(theta) = T_wrist(theta) @ inv(T_wrist(0))
             = A @ Rot_k(theta) @ inv(A)

where A is the chain up to joint k. That is a pure rotation about the axis
A@z_hat through the point A@origin -- so M's rotation axis IS the joint axis
in base coordinates, and its translation gives a point on it:

    t = (I - R) p     ->     least-squares for p (perpendicular component)

No assumption about link lengths, joint order, or the frame flip. It reads the
model's own geometry back out of the only interface we have to it.

READ THE FRAME-INDEPENDENT SECTION FIRST. Angles between axes and the
perpendicular distances between them do not depend on where the model put its
origin or which way its axes point -- they are the arm's actual shape, and a
ruler can check every one of them. Anything expressed as a coordinate is
relative to the CAD origin, which is arbitrary: this model's base yaw axis
does NOT pass through its origin, so "70 mm forward of the origin" is not the
same statement as "70 mm in front of the base column".
"""

import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from vision_pipeline import config

HOST, PORT = "127.0.0.1", 9999

# The model frame is upside-down relative to the physical arm: a 180 deg
# rotation about the shared X axis. See CLAUDE.md "COORDINATE FRAMES".
F_PHYS = np.diag([1.0, -1.0, -1.0, 1.0])

# Servos are wired J1..J6 = bus IDs 1..6, counting up the arm from the base.
# J5 is a wrist ROLL, confirmed physically 2026-08-06 -- init_arm.m's rangeDeg
# comment calls it "wrist pitch", which is a mislabel, not a geometry error.
JOINT_ROLE = {1: "base yaw", 2: "shoulder", 3: "elbow",
              4: "wrist pitch", 5: "wrist roll"}


class Server:
    def __init__(self):
        self.sock = socket.create_connection((HOST, PORT), timeout=30.0)
        self.f = self.sock.makefile("rwb")

    def fk(self, angles):
        req = {"cmd": "fk", "angles_rad": [float(a) for a in angles]}
        self.f.write((json.dumps(req) + "\n").encode())
        self.f.flush()
        resp = json.loads(self.f.readline().decode())
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error"))
        return (np.array(resp["T"]).reshape(4, 4),
                np.array(resp["T_tip"]).reshape(4, 4))

    def close(self):
        self.sock.close()


def rotation_axis(R):
    """Unit axis and angle of a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    w, v = np.linalg.eig(R)
    axis = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    axis /= np.linalg.norm(axis)
    # Sign it so a positive joint angle gives a positive rotation.
    skew = (R - R.T) / 2.0
    sense = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    if np.dot(sense, axis) < 0:
        axis = -axis
    return axis, angle


def axis_line(M):
    """(direction, point-on-axis, angle) of a rigid motion."""
    R, t = M[:3, :3], M[:3, 3]
    n, angle = rotation_axis(R)
    # t = (I - R) p is rank 2; lstsq gives the minimum-norm p, which is the
    # point on the axis closest to the origin (the component along n is free).
    p, *_ = np.linalg.lstsq(np.eye(3) - R, t, rcond=None)
    return n, p - np.dot(p, n) * n, angle


def axis_distance(n1, p1, n2, p2):
    """Perpendicular distance between two axis LINES, and whether they meet."""
    c = np.cross(n1, n2)
    nc = np.linalg.norm(c)
    if nc < 1e-6:            # parallel: distance is the perpendicular offset
        d = (p2 - p1) - np.dot(p2 - p1, n1) * n1
        return float(np.linalg.norm(d)), "parallel"
    return float(abs(np.dot(p2 - p1, c / nc))), "skew"


def screw_axes(srv, base_angles, delta=np.deg2rad(4.0), physical=True):
    out = {}
    T0_w, T0_t = srv.fk(base_angles)
    if physical:
        T0_w, T0_t = F_PHYS @ T0_w, F_PHYS @ T0_t
    for k in range(5):
        a = list(base_angles)
        a[k] += delta
        T1_w, _ = srv.fk(a)
        if physical:
            T1_w = F_PHYS @ T1_w
        n, p, _ = axis_line(T1_w @ np.linalg.inv(T0_w))
        out[k + 1] = (n, p)
    return out, T0_w, T0_t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-frame", action="store_true",
                    help="report in the MATLAB model frame instead of physical")
    args = ap.parse_args()

    phys = not args.model_frame
    srv = Server()
    try:
        ax, T_w, T_t = screw_axes(srv, [0.0] * 5, physical=phys)
        wrist, tip = T_w[:3, 3] * 1000, T_t[:3, 3] * 1000

        print(f"\nFrame: {'PHYSICAL (X fwd, Y left, Z up)' if phys else 'MODEL (raw)'}")
        print("All angles below are at the all-zero home configuration.\n")

        print("=" * 78)
        print("A.  FRAME-INDEPENDENT STRUCTURE  — a ruler can check every number here")
        print("=" * 78)

        print("\nAngles between consecutive joint axes:")
        print("  (yaw-pitch and pitch-roll junctions are 90 deg; a shoulder,")
        print("   elbow and forearm that fold in one plane are 0 deg apart)")
        expect = {(1, 2): 90, (2, 3): 0, (3, 4): 0, (4, 5): 90}
        for (a, b), want in expect.items():
            d = np.degrees(np.arccos(np.clip(abs(np.dot(ax[a][0], ax[b][0])), 0, 1)))
            got = d if want else d
            ok = "OK" if abs(got - want) < 10 else "*** MISMATCH"
            print(f"  J{a} ({JOINT_ROLE[a]:9}) vs J{b} ({JOINT_ROLE[b]:9}): "
                  f"{got:5.1f} deg   expect {want:3d}   {ok}")

        print("\nPerpendicular distance between consecutive axes = LINK LENGTHS.")
        print("These are the numbers to measure on the physical arm.")
        for a, b in ((1, 2), (2, 3), (3, 4), (4, 5)):
            d, kind = axis_distance(*ax[a], *ax[b])
            print(f"  J{a} -> J{b}: {d * 1000:7.1f} mm   ({kind})")
        print(f"  J5 -> claw tip: {np.linalg.norm(tip - wrist):7.1f} mm   "
              f"(CLAW_LEN, set by hand in init_arm.m)")

        # Is J5 a wrist ROLL (axis along the forearm) or a wrist PITCH
        # (perpendicular to it)? Both are 90 deg from J4, so the J4-J5 angle
        # above cannot tell them apart -- this can.
        fore = ax[5][1] - ax[4][1]
        fore = fore - np.dot(fore, ax[4][0]) * ax[4][0]
        if np.linalg.norm(fore) > 1e-4:
            fore /= np.linalg.norm(fore)
            d = np.degrees(np.arccos(np.clip(abs(np.dot(ax[5][0], fore)), 0, 1)))
            print(f"\nJ5 axis vs the forearm direction: {d:.1f} deg  -> "
                  f"{'ROLL (spins the claw)' if d < 45 else 'PITCH (tilts the claw)'}")

        print("\n" + "=" * 78)
        print("B.  WHERE THE MODEL PUTS THINGS  — relative to the BASE YAW AXIS,")
        print("    not the CAD origin (which is arbitrary and is NOT the column)")
        print("=" * 78)
        n1, p1 = ax[1]
        print(f"\nThe CAD origin sits {np.hypot(*(p1[:2] * 1000)):.1f} mm "
              f"horizontally from the yaw axis.")
        print("Every 'base frame' coordinate in this repo is measured from that")
        print("origin, so it is NOT a distance from the base column.\n")

        for name, pt in (("wrist", wrist), ("claw tip", tip)):
            v = pt - p1 * 1000
            out = np.hypot(v[0], v[1])
            print(f"  {name:9}: {out:6.1f} mm out from the yaw axis, "
                  f"{v[2]:+7.1f} mm above the axis' origin height")

        print("\n" + "=" * 78)
        print("C.  ARE THE PITCH JOINTS CLEAN? — decomposed about the PITCH AXIS")
        print("=" * 78)
        print("\nDo NOT decompose this about the yaw axis. The arm's plane is offset")
        print(f"from that axis by the J1->J2 link ({axis_distance(*ax[1], *ax[2])[0]*1000:.1f} mm),")
        print("so a perfectly clean pitch still shows a large apparent 'sideways'")
        print("component there. The test that means something: a pitch joint moves")
        print("the tip WITHIN its own plane, i.e. zero motion ALONG the pitch axis.\n")

        pitch_n = ax[2][0]
        inplane = np.cross(pitch_n, [0, 0, 1.0])
        inplane /= np.linalg.norm(inplane)

        print(f"   {'':4} {'along axis':>11} {'in-plane':>9} {'up':>8}   verdict")
        for k in range(1, 6):
            a = [0.0] * 5
            a[k - 1] = np.deg2rad(10.0)
            _, T1_t = srv.fk(a)
            if phys:
                T1_t = F_PHYS @ T1_t
            d = T1_t[:3, 3] * 1000 - tip
            along, across, up = np.dot(d, pitch_n), np.dot(d, inplane), d[2]
            flag = ""
            if k in (2, 3, 4):
                flag = "clean pitch" if abs(along) < 0.5 else "*** LEAKS OUT OF PLANE"
            print(f"   J{k}: {along:+11.2f} {across:+9.1f} {up:+8.1f}   {flag}")

        # Which way does the arm actually point at all-zero joints? The tip's
        # bearing from the CAD ORIGIN answers a different question and is
        # meaningless here -- the origin is 81 mm off the yaw axis, so the tip
        # landing near +X is an accident of where the CAD origin fell.
        shoulder_to_wrist = wrist - ax[2][1] * 1000
        fwd = shoulder_to_wrist - np.dot(shoulder_to_wrist, pitch_n) * pitch_n
        fwd[2] = 0.0
        bearing = np.degrees(np.arctan2(fwd[1], fwd[0]))
        print(f"\nAt all-zero joints the arm's plane points along bearing "
              f"{bearing:+.1f} deg,")
        print("measured from the +X axis that everything Python-side calls FORWARD.")
        if abs(bearing) > 30:
            print("\n  *** The arm does NOT reach along +X. Every Cartesian x/y")
            print(f"      target is rotated by {bearing:+.0f} deg relative to the arm's")
            print("      own forward. Stage D only ever validated the VERTICAL axis,")
            print("      which a rotation about Z leaves untouched -- so nothing has")
            print("      caught this. NOTE: hand-eye is immune (AX=XB uses RELATIVE")
            print("      gripper poses, so a fixed base-frame change cancels exactly),")
            print("      which rules this out as the calibration failure.")

        print("\n" + "=" * 78)
        print("D.  MOMENT ARMS — how far the tip sits from each joint's own axis")
        print("=" * 78)
        print("""
THE CHECK THIS SCRIPT WAS MISSING, and it is the one a jog can falsify with a
ruler in ten seconds. Sections A-C test axis DIRECTIONS and link LENGTHS at
home; both were clean while the model still misplaced the claw by 74 mm.

For a serial arm the tip's distance from a joint's axis is what converts that
joint's rotation into tip travel: sweep = 2*r*sin(theta/2). J2 is upstream of
J3, which is upstream of J4, so in any ordinary posture J2 should have the
LARGEST arm of the three -- it carries everything beyond it. A downstream joint
showing a bigger arm than the joint above it is not proof of a fault (a folded
arm can swing the tip back toward an upper axis), but it is the shape of one,
and it is what an operator notices immediately: "J2 moves the claw much more
than J3, and far more than J4".

Measured 2026-08-07 at the pose a jog started from, the model gave J2 63.1 mm
while a 12.6 deg jog swept the claw 30.0 mm -- which needs 137 mm. The model
understated J2's arm by 74 mm, and reported J3 at 151 mm, larger than the joint
above it. That single inversion is worth more than every statistical fit run
against the touch data, all of which pointed elsewhere.

AT SEVERAL POSES, not just home. At home the inversion is marginal (157 vs 160
mm) and reads as noise; at a working posture it is 63 vs 151.
""")
        from vision_pipeline.robot_interface import servo_calibration as sc

        cal = sc.load_calibration()
        poses = {"home (all joints zero)": [0.0] * 5}
        try:
            poses["hover (SERVO_HOVER_TICKS)"] = [
                sc.ticks_to_rad(cal, j, config.SERVO_HOVER_TICKS[j])
                for j in range(1, 6)]
        except (KeyError, TypeError):
            pass                      # no hover recorded; home alone still says a lot

        for label, angles in poses.items():
            axp, _Tw, Tt = screw_axes(srv, angles, physical=phys)
            tip_p = Tt[:3, 3]
            arms = {}
            for j in range(1, 6):
                n, p = axp[j]
                v = tip_p - p
                arms[j] = float(np.linalg.norm(v - np.dot(v, n) * n))

            print(f"  {label}:")
            inverted = [j for j in (3, 4) if arms[j] > arms[2]]
            for j in range(1, 6):
                note = "   <== LARGER than J2, which is UPSTREAM of it" \
                    if j in inverted else ""
                print(f"    J{j} ({JOINT_ROLE[j]:11}): tip is {arms[j] * 1000:6.1f} mm "
                      f"from its axis{note}")
            if inverted:
                worst = max(inverted, key=lambda j: arms[j])
                sweep2 = 2 * arms[2] * np.sin(np.deg2rad(6.0))
                sweepw = 2 * arms[worst] * np.sin(np.deg2rad(6.0))
                print(f"    -> FALSIFY IT WITH A RULER. Jog J2 and J{worst} by the same")
                print(f"       12 deg and measure the claw: the model predicts "
                      f"{sweep2 * 1000:.0f} mm")
                print(f"       for J2 and {sweepw * 1000:.0f} mm for J{worst}. If J2 moves "
                      f"the claw FURTHER,")
                print(f"       the model's J2 geometry is wrong -- not your measurement.")
            print()

        print("\n" + "=" * 78)
        print("E.  RULER SHEET — every number here is measurable on the real arm")
        print("=" * 78)
        print("\nPark the arm at HOME (python scripts/goto_pose.py --pose home)")
        print("before measuring the heights. Shaft centre to shaft centre.\n")

        print("  Link lengths (pose-independent — measure at any pose):")
        for a, b in ((2, 3), (3, 4)):
            d, _ = axis_distance(*ax[a], *ax[b])
            print(f"    servo {a} shaft -> servo {b} shaft "
                  f"({JOINT_ROLE[a]} -> {JOINT_ROLE[b]}): {d * 1000:6.1f} mm")
        print(f"    servo 5 shaft -> claw tip: "
              f"{np.linalg.norm(tip - wrist):6.1f} mm   (CLAW_LEN in init_arm.m)")

        # Heights are easier to hit with a ruler than a shaft-to-shaft span, and
        # they are referenced to the tabletop rather than the CAD origin, so
        # they do not inherit that origin's 81 mm offset.
        table = config.TABLE_Z_IN_BASE * 1000
        print(f"\n  Heights above the TABLETOP at home "
              f"(TABLE_Z_IN_BASE = {table:.0f} mm):")
        for k in (2, 3, 4):
            print(f"    servo {k} shaft centre ({JOINT_ROLE[k]:11}): "
                  f"{ax[k][1][2] * 1000 - table:6.1f} mm")
        print(f"    wrist  (servo 5 shaft)          : {wrist[2] - table:6.1f} mm")
        print(f"    claw tip                        : {tip[2] - table:6.1f} mm")
    finally:
        srv.close()


if __name__ == "__main__":
    main()

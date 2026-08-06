"""TCP client for the MATLAB IK/FK server (ik_fk_server.m).

FRAME CONVERSION SEAM — this module is the ONE place the model/physical frame
flip is handled (see CLAUDE.md "COORDINATE FRAMES"). The Simscape import's base
frame is upside-down relative to the physical robot: model +Z points physically
DOWN, model +Y points physically RIGHT, model +X is forward. The relationship
is a 180° rotation about X, self-inverse:

    physical (x, y, z) = model (x, -y, -z)

request_ik converts the physical target to model coordinates before sending;
request_fk / request_fk_tip convert returned transforms to physical before
returning. Everything above this client therefore works purely in the physical
frame (+Z up, table below the base origin), while the wire protocol and all
MATLAB-side code stay in the model frame. Joint ANGLES are unaffected by the
Cartesian flip and pass through unchanged. Do not convert anywhere else.
"""

import json
import logging
import socket
import numpy as np

# Homogeneous 180-degree rotation about X: maps model-frame poses to physical
# and vice versa (it is its own inverse). Left-multiply transforms; for bare
# points it is just (x, -y, -z).
_F_PHYS_FROM_MODEL = np.diag([1.0, -1.0, -1.0, 1.0])


class IKUnreachableError(Exception):
    """Raised when the IK solver reports a target outside the workspace."""
    pass


logger = logging.getLogger(__name__)


class MatlabIKClient:
    """TCP JSON client for IK/FK requests to the MATLAB server.

    Connects to a persistent ik_fk_server.m running on localhost.
    Requests are stateless: same inputs always produce the same outputs.
    """

    # A locked joint drifting more than this means the pin did not hold. 0.5 deg
    # is comfortably above solver noise and far below anything that matters to
    # the image (the failure being guarded against was 8 degrees of wrist roll).
    LOCK_DRIFT_WARN_RAD = 0.0087

    # What ik_fk_server.m reports when it is running the same code as this repo.
    # A MATLAB server keeps executing whatever it was started with, and nothing
    # else at the protocol level tells you so -- which cost two rounds of
    # "restart it and try again" against a server that HAD been restarted.
    # Bump both this and the string in ik_fk_server.m together.
    SERVER_BUILD = "2026-08-06-lockrelease"

    def __init__(self, host: str = "localhost", port: int = 9999):
        """Connect to the MATLAB server.

        Args:
            host: server hostname (default localhost).
            port: server port (default 9999).

        Raises:
            ConnectionRefusedError: if the server is not listening.
        """
        self.host = host
        self.port = port
        self.socket = None
        self._yaw_axis_xy = None      # measured lazily; see base_yaw_axis_xy
        self._connect()

    def _connect(self):
        """Establish TCP connection to the server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def _send_request(self, request_dict: dict) -> dict:
        """Send a JSON request and receive a JSON response.

        Args:
            request_dict: dict with at least 'cmd' key.

        Returns:
            The parsed JSON response as a dict.

        Raises:
            IKUnreachableError: if server responds with ok:false.
            socket.error: if connection fails.
        """
        # Send request as JSON + newline
        request_json = json.dumps(request_dict)
        self.socket.sendall((request_json + "\n").encode("utf-8"))

        # Read response line
        response_bytes = b""
        while not response_bytes.endswith(b"\n"):
            chunk = self.socket.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            response_bytes += chunk

        response_json = response_bytes.decode("utf-8").strip()
        response = json.loads(response_json)

        if not response.get("ok", False):
            raise IKUnreachableError(response.get("error", "Unknown error"))

        return response

    def request_ik(
        self,
        x: float,
        y: float,
        z: float,
        seed_rad: list[float] | None = None,
        lock: list[int] | None = None,
    ) -> tuple[list[float], float]:
        """Request inverse kinematics for a position.

        Args:
            x, y, z: target position in meters, PHYSICAL base frame (+Z up,
                table below the origin). Converted to the solver's model frame
                (y and z negated) on the wire — see the module docstring.
            seed_rad: the arm's CURRENT J1..J5 angles, strongly recommended for
                any real move. Without it the solver seeds from the model's
                home configuration and can return a valid solution in a wholly
                different posture — a 45mm target once came back demanding
                ~2600 ticks (~227 deg) of base rotation. Angles are
                frame-independent, so no conversion applies to them.
            lock: joint numbers (1..5) to HOLD at their seed angle. Five joints
                against a 3-DOF position target leaves a 2-dimensional null
                space, and a position-only solve has no preference within it —
                so the solver is free to spend base yaw on a move that does not
                need any. The camera rides on the wrist, so that pans the whole
                image. Locking J1 for a pure descent asks for the answer in the
                plane the arm is already in. Requires seed_rad.

        Returns:
            (angles_rad, err_mm): J1..J5 angles in radians, and IK error in mm.
            Angles are frame-independent scalars; err_mm is a norm, unchanged
            by the rotation.

        Raises:
            IKUnreachableError: if target is outside workspace or IK tolerance.
            ValueError: if lock is given without seed_rad.
        """
        if lock and seed_rad is None:
            raise ValueError(
                "lock requires seed_rad: a joint can only be held at a known "
                "angle, and without a seed the server has none to hold it at."
            )
        req = {"cmd": "ik", "x": x, "y": -y, "z": -z}
        if seed_rad is not None:
            req["seed_rad"] = list(seed_rad)
        if lock:
            req["lock"] = [int(j) for j in lock]
        resp = self._send_request(req)
        angles_rad = resp["angles_rad"]  # list of 5 floats
        err_mm = resp["err_mm"]  # float

        # A LOCK THAT SILENTLY FAILS IS WORSE THAN NO LOCK: the caller believes
        # a disturbance is suppressed and tunes its gains against that belief.
        # Observed 2026-08-06 -- a run locking J5 came back moving it 94 ticks,
        # because the server pinned it with a degenerate [v, v] interval the
        # solver did not honour, and nothing in the protocol could report it.
        # Older servers omit the field entirely, so absence is not a failure.
        drift = resp.get("lock_drift_rad")
        if lock and drift is not None and drift > self.LOCK_DRIFT_WARN_RAD:
            logger.warning(
                "IK was asked to hold joint(s) %s but moved one by %.1f deg "
                "(%.0f ticks). The lock is not holding — treat any gain measured "
                "through this solve as unreliable, and check that MATLAB was "
                "restarted after the last change to matlab/.",
                list(lock), np.degrees(drift), drift * 651.89)
        return angles_rad, err_mm

    def request_fk(self, angles_rad: list[float]) -> np.ndarray:
        """Request forward kinematics for joint angles.

        Args:
            angles_rad: list of 5 joint angles in radians (J1..J5).

        Returns:
            4x4 numpy array (WRIST pose, Body08) in the PHYSICAL base frame
            (converted from the server's model frame — see module docstring).
            This is the frame hand-eye calibration is solved against. For the
            claw tip — what the IK solver targets — use request_fk_tip().
        """
        req = {"cmd": "fk", "angles_rad": angles_rad}
        resp = self._send_request(req)
        # Response is 16 floats in row-major order, model frame
        T_model = np.array(resp["T"]).reshape((4, 4), order="C")
        return _F_PHYS_FROM_MODEL @ T_model

    def request_fk_tip(self, angles_rad: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics returning BOTH the wrist and the claw tip.

        The two frames sit ~70mm apart (CLAW_LEN) and are not interchangeable:
        hand-eye calibration is solved against the wrist, while the IK solver
        targets the tip. Round-trip validation must compare a commanded target
        against the TIP, or it measures the frame offset rather than real
        positioning error.

        Args:
            angles_rad: list of 5 joint angles in radians (J1..J5).

        Returns:
            (T_wrist, T_tip), each a 4x4 numpy array in the PHYSICAL base
            frame (converted from the server's model frame). Physically the
            tip hangs BELOW the wrist — if it ever comes back above, a frame
            conversion has been dropped.

        Raises:
            RuntimeError: if the server predates the T_tip field — it must be
                restarted to pick up the new handler.
        """
        req = {"cmd": "fk", "angles_rad": angles_rad}
        resp = self._send_request(req)

        if "T_tip" not in resp:
            raise RuntimeError(
                "MATLAB server returned no 'T_tip' field — it is running an "
                "older ik_fk_server.m. Restart it in MATLAB (Ctrl+C, then "
                "`ik_fk_server` from the matlab/ folder) to pick up the "
                "ClawTip transform."
            )

        T_wrist = _F_PHYS_FROM_MODEL @ np.array(resp["T"]).reshape((4, 4), order="C")
        T_tip = _F_PHYS_FROM_MODEL @ np.array(resp["T_tip"]).reshape((4, 4), order="C")
        return T_wrist, T_tip

    def base_yaw_axis_xy(self) -> np.ndarray:
        """Where J1's rotation axis actually is, in base-frame XY METRES.

        NOT the origin, and assuming otherwise is a live bug source. The
        imported model's origin is a CAD artefact sitting 81 mm from the column
        the arm turns about (measured 2026-08-06, scripts/audit_model_axes.py),
        while at the home pose the claw tip is only ~25 mm from that column. So
        a "radial" direction computed as `tip_xy / |tip_xy|` points up to 108 deg
        away from truly outward. That error produced jog predictions announcing
        sideways motion for joints that cannot move sideways, and it sat inside
        the visual-servo descent, where it turns a reach correction into a
        sideways one.

        MEASURED, NOT STORED. Jog J1 a little and the axis it rotates the wrist
        about IS the base yaw axis, so this cannot go stale if the model, the
        joint mapping or the frame convention changes. Cached because it is a
        property of the robot, not of the pose: two FK calls, once per session.
        """
        if self._yaw_axis_xy is None:
            from vision_pipeline.calibration import geometry

            t0 = self.request_fk([0.0] * 5)
            t1 = self.request_fk([np.deg2rad(4.0), 0.0, 0.0, 0.0, 0.0])
            _, point, _ = geometry.screw_axis(t1 @ geometry.invert_transform(t0))
            self._yaw_axis_xy = point[:2]
        return self._yaw_axis_xy

    def close(self):
        """Close the TCP connection."""
        if self.socket:
            self.socket.close()
            self.socket = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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
import socket
import numpy as np

# Homogeneous 180-degree rotation about X: maps model-frame poses to physical
# and vice versa (it is its own inverse). Left-multiply transforms; for bare
# points it is just (x, -y, -z).
_F_PHYS_FROM_MODEL = np.diag([1.0, -1.0, -1.0, 1.0])


class IKUnreachableError(Exception):
    """Raised when the IK solver reports a target outside the workspace."""
    pass


class MatlabIKClient:
    """TCP JSON client for IK/FK requests to the MATLAB server.

    Connects to a persistent ik_fk_server.m running on localhost.
    Requests are stateless: same inputs always produce the same outputs.
    """

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

    def request_ik(self, x: float, y: float, z: float) -> tuple[list[float], float]:
        """Request inverse kinematics for a position.

        Args:
            x, y, z: target position in meters, PHYSICAL base frame (+Z up,
                table below the origin). Converted to the solver's model frame
                (y and z negated) on the wire — see the module docstring.

        Returns:
            (angles_rad, err_mm): J1..J5 angles in radians, and IK error in mm.
            Angles are frame-independent scalars; err_mm is a norm, unchanged
            by the rotation.

        Raises:
            IKUnreachableError: if target is outside workspace or IK tolerance.
        """
        req = {"cmd": "ik", "x": x, "y": -y, "z": -z}
        resp = self._send_request(req)
        angles_rad = resp["angles_rad"]  # list of 5 floats
        err_mm = resp["err_mm"]  # float
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

    def close(self):
        """Close the TCP connection."""
        if self.socket:
            self.socket.close()
            self.socket = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

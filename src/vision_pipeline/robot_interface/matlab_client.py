"""TCP client for the MATLAB IK/FK server (ik_fk_server.m)."""

import json
import socket
import numpy as np


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
            x, y, z: target position in meters (base frame).

        Returns:
            (angles_rad, err_mm): J1..J5 angles in radians, and IK error in mm.

        Raises:
            IKUnreachableError: if target is outside workspace or IK tolerance.
        """
        req = {"cmd": "ik", "x": x, "y": y, "z": z}
        resp = self._send_request(req)
        angles_rad = resp["angles_rad"]  # list of 5 floats
        err_mm = resp["err_mm"]  # float
        return angles_rad, err_mm

    def request_fk(self, angles_rad: list[float]) -> np.ndarray:
        """Request forward kinematics for joint angles.

        Args:
            angles_rad: list of 5 joint angles in radians (J1..J5).

        Returns:
            4x4 numpy array (WRIST pose, Body08, in base frame). This is the
            frame hand-eye calibration is solved against. For the claw tip —
            what the IK solver targets — use request_fk_tip().
        """
        req = {"cmd": "fk", "angles_rad": angles_rad}
        resp = self._send_request(req)
        # Response is 16 floats in row-major order
        T_flat = resp["T"]
        T = np.array(T_flat).reshape((4, 4), order="C")
        return T

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
            (T_wrist, T_tip), each a 4x4 numpy array in the base frame.

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

        T_wrist = np.array(resp["T"]).reshape((4, 4), order="C")
        T_tip = np.array(resp["T_tip"]).reshape((4, 4), order="C")
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

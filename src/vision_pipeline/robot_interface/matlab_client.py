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
            4x4 numpy array (wrist pose in base frame).
        """
        req = {"cmd": "fk", "angles_rad": angles_rad}
        resp = self._send_request(req)
        # Response is 16 floats in row-major order
        T_flat = resp["T"]
        T = np.array(T_flat).reshape((4, 4), order="C")
        return T

    def close(self):
        """Close the TCP connection."""
        if self.socket:
            self.socket.close()
            self.socket = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

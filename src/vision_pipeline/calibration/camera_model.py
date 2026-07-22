"""
Pinhole camera model: convert between pixels and 3D rays in the camera frame.

This is the optics half of pixel-to-world. It knows nothing about the robot or
the table — only about the camera's own lens (its intrinsics). Given a pixel it
produces the 3D ray, in the camera's own coordinate frame, that the pixel could
have come from. The calibration code then transforms that ray into the robot
base frame and intersects it with the table.

Camera frame convention (OpenCV standard): +X right, +Y down, +Z forward out of
the lens. So a ray always has positive Z (it goes out in front of the camera).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vision_pipeline import config


@dataclass
class CameraIntrinsics:
    """The camera's internal optics: focal lengths, optical center, distortion.

    fx, fy, cx, cy are in pixels. distortion is OpenCV's (k1, k2, p1, p2, k3).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 camera matrix K used by OpenCV projection functions."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array(self.distortion, dtype=float)

    def pixel_to_ray(self, pixel_xy: tuple[float, float]) -> np.ndarray:
        """Convert a pixel to a unit ray direction in the camera frame.

        Undistorts the pixel first (a no-op when distortion is all zeros), then
        back-projects through the pinhole model. The returned vector is unit
        length and points out in front of the camera (+Z).
        """
        pts = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float64)
        # undistortPoints with P=None returns normalized coords (x', y') where
        # x' = (u - cx)/fx already corrected for lens distortion.
        normalized = cv2.undistortPoints(pts, self.matrix, self.dist_coeffs)
        x, y = normalized[0, 0]
        ray = np.array([x, y, 1.0], dtype=float)
        return ray / np.linalg.norm(ray)

    def project_point(self, point_cam: np.ndarray) -> tuple[float, float]:
        """Project a 3D point in the CAMERA frame back to a pixel.

        The inverse of pixel_to_ray's back-projection, useful for tests and for
        drawing predicted positions. Assumes the point is in front of the lens.
        """
        point_cam = np.asarray(point_cam, dtype=float).reshape(1, 3)
        image_points, _ = cv2.projectPoints(
            point_cam,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=self.matrix,
            distCoeffs=self.dist_coeffs,
        )
        u, v = image_points[0, 0]
        return (float(u), float(v))


def default_intrinsics() -> CameraIntrinsics:
    """Build intrinsics from the inline values in config.py (the fallback)."""
    return CameraIntrinsics(
        fx=config.CAMERA_FX,
        fy=config.CAMERA_FY,
        cx=config.CAMERA_CX,
        cy=config.CAMERA_CY,
        distortion=tuple(config.CAMERA_DISTORTION),
    )


def load_intrinsics(path: str | Path | None = None) -> CameraIntrinsics:
    """Load intrinsics from a JSON file, falling back to config.py defaults.

    The file (written by your calibration step) looks like:
        {"fx": 550.0, "fy": 550.0, "cx": 320.0, "cy": 240.0,
         "distortion": [0, 0, 0, 0, 0]}

    If ``path`` is None we use config.CAMERA_INTRINSICS_PATH. A missing file is
    NOT an error — we return the config defaults so the pipeline still runs
    (with placeholder optics) before you've calibrated.
    """
    if path is None:
        path = config.CAMERA_INTRINSICS_PATH
    path = Path(path)
    if not path.exists():
        return default_intrinsics()

    data = json.loads(path.read_text())
    return CameraIntrinsics(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        distortion=tuple(data.get("distortion", (0.0, 0.0, 0.0, 0.0, 0.0))),
    )


def save_intrinsics(intrinsics: CameraIntrinsics, path: str | Path) -> None:
    """Write intrinsics to JSON (for use after running a calibration)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "distortion": list(intrinsics.distortion),
            },
            indent=2,
        )
    )

"""Camera axes and the pinhole projection, shared by the renderer, the targets and the evaluation.

Datasets and learning code use OpenCV camera axes: x right, y down, z forward (the viewing
direction). Blender cameras use x right, y up, z backward (Blender looks along -z). A camera
rotation R maps camera-axis vectors to world vectors in both conventions.

``flip_camera_axes`` is the only place where the two conventions are converted. NumPy only, so
the module can be imported anywhere (including next to Blender).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Diagonal flip of the y and z camera axes; it is its own inverse.
_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])
WORLD_UP = np.array([0.0, 0.0, 1.0])


def flip_camera_axes(rotation: np.ndarray) -> np.ndarray:
    """Convert camera rotations [..., 3, 3] between Blender and OpenCV axes (either direction).

    The world frame is unchanged; only the camera's own y and z axes are negated.
    """
    return np.asarray(rotation, dtype=float) @ _AXIS_FLIP


def look_at_blender(location: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Blender camera rotation [3, 3] looking from ``location`` to ``target`` with world z up.

    Equals Blender's ``(target - location).to_track_quat("-Z", "Y")``: camera -z towards the
    target, camera y in the vertical plane through the viewing direction, pointing up.
    """
    forward = np.asarray(target, dtype=float) - np.asarray(location, dtype=float)
    forward = forward / np.linalg.norm(forward)
    up = WORLD_UP - (WORLD_UP @ forward) * forward
    if np.linalg.norm(up) < 1.0e-6:
        raise ValueError("The viewing direction is vertical; look-at with world z up is undefined.")
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)
    return np.column_stack((right, up, -forward))


def look_at_opencv(location: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV camera rotation [3, 3] (x right, y down, z towards the target), world z up."""
    return flip_camera_axes(look_at_blender(location, target))


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics of the rendered camera.

    Blender's automatic sensor fit applies the sensor width to the larger image dimension, so one
    focal length in pixels holds for both axes. Image coordinates are continuous, with (0, 0) at
    the top-left corner of the image: pixel (i, j) covers [i, i + 1) x [j, j + 1).
    """

    width: int
    height: int
    focal_length_mm: float
    sensor_width_mm: float

    @classmethod
    def from_render_config(cls, render: dict) -> CameraIntrinsics:
        """From a rendering config mapping (``width``, ``height`` and ``camera``)."""
        camera = render["camera"]
        return cls(
            width=int(render["width"]),
            height=int(render["height"]),
            focal_length_mm=float(camera["focal_length_mm"]),
            sensor_width_mm=float(camera["sensor_width_mm"]),
        )

    @property
    def focal_pixels(self) -> float:
        return max(self.width, self.height) * self.focal_length_mm / self.sensor_width_mm

    def project(self, points_camera: np.ndarray) -> np.ndarray:
        """Points [..., 3] (m, OpenCV camera axes) -> image coordinates [..., 2] (column, row)."""
        points = np.asarray(points_camera, dtype=float)
        depth = points[..., 2:3]
        centre = np.array([self.width / 2.0, self.height / 2.0])
        return centre + self.focal_pixels * points[..., :2] / depth


def world_to_camera(
    points_world: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray
) -> np.ndarray:
    """World points [..., 3] -> camera-axis coordinates [..., 3] for a camera pose (p, R)."""
    relative = np.asarray(points_world, dtype=float) - np.asarray(camera_position, dtype=float)
    return np.einsum("...ji,...j->...i", np.asarray(camera_rotation, dtype=float), relative)

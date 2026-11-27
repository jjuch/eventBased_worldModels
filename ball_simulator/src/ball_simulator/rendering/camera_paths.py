"""Camera paths: one smooth camera trajectory per object trajectory, world frame, OpenCV axes.

Modes, drawn per trajectory with ``camera_motion.fractions``:

    static     the base camera of the rendering config
    drift      base camera plus smooth translation and rotation (sums of sinusoids)
    tracking   drifting position; the camera looks at a lagged copy of the object position
    co_moving  base orientation; the position follows a lagged copy of the object displacement

All motion is C2-continuous: sinusoids are smooth, and the lag is a critically damped
second-order filter, whose output has a continuous second derivative for a continuous input.
A path is rejected and redrawn when the object leaves the image or comes closer than
``minimum_distance``, or when the camera moves further than ``maximum_offset`` from the base
position. After ``maximum_attempts`` rejected paths the trajectory is skipped.

Camera states are evaluated on a dense grid (``DENSE_SUBSTEPS`` per observation interval) for the
filter and the velocities, then sampled at the object observation times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .camera_axes import CameraIntrinsics, look_at_opencv, world_to_camera
from .config import CameraMotionConfig, RenderConfig

DENSE_SUBSTEPS = 20
SINUSOIDS_PER_AXIS = 3


@dataclass(frozen=True)
class CameraPath:
    """Camera states at the object observation times; world frame, OpenCV camera axes.

    time [N] s, position [N, 3] m, quaternion_xyzw [N, 4] camera-to-world rotation,
    linear_velocity [N, 3] m/s, angular_velocity [N, 3] rad/s (world frame, relative to the
    static scene, R[k+1] = exp(dt * angular_velocity) R[k]).
    """

    mode: str
    time: np.ndarray
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    attempts: int


def _sinusoids(
    rng: np.random.Generator, time: np.ndarray, frequency_hz: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Random sum of sinusoids per axis, zero at the first time: value and derivative [M, 3]."""
    frequency = rng.uniform(*frequency_hz, size=(SINUSOIDS_PER_AXIS, 3))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(SINUSOIDS_PER_AXIS, 3))
    amplitude = rng.normal(size=(SINUSOIDS_PER_AXIS, 3))
    angle = 2.0 * np.pi * frequency * (time - time[0])[:, None, None] + phase
    value = (amplitude * np.sin(angle)).sum(axis=1)
    derivative = (amplitude * 2.0 * np.pi * frequency * np.cos(angle)).sum(axis=1)
    return value - value[0], derivative


def _scaled_to_speed(value: np.ndarray, derivative: np.ndarray, speed: float) -> np.ndarray:
    """Scale a signal so that the largest norm of its derivative equals ``speed``."""
    largest = np.linalg.norm(derivative, axis=1).max()
    return value * (speed / largest) if largest > 0.0 else value


def critically_damped_lag(signal: np.ndarray, dt: float, lag: float) -> np.ndarray:
    """Second-order critically damped low-pass of a signal [M, 3] sampled every ``dt`` s.

    Natural frequency 2 / lag, so a constant velocity v is followed a distance lag * v behind.
    The filter starts in that steady state (delayed by lag * initial velocity, moving with it),
    so there is no start-up transient.
    """
    natural_frequency = 2.0 / lag
    output = np.empty_like(signal)
    velocity = (signal[1] - signal[0]) / dt
    position = signal[0] - lag * velocity
    for index in range(len(signal)):
        output[index] = position
        acceleration = (
            natural_frequency**2 * (signal[index] - position) - 2.0 * natural_frequency * velocity
        )
        velocity = velocity + dt * acceleration
        position = position + dt * velocity
    return output


def _angular_velocity(rotations: Rotation, dt: float) -> np.ndarray:
    """World-frame angular velocity [M, 3] by central differences of R[k+1] R[k-1]^T."""
    rotation_vectors = np.empty((len(rotations), 3))
    rotation_vectors[1:-1] = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2.0 * dt)
    rotation_vectors[0] = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
    rotation_vectors[-1] = (rotations[-1] * rotations[-2].inv()).as_rotvec() / dt
    return rotation_vectors


def _dense_path(
    mode: str,
    rng: np.random.Generator,
    dense_time: np.ndarray,
    object_position: np.ndarray,
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    motion: CameraMotionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera positions [M, 3] and OpenCV rotations [M, 3, 3] on the dense grid."""
    count = len(dense_time)
    dt = float(dense_time[1] - dense_time[0])

    def drift_translation() -> np.ndarray:
        value, derivative = _sinusoids(rng, dense_time, motion.drift_frequency_hz)
        speed = rng.uniform(0.5, 1.0) * motion.maximum_speed
        return base_position + _scaled_to_speed(value, derivative, speed)

    if mode == "static":
        return np.repeat(base_position[None], count, axis=0), np.repeat(
            base_rotation[None], count, axis=0
        )

    if mode == "drift":
        rotation_value, rotation_derivative = _sinusoids(rng, dense_time, motion.drift_frequency_hz)
        angular_speed = rng.uniform(0.5, 1.0) * motion.maximum_angular_speed
        rotation_vector = _scaled_to_speed(rotation_value, rotation_derivative, angular_speed)
        # Perturbation about the camera's own axes: R(t) = R_base exp(theta(t)).
        rotations = base_rotation @ Rotation.from_rotvec(rotation_vector).as_matrix()
        return drift_translation(), rotations

    lag = rng.uniform(*motion.tracking_lag_s)
    lagged_object = critically_damped_lag(object_position, dt, lag)
    if mode == "tracking":
        positions = drift_translation()
        rotations = np.stack(
            [
                look_at_opencv(position, target)
                for position, target in zip(positions, lagged_object, strict=True)
            ]
        )
        return positions, rotations

    if mode == "co_moving":
        gain = rng.uniform(*motion.co_moving_gain)
        positions = base_position + gain * (lagged_object - lagged_object[0])
        return positions, np.repeat(base_rotation[None], count, axis=0)

    raise ValueError(f"Unknown camera motion mode: {mode!r}.")


def path_is_valid(
    positions: np.ndarray,
    rotations: np.ndarray,
    object_position: np.ndarray,
    object_radius: float,
    base_position: np.ndarray,
    intrinsics: CameraIntrinsics,
    motion: CameraMotionConfig,
) -> bool:
    """Bounded offset, object far enough away, object disc fully inside the image with margin."""
    if np.linalg.norm(positions - base_position, axis=1).max() > motion.maximum_offset:
        return False
    in_camera = world_to_camera(object_position, positions, rotations)
    depth = in_camera[:, 2]
    if np.any(depth < motion.minimum_distance):
        return False
    pixels = intrinsics.project(in_camera)
    disc = intrinsics.focal_pixels * object_radius / depth + motion.margin_px
    inside = (
        (pixels[:, 0] - disc >= 0.0)
        & (pixels[:, 0] + disc <= intrinsics.width)
        & (pixels[:, 1] - disc >= 0.0)
        & (pixels[:, 1] + disc <= intrinsics.height)
    )
    return bool(inside.all())


def generate_camera_path(
    object_time: np.ndarray,
    object_position: np.ndarray,
    object_radius: float,
    config: RenderConfig,
    trajectory_index: int,
) -> CameraPath | None:
    """Camera path for one object trajectory, or None when no valid path was found.

    object_time [N] s, object_position [N, 3] m (world frame), object_radius m (bounding radius).
    Reproducible: the random stream depends only on ``camera_motion.seed`` and the index.
    """
    motion = config.camera_motion
    rng = np.random.default_rng(np.random.SeedSequence([motion.seed, trajectory_index]))
    modes = sorted(motion.fractions)
    mode = str(rng.choice(modes, p=[motion.fractions[name] for name in modes]))

    base_position = np.asarray(config.camera.location, dtype=float)
    base_rotation = look_at_opencv(base_position, np.asarray(config.camera.target, dtype=float))
    intrinsics = CameraIntrinsics.from_render_config(config.model_dump(mode="json"))

    observation_count = len(object_time)
    dense_time = np.linspace(
        object_time[0], object_time[-1], (observation_count - 1) * DENSE_SUBSTEPS + 1
    )
    dense_object = np.column_stack(
        [np.interp(dense_time, object_time, object_position[:, axis]) for axis in range(3)]
    )
    observed = slice(None, None, DENSE_SUBSTEPS)
    attempts = 1 if mode == "static" else motion.maximum_attempts

    for attempt in range(1, attempts + 1):
        positions, rotations = _dense_path(
            mode, rng, dense_time, dense_object, base_position, base_rotation, motion
        )
        if not path_is_valid(
            positions[observed],
            rotations[observed],
            object_position,
            object_radius,
            base_position,
            intrinsics,
            motion,
        ):
            continue
        dt = float(dense_time[1] - dense_time[0])
        rotation = Rotation.from_matrix(rotations)
        return CameraPath(
            mode=mode,
            time=np.asarray(object_time, dtype=float).copy(),
            position=positions[observed],
            quaternion_xyzw=rotation[observed].as_quat(canonical=False),
            linear_velocity=np.gradient(positions, dt, axis=0)[observed],
            angular_velocity=_angular_velocity(rotation, dt)[observed],
            attempts=attempt,
        )
    return None


def write_camera_paths(
    dataset: str | Path, config: RenderConfig, output: str | Path
) -> tuple[int, list[str]]:
    """Camera paths for every trajectory of a dataset; returns (written, skipped trajectory ids).

    File layout: /trajectories/<id>/{time, position, quaternion_xyzw, linear_velocity,
    angular_velocity} with attributes ``mode`` and ``attempts``.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    written = 0
    temporary = output.with_suffix(".partial.h5")
    with h5py.File(dataset, "r") as source, h5py.File(temporary, "w") as target:
        target.attrs["camera_axes"] = "opencv: x right, y down, z forward"
        target.attrs["frame"] = "world"
        target.attrs["quaternion_order"] = "xyzw"
        target.attrs["config_json"] = json.dumps(
            {
                "camera": config.camera.model_dump(mode="json"),
                "camera_motion": config.camera_motion.model_dump(mode="json"),
                "width": config.width,
                "height": config.height,
            }
        )
        for trajectory_id in sorted(source["trajectories"]):
            group = source["trajectories"][trajectory_id]
            path = generate_camera_path(
                np.asarray(group["observations/time"], dtype=float),
                np.asarray(group["observations/position"], dtype=float),
                float(group["parameters"].attrs["radius"]),
                config,
                int(trajectory_id),
            )
            if path is None:
                skipped.append(trajectory_id)
                continue
            written_group = target.create_group(f"trajectories/{trajectory_id}")
            for name in (
                "time",
                "position",
                "quaternion_xyzw",
                "linear_velocity",
                "angular_velocity",
            ):
                written_group.create_dataset(name, data=getattr(path, name))
            written_group.attrs["mode"] = path.mode
            written_group.attrs["attempts"] = path.attempts
            written += 1
        target.attrs["skipped_json"] = json.dumps(skipped)
    temporary.replace(output)
    return written, skipped


def load_camera_path(path: str | Path, trajectory_id: str) -> CameraPath:
    with h5py.File(path, "r") as handle:
        group = handle["trajectories"][trajectory_id]
        return CameraPath(
            mode=str(group.attrs["mode"]),
            attempts=int(group.attrs["attempts"]),
            **{
                name: np.asarray(group[name], dtype=float)
                for name in (
                    "time",
                    "position",
                    "quaternion_xyzw",
                    "linear_velocity",
                    "angular_velocity",
                )
            },
        )


def camera_path_ids(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted(handle["trajectories"]) if "trajectories" in handle else []

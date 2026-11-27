"""Camera paths: smooth, bounded, object in view, consistent velocities, reproducible."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ball_simulator.rendering.camera_axes import CameraIntrinsics, look_at_opencv, world_to_camera
from ball_simulator.rendering.camera_paths import (
    critically_damped_lag,
    generate_camera_path,
    path_is_valid,
)
from ball_simulator.rendering.config import RenderConfig

MODES = ("static", "drift", "tracking", "co_moving")


def _config(mode: str) -> RenderConfig:
    return RenderConfig.model_validate(
        {
            "width": 384,
            "height": 384,
            "camera": {
                "location": [4.0, -5.0, 3.0],
                "target": [0.0, 0.0, 0.65],
                "focal_length_mm": 42.0,
                "sensor_width_mm": 36.0,
            },
            "camera_motion": {"fractions": {mode: 1.0}, "seed": 3},
        }
    )


def _ballistic_object(steps: int = 21, dt: float = 0.01):
    time = np.arange(steps) * dt
    position = (
        np.array([0.0, 0.0, 0.65])
        + np.outer(time, [1.2, -0.8, 1.5])
        + 0.5 * np.outer(time**2, [0.0, 0.0, -9.81])
    )
    return time, position


@pytest.mark.parametrize("mode", MODES)
def test_paths_are_reproducible_bounded_and_keep_the_object_in_view(mode):
    config = _config(mode)
    time, position = _ballistic_object()
    first = generate_camera_path(time, position, 0.1, config, trajectory_index=7)
    second = generate_camera_path(time, position, 0.1, config, trajectory_index=7)
    assert first is not None and first.mode == mode
    np.testing.assert_array_equal(first.position, second.position)
    np.testing.assert_array_equal(first.quaternion_xyzw, second.quaternion_xyzw)

    rotations = Rotation.from_quat(first.quaternion_xyzw).as_matrix()
    base = np.asarray(config.camera.location)
    assert (
        np.linalg.norm(first.position - base, axis=1).max() <= config.camera_motion.maximum_offset
    )
    intrinsics = CameraIntrinsics.from_render_config(config.model_dump(mode="json"))
    assert path_is_valid(
        first.position, rotations, position, 0.1, base, intrinsics, config.camera_motion
    )
    pixels = intrinsics.project(world_to_camera(position, first.position, rotations))
    assert np.all((pixels > 0) & (pixels < 384))


@pytest.mark.parametrize("mode", MODES)
def test_paths_are_smooth_and_velocities_match_poses(mode):
    time, position = _ballistic_object()
    path = generate_camera_path(time, position, 0.1, _config(mode), trajectory_index=11)
    dt = time[1] - time[0]
    # C2: bounded second differences of position (m/s^2), no frame-to-frame jumps.
    acceleration = np.diff(path.position, 2, axis=0) / dt**2
    assert np.abs(acceleration).max() < 60.0
    # Mean velocity over each interval equals the mean of the end-point velocities to O(dt^2).
    mean_velocity = np.diff(path.position, axis=0) / dt
    np.testing.assert_allclose(
        mean_velocity, 0.5 * (path.linear_velocity[1:] + path.linear_velocity[:-1]), atol=2e-3
    )
    rotations = Rotation.from_quat(path.quaternion_xyzw)
    mean_angular = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt
    np.testing.assert_allclose(
        mean_angular, 0.5 * (path.angular_velocity[1:] + path.angular_velocity[:-1]), atol=2e-3
    )


def test_drift_respects_speed_limits():
    config = _config("drift")
    time, position = _ballistic_object()
    for index in range(5):
        path = generate_camera_path(time, position, 0.1, config, trajectory_index=index)
        motion = config.camera_motion
        assert np.linalg.norm(path.linear_velocity, axis=1).max() <= motion.maximum_speed * 1.01
        assert (
            np.linalg.norm(path.angular_velocity, axis=1).max()
            <= motion.maximum_angular_speed * 1.05
        )


def test_mode_fractions_are_honoured():
    config = RenderConfig.model_validate(
        {
            "width": 384,
            "height": 384,
            "camera": {
                "location": [4.0, -5.0, 3.0],
                "target": [0.0, 0.0, 0.65],
                "focal_length_mm": 42.0,
                "sensor_width_mm": 36.0,
            },
            "camera_motion": {"fractions": {mode: 0.25 for mode in MODES}, "seed": 5},
        }
    )
    time, position = _ballistic_object()
    modes = [generate_camera_path(time, position, 0.1, config, index).mode for index in range(200)]
    counts = {mode: modes.count(mode) for mode in MODES}
    assert all(30 <= count <= 70 for count in counts.values()), counts


def test_object_leaving_the_image_is_rejected():
    config = _config("static")
    time, _ = _ballistic_object()
    location, target = np.asarray(config.camera.location), np.asarray(config.camera.target)
    rotation = look_at_opencv(location, target)
    # 3 m to the side at 4 m depth: about 37 degrees off-axis, outside the 23 degree half-angle.
    beside = location + 4.0 * rotation[:, 2] + 3.0 * rotation[:, 0]
    assert (
        generate_camera_path(time, np.repeat(beside[None], len(time), axis=0), 0.1, config, 0)
        is None
    )


def test_lag_filter_follows_a_ramp_with_constant_delay_from_the_start():
    dt, lag, speed = 0.0005, 0.08, 2.0
    time = np.arange(400) * dt
    signal = np.column_stack((speed * time, np.zeros_like(time), np.zeros_like(time)))
    output = critically_damped_lag(signal, dt, lag)
    np.testing.assert_allclose(output[:, 0], signal[:, 0] - lag * speed, atol=1e-9)

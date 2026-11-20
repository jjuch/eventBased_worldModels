"""Exactness of the interval targets and the rollout on an analytic ballistic trajectory."""

import torch

from ball_world_model.geometry.se3 import rollout_world, so3_exp
from ball_world_model.training.targets import window_targets


def _matrix_to_quaternion_xyzw(rotation: torch.Tensor) -> torch.Tensor:
    """Valid for rotations far from 180 degrees, which is all this test needs."""
    w = 0.5 * torch.sqrt(1.0 + rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2])
    x = (rotation[..., 2, 1] - rotation[..., 1, 2]) / (4 * w)
    y = (rotation[..., 0, 2] - rotation[..., 2, 0]) / (4 * w)
    z = (rotation[..., 1, 0] - rotation[..., 0, 1]) / (4 * w)
    return torch.stack((x, y, z, w), dim=-1)


def _ballistic_batch(frames: int = 8, dt: float = 0.01) -> dict[str, torch.Tensor]:
    """Constant gravity, constant spin about a fixed axis (torque-free sphere)."""
    dtype = torch.float64
    time = torch.arange(frames, dtype=dtype) * dt
    p0 = torch.tensor([0.1, -0.2, 0.6], dtype=dtype)
    v0 = torch.tensor([1.0, 0.5, 2.0], dtype=dtype)
    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=dtype)
    omega = torch.tensor([3.0, -12.0, 20.0], dtype=dtype)
    r0 = so3_exp(torch.tensor([0.2, 0.1, -0.3], dtype=dtype))

    position = p0 + time[:, None] * v0 + 0.5 * time[:, None] ** 2 * gravity
    velocity = v0 + time[:, None] * gravity
    rotation = so3_exp(time[:, None] * omega) @ r0
    return {
        "context_time": time[None],
        "context_position": position[None],
        "context_linear_velocity": velocity[None],
        "context_angular_velocity": omega.expand(frames, 3)[None],
        "context_quaternion_xyzw": _matrix_to_quaternion_xyzw(rotation)[None],
    }


def test_interval_targets_are_exact_for_ballistic_motion():
    batch = _ballistic_batch()
    targets = window_targets(batch)
    velocity = batch["context_linear_velocity"].float()

    # Mean velocity over an interval equals the average of its endpoint velocities under constant acceleration.
    expected_mean_velocity = 0.5 * (velocity[:, 1:] + velocity[:, :-1])
    assert torch.allclose(targets.interval_linear_velocity, expected_mean_velocity, atol=1e-5)
    assert torch.allclose(targets.interval_linear_acceleration, torch.tensor([0.0, 0.0, -9.81]).expand(1, 7, 3), atol=1e-4)
    assert torch.allclose(targets.interval_angular_velocity, torch.tensor([3.0, -12.0, 20.0]).expand(1, 7, 3), atol=1e-4)
    assert torch.allclose(targets.interval_angular_acceleration, torch.zeros(1, 7, 3), atol=1e-6)


def test_rollout_reproduces_ballistic_motion():
    batch = _ballistic_batch(frames=6)
    targets = window_targets(batch)
    current = targets.current
    horizons = batch["context_time"][:, 1:].float()  # rollout from frame 0
    positions, rotations = rollout_world(
        targets.frame_position[:, 0], targets.frame_rotation[:, 0],
        targets.frame_linear_velocity[:, 0], targets.frame_angular_velocity[:, 0],
        current.linear_acceleration, current.angular_acceleration, horizons,
    )
    assert torch.allclose(positions, targets.frame_position[:, 1:], atol=1e-5)
    assert torch.allclose(rotations, targets.frame_rotation[:, 1:], atol=1e-5)


def test_perfect_prediction_has_zero_loss_including_kinematic_consistency():
    """Feeding the targets back as predictions must give (near) zero for every term."""
    from ball_world_model.models.observer import current_state
    from ball_world_model.models.state import Acceleration, ObserverPrediction, Pose, Twist
    from ball_world_model.training.losses import LossSettings, observer_losses

    targets = window_targets(_ballistic_batch())
    poses = Pose(targets.frame_position, targets.frame_rotation)
    twists = Twist(targets.interval_linear_velocity, targets.interval_angular_velocity)
    accelerations = Acceleration(targets.interval_linear_acceleration, targets.interval_angular_acceleration)
    prediction = ObserverPrediction(
        frame_poses=poses,
        interval_twists=twists,
        interval_accelerations=accelerations,
        interval_dt=targets.interval_dt,
        current=current_state(poses, twists, accelerations, targets.interval_dt),
        keypoint_coordinates=torch.zeros(1, 8, 4, 2),
    )
    scales = {name: torch.ones(3) for name in (
        "position", "linear_velocity", "angular_velocity", "linear_acceleration", "angular_acceleration")}
    total, components = observer_losses(prediction, targets, scales, LossSettings(rollout=0.0, kinematic=1.0))
    assert components["kinematic_linear"] < 1e-8
    assert components["kinematic_angular"] < 1e-8
    assert total < 1e-6
"""Observer losses. Every term is dimensionless and O(1) when errors are of typical size.

    position, linear/angular velocity, linear/angular acceleration:
        squared error divided by the training-set std of that quantity
    orientation:
        squared angle divided by orientation_scale^2
    rollout:
        the current state is propagated with constant acceleration to the
        future frames; its pose errors are scaled like the pose terms

The total is
    pose * (position + orientation) + twist * (linear + angular velocity)
    + acceleration * (linear + angular acceleration) + rollout * (rollout position + orientation)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ball_world_model.geometry.se3 import chordal_angle_squared, rollout_world, rotation_angle
from ball_world_model.models.state import ObserverPrediction
from .targets import FutureTargets, WindowTargets

@dataclass(frozen=True)
class LossSettings:
    pose: float = 1.0
    twist: float = 1.0
    acceleration: float = 1.0
    rollout: float = 0.1
    rollout_steps: int = 5
    orientation_scale_deg: float = 5.0

    @property
    def orientation_scale_rad(self) -> float:
        return math.radians(self.orientation_scale_deg)


def scaled_mse(prediction: Tensor, target: Tensor, scale: Tensor) -> Tensor:
    return ((prediction - target) / scale).square().mean()


def orientation_loss(prediction: Tensor, target: Tensor, scale_rad: float) -> Tensor:
    return chordal_angle_squared(prediction, target).mean() / scale_rad**2


def observer_losses(
    prediction: ObserverPrediction,
    targets: WindowTargets,
    scales: dict[str, Tensor],
    settings: LossSettings,
    future: FutureTargets | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    zero = prediction.frame_poses.position.new_zeros(())
    poses, twists, accelerations = (
        prediction.frame_poses, prediction.interval_twists, prediction.interval_accelerations
    )

    components = {
        "position": scaled_mse(poses.position, targets.frame_position, scales["position"]),
        "orientation": orientation_loss(poses.rotation, targets.frame_rotation, settings.orientation_scale_rad),
        "linear_velocity": scaled_mse(twists.linear, targets.interval_linear_velocity, scales["linear_velocity"]),
        "angular_velocity": scaled_mse(twists.angular, targets.interval_angular_velocity, scales["angular_velocity"]),
        "linear_acceleration": scaled_mse(
            accelerations.linear, targets.interval_linear_acceleration, scales["linear_acceleration"]
        ),
        "angular_acceleration": scaled_mse(
            accelerations.angular, targets.interval_angular_acceleration, scales["angular_acceleration"]
        ),
        "rollout_position": zero,
        "rollout_orientation": zero,
    }

    if settings.rollout > 0 and future is not None:
        state = prediction.current
        rollout_position, rollout_rotation = rollout_world(
            state.pose.position, state.pose.rotation,
            state.twist.linear, state.twist.angular,
            state.acceleration.linear, state.acceleration.angular,
            future.horizons,
        )
        components["rollout_position"] = scaled_mse(rollout_position, future.position, scales["position"])
        components["rollout_orientation"] = orientation_loss(
            rollout_rotation, future.rotation, settings.orientation_scale_rad
        )
    
    total = (
        settings.pose * (components["position"] + components["orientation"])
        + settings.twist * (components["linear_velocity"] + components["angular_velocity"])
        + settings.acceleration * (components["linear_acceleration"] + components["angular_acceleration"])
        + settings.rollout * (components["rollout_position"] + components["rollout_orientation"])
    )
    return total, components


@torch.no_grad()
def physical_metrics(prediction: ObserverPrediction, targets: WindowTargets) -> dict[str, Tensor]:
    """Errors in physical units for logging: frame/interval quantities and the current state."""

    def rmse(error: Tensor) -> Tensor:
        return error.square().sum(-1).mean().sqrt()

    current, current_target = prediction.current, targets.current
    return {
        "position_rmse_m": rmse(prediction.frame_poses.position - targets.frame_position),
        "orientation_error_deg": torch.rad2deg(
            rotation_angle(prediction.frame_poses.rotation, targets.frame_rotation)
        ).mean(),
        "interval_linear_velocity_rmse_mps": rmse(
            prediction.interval_twists.linear - targets.interval_linear_velocity
        ),
        "interval_angular_velocity_rmse_radps": rmse(
            prediction.interval_twists.angular - targets.interval_angular_velocity
        ),
        "interval_linear_acceleration_rmse_mps2": rmse(
            prediction.interval_accelerations.linear - targets.interval_linear_acceleration
        ),
        "interval_angular_acceleration_rmse_radps2": rmse(
            prediction.interval_accelerations.angular - targets.interval_angular_acceleration
        ),
        "current_linear_velocity_rmse_mps": rmse(current.twist.linear - current_target.linear_velocity),
        "current_angular_velocity_rmse_radps": rmse(current.twist.angular - current_target.angular_velocity),
    }
"""Observer losses. Every term is dimensionless and O(1) when errors are of typical size.

    position, linear/angular velocity, linear/angular acceleration:
        squared error divided by the training-set std of that quantity
    orientation:
        squared angle divided by orientation_scale^2
    rollout:
        the current state is propagated with constant acceleration to the
        future frames; its pose errors are scaled like the pose terms
    kinematic (optional, default weight 0):
        the predicted velocity change between neighbouring intervals must match the predicted accelerations, (xi[k+1] - xi[k]) = dt_centre * (xi_dot[k] + xi_dot[k+1]) / 2;
        the error is a velocity, so it is scaled by the velocity std

The total is
    pose * (position + orientation) + twist * (linear + angular velocity)
    + acceleration * (linear + angular acceleration) + rollout * (rollout position + orientation) + kinematic * (linear + angular kinematic consistency)
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
    kinematic: float = 0.0

    @property
    def orientation_scale_rad(self) -> float:
        return math.radians(self.orientation_scale_deg)


def scaled_mse(prediction: Tensor, target: Tensor, scale: Tensor) -> Tensor:
    return ((prediction - target) / scale).square().mean()


def orientation_loss(prediction: Tensor, target: Tensor, scale_rad: float) -> Tensor:
    return chordal_angle_squared(prediction, target).mean() / scale_rad**2


def kinematic_consistency(rates: Tensor, accelerations: Tensor, dt: Tensor, velocity_scale: Tensor) -> Tensor:
    """Mismatch between the velocity change across neighbouring intervals and the accelerations.

    rates [B, I, 3] interval-mean velocities, accelerations [B, I, 3] interval-mean accelerations, dt [B, I]. Interval means sit at interval centres, which are dt_centre = (dt[k] + dt[k+1]) / 2 apart; under a linearly varying acceleration the velocity change between them is dt_centre * (a[k] + a[k+1]) / 2.
    The error is a velocity and is scaled by the velocity std, so the term stays well conditioned even when the data contain no acceleration.
    """
    centre_dt = 0.5 * (dt[:, 1:] + dt[:, :-1]).unsqueeze(-1)
    velocity_change = rates[:, 1:] - rates[:, :-1]
    implied_change = centre_dt * 0.5 * (accelerations[:, 1:] + accelerations[:, :-1])
    return scaled_mse(velocity_change, implied_change, velocity_scale)


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
        "kinematic_linear": zero,
        "kinematic_angular": zero,
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

    if settings.kinematic > 0:
        components["kinematic_linear"] = kinematic_consistency(
            twists.linear, accelerations.linear, prediction.interval_dt, scales["linear_velocity"]
        )
        components["kinematic_angular"] = kinematic_consistency(
            twists.angular, accelerations.angular, prediction.interval_dt, scales["angular_velocity"]
        )
    
    total = (
        settings.pose * (components["position"] + components["orientation"])
        + settings.twist * (components["linear_velocity"] + components["angular_velocity"])
        + settings.acceleration * (components["linear_acceleration"] + components["angular_acceleration"])
        + settings.rollout * (components["rollout_position"] + components["rollout_orientation"])
        + settings.kinematic * (components["kinematic_linear"] + components["kinematic_angular"])
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
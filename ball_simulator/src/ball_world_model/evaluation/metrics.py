from __future__ import annotations

from torch import Tensor

from ball_world_model.geometry.se3 import rotation_angle
from ball_world_model.models.state import ObserverPrediction
from ball_world_model.training.targets import CurrentTargets


def rmse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean(0).sqrt()


def r2(prediction: Tensor, target: Tensor) -> Tensor:
    """Per-component R^2; meaningless (very negative) when the target barely varies."""
    residual = (prediction - target).square().sum(0)
    total = (target - target.mean(0)).square().sum(0).clamp_min(1.0e-12)
    return 1.0 - residual / total

def state_metrics(prediction: ObserverPrediction, target: CurrentTargets) -> dict[str, Tensor]:
    """Metrics of the current-state estimate (last context frame)."""
    state = prediction.current
    return {
        "position_rmse": rmse(state.pose.position, target.position),
        "orientation_rmse_rad": rotation_angle(target.rotation, state.pose.rotation).square().mean().sqrt(),
        "linear_velocity_rmse": rmse(state.twist.linear, target.linear_velocity),
        "angular_velocity_rmse": rmse(state.twist.angular, target.angular_velocity),
        "linear_acceleration_rmse": rmse(state.acceleration.linear, target.linear_acceleration),
        "angular_acceleration_rmse": rmse(state.acceleration.angular, target.angular_acceleration),
        "linear_velocity_r2": r2(state.twist.linear, target.linear_velocity),
        "angular_velocity_r2": r2(state.twist.angular, target.angular_velocity),
        "linear_acceleration_r2": r2(state.acceleration.linear, target.linear_acceleration),
        "angular_acceleration_r2": r2(state.acceleration.angular, target.angular_acceleration),
    }
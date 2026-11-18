from __future__ import annotations

import torch
from torch import Tensor

from ball_world_model.geometry.se3 import so3_log

def rmse(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt((prediction - target).square().mean(0))

def r2(prediction: Tensor, target: Tensor) -> Tensor:
    residual = (prediction - target).square().sum(0)
    total = (target - target.mean(0)).square().sum(0).clamp_min(1.0e-12)
    return 1.0 - residual / total

def state_metrics(prediction, target) -> dict[str, Tensor]:
    return {
        "position_rmse": rmse(prediction.pose.position, target.position),
        "orientation_rmse_rad": torch.sqrt(so3_log(target.rotation.transpose(-1, -2) @ prediction.pose.rotation).square().sum(-1).mean()),
        "linear_velocity_rmse": rmse(prediction.twist.linear, target.linear_velocity),
        "angular_velocity_rmse": rmse(prediction.twist.angular, target.angular_velocity),
        "linear_acceleration_rmse": rmse(prediction.acceleration.linear, target.linear_acceleration),
        "angular_acceleration_rmse": rmse(prediction.acceleration.angular, target.angular_acceleration),
        "linear_velocity_r2": r2(prediction.twist.linear, target.linear_velocity),
        "angular_velocity_r2": r2(prediction.twist.angular, target.angular_velocity),
        "linear_acceleration_r2": r2(prediction.acceleration.linear, target.linear_acceleration),
        "angular_acceleration_r2": r2(prediction.acceleration.angular, target.angular_acceleration),
    }
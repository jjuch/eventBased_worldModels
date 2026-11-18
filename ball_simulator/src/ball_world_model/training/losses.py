"""Compact loss set for physical state and coupled SE(3) carriers."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor

from ball_world_model.geometry.se3 import propagate_world, so3_log
from ball_world_model.models.state import ObserverPrediction
from ball_world_model.models.teacher import SE3Teacher
from .targets import CurrentTargets

@dataclass(frozen=True)
class LossWeights:
    pose: float = 1.0
    twist: float = 1.0
    acceleration: float = 1.0
    carrier: float = 0.05
    carrier_velocity: float = 0.05
    carrier_acceleration: float = 0.05
    rollout: float = 0.05


def scaled_mse(prediction: Tensor, target: Tensor, scale: Tensor) -> Tensor:
    return F.mse_loss((prediction - target) / scale.clamp_min(1.0e-6), torch.zeros_like(target))


def observer_losses(
    prediction: ObserverPrediction,
    target: CurrentTargets,
    teacher: SE3Teacher,
    weights: LossWeights,
    scales: dict[str, Tensor],
    previous: CurrentTargets | None = None,
    dt: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    target_carriers = teacher(
        target.position, 
        target.rotation, 
        target.linear_velocity, 
        target.angular_velocity,
        target.linear_acceleration, 
        target.angular_acceleration,
    )

    position = scaled_mse(prediction.pose.position, target.position, scales["position"])
    orientation = so3_log(target.rotation.transpose(-1, -2) @ prediction.pose.rotation).square().sum(-1).mean()

    linear_velocity = scaled_mse(prediction.twist.linear, target.linear_velocity, scales["linear_velocity"])
    angular_velocity = scaled_mse(prediction.twist.angular, target.angular_velocity, scales["angular_velocity"])

    linear_acceleration = scaled_mse(prediction.acceleration.linear, target.linear_acceleration, scales["linear_acceleration"])
    angular_acceleration = scaled_mse(prediction.acceleration.angular, target.angular_acceleration, scales["angular_acceleration"])

    carrier = F.mse_loss(prediction.carrier, target_carriers.carrier)
    carrier_velocity = F.mse_loss(prediction.carrier_velocity, target_carriers.carrier_velocity)
    carrier_acceleration = F.mse_loss(prediction.carrier_acceleration, target_carriers.carrier_acceleration)

    rollout = position.new_zeros(())
    if previous is not None and dt is not None:
        p, r, v, w = propagate_world(
            previous.position, previous.rotation, previous.linear_velocity, previous.angular_velocity, dt,
            prediction.acceleration.linear, prediction.acceleration.angular,
        )
        rollout = (
            scaled_mse(p, target.position, scales["position"])
            + so3_log(target.rotation.transpose(-1, -2) @ r).square().sum(-1).mean()
            + scaled_mse(v, target.linear_velocity, scales["linear_velocity"])
            + scaled_mse(w, target.angular_velocity, scales["angular_velocity"])
        )
    components = {
        "position": position, "orientation": orientation,
        "linear_velocity": linear_velocity, "angular_velocity": angular_velocity,
        "linear_acceleration": linear_acceleration, "angular_acceleration": angular_acceleration,
        "carrier": carrier, "carrier_velocity": carrier_velocity,
        "carrier_acceleration": carrier_acceleration, "rollout": rollout,
    }
    
    total = (
        weights.pose * (position + orientation)
        + weights.twist * (linear_velocity + angular_velocity)
        + weights.acceleration * (linear_acceleration + angular_acceleration)
        + weights.carrier * carrier
        + weights.carrier_velocity * carrier_velocity
        + weights.carrier_acceleration * carrier_acceleration
        + weights.rollout * rollout
    )
    return total, components

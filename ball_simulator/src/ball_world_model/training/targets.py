"""Current-state targets and acceleration fallback for existing datasets."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from ball_world_model.geometry.se3 import rotation_6d_to_matrix

@dataclass(frozen=True)
class CurrentTargets:
    position: Tensor
    rotation: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor
    linear_acceleration: Tensor
    angular_acceleration: Tensor


def quaternion_xyzw_to_matrix(q: Tensor) -> Tensor:
    q = torch.nn.functional.normalize(q, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def _latest_rate(values: Tensor, time: Tensor) -> Tensor:
    dt = (time[:, -1] - time[:, -2]).clamp_min(1.0e-6).unsqueeze(-1)
    return (values[:, -1] - values[:, -2]) / dt


def current_targets(batch: dict[str, Tensor]) -> CurrentTargets:
    """Build targets at the latest observed time.

    Prefer simulator-provided accelerations. Finite differences are a compatibility fallback and should not be used for impulsive/contact experiments.
    """
    time = batch["context_time"]
    linear = batch["context_linear_velocity"]
    angular = batch["context_angular_velocity"]
    linear_acceleration = batch.get("context_linear_acceleration")
    angular_acceleration = batch.get("context_angular_acceleration")
    if linear_acceleration is None:
        linear_acceleration = _latest_rate(linear, time)
    else:
        linear_acceleration = linear_acceleration[:, -1]
    if angular_acceleration is None:
        angular_acceleration = _latest_rate(angular, time)
    else:
        angular_acceleration = angular_acceleration[:, -1]
        
    quaternion = batch["context_quaternion_xyzw"][:, -1]
    return CurrentTargets(
        position=batch["context_position"][:, -1],
        rotation=quaternion_xyzw_to_matrix(quaternion),
        linear_velocity=linear[:, -1], angular_velocity=angular[:, -1],
        linear_acceleration=linear_acceleration,
        angular_acceleration=angular_acceleration,
    )

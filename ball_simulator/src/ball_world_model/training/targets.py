"""Ground-truth targets for one context window, built from the simulator states.

Interval targets are means over each interval and are computed from the stored states without approximation:

    mean linear velocity      = (p[k+1] - p[k]) / dt
    mean angular velocity     = log(R[k+1] R[k]^T) / dt      (world frame)
    mean linear acceleration  = (v[k+1] - v[k]) / dt
    mean angular acceleration = (omega[k+1] - omega[k]) / dt

The rotation-vector form is exact when the rotation axis is constant over the interval, which holds for torque-free spheres and is a tight approximation at 100 fps otherwise. No acceleration needs to be stored in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ball_world_model.geometry.se3 import so3_log


def quaternion_xyzw_to_matrix(q: Tensor) -> Tensor:
    q = torch.nn.functional.normalize(q, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


@dataclass(frozen=True)
class CurrentTargets:
    """Instantaneous state at the last context frame, [B, ...].

    The acceleration is the mean over the last interval, because the dataset stores no instantaneous acceleration (identical for constant acceleration).
    """
    position: Tensor
    rotation: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor
    linear_acceleration: Tensor
    angular_acceleration: Tensor


@dataclass(frozen=True)
class WindowTargets:
    """Frame targets [B, T, ...] and interval-mean targets [B, I, 3]."""
 
    frame_position: Tensor
    frame_rotation: Tensor
    frame_linear_velocity: Tensor
    frame_angular_velocity: Tensor
    interval_dt: Tensor
    interval_linear_velocity: Tensor
    interval_angular_velocity: Tensor
    interval_linear_acceleration: Tensor
    interval_angular_acceleration: Tensor

    @property
    def current(self) -> CurrentTargets:
        return CurrentTargets(
            position=self.frame_position[:, -1],
            rotation=self.frame_rotation[:, -1],
            linear_velocity=self.frame_linear_velocity[:, -1],
            angular_velocity=self.frame_angular_velocity[:, -1],
            linear_acceleration=self.interval_linear_acceleration[:, -1],
            angular_acceleration=self.interval_angular_acceleration[:, -1],
        )

@dataclass(frozen=True)
class FutureTargets:
    """Poses after the context window: horizons [B, F] in s since the last context frame."""
    horizons: Tensor
    position: Tensor
    rotation: Tensor


def _interval_rate(values: Tensor, dt: Tensor) -> Tensor:
    return (values[:, 1:] - values[:, :-1]) / dt.unsqueeze(-1)


def window_targets(batch: dict[str, Tensor]) -> WindowTargets:
    # Finite differences in float64: the per-frame changes are small compared to the values.
    time = batch["context_time"].double()
    position = batch["context_position"].double()
    rotation = quaternion_xyzw_to_matrix(batch["context_quaternion_xyzw"].double())
    linear_velocity = batch["context_linear_velocity"].double()
    angular_velocity = batch["context_angular_velocity"].double()

    dt = torch.diff(time, dim=1).clamp_min(1.0e-9)
    relative_rotation = rotation[:, 1:] @ rotation[:, :-1].transpose(-1, -2)
    targets = {
        "frame_position": position,
        "frame_rotation": rotation,
        "frame_linear_velocity": linear_velocity,
        "frame_angular_velocity": angular_velocity,
        "interval_dt": dt,
        "interval_linear_velocity": _interval_rate(position, dt),
        "interval_angular_velocity": so3_log(relative_rotation) / dt.unsqueeze(-1),
        "interval_linear_acceleration": _interval_rate(linear_velocity, dt),
        "interval_angular_acceleration": _interval_rate(angular_velocity, dt),
    }
    return WindowTargets(**{name: value.float() for name, value in targets.items()})


def future_targets(batch: dict[str, Tensor], steps: int) -> FutureTargets:
    """First ``steps`` future poses relative to the last context frame."""
    available = batch["future_time"].shape[1]
    if steps > available:
        raise ValueError(f"Requested {steps} rollout steps but the batch holds {available} future frames.")
    horizons = batch["future_time"][:, :steps] - batch["context_time"][:, -1:]
    return FutureTargets(
        horizons=horizons.float(),
        position=batch["future_position"][:, :steps].float(),
        rotation=quaternion_xyzw_to_matrix(batch["future_quaternion_xyzw"][:, :steps].float()),
     )
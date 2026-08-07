from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class StateStatistics:
    position_mean: torch.Tensor
    position_std: torch.Tensor
    linear_velocity_mean: torch.Tensor
    linear_velocity_std: torch.Tensor
    angular_velocity_mean: torch.Tensor
    angular_velocity_std: torch.Tensor


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    return values.std(dim=0, unbiased=False).clamp_min(1.0e-6)


def compute_state_statics(loader: DataLoader) -> StateStatistics:
    positions: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    angular_velocities: list[torch.Tensor] = []

    for batch in loader:
        positions.append(batch["context_position"][:, -1].float())
        velocities.append(batch["context_linear_velocity"][:, -1].float())
        angular_velocities.append(batch["context_angular_velocity"][:, -1].float())

    position = torch.cat(positions)
    velocity = torch.cat(velocities)
    angular_velocity = torch.cat(angular_velocities)
    return StateStatistics(
        position.mean(0), _safe_std(position),
        velocity.mean(0), _safe_std(velocity),
        angular_velocity.mean(0), _safe_std(angular_velocity),
    )
"""Streaming physical statistics used only for regression conditioning."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from .targets import current_targets

@dataclass(frozen=True)
class StateStatistics:
    means: dict[str, Tensor]
    stds: dict[str, Tensor]

@torch.no_grad()
def compute_statistics(loader) -> StateStatistics:
    values = {name: [] for name in (
        "position", "linear_velocity", "angular_velocity",
        "linear_acceleration", "angular_acceleration",
    )}

    for batch in loader:
        target = current_targets(batch)
        for name in values:
            values[name].append(getattr(target, name).float().cpu())

    joined = {name: torch.cat(items) for name, items in values.items()}
    means = {name: value.mean(0) for name, value in joined.items()}
    stds = {name: value.std(0, unbiased=False).clamp_min(1.0e-6) for name, value in joined.items()}
    
    return StateStatistics(means, stds)
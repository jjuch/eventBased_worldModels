"""Training-set statistics that standardise the observer heads and losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .targets import WindowTargets, window_targets

# Model quantity name -> the target it is compared with.
TARGET_FIELDS = {
    "position": "frame_position",
    "linear_velocity": "interval_linear_velocity",
    "angular_velocity": "interval_angular_velocity",
    "linear_acceleration": "interval_linear_acceleration",
    "angular_acceleration": "interval_angular_acceleration",
}

@dataclass(frozen=True)
class StateStatistics:
    means: dict[str, Tensor]
    stds: dict[str, Tensor]
    informative: dict[str, bool] # False when a quantity never varies


@torch.no_grad()
def compute_statistics(loader, minimum_std: float = 1.0e-3, informative_std: float = 1.0e-6) -> StateStatistics:
    """Per-component mean and std of every target over the loader.
 
    ``minimum_std`` floors the scale of quantities that never vary. Because the heads are decoded with the same std, the loss of such a quantity reduces to the squared raw output and stays well conditioned.
    """
    values: dict[str, list[Tensor]] = {name: [] for name in TARGET_FIELDS}

    for batch in loader:
        targets: WindowTargets = window_targets(batch)
        for name, field in TARGET_FIELDS.items():
            values[name].append(getattr(targets, field).reshape(-1, 3).cpu())

    joined = {name: torch.cat(parts) for name, parts in values.items()}
    raw_stds = {name: value.std(0, unbiased=False) for name, value in joined.items()}
    
    return StateStatistics(
        means={name: value.mean(0) for name, value in joined.items()},
        stds={name: std.clamp_min(minimum_std) for name, std in raw_stds.items()},
        informative={name: bool(std.max() > informative_std) for name, std in raw_stds.items()},
    )
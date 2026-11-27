"""Training-set statistics that standardise the observer heads and losses, per body and quantity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .targets import AnchorWindowTargets, anchor_window_targets


def target_values(targets: AnchorWindowTargets) -> dict[str, Tensor]:
    """Every standardised quantity's targets, flattened to [N, 3].

    The camera's last-frame pose is identity by construction and is left out.
    """
    values = {
        "object_in_camera_position": targets.object_in_camera.position,
        "camera_position": targets.camera.frame_position[:, :-1],
        "gravity": targets.gravity,
    }
    for body_name, body in (("object", targets.object), ("camera", targets.camera)):
        for kind in ("linear", "angular"):
            values[f"{body_name}_{kind}_velocity"] = getattr(body, f"interval_{kind}_velocity")
            values[f"{body_name}_{kind}_acceleration"] = getattr(
                body, f"interval_{kind}_acceleration"
            )
    return {name: value.reshape(-1, 3) for name, value in values.items()}


@dataclass(frozen=True)
class StateStatistics:
    means: dict[str, Tensor]
    stds: dict[str, Tensor]
    informative: dict[str, bool]  # False when a quantity never varies


@torch.no_grad()
def compute_statistics(
    loader, minimum_std: float = 1.0e-3, informative_std: float = 1.0e-6
) -> StateStatistics:
    """Per-component mean and std of every standardised quantity over the loader.

    ``minimum_std`` floors the scale of quantities that never vary (e.g. the camera of a
    fixed-camera dataset). Because the heads are decoded with the same std, the loss of such a
    quantity reduces to the squared raw output and stays well conditioned.
    """
    parts: dict[str, list[Tensor]] = {}
    for batch in loader:
        for name, value in target_values(anchor_window_targets(batch)).items():
            parts.setdefault(name, []).append(value.cpu())

    joined = {name: torch.cat(values) for name, values in parts.items()}
    raw_stds = {name: value.std(0, unbiased=False) for name, value in joined.items()}
    return StateStatistics(
        means={name: value.mean(0) for name, value in joined.items()},
        stds={name: std.clamp_min(minimum_std) for name, std in raw_stds.items()},
        informative={name: bool(std.max() > informative_std) for name, std in raw_stds.items()},
    )

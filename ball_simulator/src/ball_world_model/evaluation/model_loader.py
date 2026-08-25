from __future__ import annotations

from pathlib import Path

import torch

from ball_world_model.training.kinematic_module import (
    KinematicObservabilityModule,
    KinematicStatistics,
)

_STATISTIC_NAMES = (
    "position_mean",
    "position_std",
    "linear_velocity_mean",
    "linear_velocity_std",
    "angular_velocity_mean",
    "angular_velocity_std",
)


def load_kinematic_module(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> KinematicObservabilityModule:
    """Reconstruct the Lightning module, including training-set normalisation."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    missing = [name for name in _STATISTIC_NAMES if name not in state_dict]
    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing normalization buffers: {missing}"
        )

    statistics = KinematicStatistics(
        **{name: state_dict[name].detach().cpu() for name in _STATISTIC_NAMES}
    )
    hyperparameters = dict(checkpoint.get("hyper_parameters", {}))
    allowed = {
        "task",
        "embedding_dim",
        "temporal_depth",
        "keypoints",
        "dropout",
        "learning_rate",
        "weight_decay",
        "state_weight",
        "translation_consistency_weight",
        "rotation_consistency_weight",
        "world_angular_velocity",
    }
    arguments = {key: value for key, value in hyperparameters.items() if key in allowed}
    module = KinematicObservabilityModule(statistics, **arguments)
    module.load_state_dict(state_dict, strict=True)
    module.eval()
    module.to(device)
    return module


def denormalised_prediction(module, prediction):
    result = {}
    if prediction.position is not None:
        result["position"] = module._denormalise(
            prediction.position, module.position_mean, module.position_std
        )
    if prediction.linear_velocity is not None:
        result["linear_velocity"] = module._denormalise(
            prediction.linear_velocity,
            module.linear_velocity_mean,
            module.linear_velocity_std,
        )
    if prediction.angular_velocity is not None:
        result["angular_velocity"] = module._denormalise(
            prediction.angular_velocity,
            module.angular_velocity_mean,
            module.angular_velocity_std,
        )
    if prediction.rotation_matrix is not None:
        result["rotation_matrix"] = prediction.rotation_matrix
    return result

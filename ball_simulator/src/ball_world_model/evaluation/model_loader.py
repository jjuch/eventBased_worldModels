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

_ALLOWED_HYPERPARAMETERS = {
    "task",
    "embedding_dim",
    "keypoints",
    "motion_dim",
    "decoder_hidden_dim",
    "refinement_hidden_dim",
    "refinement_iterations",
    "delta_momentum",
    "default_frame_dt",
    "dropout",
    "temporal_depth",
    "learning_rate",
    "weight_decay",
    "position_weight",
    "velocity_weight",
    "latent_prediction_weight",
    "kinematic_weight",
    "reverse_weight",
    "variance_weight",
    "rotation_weight",
    "angular_velocity_weight",
    "state_weight",
    "translation_consistency_weight",
    "rotation_consistency_weight",
    "world_angular_velocity",
}

def _statistics_from_state_dict(state_dict: dict[str, torch.Tensor]) -> KinematicStatistics:
    missing = [name for name in _STATISTIC_NAMES if name not in state_dict]
    if missing:
        raise ValueError(f"Checkpoint is missing normalisation buffers: {missing}.")
    return KinematicStatistics(
        **{name: state_dict[name].detach().cpu() for name in _STATISTIC_NAMES}
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

    statistics = _statistics_from_state_dict(state_dict)
    hyperparameters = dict(checkpoint.get("hyper_parameters", {}))
    arguments = {key: value for key, value in hyperparameters.items() if key in _ALLOWED_HYPERPARAMETERS}
    module = KinematicObservabilityModule(statistics, **arguments)
    module.load_state_dict(state_dict, strict=True)
    module.eval()
    module.to(device)
    return module


def denormalised_prediction(module, prediction) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
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

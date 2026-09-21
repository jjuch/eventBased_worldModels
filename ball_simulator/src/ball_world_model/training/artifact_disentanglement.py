"""Loss utilities for physics-free residual artifact sectors."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ball_world_model.models.artifact_residual import cross_covariance_loss
from ball_world_model.models.rotation import rotation_6d_to_matrix, rotation_geodesic_error


def artifact_disentanglement_loss(
    *,
    artifact: torch.Tensor,
    reconstructed_residual: torch.Tensor,
    residual_target: torch.Tensor,
    adversary_omega: torch.Tensor,
    adversary_relative_rotation_6d: torch.Tensor,
    target_omega_normalised: torch.Tensor,
    target_relative_rotation: torch.Tensor,
    physical_motion: torch.Tensor,
) -> dict[str, torch.Tensor]:
    residual_reconstruction = F.smooth_l1_loss(
        reconstructed_residual, residual_target.detach()
    )
    adversary_omega_loss = F.mse_loss(adversary_omega, target_omega_normalised)
    adversary_rotation_loss = rotation_geodesic_error(
        rotation_6d_to_matrix(adversary_relative_rotation_6d),
        target_relative_rotation,
    ).square().mean()
    covariance = cross_covariance_loss(physical_motion, artifact)
    return {
        "artifact_residual_reconstruction_loss": residual_reconstruction,
        "artifact_adversary_omega_loss": adversary_omega_loss,
        "artifact_adversary_rotation_loss": adversary_rotation_loss,
        "artifact_cross_covariance_loss": covariance,
    }
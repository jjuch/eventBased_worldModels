"""Scale-balanced losses and diagnostics for a unified SE(3) twist."""
from __future__ import annotations

import torch

from ball_world_model.models.rotation import rotation_geodesic_error

def safe_scale(scale: torch.Tensor, minimum: float = 1.0e-6) -> torch.Tensor:
    return scale.detach().abs().clamp_min(minimum)

def standardised_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scale = safe_scale(scale).to(prediction)
    return (prediction - target) / scale


def balanced_mean(*losses: torch.Tensor | None) -> torch.Tensor:
    active = [loss for loss in losses if loss is not None]
    if not active:
        raise ValueError("balanced_mean required at least one active loss.")
    return torch.stack(active).mean()


def dimensionless_twist_loss(
    predicted_velocity: torch.Tensor | None,
    target_velocity: torch.Tensor | None,
    velocity_scale: torch.Tensor | None,
    predicted_omega: torch.Tensor | None,
    target_omega: torch.Tensor | None,
    omega_scale: torch.Tensor | None,
) -> torch.Tensor:
    errors = []
    if predicted_velocity is not None:
        errors.append(standardised_error(predicted_velocity, target_velocity, velocity_scale))
    if predicted_omega is not None:
        errors.append(standardised_error(predicted_omega, target_omega, omega_scale))
    if not errors:
        raise ValueError("At least one twist sector must be active")
    return torch.cat(errors, dim=-1).square().mean()

def dimensionless_reverse_loss(
    forward_velocity: torch.Tensor | None,
    backward_velocity: torch.Tensor | None,
    velocity_scale: torch.Tensor | None,
    forward_omega: torch.Tensor | None,
    backward_omega: torch.Tensor | None,
    omega_scale: torch.Tensor | None,
) -> torch.Tensor:
    errors = []
    if forward_velocity is not None:
        errors.append((forward_velocity + backward_velocity) / safe_scale(velocity_scale).to(forward_velocity))
    if forward_omega is not None:
        errors.append((forward_omega + backward_omega) / safe_scale(omega_scale).to(forward_omega))
    if not errors:
        raise ValueError("At least one twist sector must be active")
    return torch.cat(errors, dim=-1).square().mean()


def rms_normalised_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_scale: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a dimensionless loss and the detached RMS target scale.

    One scalar scale is used per active loss family. This prevents large
    rotational tangents from dominating translational tangents while retaining the relative weighting of coordinates within each geometric sector.
    """
    scale = torch.sqrt(target.detach().square().mean() + minimum_scale**2)
    return ((prediction - target) / scale).square().mean(), scale


def dimensionless_group_loss(
    predicted_next_position: torch.Tensor | None,
    target_next_position: torch.Tensor | None,
    predicted_previous_position: torch.Tensor | None,
    target_previous_position: torch.Tensor | None,
    position_step_scale: torch.Tensor | None,
    predicted_next_rotation: torch.Tensor | None,
    target_next_rotation: torch.Tensor | None,
    predicted_previous_rotation: torch.Tensor | None,
    target_previous_rotation: torch.Tensor | None,
    rotation_step_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    translation_loss = None
    rotation_loss = None
    if predicted_next_position is not None:
        scale = safe_scale(position_step_scale).to(predicted_next_position)
        translation_loss = 0.5 * (
            ((predicted_next_position - target_next_position) / scale).square().mean()
            + ((predicted_previous_position - target_previous_position) / scale).square().mean()
        )
    if predicted_next_rotation is not None:
        scale = safe_scale(rotation_step_scale).to(predicted_next_rotation)
        rotation_loss = 0.5 * (
            (rotation_geodesic_error(predicted_next_rotation, target_next_rotation) / scale).square().mean()
            + (rotation_geodesic_error(predicted_previous_rotation, target_previous_rotation) / scale).square().mean()
        )
    return balanced_mean(translation_loss, rotation_loss), translation_loss, rotation_loss


def camera_basis(
    location: torch.Tensor,
    target: torch.Tensor,
    world_up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return rows [camera-right, camera-up, camera-depth] in world coordinates."""
    if world_up is None:
        world_up = location.new_tensor((0.0, 0.0, 1.0))
    depth = target - location
    depth = depth / depth.norm().clamp_min(1.0e-8)
    right = torch.linalg.cross(depth, world_up)
    right = right / right.norm().clamp_min(1.0e-8)
    up = torch.linalg.cross(right, depth)
    up = up / up.norm().clamp_min(1.0e-8)
    return torch.stack((right, up, depth), dim=0)


def world_to_camera_vector(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...j,ij->...i", vector, basis.to(vector))


def component_rmse(error: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(error.square().mean(dim=tuple(range(error.ndim - 1))))


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    """Diagnostic gradient norm without modifying parameter gradients."""
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return torch.sqrt(squared)


def context_twist_loss(
    predicted_velocity: torch.Tensor | None,
    target_velocity: torch.Tensor | None,
    velocity_scale: torch.Tensor |  None,
    predicted_omega: torch.Tensor | None,
    target_omega: torch.Tensor | None,
    omega_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Supervise the mean twist across a window.

    This objective is appropriate when the trajectory generator declares the
    twist constant over the context. It keeps one unified se(3) coordinate and
    pools evidence across every observed interval.
    """
    return dimensionless_twist_loss(
        predicted_velocity.mean(dim=1) if predicted_velocity is not None else None,
        target_velocity.mean(dim=1) if target_velocity is not None else None,
        predicted_omega.mean(dim=1) if predicted_omega is not None else None,
        target_omega.mean(dim=1) if target_omega is not None else None,
        omega_scale,
    )


def interval_twist_variance_loss(
    predicted_velocity: torch.Tensor | None,
    velocity_scale: torch.Tensor | None,
    predicted_omega: torch.Tensor | None,
    omega_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Penalize interval-to-interval twist variation in constant-twist windows."""
    residuals = []
    if predicted_velocity is not None:
        centered = predicted_velocity - predicted_velocity.mean(dim=1, keepdim=True)
        residuals.append(centered / safe_scale(velocity_scale).to(centered))

    if predicted_omega is not None:
        centered = predicted_omega - predicted_omega.mean(dim=1, keepdim=True)
        residuals.append(centered / safe_scale(omega_scale).to(centered))
    if not residuals:
        raise ValueError("At least one twist sector must be active.")
    return torch.cat(residuals, dim=-1).square().mean()
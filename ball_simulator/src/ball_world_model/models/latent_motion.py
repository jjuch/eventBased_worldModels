from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .rotation import matrix_to_rotation_6d, so3_exp, so3_residual_vector


class RunningDeltaNormaliser(nn.Module):
    """Per-channel normalisation preserving the magnitude of temporal feature change."""

    def __init__(self, channels: int, momentum: float = 0.01, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))


    @torch.no_grad()
    def _update(self, difference: torch.Tensor) -> None:
        dimensions = (0, 1, 3, 4)
        mean = difference.mean(dim=dimensions)
        variance = difference.var(dim=dimensions, unbiased=False)
        if self.updates.item() == 0:
            self.running_mean.copy_(mean)
            self.running_var.copy_(variance.clamp_min(self.epsilon))
        else:
            self.running_mean.lerp_(mean, self.momentum)
            self.running_var.lerp_(variance, self.momentum)
        self.updates.add_(1)

    def forward(self, difference: torch.Tensor) -> torch.Tensor:
        if difference.ndim != 5:
            raise ValueError("Expected temporal differences with shape (B, T-1, C, H, W).")
        if self.training:
            self._update(difference.detach())
        mean = self.running_mean.view(1, 1, -1, 1, 1)
        std = torch.sqrt(self.running_var + self.epsilon).view(1, 1, -1, 1, 1)
        return (difference - mean) / std


class SpatialMotionEncoder(nn.Module):
    """Learn motion from consecutive content maps and their normalised difference."""

    def __init__(self, feature_channels: int, motion_dim: int = 256) -> None:
        super().__init__()
        hidden = feature_channels
        self.network = nn.Sequential(
            nn.Conv2d(3 * feature_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, motion_dim),
            nn.LayerNorm(motion_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        normalised_difference: torch.Tensor,
    ) -> torch.Tensor:
        batch, pairs = first.shape[:2]
        inputs = torch.cat((first, second, normalised_difference), dim=2).flatten(0, 1)
        motion = self.network(inputs)
        return motion.reshape(batch, pairs, -1)


class LatentTransition(nn.Module):
    def __init__(self, motion_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(motion_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        return self.network(motion)


class SharedTranslationalCorrector(nn.Module):
    """One shared correction step driven by the translational forward kinematic residual."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9),
        )

        # Begin as an exact identity refinement, learning must justify corrections.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


    def forward(
        self,
        first_position: torch.Tensor,
        second_postion: torch.Tensor,
        velocity: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = second_postion - first_position - velocity * dt
        inputs = torch.cat((first_position, second_postion, velocity, residual), dim=-1)
        correction = self.network(inputs)
        delta_first, delta_second, delta_velocity = correction.chunk(3, dim=-1)
        return (
            first_position + delta_first,
            second_postion + delta_second,
            velocity + delta_velocity,
            residual,
        )


class SharedRotationalCorrector(nn.Module):
    """Shared SO(3) correction step for worldd-frame angular velocity."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(18, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9),
        )

        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


    def forward(
        self,
        first_rotation: torch.Tensor,
        second_rotation: torch.Tensor,
        angular_velocity: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_second = so3_exp(angular_velocity * dt) @ first_rotation
        relative = second_rotation @ expected_second.transpose(-1, -2)
        residual = so3_residual_vector(relative)
        inputs = torch.cat(
            (
                matrix_to_rotation_6d(first_rotation),
                matrix_to_rotation_6d(second_rotation),
                angular_velocity,
                residual,
            ),
            dim=-1,
        )
        correction = self.network(inputs)
        delta_first, delta_second, delta_omega = correction.chunk(3, dim=-1)
        return (
            so3_exp(delta_first) @ first_rotation,
            so3_exp(delta_second) @ second_rotation,
            angular_velocity + delta_omega,
            residual,
        )


@dataclass(frozen=True)
class MotionDiagnostics:
    normalised_forward_difference: torch.Tensor
    forward_motion: torch.Tensor
    backward_motion: torch.Tensor
    predicted_next_embedding: torch.Tensor
    predicted_previous_embedding: torch.Tensor
    forward_velocity: torch.Tensor | None
    backward_velocity: torch.Tensor | None
    forward_angular_velocity: torch.Tensor | None
    backward_angular_velocity: torch.Tensor | None
    linear_refinement_residuals: tuple[torch.Tensor, ...]
    rotational_refinement_residuals: tuple[torch.Tensor, ...]
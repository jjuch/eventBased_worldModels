"""Residual artifact modelling and adversarial physical-information removal."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def gradient_reverse(value: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(value, strength)


def cross_covariance_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Squared cross-covariance after per-coordinate standardisation."""
    first = first.flatten(0, 1)
    second = second.flatten(0, 1)
    if first.shape[-1] == 0 or second.shape[-1] == 0:
        return first.new_zeros(())

    first = (first - first.mean(0)) / first.std(0, unbiased=False).clamp_min(1e-4)
    second = (second - second.mean(0)) / second.std(0, unbiased=False).clamp_min(1e-4)
    covariance = first.T @ second / max(first.shape[0] - 1, 1)
    return covariance.square().mean()


class PhysicalFeatureRatePredictor(nn.Module):
    """Predict the feature rate explained by a structured physical condition."""
    def __init__(self, channels: int, hidden_channels: int = 128, conditioning_dim: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels + conditioning_dim, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = feature.shape
        condition = condition[..., None, None].expand(-1, -1, -1, h, w)
        value = torch.cat((feature, condition), dim=2).flatten(0, 1)
        return self.network(value).reshape(b, t, c, h, w)


class ArtifactResidualEncoder(nn.Module):
    """Encode only residual feature rate after the physics prediction."""
    def __init__(self, channels: int, artifact_dim: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.artifact_dim = artifact_dim
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, artifact_dim),
            nn.LayerNorm(artifact_dim),
            nn.SiLU(),
        )

    def forward(self, residual_rate: torch.Tensor) -> torch.Tensor:
        b, t = residual_rate.shape[:2]
        return self.network(residual_rate.flatten(0, 1)).reshape(b, t, self.artifact_dim)


class ArtifactResidualDecoder(nn.Module):
    """Positive task: reconstruct the unexplained feature-rate map from nu."""
    def __init__(self, artifact_dim: int, channels: int, spatial_size: int = 16, hidden_channels: int = 128) -> None:
        super().__init__()
        if spatial_size % 4:
            raise ValueError("spatial_size must be divisible by four.")
        
        self.channels, self.spatial_size = channels, spatial_size
        self.initial_size = spatial_size // 4
        self.linear = nn.Linear(artifact_dim, hidden_channels * self.initial_size**2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, channels, 3, padding=1),
        )


    def forward(self, artifact: torch.Tensor) -> torch.Tensor:
        b, t = artifact.shape[:2]
        value = self.linear(artifact.flatten(0, 1))
        value = value.reshape(b * t, -1, self.initial_size, self.initial_size)
        value = self.decoder(value)
        return value.reshape(b, t, self.channels, self.spatial_size, self.spatial_size)


class PhysicsAdversary(nn.Module):
    """Predict standardised omega and relative rotation from artifact features."""
    def __init__(self, artifact_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(artifact_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.omega = nn.Linear(hidden_dim, 3)
        self.relative_rotation_6d = nn.Linear(hidden_dim, 6)

    def forward(self, artifact: torch.Tensor, strength: float = 0.0):
        value = gradient_reverse(artifact, strength) if strength else artifact
        value = self.trunk(value)
        return self.omega(value), self.relative_rotation_6d(value)


def adversary_strength(current_epoch: int, warmup_epochs: int, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    progress = min(max(current_epoch / max(warmup_epochs, 1), 0.0), 1.0)

    # Smooth DANN-style ramp from zero.
    return float(maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0))
"""Small context-level aggregator for a unified SE(3) twist."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ContextTwist:
    """One twist inferred from all interval evidence in a window."""
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    attention: torch.Tensor
    pooled_evidence: torch.Tensor


class ContextTwistAggregator(nn.Module):
    """Infer one six-dimensional twist by weighted pooling.
    
    The module is intentionally small. It scores each interval, forms one
    convex combination of the interval features, and decodes a single twist.
    Translation and rotation are decoded together from the same pooled
    evidence vector.
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        *,
        use_translation: bool = True,
        use_rotation: bool = True,
    ) -> None:
        super().__init__()
        self.use_translation = bool(use_translation)
        self.use_rotation = bool(use_rotation)
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
        )
        self.score = nn.Linear(hidden_dim, 1)
        self.twist = nn.Linear(hidden_dim, 6)

    def forward(self, interval_features: torch.Tensor) -> ContextTwist:
        if interval_features.ndim != 3:
            raise ValueError(
                "interval_features must have shape [batch, intervals, features]."
            )

        evidence = self.projection(interval_features)
        attention = torch.softmax(self.score(evidence).squeeze(-1), dim=1)
        pooled = torch.sum(attention.unsqueeze(-1) * evidence, dim=1)
        twist = self.twist(pooled)
        linear_velocity = twist[..., :3]
        angular_velocity = twist[..., 3:]

        if not self.use_translation:
            linear_velocity = torch.zeros_like(linear_velocity)
        if not self.use_rotation:
            angular_velocity = torch.zeros_like(angular_velocity)

        return ContextTwist(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            attention=attention,
            pooled_evidence=pooled,
        )
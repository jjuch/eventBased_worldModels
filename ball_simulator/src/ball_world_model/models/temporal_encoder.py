"""Minimal causal temporal heads for twist and acceleration."""

from __future__ import annotations

import torch
from torch import Tensor, nn

class CausalAttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * self.value(sequence)).sum(dim=-2)
        return self.output(pooled), weights

class TwistEncoder(nn.Module):
    """Uses first feature rates only; no pose shortcut is supplied."""
    def __init__(self, feature_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(feature_channels * 3, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim), 
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
        )
        self.pool = CausalAttentionPool(hidden_dim, hidden_dim, 6)

    def forward(self, maps: Tensor, first_rate: Tensor) -> tuple[Tensor, Tensor]:
        batch, intervals = first_rate.shape[:2]
        inputs = torch.cat((maps[:, :-1], maps[:, 1:], first_rate), dim=2)
        encoded = self.spatial(inputs.flatten(0, 1)).reshape(batch, intervals, -1)
        return self.pool(encoded)


class AccelerationEncoder(nn.Module):
    """Uses only second feature rates and their local first-rate context.

    Pose and decoded twist are deliberately not passed to this branch. The branch must identify change of motion from visual temporal curvature. The analytic carrier teacher introduces the required omega-dependent centripetal term.
    """
    def __init__(self, feature_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(feature_channels * 3, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim), 
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
        )
        self.pool = CausalAttentionPool(hidden_dim, hidden_dim, 6)
        
    def forward(self, first_rate: Tensor, second_rate: Tensor) -> tuple[Tensor, Tensor]:
        # second_rate[t] is centred at frame t+1; adjacent first rates bracket it.
        inputs = torch.cat((first_rate[:, :-1], first_rate[:, 1:], second_rate), dim=2)
        batch, samples = inputs.shape[:2]
        encoded = self.spatial(inputs.flatten(0, 1)).reshape(batch, samples, -1)
        return self.pool(encoded)

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
    def __init__(self, keypoints: int, keypoint_dim: int, hidden_dim: int) -> None:
        super().__init__()
        pair_dim = keypoints * (2 * 2 + 3 * keypoint_dim + 1)
        self.interval = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.SiLU(),
        )
        self.pool = CausalAttentionPool(hidden_dim, hidden_dim, 6)

    def forward(self, coordinates: Tensor, descriptors: Tensor, confidence: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
        dq = (coordinates[:, 1:] - coordinates[:, :-1]) / dt[..., None, None]
        dd = (descriptors[:, 1:] - descriptors[:, :-1]) / dt[..., None, None]
        pair = torch.cat((coordinates[:, :-1], coordinates[:, 1:], dq, descriptors[:, :-1], descriptors[:, 1:], dd, confidence[:, 1:, :, None]), dim=-1)
        return self.pool(self.interval(pair.flatten(-2)))


class AccelerationEncoder(nn.Module):
    """Uses only second feature rates and their local first-rate context.

    Pose and decoded twist are deliberately not passed to this branch. The branch must identify change of motion from visual temporal curvature. The analytic carrier teacher introduces the required omega-dependent centripetal term.
    """
    def __init__(self, keypoints: int, keypoint_dim: int, hidden_dim: int) -> None:
        super().__init__()
        sample_dim = keypoints * (2 + keypoint_dim)
        self.sample = nn.Sequential(
            nn.Linear(sample_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.SiLU(),
        )
        self.pool = CausalAttentionPool(hidden_dim, hidden_dim, 6)
        
    def forward(self, coordinates: Tensor, descriptors: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
        first_q = (coordinates[:, 1:] - coordinates[:, :-1]) / dt[..., None, None]
        first_d = (descriptors[:, 1:] - descriptors[:, :-1]) / dt[..., None, None]
        centre_dt = 0.5 * (dt[:, 1:] + dt[:, :-1])
        second_q = (first_q[:, 1:] - first_q[:, :-1]) / centre_dt[..., None, None]
        second_d = (first_d[:, 1:] - first_d[:, :-1]) / centre_dt[..., None, None]
        sample = torch.cat((second_q, second_d), dim=-1).flatten(-2)
        return self.pool(self.sample(sample))

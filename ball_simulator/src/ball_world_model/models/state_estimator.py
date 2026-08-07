from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .frame_encoder import SmallFrameEncoder
from .rotation import rotation_6d_to_matrix
from .temporal_encoder import TemporalContextEncoder


@dataclass(frozen=True)
class StatePrediction:
    position: torch.Tensor
    rotation_6d: torch.Tensor
    rotation_matrix: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor


class StateEstimator(nn.Module):
    """Observability baseline: ten RGB frames to state at the final context frame."""

    def __init__(
        self,
        embedding_dim: int = 384,
        temporal_depth: int = 4,
        temporal_heads: int = 6,
        maximum_frames: int = 32,
    ) -> None:
        super().__init__()
        self.frame_encoder = SmallFrameEncoder(embedding_dim=embedding_dim)
        self.temporal_encoder = TemporalContextEncoder(
            embedding_dim=embedding_dim,
            depth=temporal_depth,
            heads=temporal_heads,
            maximum_frames=maximum_frames,
        )
        self.state_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 15),
        )


    def encode_context(self, context_rgb: torch.Tensor) -> torch.Tensor:
        return self.temporal_encoder(self.frame_encoder(context_rgb))

    def forward(self, context_rgb: torch.Tensor) -> StatePrediction:
        values = self.state_head(self.encode_context(context_rgb))
        position = values[..., 0:3]
        rotation_6d = values[..., 3:9]
        linear_velocity = values[..., 9:12]
        angular_velocity = values[..., 12:15]
        return StatePrediction(
            position=position,
            rotation_6d=rotation_6d,
            rotation_matrix=rotation_6d_to_matrix(rotation_6d),
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
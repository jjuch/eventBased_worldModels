from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .kinematic_encoder import CoordinateAwareFrameEncoder
from .rotation import rotation_6d_to_matrix
from .temporal_conv import TemporalConvEncoder

TaskKind = Literal["translation", "rotation", "combined"]


@dataclass(frozen=True)
class KinematicPrediction:
    position: torch.Tensor | None = None
    linear_velocity: torch.Tensor | None = None
    rotation_6d: torch.Tensor | None = None
    rotation_matrix: torch.Tensor | None = None
    angular_velocity: torch.Tensor | None = None
    latent: torch.Tensor | None = None


class KinematicStateEstimator(nn.Module):
    """Observability baseline: ten RGB frames to state at the final context frame."""

    def __init__(
        self,
        task: TaskKind,
        embedding_dim: int = 256,
        temporal_depth: int = 4,
        keypoints: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.frame_encoder = CoordinateAwareFrameEncoder(
            embedding_dim=embedding_dim,
            keypoints=keypoints,
            )
        self.temporal_encoder = TemporalConvEncoder(
            embedding_dim=embedding_dim,
            depth=temporal_depth,
            dropout=dropout,
        )
        self.position_head = self._head(embedding_dim, 3) if task in {"translation", "combined"} else None
        self.velocity_head = self._head(embedding_dim, 3) if task in {"translation", "combined"} else None
        self.rotation_head = self._head(embedding_dim, 6) if task in {"rotation", "combined"} else None
        self.omega_head = self._head(embedding_dim, 3) if task in {"rotation", "combined"} else None


    @staticmethod
    def _head(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )   


    def forward(self, context_rgb: torch.Tensor) -> KinematicPrediction:
        latent = self.temporal_encoder(self.frame_encoder(context_rgb))
        position = self.position_head(latent) if self.position_head is not None else None
        velocity = self.velocity_head(latent) if self.velocity_head is not None else None
        rotation_6d = self.rotation_head(latent) if self.rotation_head is not None else None
        rotation_matrix = rotation_6d_to_matrix(rotation_6d) if rotation_6d is not None else None
        omega = self.omega_head(latent) if self.omega_head is not None else None

        return KinematicPrediction(
            position=position,
            linear_velocity=velocity,
            rotation_6d=rotation_6d,
            rotation_matrix=rotation_matrix,
            angular_velocity=omega,
            latent=latent,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .kinematic_encoder import CoordinateAwareFrameEncoder
from .latent_motion import (
    LatentTransition,
    MotionDiagnostics,
    RunningDeltaNormaliser,
    SharedKinematicCorrector,
    SpatialMotionEncoder,
)
from .rotation import rotation_6d_to_matrix

TaskKind = Literal["translation", "rotation", "combined"]


@dataclass(frozen=True)
class KinematicPrediction:
    position: torch.Tensor | None = None
    linear_velocity: torch.Tensor | None = None
    rotation_6d: torch.Tensor | None = None
    rotation_matrix: torch.Tensor | None = None
    angular_velocity: torch.Tensor | None = None
    frame_latent: torch.Tensor | None = None
    feature_maps: torch.Tensor | None = None
    motion: MotionDiagnostics | None = None


class KinematicStateEstimator(nn.Module):
    """Bidirectional content-motion observer with optional shared correction iterations."""

    def __init__(
        self,
        task: TaskKind,
        embedding_dim: int = 256,
        keypoints: int = 8,
        motion_dim: int = 256,
        decoder_hidden_dim: int = 256,
        refinement_hidden_dim: int = 128,
        refinement_iterations: int = 1,
        delta_momentum: float = 0.01,
        default_frame_dt: float = 0.01,
        dropout: float = 0.0,
        temporal_depth: int | None = None,
        position_mean: torch.Tensor | None = None,
        position_std: torch.Tensor | None = None,
        velocity_mean: torch.Tensor | None = None,
        velocity_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        del temporal_depth # Accepted for configuration compatibility
        if refinement_iterations < 0:
            raise ValueError("refinement_iterations must be non-negative.")
        self.task = task
        self.refinement_iterations = refinement_iterations
        self.default_frame_dt = float(default_frame_dt)
        self.register_buffer(
            "position_mean",
            torch.zeros(3) if position_mean is None else position_mean.detach().float().clone(),
        )
        self.register_buffer(
            "position_std",
            torch.ones(3) if position_std is None else position_std.detach().float().clone(),
        )
        self.register_buffer(
            "velocity_mean",
            torch.zeros(3) if velocity_mean is None else velocity_mean.detach().float().clone(),
        )
        self.register_buffer(
            "velocity_std",
            torch.ones(3) if velocity_std is None else velocity_std.detach().float().clone(),
        )

        self.frame_encoder = CoordinateAwareFrameEncoder(
            embedding_dim=embedding_dim,
            keypoints=keypoints,
            )
        feature_channels = self.frame_encoder.feature_channels
        self.delta_normaliser = RunningDeltaNormaliser(
            feature_channels, momentum=delta_momentum,
        )
        self.motion_encoder = SpatialMotionEncoder(feature_channels, motion_dim)
        self.latent_transition = LatentTransition(motion_dim, embedding_dim)
        
        translation = task in {"translation", "combined"}
        rotation = task in {"rotation", "combined"}
        self.position_head = self._head(embedding_dim, decoder_hidden_dim, 3, dropout) if translation else None
        velocity_input_dim = 2 * embedding_dim + motion_dim
        self.velocity_head = self._head(velocity_input_dim, decoder_hidden_dim, 3, dropout) if translation else None

        self.rotation_head = self._head(embedding_dim, decoder_hidden_dim, 6, dropout) if rotation else None
        self.omega_head = self._head(velocity_input_dim, decoder_hidden_dim, 3, dropout) if rotation else None
        self.corrector = SharedKinematicCorrector(refinement_hidden_dim) if translation else None


    @staticmethod
    def _head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


    @staticmethod
    def _pair_decoder_input(
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((first_embedding, second_embedding, motion), dim=-1)

    @staticmethod
    def _pair_velocity_to_frame_velocity(pair_velocity: torch.Tensor) -> torch.Tensor:
        # Pair t describes the interval [t, t+1]. The last frame receives the last available interval velocity so the historic (B, T, 3) contract is preserved.
        return torch.cat((pair_velocity, pair_velocity[:, -1:]), dim=1)
    

    def forward(
        self, 
        context_rgb: torch.Tensor,
        context_time: torch.Tensor | None = None,
    ) -> KinematicPrediction:
        feature_maps = self.frame_encoder.encode_feature_maps(context_rgb)
        frame_latent = self.frame_encoder.embeddings_from_maps(feature_maps)
        batch, frames = frame_latent.shape[:2]
        if frames < 2:
            raise ValueError("At least two frames are required for motion estimation.")

        if context_time is None:
            context_time = (
                torch.arange(frames, device=context_rgb.device, dtype=context_rgb.dtype) * self.default_frame_dt
            ).unsqueeze(0).expand(batch, -1)
        context_time = context_time.to(device=context_rgb.device, dtype=context_rgb.dtype)
        dt = torch.diff(context_time, dim=1).unsqueeze(-1).clamp_min(1.0e-8)

        difference = feature_maps[:, 1:] - feature_maps[:, :-1]
        # Normalise a feature-rate rather than a raw difference. This supplies dt explicitly and makes the motion representation comparable across frame strides.
        feature_rate = difference / dt.unsqueeze(-1).unsqueeze(-1)
        normalised_difference = self.delta_normaliser(feature_rate)

        forward_motion = self.motion_encoder(
            feature_maps[:, :-1], feature_maps[:, 1:], normalised_difference
        )
        backward_motion = self.motion_encoder(
            feature_maps[:, 1:], feature_maps[:, :-1], -normalised_difference
        )

        latent_rate_forward = self.latent_transition(forward_motion)
        latent_rate_backward = self.latent_transition(backward_motion)
        predicted_next_embedding = frame_latent[:, :-1] + latent_rate_forward * dt
        predicted_previous_embedding = frame_latent[:, 1:] + latent_rate_backward * dt

        position = self.position_head(frame_latent) if self.position_head is not None else None
        rotation_6d = self.rotation_head(frame_latent) if self.rotation_head is not None else None
        rotation_matrix = rotation_6d_to_matrix(rotation_6d) if rotation_6d is not None else None

        forward_input = self._pair_decoder_input(
            frame_latent[:, :-1], frame_latent[:, 1:], forward_motion
        )
        backward_input = self._pair_decoder_input(
            frame_latent[:, 1:], frame_latent[:, :-1], backward_motion
        )
        forward_velocity = self.velocity_head(forward_input) if self.velocity_head is not None else None
        backward_velocity = self.velocity_head(backward_input) if self.velocity_head is not None else None
        pair_omega = self.omega_head(forward_input) if self.omega_head is not None else None

        # Optional refinement procedure - TODO implement for rotation as well
        refinement_residuals: list[torch.Tensor] = []
        if position is not None and forward_velocity is not None:
            # Heads predict normalised state. Iterative correction is performed in physical units so p[t+1] - p[t] - v[t] * dt is dimensionally meaningful.
            refined_position = position * self.position_std + self.position_mean
            refined_velocity = forward_velocity * self.velocity_std + self.velocity_mean
            for _ in range(self.refinement_iterations):
                first, second, refined_velocity, residual = self.corrector(
                    refined_position[:, :-1],
                    refined_position[:, 1:],
                    refined_velocity,
                    dt,
                )
                refinement_residuals.append(residual)
                accumulated = torch.zeros_like(refined_position)
                counts = torch.zeros_like(refined_position[..., :1])
                accumulated[:, :-1] += first
                counts[:, :-1] += 1.0
                accumulated[:, 1:] += second
                counts[:, 1:] += 1.0
                refined_position = accumulated / counts.clamp_min(1.0)
            position = (refined_position - self.position_mean) / self.position_std
            forward_velocity = (refined_velocity - self.velocity_mean) / self.velocity_std

        linear_velocity = (
            self._pair_velocity_to_frame_velocity(forward_velocity)
            if forward_velocity is not None
            else None
        )
        angular_velocity = (
            self._pair_velocity_to_frame_velocity(pair_omega)
            if pair_omega is not None
            else None
        )
        diagnostics = MotionDiagnostics(
            raw_forward_difference=difference,
            raw_forward_feature_rate=feature_rate,
            normalised_forward_difference=normalised_difference,
            forward_motion=forward_motion,
            backward_motion=backward_motion,
            predicted_next_embedding=predicted_next_embedding,
            predicted_previous_embedding=predicted_previous_embedding,
            forward_velocity=forward_velocity,
            backward_velocity=backward_velocity,
            refinement_residuals=tuple(refinement_residuals),
        )

        return KinematicPrediction(
            position=position,
            linear_velocity=linear_velocity,
            rotation_6d=rotation_6d,
            rotation_matrix=rotation_matrix,
            angular_velocity=angular_velocity,
            frame_latent=frame_latent,
            feature_maps=feature_maps,
            motion=diagnostics,
        )

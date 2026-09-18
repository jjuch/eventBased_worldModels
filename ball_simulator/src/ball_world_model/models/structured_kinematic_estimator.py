"""Rotation-only observer with SO(3) x artifact and so(3) x artifact latents.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .kinematic_encoder import CoordinateAwareFrameEncoder
from .latent_motion import RunningDeltaNormaliser, SpatialMotionEncoder
from .structured_so3 import RichSO3ContextHead, RichSO3MotionHead, world_step, orbit, MotionSector, ContextSectors

@dataclass(frozen=True)
class RichMotionDiagnostics:
    normalised_forward_difference: torch.Tensor
    forward_motion: torch.Tensor
    backward_motion: torch.Tensor
    predicted_next_embedding: torch.Tensor
    predicted_previous_embedding: torch.Tensor
    forward_angular_velocity: torch.Tensor
    backward_angular_velocity: torch.Tensor
    forward_sectors: MotionSector
    backward_sectors: MotionSector
    predicted_next_rotation: torch.Tensor
    predicted_previous_rotation: torch.Tensor
    predicted_next_invariants: torch.Tensor
    predicted_previous_invariants: torch.Tensor
    predicted_next_artifacts: torch.Tensor
    predicted_previous_artifacts: torch.Tensor
    linear_refinement_residuals: tuple = ()
    rotational_refinement_residuals: tuple = ()


@dataclass(frozen=True)
class RichPrediction:
    position: None
    linear_velocity: None
    rotation_6d: torch.Tensor
    rotation_matrix: torch.Tensor
    angular_velocity: torch.Tensor
    frame_latent: torch.Tensor
    feature_maps: torch.Tensor
    motion: RichMotionDiagnostics
    context_sectors: ContextSectors


class StructuredSO3StateEstimator(nn.Module):
    def __init__(
        self, 
        embedding_dim=256, 
        motion_dim=256, 
        descriptor_dim=256, 
        keypoints=8, 
        geometric_channels=24, 
        physical_invariant_dim=16, 
        default_frame_dt=0.01, 
        delta_momentum=0.01, 
        **_: object) -> None:
        super().__init__()
        self.default_frame_dt = float(default_frame_dt)
        self.frame_encoder = CoordinateAwareFrameEncoder(
            embedding_dim=descriptor_dim, 
            keypoints=keypoints
        )

        channels = self.frame_encoder.feature_channels
        self.delta_normaliser = RunningDeltaNormaliser(channels, momentum=delta_momentum)
        self.motion_encoder = SpatialMotionEncoder(channels, motion_dim)
        self.context_head = RichSO3ContextHead(descriptor_dim, embedding_dim, geometric_channels, physical_invariant_dim)
        self.motion_head = RichSO3MotionHead(motion_dim, motion_dim, geometric_channels, physical_invariant_dim)
        context_layout = self.context_head.layout
        motion_layout = self.motion_head.layout
        self.invariant_transition = self._transition(
            motion_layout.invariant_dim,
            context_layout.invariant_dim
        )
        self.artifact_transition = self._transition(
            motion_layout.motion_artifact_dim,
            context_layout.context_artifact_dim
        )

    @staticmethod
    def _transition(input_dim: int, output_dim: int) -> nn.Module:
        hidden = max(input_dim, output_dim, 16)
        network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
        return network


    @staticmethod
    def _frame_velocity(pair: torch.Tensor) -> torch.Tensor:
        return torch.cat((pair, pair[:, -1:]), dim=1)

    @staticmethod
    def _pack_context(rotation_6d, amplitudes, carrier, invariants, artifacts):
        return torch.cat(
            (
                rotation_6d,
                amplitudes,
                carrier.flatten(-2),
                invariants,
                artifacts,
            ), dim=-1,
        )

    
    def forward(self, context_rgb, context_time=None) -> RichPrediction:
        maps = self.frame_encoder.encode_feature_maps(context_rgb)
        descriptor = self.frame_encoder.embeddings_from_maps(maps)
        context = self.context_head(descriptor)
        batch, frames = descriptor.shape[:2]
        if frames < 2:
            raise ValueError("At least two frames are required.")


        if context_time is None:
            context_time = (
                torch.arange(frames, device=descriptor.device, dtype=descriptor.dtype) * self.default_frame_dt
            ).unsqueeze(0).expand(batch, -1)

        dt = torch.diff(context_time.to(descriptor), dim=1).unsqueeze(-1).clamp_min(1e-8)
        raw_rate = (maps[:, 1:] - maps[:, :-1]) / dt.unsqueeze(-1).unsqueeze(-1)
        normalised_rate = self.delta_normaliser(raw_rate)

        forward_features = self.motion_encoder(maps[:, :-1], maps[:, 1:], normalised_rate)
        backward_features = self.motion_encoder(maps[:, 1:], maps[:, :-1], -normalised_rate)

        forward = self.motion_head(forward_features, context.carrier[:, :-1])
        backward = self.motion_head(backward_features, context.carrier[:, 1:])

        next_rotation = world_step(context.rotation[:, :-1], forward.omega, dt)
        previous_rotation = world_step(context.rotation[:, 1:], backward.omega, dt)

        next_amplitudes = (context.amplitudes[:, :-1] + dt * forward.amplitude_rate).clamp_min(0.0)
        previous_amplitudes = (context.amplitudes[:, 1:] + dt * backward.amplitude_rate).clamp_min(0.0)

        next_carrier = next_amplitudes.unsqueeze(-1) * orbit(next_rotation, self.context_head.templates)
        previous_carrier = previous_amplitudes.unsqueeze(-1) * orbit(previous_rotation, self.context_head.templates)

        # These coordinates are invariant under changing the SO(3) basis, not constant in time. They may encode scalar observability/confidence and therefore receive a learned temporal rate from the invariant motion sector.
        next_invariants = (
            context.physical_invariants[:, :-1] + dt * self.invariant_transition(forward.physical_invariants)
        )
        previous_invariants = (
            context.physical_invariants[:, 1:] + dt * self.invariant_transition(backward.physical_invariants)
        )

        next_artifacts = context.artifacts[:, :-1] + dt * self.artifact_transition(forward.artifacts)
        previous_artifacts = context.artifacts[:, 1:] + dt * self.artifact_transition(backward.artifacts)

        next_rotation_6d = next_rotation[..., :, :2].transpose(-1, -2).flatten(-2)
        previous_rotation_6d = previous_rotation[..., :, :2].transpose(-1, -2).flatten(-2)

        next_packed = self._pack_context(
            next_rotation_6d,
            next_amplitudes, 
            next_carrier,
            next_invariants,
            next_artifacts,
        )
        previous_packed = self._pack_context(
            previous_rotation_6d,
            previous_amplitudes, 
            previous_carrier,
            previous_invariants,
            previous_artifacts,
        )
        
        diagnostics = RichMotionDiagnostics(
            normalised_forward_difference=normalised_rate, 
            forward_motion=forward.packed, 
            backward_motion=backward.packed, 
            predicted_next_embedding=next_packed, 
            predicted_previous_embedding=previous_packed, 
            forward_angular_velocity=forward.omega, 
            backward_angular_velocity=backward.omega, 
            forward_sectors=forward, 
            backward_sectors=backward, 
            predicted_next_rotation=next_rotation, 
            predicted_previous_rotation=previous_rotation,
            predicted_next_invariants=next_invariants,
            predicted_previous_invariants=previous_invariants,
            predicted_next_artifacts=next_artifacts,
            predicted_previous_artifacts=previous_artifacts,
        )
        return RichPrediction(
            position=None, 
            linear_velocity=None, 
            rotation_6d=context.rotation_6d, 
            rotation_matrix=context.rotation, 
            angular_velocity=self._frame_velocity(forward.omega), 
            frame_latent=context.packed, 
            feature_maps=maps, 
            motion=diagnostics, 
            context_sectors=context
        )
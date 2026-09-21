"""Rotation-only observer with SO(3) x artifact and so(3) x artifact latents.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from .kinematic_encoder import CoordinateAwareFrameEncoder
from .latent_motion import RunningDeltaNormaliser, SpatialMotionEncoder
from .structured_so3 import (
    RichSO3ContextHead, 
    RichSO3MotionHead, 
    world_step, orbit, 
    MotionSector, 
    ContextSectors,
)

from .artifact_residual import (
    ArtifactResidualDecoder,
    ArtifactResidualEncoder,
    PhysicalFeatureRatePredictor,
)

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
    physical_feature_rate_forward: torch.Tensor
    physical_feature_rate_backward: torch.Tensor
    artifact_residual_target_forward: torch.Tensor
    artifact_residual_target_backward: torch.Tensor
    artifact_residual_reconstruction_forward: torch.Tensor
    artifact_residual_reconstruction_backward: torch.Tensor
    linear_refinement_residuals: tuple = ()
    rotational_refinement_residuals: tuple = ()

    @property
    def artifact_residual_target(self) -> torch.Tensor:
        return self.artifact_residual_target_forward

    @property
    def artifact_residual_reconstruction(self) -> torch.Tensor:
        return self.artifact_residual_reconstruction_forward



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
        artifact_mode="physical_residual_adversarial",
        artifact_residual_hidden_channels=128,
        **_: object) -> None:
        super().__init__()
        self.default_frame_dt = float(default_frame_dt)
        self.artifact_mode = artifact_mode
        self.frame_encoder = CoordinateAwareFrameEncoder(
            embedding_dim=descriptor_dim, 
            keypoints=keypoints
        )

        channels = self.frame_encoder.feature_channels
        self.delta_normaliser = RunningDeltaNormaliser(channels, momentum=delta_momentum)
        self.motion_encoder = SpatialMotionEncoder(channels, motion_dim)
        self.context_head = RichSO3ContextHead(
            descriptor_dim, embedding_dim, geometric_channels, physical_invariant_dim
        )
        self.motion_head = RichSO3MotionHead(
            motion_dim, motion_dim, geometric_channels, physical_invariant_dim
        )

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
        self.physical_feature_rate_predictor = PhysicalFeatureRatePredictor(
            channels, artifact_residual_hidden_channels
        )
        self.artifact_residual_encoder = ArtifactResidualEncoder(
            channels,
            motion_layout.motion_artifact_dim,
            artifact_residual_hidden_channels,
        )
        self.artifact_residual_decoder = ArtifactResidualDecoder(
            motion_layout.motion_artifact_dim,
            channels,
            spatial_size=16,
            hidden_channels=artifact_residual_hidden_channels,
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

    @staticmethod
    def _replace_motion_artifacts(sector: MotionSector, artifacts: torch.Tensor) -> MotionSector:
        packed = torch.cat(
            (
                sector.omega,
                sector.amplitude_rate,
                sector.carrier_tangent.flatten(-2),
                sector.physical_invariants,
                artifacts,
            ),
            dim=-1,
        )
        return replace(sector, artifacts=artifacts, packed=packed)

    def _residual_artifacts(
        self,
        maps: torch.Tensor,
        rotation_6d: torch.Tensor,
        omega: torch.Tensor,
        dt: torch.Tensor,
        observed_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        physical_rate = self.physical_feature_rate_predictor(maps, rotation_6d, omega, dt)
        residual_target = observed_rate - physical_rate.detach()
        artifacts = self.artifact_residual_encoder(residual_target)
        reconstruction = self.artifact_residual_decoder(artifacts)
        return artifacts, physical_rate, residual_target, reconstruction

    
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

        if self.artifact_mode == "physical_residual_adverserial":
            (forward_artifacts, physical_rate_forward, residual_forward, reconstruction_forward) = self._residual_artifacts(
                maps[:, :-1], context.rotation_6d[:, :-1], forward.omega, dt, normalised_rate,
            )
            (backward_artifacts, physical_rate_backward, residual_backward, reconstruction_backward) = self._residual_artifacts(
                maps[:, 1:], context.rotation_6d[:, 1:], backward.omega, dt, -normalised_rate,
            )
            forward = self._replace_motion_artifacts(forward, forward_artifacts)
            backward = self._replace_motion_artifacts(backward, backward_artifacts)
        else:
            physical_rate_forward = torch.zeros_like(normalised_rate)
            physical_rate_backward = torch.zeros_like(normalised_rate)
            residual_forward = normalised_rate
            residual_backward = -normalised_rate
            reconstruction_forward = torch.zeros_like(normalised_rate)
            reconstruction_backward = torch.zeros_like(normalised_rate)

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
            physical_feature_rate_forward=physical_rate_forward,
            physical_feature_rate_backward=physical_rate_backward,
            artifact_residual_target_forward=residual_forward,
            artifact_residual_target_backward=residual_backward,
            artifact_residual_reconstruction_forward=reconstruction_forward,
            artifact_residual_reconstruction_backward=reconstruction_backward,
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

    def forward_with_zero_artifacts(self, context_rgb, context_time=None) -> RichPrediction:
        """Diagnostic route. Physical outputs are upstream of artifact replacement."""
        prediction = self.forward(context_rgb, context_time)
        motion = prediction.motion

        forward = self._replace_motion_artifacts(
            motion.forward_sectors, torch.zeros_like(motion.forward_sectors.artifacts)
        )
        backward = self._replace_motion_artifacts(
            motion.backward_sectors, torch.zeros_like(motion.backward_sectors.artifacts)
        )
        return replace(
            prediction,
            motion=replace(motion, forward_sectors=forward, backward_sectors=backward),
        )
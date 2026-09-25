"""Latent estimator for translation-only, rotation-only and combined SE(3) modes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from .kinematic_encoder import CoordinateAwareFrameEncoder
from .latent_motion import RunningDeltaNormaliser, SpatialMotionEncoder
from .structured_se3 import (
    SE3ContextHead,
    SE3MotionHead,
    SE3ContextLatent,
    SE3MotionLatent,
    pack_context,
    pack_motion,
    translation_step,
    rotation_step,
    orbit,
)
from .rotation import matrix_to_rotation_6d

from .artifact_residual import (
    ArtifactResidualDecoder,
    ArtifactResidualEncoder,
    PhysicalFeatureRatePredictor,
)


@dataclass(frozen=True)
class SE3MotionDiagnostics:
    normalised_forward_difference: torch.Tensor
    forward_motion: torch.Tensor
    backward_motion: torch.Tensor
    predicted_next_embedding: torch.Tensor
    predicted_previous_embedding: torch.Tensor
    forward_sectors: SE3MotionLatent
    backward_sectors: SE3MotionLatent
    predicted_next_position: torch.Tensor
    predicted_previous_position: torch.Tensor
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
    def forward_linear_velocity(self) -> torch.Tensor:
        return self.forward_sectors.linear_velocity

    @property
    def backward_linear_velocity(self) -> torch.Tensor:
        return self.backward_sectors.linear_velocity
    
    @property
    def forward_angular_velocity(self) -> torch.Tensor:
        return self.forward_sectors.angular_velocity
    
    @property
    def backward_angular_velocity(self) -> torch.Tensor:
        return self.backward_sectors.angular_velocity

    @property
    def artifact_residual_target(self) -> torch.Tensor:
        return self.artifact_residual_target_forward

    @property
    def artifact_residual_reconstruction(self) -> torch.Tensor:
        return self.artifact_residual_reconstruction_forward



@dataclass(frozen=True)
class SE3Prediction:
    position: torch.Tensor
    linear_velocity: torch.Tensor
    rotation_6d: torch.Tensor
    rotation_matrix: torch.Tensor
    angular_velocity: torch.Tensor
    frame_latent: torch.Tensor
    feature_maps: torch.Tensor
    motion: SE3MotionDiagnostics
    context_sectors: SE3ContextLatent


class StructuredSE3StateEstimator(nn.Module):
    def __init__(
        self,
        task='combined',
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
        self.task = task
        self.default_frame_dt = float(default_frame_dt)
        self.artifact_mode = artifact_mode
        self.frame_encoder = CoordinateAwareFrameEncoder(
            embedding_dim=descriptor_dim, 
            keypoints=keypoints
        )

        channels = self.frame_encoder.feature_channels
        self.delta_normaliser = RunningDeltaNormaliser(channels, momentum=delta_momentum)
        self.motion_encoder = SpatialMotionEncoder(channels, motion_dim)
        self.context_head = SE3ContextHead(
            descriptor_dim, embedding_dim, geometric_channels, physical_invariant_dim, task
        )
        self.motion_head = SE3MotionHead(
            motion_dim, motion_dim, geometric_channels, physical_invariant_dim, task
        )

        context_layout = self.context_head.layout
        motion_layout = self.motion_head.layout
        self.invariant_transition = self._transition(
            physical_invariant_dim,
            physical_invariant_dim
        )
        self.artifact_transition = self._transition(
            motion_layout.motion_artifact_dim,
            context_layout.context_artifact_dim
        )
        self.physical_feature_rate_predictor = PhysicalFeatureRatePredictor(
            channels, artifact_residual_hidden_channels, conditioning_dim=16
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
    def _frame(pair: torch.Tensor) -> torch.Tensor:
        return torch.cat((pair, pair[:, -1:]), dim=1)


    @staticmethod
    def _replace_motion_artifacts(sector: SE3MotionLatent, artifacts: torch.Tensor) -> SE3MotionLatent:
        packed = pack_motion(
            sector.linear_velocity,
            sector.angular_velocity,
            sector.translation_amplitude_rates,
            sector.rotation_amplitude_rates,
            sector.translation_carrier_tangents,
            sector.rotation_carrier_tangents,
            sector.physical_scalars,
            artifacts
        )
        return replace(
            sector, 
            artifacts=artifacts, 
            packed=packed,
        )

    def _condition(self, context: SE3ContextLatent, motion: SE3MotionLatent, dt: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (context.position, context.rotation_6d, motion.linear_velocity, motion.angular_velocity, dt),
            dim=-1,
        )

    def _residual_artifacts(
        self,
        maps: torch.Tensor,
        context: SE3ContextLatent,
        motion: SE3MotionLatent,
        dt: torch.Tensor,
        observed_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = self._condition(context, motion, dt)
        physical_rate = self.physical_feature_rate_predictor(maps, condition)
        residual_target = observed_rate - physical_rate.detach()
        artifacts = self.artifact_residual_encoder(residual_target)
        reconstruction = self.artifact_residual_decoder(artifacts)
        return artifacts, physical_rate, residual_target, reconstruction


    @staticmethod
    def _slice_context(context: SE3ContextLatent, slice: slice) -> SE3ContextLatent:
        return replace(
            context, 
            position=context.position[:, slice],
            rotation_6d=context.rotation_6d[:, slice],
            rotation=context.rotation[:, slice],
            translation_amplitudes=context.translation_amplitudes[:, slice],
            rotation_amplitudes=context.rotation_amplitudes[:, slice],
            translation_carrier=context.translation_carrier[:, slice], rotation_carrier=context.rotation_carrier[:, slice],
            physical_scalars=context.physical_scalars[:, slice],
            artifacts=context.artifacts[:, slice],
            packed=context.packed[:, slice],
        )

    
    def forward(self, context_rgb, context_time=None) -> SE3Prediction:
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

        context_forward = self._slice_context(context, slice(None, -1))
        context_backward = self._slice_context(context, slice(1, None))

        forward = self.motion_head(forward_features, context_forward)
        backward = self.motion_head(backward_features, context_backward)

        if self.artifact_mode == "physical_residual_adverserial":
            (forward_artifacts, physical_rate_forward, residual_forward, reconstruction_forward) = self._residual_artifacts(
                maps[:, :-1], context_forward, forward, dt, normalised_rate,
            )
            (backward_artifacts, physical_rate_backward, residual_backward, reconstruction_backward) = self._residual_artifacts(
                maps[:, 1:], context_backward, backward, dt, -normalised_rate,
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

        next_position = translation_step(context_forward.position, forward.linear_velocity, dt)
        previous_position = translation_step(context_backward.position, backward.linear_velocity, dt)

        next_rotation = rotation_step(context_forward.rotation, forward.angular_velocity, dt)
        previous_rotation = rotation_step(context_backward.rotation, backward.angular_velocity, dt)

        next_translation_amplitudes = (context_forward.translation_amplitudes + dt * forward.translation_amplitude_rates).clamp_min(0.0)
        previous_translation_amplitudes = (context_backward.translation_amplitudes + dt * backward.translation_amplitude_rates).clamp_min(0.0)
        next_rotation_amplitudes = (context_forward.rotation_amplitudes + dt * forward.rotation_amplitude_rates).clamp_min(0.0)
        previous_rotation_amplitudes = (context_backward.rotation_amplitudes + dt * backward.rotation_amplitude_rates).clamp_min(0.0)

        new_templates = self.context_head.templates.to(descriptor)
        next_translation_carrier = (
            next_position.unsqueeze(-2) + next_translation_amplitudes.unsqueeze(-1) * new_templates
            if context.mask.translation
            else torch.zeros_like(context_forward.translation_carrier)
        )
        previous_translation_carrier = (
            previous_position.unsqueeze(-2) + previous_translation_amplitudes.unsqueeze(-1) * new_templates
            if context.mask.translation
            else torch.zeros_like(context_backward.translation_carrier)
        )
        next_rotation_carrier = (
            next_rotation_amplitudes.unsqueeze(-1) * orbit(next_rotation, new_templates) if context.mask.rotation 
            else torch.zeros_like(context_forward.rotation_carrier)
        )
        previous_rotation_carrier = (
            previous_rotation_amplitudes.unsqueeze(-1) * orbit(previous_rotation, new_templates)
            if context.mask.rotation
            else torch.zeros_like(context_backward.rotation)
        )
        
        next_invariants = (
            context_forward.physical_scalars + dt * self.invariant_transition(forward.physical_scalars)
        )
        previous_invariants = (
            context_backward.physical_scalars + dt * self.invariant_transition(backward.physical_scalars)
        )

        next_artifacts = context_forward.artifacts + dt * self.artifact_transition(forward.artifacts)
        previous_artifacts = context_backward.artifacts + dt * self.artifact_transition(backward.artifacts)

        next_rotation_6d = context_forward.rotation_6d if not context.mask.rotation else matrix_to_rotation_6d(next_rotation)
        previous_rotation_6d = context_backward.rotation_6d if not context.mask.rotation else matrix_to_rotation_6d(previous_rotation)

        next_packed = pack_context(
            next_position,
            next_rotation_6d,
            next_translation_amplitudes,
            next_rotation_amplitudes,
            next_translation_carrier,
            next_rotation_carrier,
            next_invariants,
            next_artifacts,
        )
        previous_packed = pack_context(
            previous_position,
            previous_rotation_6d,
            previous_translation_amplitudes,
            previous_rotation_amplitudes,
            previous_translation_carrier,
            previous_rotation_carrier,
            previous_invariants,
            previous_artifacts,
        )
        
        diagnostics = SE3MotionDiagnostics(
            normalised_forward_difference=normalised_rate,
            forward_motion=forward.packed,
            backward_motion=backward.packed,
            predicted_next_embedding=next_packed,
            predicted_previous_embedding=previous_packed,
            forward_sectors=forward,
            backward_sectors=backward,
            predicted_next_position=next_position,
            predicted_previous_position=previous_position,
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
        return SE3Prediction(
            position=context.position, 
            linear_velocity=self._frame(forward.linear_velocity), 
            rotation_6d=context.rotation_6d, 
            rotation_matrix=context.rotation, 
            angular_velocity=self._frame(forward.omega), 
            frame_latent=context.packed, 
            feature_maps=maps, 
            motion=diagnostics, 
            context_sectors=context,
        )

    def forward_with_zero_artifacts(self, context_rgb, context_time=None) -> SE3Prediction:
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
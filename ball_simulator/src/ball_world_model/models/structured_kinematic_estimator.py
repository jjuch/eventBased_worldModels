"""Rotation-only observer with SO(3) x artifact and so(3) x artifact latents.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .kinematic_encoder import CoordinateAwareFrameEncoder
from .latent_motion import RunningDeltaNormaliser, SpatialMotionEncoder
from .structured_so3 import RichSO3ContextHead, RichSO3MotionHead, world_step, orbit

@dataclass(frozen=True)
class RichMotionDiagnostics:
    normalised_forward_difference: torch.Tensor
    forward_motion: torch.Tensor
    backward_motion: torch.Tensor
    predicted_next_embedding: torch.Tensor
    predicted_previous_embedding: torch.Tensor
    forward_angular_velocity: torch.Tensor
    backward_angular_velocity: torch.Tensor
    forward_sectors: object
    backward_sectors: object
    predicted_next_rotation: torch.Tensor
    predicted_previous_rotation: torch.Tensor
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
    context_sectors: object


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
        **_:object) -> None:
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

        la, lm = self.context_head.layout, self.motion_head.layout
        self.artifact_transition = nn.Sequential(
            nn.Linear(lm.motion_artifact_dim, la.context_artifact_dim),
            nn.LayerNorm(la.context_artifact_dim),
            nn.SiLU(),
            nn.Linear(la.context_artifact_dim, la.context_artifact_dim)
        )


    @staticmethod
    def _frame_velocity(pair: torch.Tensor) -> torch.Tensor:
        return torch.cat((pair, pair[:, -1:]), dim=1)

    def _pack_context(self, r6, alpha, U, c, a):
        return torch.cat((r6, alpha, U.flatten(-2), c, a), -1)

    def forward(self, context_rgb, context_time=None) -> RichPrediction:
        maps = self.frame_encoder.encode_feature_maps(context_rgb)
        descriptor = self.frame_encoder.embeddings_from_maps(maps)
        context = self.context_head(descriptor)
        batch, frames = descriptor.shape[:2]

        if context_time is None:
            context_time = (
                torch.arange(frames, device=descriptor.device, dtype=descriptor.dtype) * self.default_frame_dt
            ).unsqueeze(0).expand(batch, -1)

        dt = torch.diff(context_time.to(descriptor), 1).unsqueeze(-1).clamp_min(1e-8)
        rate = (maps[:, 1:] - maps[:, :-1]) / dt.unsqueeze(-1).unsqueeze(-1)
        norm = self.delta_normaliser(rate)

        f = self.motion_encoder(maps[:, :-1], maps[:, 1:], norm)
        b = self.motion_encoder(maps[:, 1:], maps[:, :-1], -norm)
        mf = self.motion_head(f, context.carrier[:, :-1])
        mb = self.motion_head(b, context.carrier[:, 1:])
        Rn = world_step(context.rotation[:, :-1], mf.omega, dt)
        Rp = world_step(context.rotation[:, 1:], mb.omega, dt)
        an = (context.amplitudes[:, :-1] + dt * mf.amplitude_rate).clamp_min(0.)
        ap = (context.amplitudes[:, 1:] + dt * mb.amplitude_rate).clamp_min(0.)
        Un = an.unsqueeze(-1) * orbit(Rn, self.context_head.templates)
        Up = ap.unsqueeze(-1) * orbit(Rp, self.context_head.templates)
        cn = context.physical_invariants[:, :-1]
        cp = context.physical_invariants[:, 1:]
        aa_n = context.artifacts[:, :-1] + dt * self.artifact_transition(mf.artifact)
        aa_p = context.artifacts[:, 1:] + dt * self.artifact_transition(mb.artifact)
        pn = self._pack_context(Rn[..., :, :2].transpose(-1, -2).flatten(-2), an, Un, cn, aa_n)
        pp = self._pack_context(Rp[..., :, :2].transpose(-1, -2).flatten(-2), ap, Up, cp, aa_p)
        diag = RichMotionDiagnostics(
            normalised_forward_difference=norm, 
            forward_motion=mf.packed, 
            backward_motion=mb.packed, 
            predicted_next_embedding=pn, 
            predicted_previous_embedding=pp, 
            forward_angular_velocity=mf.omega, 
            backward_angular_velocity=mb.omega, 
            forward_sectors=mf, 
            backward_sectors=mb, 
            predicted_next_rotation=Rn, 
            predicted_previous_rotation=Rp
        )
        return RichPrediction(
            position=None, 
            linear_velocity=None, 
            rotation_6d=context.rotation_6d, 
            rotation_matrix=context.rotation, 
            angular_velocity=self._frame_velocity(mf.omega), 
            frame_latent=context.packed, 
            feature_maps=maps, 
            motion=diag, 
            context_sectors=context
        )
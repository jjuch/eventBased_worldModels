"""
Unified fixed-width SE(3)-compatible context and motion latents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .rotation import matrix_to_rotation_6d, rotation_6d_to_matrix, so3_exp

from ball_world_model.models.kinematic_estimator import TaskKind

PHYSICAL_MARKERS = F.normalize(torch.tensor(
    [
        [1., 1., 1.],
        [1., -1., -1.],
        [-1., 1., -1.],
        [-1., -1., 1.]
    ]
), dim=-1)


def fibonacci_sphere(count: int) -> torch.Tensor:
    i = torch.arange(count, dtype=torch.float32)
    phi = (1. + 5.**0.5) / 2.
    z = 1. - 2. * (i + .5) / count
    r = torch.sqrt((1 - z*z).clamp_min(0.))
    theta = 2. * torch.pi * i / phi
    return torch.stack((r * torch.cos(theta), r * torch.sin(theta), z), -1)

def canonical_templates(count: int = 24) -> torch.Tensor:
    if count < 4:
        raise ValueError("geometric_channels must be at least four.")
    return torch.cat((PHYSICAL_MARKERS, fibonacci_sphere(count - 4)), 0)


@dataclass(frozen=True)
class SectorMask:
    translation: bool
    rotation: bool

    @classmethod
    def for_task(cls, task: TaskKind):
        if task not in {"translation", "rotation", "combined"}:
            raise ValueError(
                f"A task should be 'translation', 'rotation', or 'combined',"
                f"instead {task} was given."
            )
        return cls(task in {"translation", "combined"}, task in {"rotation", "combined"})


@dataclass(frozen=True)
class SE3Layout:
    total_dim: int = 256
    channels: int = 24
    invariant_dim: int = 16

    @property
    def context_structured_dim(self):
        return 3 + 6 + 2 * self.channels + 6 * self.channels + self.invariant_dim

    @property
    def motion_structured_dim(self):
        return 6 + 2 * self.channels + 6 * self.channels + self.invariant_dim

    @property
    def context_artifact_dim(self):
        return self.total_dim - self.context_structured_dim

    @property
    def motion_artifact_dim(self):
        return self.total_dim - self.motion_structured_dim

    def validate(self):
        if self.context_artifact_dim <= 0 or self.motion_artifact_dim <= 0:
            raise ValueError("total_dim is too small to include enough artifact dimensions.")


@dataclass(frozen=True)
class SE3ContextLatent:
    position: torch.Tensor
    rotation_6d: torch.Tensor
    rotation: torch.Tensor
    translation_amplitudes: torch.Tensor
    rotation_amplitudes: torch.Tensor
    translation_carrier: torch.Tensor
    rotation_carrier: torch.Tensor
    physical_scalars: torch.Tensor
    artifacts: torch.Tensor
    packed: torch.Tensor
    mask: SectorMask

    @property
    def physical_invariants(self):
        return self.physical_scalars
    
    @property
    def amplitudes(self):
        return self.translation_amplitudes, self.rotation_amplitudes

    @property
    def carrier(self):
        return self. translation_carrier, self.rotation_carrier


@dataclass(frozen=True)
class SE3MotionLatent:
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    translation_amplitude_rates: torch.Tensor
    rotation_amplitude_rates: torch.Tensor
    translation_carrier_tangents: torch.Tensor
    rotation_carrier_tangents: torch.Tensor
    physical_scalars: torch.Tensor
    artifacts: torch.Tensor
    packed: torch.Tensor
    mask: SectorMask

    @property
    def omega(self):
        return self.angular_velocity

    @property
    def amplitude_rate(self):
        return self. translation_amplitude_rates, self.rotation_amplitude_rates

    @property
    def carrier_tangent(self):
        return self.rotation_carrier_tangents, self.rotation_carrier_tangents

    @property
    def physical_invariants(self):
        return self.physical_scalars


def identity_rotation(shape: tuple, reference) -> torch.Tensor:
    return torch.eye(3, device=reference.device, dtype=reference.dtype).expand(*shape, 3, 3)

def pack_context(p, r6, at, ar, Ut, Ur, c, a):
    return torch.cat((
        p, r6, at, ar, Ut.flatten(-2), Ur.flatten(-2), c, a
    ), dim=-1)

def pack_motion(v, w, dat, dar, dUt, dUr, q, a):
    return torch.cat((
        v, w, dat, dar, dUt.flatten(-2), dUr.flatten(-2), q, a
    ), dim=-1)

def translation_step(p, v, dt):
    return p + dt * v

def rotation_step(R, w, dt):
    return so3_exp(w * dt) @ R

def orbit(rotation: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,kj->...ki", rotation, templates.to(rotation))

def tangent_rotation(carrier: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Induced tangent action u_dot = omega x u for every landmark."""
    return torch.cross(omega.unsqueeze(-2).expand_as(carrier), carrier, dim=-1)


class SE3ContextHead(nn.Module):
    def __init__(self, input_dim=256, total_dim=256, channels=24, invariant_dim=16, task: TaskKind="combined"):
        super().__init__()
        self.layout = SE3Layout(total_dim, channels, invariant_dim)
        self.layout.validate()

        self.mask = SectorMask.for_task(task)
        self.register_buffer("templates", canonical_templates(channels))
        self.position = nn.Linear(input_dim, 3)
        self.rotation=nn.Linear(input_dim,6)
        self.translation_amplitudes = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.Softplus()
        )
        self.rotation_amplitudes = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.Softplus()
        )
        self.scalars = nn.Sequential(
            nn.Linear(input_dim, invariant_dim),
            nn.LayerNorm(invariant_dim),
            nn.SiLU()
        )
        self.artifacts = nn.Sequential(
            nn.Linear(input_dim, self.layout.context_artifact_dim),
            nn.LayerNorm(self.layout.context_artifact_dim),
            nn.SiLU()
        )


    def forward(self, x):
        shape = x.shape[:-1]
        position = self.position(x) if self.mask.translation else x.new_zeros(*shape, 3)
        r6 = self.rotation(x) if self.mask.rotation else matrix_to_rotation_6d(identity_rotation(shape, x))
        R = rotation_6d_to_matrix(r6)
        alpha_translation = self.translation_amplitudes(x) if self.mask.translation else x.new_zeros(*shape, self.layout.channels)
        alpha_rotation = self.rotation_amplitudes(x) if self.mask.rotation else x.new_zeros(*shape, self.layout.channels)

        U_translation = (
            position.unsqueeze(-2) + alpha_translation.unsqueeze(-1) * self.templates.to(x)
            if self.mask.translation 
            else x.new_zeros(*shape, self.layout.channels, 3)
        )
        U_rotation = (
            alpha_rotation.unsqueeze(-1) * orbit(R, self.templates)
            if self.mask.rotation
            else x.new_zeros(*shape, self.layout.channels, 3)
        )
        c = self.scalars(x)
        a = self.artifacts(x)

        packed = pack_context(position, r6, alpha_translation, alpha_rotation, U_translation, U_rotation, c, a)
    
        return SE3ContextLatent(
            position,
            r6,
            R,
            alpha_translation,
            alpha_rotation,
            U_translation,
            U_rotation,
            c, a,
            packed,
            self.mask
        )


class SE3MotionHead(nn.Module):
    def __init__(self, input_dim=256, total_dim=256, channels=24, invariant_dim=16, task: TaskKind="combined"):
        super().__init__()
        self.layout = SE3Layout(total_dim, channels, invariant_dim)
        self.layout.validate()
        self.mask = SectorMask.for_task(task)

        self.velocity = nn.Linear(input_dim, 3)
        self.omega = nn.Linear(input_dim, 3)

        self.translation_amplitude_rate = nn.Linear(input_dim, channels)
        self.rotation_amplitude_rate = nn.Linear(input_dim, channels)
        self.scalars = nn.Sequential(
            nn.Linear(input_dim, invariant_dim),
            nn.LayerNorm(invariant_dim),
            nn.SiLU()
        )
        self.artifacts = nn.Sequential(
            nn.Linear(input_dim, self.layout.motion_artifact_dim), 
            nn.LayerNorm(self.layout.motion_artifact_dim),
            nn.SiLU()
        )


    def _translation_directions(self, c: SE3ContextLatent) -> torch.Tensor:
        return F.normalize(c.translation_carrier - c.position.unsqueeze(-2), dim=-1, eps=1e-6)

    def _rotation_directions(self, c: SE3ContextLatent) -> torch.Tensor:
        return F.normalize(c.rotation_carrier, dim=-1, eps=1e-6)

    
    def forward(self, x, context: SE3ContextLatent) -> SE3MotionLatent:
        shape = x.shape[:-1]
        velocity = self.velocity(x) if self.mask.translation else x.new_zeros(*shape, 3)
        w = self.omega(x) if self.mask.rotation else x.new_zeros(*shape, 3)
        translation_amplitude_rate = (
            self.translation_amplitude_rate(x)
            if self.mask.translation
            else x.new_zeros(*shape, self.layout.channels)
        )
        rotation_amplitude_rate = (
            self.rotation_amplitude_rate(x)
            if self.mask.rotation
            else x.new_zeros(*shape, self.layout.channels)
        )
        translation_dU = (
            velocity.unsqueeze(-2) + translation_amplitude_rate.unsqueeze(-1) * self._translation_directions(context)
            if self.mask.translation
            else x.new_zeros(*shape, self.layout.channels, 3)
        )
        base = context.rotation_carrier
        rotation_dU = (
            tangent_rotation(base, w) + rotation_amplitude_rate.unsqueeze(-1) * self._rotation_directions(context) 
            if self.mask.rotation
            else x.new_zeros(*shape, self.layout.channels, 3)
        )

        q = self.scalars(x)
        a = self.artifacts(x)

        packed = pack_motion(velocity, w, translation_amplitude_rate, rotation_amplitude_rate, translation_dU, rotation_dU, q, a)
        return SE3MotionLatent(
            linear_velocity=velocity,
            angular_velocity=w,
            translation_amplitude_rates=translation_amplitude_rate,
            rotation_amplitude_rates=rotation_amplitude_rate,
            translation_carrier_tangents=translation_dU,
            rotation_carrier_tangents=rotation_dU,
            physical_scalars=q,
            artifacts=a,
            packed=packed,
            mask=self.mask,
        )


class SE3PhysicalTeacher(nn.Module):
    """Parameter-free equivariant lift, richer than identity but algebra preserving."""
    def __init__(self, channels=24) -> None:
        super().__init__()
        self.register_buffer("templates", canonical_templates(channels))

    def context(self, p: torch.Tensor, R: torch.Tensor, mask: SectorMask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = p.shape[:-1]
        one = torch.ones(*shape, len(self.templates), device=p.device, dtype=p.dtype)
        zero = torch.zeros_like(one)
        translation_amplitude = one if mask.translation else zero
        rotation_amplitude = one if mask.rotation else zero
        translation_carrier = ( # Ut
            p.unsqueeze(-2) + self.templates.to(p) 
            if mask.translation
            else p.new_zeros(*shape, len(self.templates), 3)
        )
        rotation_carrier = ( #Ur
            orbit(R, self.templates)
            if mask.rotation
            else p.new_zeros(*shape, len(self.templates), 3)
        )
        return translation_amplitude, rotation_amplitude, translation_carrier, rotation_carrier
    

    def motion(self, p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, mask: SectorMask) -> tuple[torch.Tensor, torch.Tensor]:
        del p
        shape = v.shape[:-1]
        translation_carrier_rate = ( #dUt
            v.unsqueeze(-2).expand(*shape, len(self.templates), 3)
            if mask.translation
            else v.new_zeros(*shape, len(self.templates), 3)
        )
        base = orbit(R, self.templates)
        rotation_carrier_rate = ( #dUr
            tangent_rotation(base, w)
            if mask.rotation
            else v.new_zeros(*shape, len(self.templates), 3)
        )
        return translation_carrier_rate, rotation_carrier_rate
"""K=24 SO(3)/so(3) carrier latents with separate Euclidean artifacts.

The geometric sector is deliberately richer than the raw physical state.  It
contains a proper group element plus the orbit of canonical body landmarks.
The orbit is a direct-sum representation of SO(3), so finite rotations and
Lie-algebra generators act on it without an unrestricted decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .rotation import matrix_to_rotation_6d, rotation_6d_to_matrix, so3_exp

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

def orbit(rotation: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,kj->...ki", rotation, templates.to(rotation))

def tangent(carrier: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Induced tangent action u_dot = omega x u for every landmark."""
    return torch.cross(omega.unsqueeze(-2).expand_as(carrier), carrier, dim=-1)

def world_step(rotation: torch.Tensor, omega: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    return so3_exp(omega*dt) @ rotation

@dataclass(frozen=True)
class ContextSectors:
    rotation_6d: torch.Tensor
    rotation: torch.Tensor
    amplitudes: torch.Tensor
    carrier: torch.Tensor
    physical_invariants: torch.Tensor
    artifacts: torch.Tensor
    packed: torch.Tensor


@dataclass(frozen=True)
class MotionSector:
    omega: torch.Tensor
    amplitude_rate: torch.Tensor
    carrier_tangent: torch.Tensor
    physical_invariants: torch.Tensor
    artifact: torch.Tensor
    packed: torch.Tensor


class RichLayout:
    def __init__(self, total_dim=256, channels=24, invariant_dim=16):
        self.total_dim, self.channels, self.invariant_dim = total_dim, channels, invariant_dim
        self.context_structured_dim = 6 + channels + 3 * channels + invariant_dim
        self.motion_structured_dim = 3 + channels + 3 * channels + invariant_dim
        self.context_artifact_dim = total_dim - self.context_structured_dim
        self.motion_artifact_dim = total_dim  - self.motion_structured_dim
        if min(self.context_artifact_dim, self.motion_artifact_dim) <= 0:
            raise ValueError("total_sim is too small for the requested structured sectors.")


class RichSO3ContextHead(nn.Module):
    def __init__(self, input_dim=256, total_dim=256, channels=24, invariant_dim=16):
        super().__init__()
        self.layout = RichLayout(total_dim, channels, invariant_dim)

        self.rotation = nn.Linear(input_dim, 6)
        self.amplitudes = nn.Sequential(nn.Linear(input_dim, channels), nn.Softplus())
        self.invariants = nn.Sequential(nn.Linear(input_dim, invariant_dim), nn.LayerNorm(invariant_dim), nn.SiLU())
        self.artifacts = nn.Sequential(
            nn.Linear(input_dim, self.layout.context_artifact_dim), 
            nn.LayerNorm(self.layout.context_artifact_dim), 
            nn.SiLU()
        )
        self.register_buffer("templates", canonical_templates(channels))


    def forward(self, x):
        r6 = self.rotation(x)
        R = rotation_6d_to_matrix(r6)
        alpha = self.amplitudes(x)

        U = alpha.unsqueeze(-1) * orbit(R, self.templates)
        c = self.invariants(x)
        a = self.artifacts(x)

        packed = torch.cat((r6, alpha, U.flatten(-2), c, a), -1)
        return ContextSectors(r6, R, alpha, U, c, a, packed)


class RichSO3MotionHead(nn.Module):
    def __init__(self, input_dim=256, total_dim=256, channels=24, invariant_dim=16):
        super().__init__()
        self.layout = RichLayout(total_dim, channels, invariant_dim)

        self.omega = nn.Linear(input_dim, 3)
        self.amplitude_rate = nn.Linear(input_dim, channels)
        self.invariants = nn.Sequential(
            nn.Linear(input_dim, invariant_dim),
            nn.LayerNorm(invariant_dim),
            nn.SiLU()
        )
        self.artifacts = nn.Sequential(
            nn.Linear(input_dim, self.layout.motion_artifact_dim),
            nn.LayerNorm(self.layout.motion_artifact_dim),
            nn.SiLU()
        )

    def forward(self, x, carrier):
        w = self.omega(x)
        da = self.amplitude_rate(x)
        dU = tangent(carrier, w)

        c = self.invariants(x)
        a = self.artifacts(x)
        packed = torch.cat((w, da, dU.flatten(-2), c, a), -1)
        return MotionSector(w, da, dU, c, a, packed)


class RichPhysicalStateTeacher(nn.Module):
    """Parameter-free equivariant lift, richer than identity but algebra preserving."""
    def __init__(self, channels=24) -> None:
        super().__init__()
        self.register_buffer("templates", canonical_templates(channels))

    def context(self, rotation) -> tuple[torch.Tensor, torch.Tensor]:
        amplitude = torch.ones(rotation.shape[:-2] + (len(self.templates),), device=rotation.device, dtype=rotation.dtype)
        return amplitude, orbit(rotation, self.templates)

    def motion(self, rotation, omega) -> tuple[torch.Tensor, torch.Tensor]:
        return omega, tangent(orbit(rotation, self.templates), omega)
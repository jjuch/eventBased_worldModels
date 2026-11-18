"""Minimal flexible SE(3) teacher.

The physical core is exact. Flexibility remains in the student feature encoder and in the learned mapping to pose/twist/acceleration. The teacher does not dictate a complete 256-dimensional neural latent. It supervises one compact state and one redundant affine carrier that is generated analytically from that state.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn

from ball_world_model.geometry.se3 import carrier_acceleration, carrier_state, carrier_velocity

@dataclass(frozen=True)
class TeacherTargets:
    carrier: Tensor
    carrier_velocity: Tensor
    carrier_acceleration: Tensor


def fibonacci_sphere(count: int, radius: float = 1.0) -> Tensor:
    index = torch.arange(count, dtype=torch.float64)
    golden = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0, dtype=torch.float64)))
    y = 1.0 - 2.0 * (index + 0.5) / count
    radial = torch.sqrt((1.0 - y.square()).clamp_min(0.0))
    angle = golden * index
    points = torch.stack((radial * torch.cos(angle), y, radial * torch.sin(angle)), dim=-1)
    return (radius * points).float()

class SE3Teacher(nn.Module):
    def __init__(self, carrier_points: int = 16, carrier_radius: float = 0.10) -> None:
        super().__init__()
        self.register_buffer("templates", fibonacci_sphere(carrier_points, carrier_radius))

    def forward(
        self, position: Tensor, rotation: Tensor, linear: Tensor, angular: Tensor, linear_acceleration: Tensor, angular_acceleration: Tensor,
    ) -> TeacherTargets:
        templates = self.templates.to(position)
        return TeacherTargets(
            carrier_state(position, rotation, templates),
            carrier_velocity(rotation, linear, angular, templates),
            carrier_acceleration(rotation, angular, linear_acceleration, angular_acceleration, templates),
        )


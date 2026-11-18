"""Small, explicit SE(3) geometry layer.

Conventions
-----------
* ``R`` maps body vectors to world vectors.
* ``p`` is the world position of the body-frame origin.
* ``v = p_dot`` and ``omega`` are expressed in the world frame.
* ``a = v_dot`` and ``alpha = omega_dot`` are expressed in the world frame.

The affine carrier is X_k = p + R b_k. Its derivatives couple translation and
rotation and are the teacher's structured high-dimensional representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

_EPS = 1.0e-7

def skew(vector: Tensor) -> Tensor:
    """Return the cross-product matrix for vectors shaped ``[..., 3]``."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )


def so3_exp(rotation_vector: Tensor) -> Tensor:
    """Stable exponential map from axis-angle vectors to rotation matrices."""
    theta2 = (rotation_vector * rotation_vector).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(_EPS * _EPS))
    K = skew(rotation_vector)

    eye = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    eye = eye.expand(*rotation_vector.shape[:-1], 3, 3)

    small = theta2 < 1.0e-8
    sinc = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120., torch.sin(theta) / theta)
    one_minus_cos = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(_EPS * _EPS),
    )
    return eye + sinc.unsqueeze(-1) * K + one_minus_cos.unsqueeze(-1) * (K @ K)


def so3_log(rotation:Tensor) -> Tensor:
    """Logarithm map for ordinary training/evaluation errors."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    theta = torch.acos(cosine)
    vee = torch.stack(
        (rotation[..., 2, 1] - rotation[..., 1, 2],
         rotation[..., 0, 2] - rotation[..., 2, 0],
         rotation[..., 1, 0] - rotation[..., 0, 1]), dim=-1,
    )
    scale = theta / (2.0 * torch.sin(theta).clamp_min(_EPS))
    small = theta.abs() < 1.0e-4
    return torch.where(small.unsqueeze(-1), 0.5 * vee, scale.unsqueeze(-1) * vee)


def rotation_6d_to_matrix(value: Tensor) -> Tensor:
    """Continuous 6D rotation representation of Zhou et al."""
    first, second = value[..., :3], value[..., 3:]
    b1 = torch.nn.functional.normalize(first, dim=-1, eps=1.0e-6)
    b2 = torch.nn.functional.normalize(second - (b1 * second).sum(-1, keepdim=True) * b1, dim=-1, eps=1.0e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(rotation: Tensor) -> Tensor:
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def compose(p1: Tensor, r1: Tensor, p2: Tensor, r2: Tensor) -> tuple[Tensor, Tensor]:
    """Compose ``(p1, r1) * (p2, r2)``."""
    return p1 + torch.einsum("...ij,...j->...i", r1, p2), r1 @ r2


def inverse(position: Tensor, rotation: Tensor) -> tuple[Tensor, Tensor]:
    r_inv = rotation.transpose(-1, -2)
    return -torch.einsum("...ij,...j->...i", r_inv, position), r_inv


def transform_points(position: Tensor, rotation: Tensor, points: Tensor) -> Tensor:
    return position.unsqueeze(-2) + torch.einsum("...ij,kj->...ki", rotation, points)


def carrier_state(position: Tensor, rotation: Tensor, templates: Tensor) -> Tensor:
    return transform_points(position, rotation, templates)


def carrier_velocity(rotation: Tensor, linear: Tensor, angular: Tensor, templates: Tensor) -> Tensor:
    offsets = torch.einsum("...ij,kj->...ki", rotation, templates)
    omega = angular.unsqueeze(-2).expand_as(offsets)
    return linear.unsqueeze(-2) + torch.cross(omega, offsets, dim=-1)


def carrier_acceleration(
    rotation: Tensor,
    angular: Tensor,
    linear_acceleration: Tensor,
    angular_acceleration: Tensor,
    templates: Tensor,
) -> Tensor:
    offsets = torch.einsum("...ij,kj->...ki", rotation, templates)
    omega = angular.unsqueeze(-2).expand_as(offsets)
    alpha = angular_acceleration.unsqueeze(-2).expand_as(offsets)
    tangential = torch.cross(alpha, offsets, dim=-1)
    centripetal = torch.cross(omega, torch.cross(omega, offsets, dim=-1), dim=-1)
    return linear_acceleration.unsqueeze(-2) + tangential + centripetal

def propagate_world(
    position: Tensor,
    rotation: Tensor,
    linear: Tensor,
    angular: Tensor,
    dt: Tensor,
    linear_acceleration: Tensor | None = None,
    angular_acceleration: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Second-order world-frame propagation.

    The angular update assumes constant world angular acceleration over one short interval. For sufficiently fast sampling frequencies this is a transparent and stable approximation.
    """
    while dt.ndim < position.ndim:
        dt = dt.unsqueeze(-1)
    a = torch.zeros_like(linear) if linear_acceleration is None else linear_acceleration
    alpha = torch.zeros_like(angular) if angular_acceleration is None else angular_acceleration
    next_position = position + dt * linear + 0.5 * dt.square() * a
    next_linear = linear + dt * a
    rotation_increment = so3_exp(dt * angular + 0.5 * dt.square() * alpha)
    next_rotation = rotation_increment @ rotation
    next_angular = angular + dt * alpha
    return next_position, next_rotation, next_linear, next_angular

from __future__ import annotations

import torch
import torch.nn.functional as F

def quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = F.normalize(quaternion, dim=-1)
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    first, second = rotation[..., :3], rotation[..., 3:]
    basis_1 = F.normalize(first, dim=-1)
    # Construct a basis perpendicular to basis_1
    basis_2 = F.normalize(second - (basis_1 * second).sum(-1, keepdim=True) * basis_1, dim=-1)
    # Third basis element is the cross product
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-1)


def rotation_geodesic_error(
    predicted_matrix: torch.Tensor,
    target_matrix: torch.Tensor,
) -> torch.Tensor:
    relative = predicted_matrix.transpose(-1, -2) @ target_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
        -1.0 + 1.0e-7, 1.0 - 1.0e-7
    )
    return torch.acos(cosine)


def so3_exp(angular_increment: torch.Tensor) -> torch.Tensor:
    """Stable exponential map from axis-angle vectors to SO(3)."""
    theta_squared = (angular_increment * angular_increment).sum(-1, keepdim=True)
    theta = torch.sqrt(theta_squared.clamp_min(1.0e-16))

    x, y, z = angular_increment.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(angular_increment.shape[:-1] + (3, 3))

    identity = torch.eye(3, device=angular_increment.device, dtype=angular_increment.dtype)
    identity = identity.expand(angular_increment.shape[:-1] + (3, 3))
    small = theta < 1.0e-4
    a = torch.where(small, 1.0 - theta_squared / 6.0, torch.sin(theta) / theta)
    b = torch.where(
        small,
        0.5 - theta_squared / 24.0,
        (1.0 - torch.cos(theta)) / theta_squared.clamp_min(1.0e-16),
    )
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)

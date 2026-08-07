from __future__ import annotations

import torch
import torch.nn.functional as F

def quaternion_xyzw_to_matrix(quaternion: torch.Torch) -> torch.Tensor:
    quaternion = F.normalize(quaternion, dim=-1)
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
            2.0 * (x * y + z * w), 1 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
            2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1 - 2.0 * (x * x + y * y), 
        ), dim=-1
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :, :2].transpose(-1, -2).reshape(matrix.shape[:2] + (6,))


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
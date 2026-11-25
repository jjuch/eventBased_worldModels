"""Orientation errors for bodies that look identical under some rotations (e.g. a plain cube)."""

from __future__ import annotations

import torch
from torch import Tensor


def geodesic_angle(first: Tensor, second: Tensor) -> Tensor:
    """Angle in rad between rotation matrices [..., 3, 3], accurate over the full range [0, π].

    Uses atan2(sin, cos) of the relative rotation rather than acos or a log map, so an angle of
    exactly π (common between symmetry-equivalent orientations) is not mistaken for zero.
    """
    relative = first.transpose(-1, -2) @ second
    cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    vee = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(vee, dim=-1)
    return torch.atan2(sine, cosine)


def symmetric_rotation_angle(first: Tensor, second: Tensor, symmetries: Tensor) -> Tensor:
    """Smallest angle in rad between ``first`` and any orientation equivalent to ``second``.

    first, second  [..., 3, 3] rotations (body to world)
    symmetries     [G, 3, 3] body-frame rotations S under which the body looks identical, so R
                   and R S are indistinguishable; must include the identity
    returns        [...]
    """
    symmetries = symmetries.to(device=second.device, dtype=second.dtype)
    candidates = second.unsqueeze(-3) @ symmetries
    return geodesic_angle(first.unsqueeze(-3), candidates).amin(dim=-1)

from __future__ import annotations

import math

import pytest
import torch
from scipy.spatial.transform import Rotation

from ball_simulator.trajectories.config import ShapeConfig
from ball_simulator.trajectories.shapes import appearance_symmetries, build_shape, marker_layout
from ball_world_model.geometry.symmetry import geodesic_angle, symmetric_rotation_angle


def _cube_symmetries() -> torch.Tensor:
    cube = build_shape(ShapeConfig(kind="cube"), 0.1)
    return torch.as_tensor(appearance_symmetries(cube, marker_layout(cube, "none")))


def test_geodesic_angle_covers_half_turns_and_small_angles():
    identity = torch.eye(3, dtype=torch.float64)
    half_turn = torch.as_tensor(Rotation.from_rotvec([0.0, 0.0, math.pi]).as_matrix())
    small = torch.as_tensor(Rotation.from_rotvec([1e-5, 0.0, 0.0]).as_matrix())
    assert geodesic_angle(identity, half_turn).item() == pytest.approx(math.pi)
    assert geodesic_angle(identity, small).item() == pytest.approx(1e-5, rel=1e-6)


def test_symmetry_equivalent_orientations_have_zero_error():
    symmetries = _cube_symmetries()
    rotation = torch.as_tensor(Rotation.random(16, random_state=3).as_matrix())
    equivalent = rotation @ symmetries[torch.arange(16) % len(symmetries)]
    error = symmetric_rotation_angle(rotation, equivalent, symmetries)
    assert error.abs().max().item() < 1e-6


def test_symmetric_error_never_exceeds_raw_error():
    symmetries = _cube_symmetries()
    first = torch.as_tensor(Rotation.random(32, random_state=4).as_matrix())
    second = torch.as_tensor(Rotation.random(32, random_state=5).as_matrix())
    raw = geodesic_angle(first, second)
    reduced = symmetric_rotation_angle(first, second, symmetries)
    assert torch.all(reduced <= raw + 1e-9)
    # Once cube symmetry is removed, no two orientations are more than ~62.8° apart.
    assert reduced.max().item() < math.radians(63.0)

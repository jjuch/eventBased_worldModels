import torch
import pytest

from ball_world_model.models.rotation import rotation_geodesic_error, so3_exp
from ball_world_model.models.structured_kinematic_estimator import StructuredSO3StateEstimator
from ball_world_model.models.structured_so3 import (
    RichPhysicalStateTeacher,
    PHYSICAL_MARKERS,
    tangent,
    orbit,
    world_step,
    canonical_templates,
    RichLayout,
)


def test_orbits_lift_is_equivariant_under_group_composition():
    C = canonical_templates(24)
    first = so3_exp(torch.randn(5, 3))
    second = so3_exp(torch.randn(5, 3))
    direct = orbit(second @ first, C)
    composed = torch.einsum(
        "...ij, ...mj->...mi", second, orbit(first, C)
    )
    torch.testing.assert_close(direct, composed, atol=1e-6, rtol=1e-6) 


def test_rich_layout():
    x = RichLayout(256, 24, 16)
    assert x.context_structured_dim == 118 and x.context_artifact_dim == 138
    assert x.motion_structured_dim == 115 and x.motion_artifact_dim == 141


def test_tangent_matches_finite_difference():
    C = canonical_templates(24)
    rotation = so3_exp(torch.randn(2,3)*.2)
    omega = torch.randn(2, 3)
    U = orbit(rotation, C)
    analytic = tangent(U, omega)
    dt = torch.full((2, 1), 1e-4)
    next_orbit = orbit(world_step(rotation, omega, dt), C)
    numerical = (next_orbit - U) / dt.unsqueeze(-1)
    torch.testing.assert_close(analytic, numerical, atol=3e-3, rtol=8e-4)


def test_teacher_is_rich_but_geometry_preserving():
    teacher = RichPhysicalStateTeacher()
    rotation = so3_exp(torch.randn(5, 3))
    omega = torch.randn(5, 3)
    group, orb = teacher.context(rotation)
    algebra, tang = teacher.motion(rotation, omega)
    assert group.shape == (5, 24)
    assert orb.shape == (5, 24, 3)
    assert algebra.shape == (5, 3)
    assert tang.shape == (5, 24, 3)


def test_structured_model_shapes_and_valid_group():
    model = StructuredSO3StateEstimator(
        embedding_dim=64,
        motion_dim=48,
        descriptor_dim=64,
        keypoints=4,
        geometric_channels=8,
        physical_invariant_dim=8,
    )
    images = torch.randn(2, 5, 3, 64, 64)
    time = torch.arange(5, dtype=torch.float32).unsqueeze(0).expand(2, -1) * 0.01
    prediction = model(images, time)
    assert prediction.frame_latent.shape == (2, 5, 64)
    assert prediction.motion.forward_motion.shape == (2, 4, 48)
    assert prediction.context_sectors.artifacts.shape[-1] == 18
    assert prediction.motion.forward_sectors.artifact.shape[-1] == 5

    identity = torch.eye(3).expand_as(prediction.rotation_matrix)
    torch.testing.assert_close(
        prediction.rotation_matrix.transpose(-1, -2) @ prediction.rotation_matrix,
        identity,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.max(rotation_geodesic_error(
        prediction.motion.predicted_next_rotation,
        prediction.motion.predicted_next_rotation,
    )) < 1e-3

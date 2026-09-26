import torch

from dataclasses import replace

from ball_world_model.models.rotation import rotation_geodesic_error, so3_exp
from ball_world_model.models.structured_se3_estimator import StructuredSE3StateEstimator
from ball_world_model.models.structured_se3 import (
    SE3PhysicalTeacher,
    PHYSICAL_MARKERS,
    tangent_rotation,
    orbit,
    rotation_step,
    canonical_templates,
    SE3Layout,
    SE3ContextHead,
    SE3MotionHead,
    SectorMask,
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


def test_se3_layout():
    x = SE3Layout(256, 24, 16)
    assert x.context_structured_dim == 217 and x.context_artifact_dim == 39
    assert x.motion_structured_dim == 214 and x.motion_artifact_dim == 42


def test_tangent_matches_finite_difference():
    C = canonical_templates(24)
    rotation = so3_exp(torch.randn(2,3)*.2)
    omega = torch.randn(2, 3)
    U = orbit(rotation, C)
    analytic = tangent_rotation(U, omega)
    dt = torch.full((2, 1), 1e-4)
    next_orbit = orbit(rotation_step(rotation, omega, dt), C)
    numerical = (next_orbit - U) / dt.unsqueeze(-1)
    torch.testing.assert_close(analytic, numerical, atol=3e-3, rtol=8e-4)


def test_teacher_is_rich_but_geometry_preserving():
    teacher = SE3PhysicalTeacher()
    position = torch.randn(5, 3)
    rotation = so3_exp(torch.randn(5, 3))
    velocity = torch.randn(5, 3)
    omega = torch.randn(5, 3)
    mask = SectorMask(translation=True, rotation=True)
    at, ar, ct, cr = teacher.context(position, rotation, mask)
    crt, crr = teacher.motion(position, rotation, velocity, omega, mask)
    assert at.shape == (5, 24)
    assert ar.shape == (5, 24)
    assert ct.shape == (5, 24, 3)
    assert cr.shape == (5, 24, 3)
    assert crt.shape == (5, 24, 3)
    assert crr.shape == (5, 24, 3)


def test_structured_model_shapes_and_valid_group():
    model = StructuredSE3StateEstimator(
        embedding_dim=64,
        motion_dim=64,
        descriptor_dim=64,
        keypoints=4,
        geometric_channels=4,
        physical_invariant_dim=8,
    )
    images = torch.randn(2, 5, 3, 64, 64)
    time = torch.arange(5, dtype=torch.float32).unsqueeze(0).expand(2, -1) * 0.01
    prediction = model(images, time)
    assert prediction.frame_latent.shape == (2, 5, 64)
    assert prediction.motion.forward_motion.shape == (2, 4, 64)
    assert prediction.context_sectors.artifacts.shape[-1] == 15
    assert prediction.motion.forward_sectors.artifacts.shape[-1] == 18
    assert prediction.motion.predicted_next_invariants.shape == (2, 4, 8)
    assert prediction.motion.predicted_previous_invariants.shape == (2, 4, 8)
    assert prediction.motion.predicted_next_artifacts.shape == (2, 4, prediction.context_sectors.artifacts.shape[-1])
    assert prediction.motion.predicted_next_embedding.shape == (2, 4, 64)
    assert prediction.motion.predicted_previous_embedding.shape == (2, 4, 64)

    identity = torch.eye(3).expand_as(prediction.rotation_matrix)
    torch.testing.assert_close(
        prediction.rotation_matrix.transpose(-1, -2) @ prediction.rotation_matrix,
        identity,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.isfinite(prediction.motion.predicted_next_rotation).all()

    next_rotation = prediction.motion.predicted_next_rotation
    identity2 = torch.eye(3, device=next_rotation.device, dtype=next_rotation.dtype).expand_as(next_rotation)

    torch.testing.assert_close(next_rotation.transpose(-1, -2) @ next_rotation, identity2, atol=1e-4, rtol=1e-4)

    torch.testing.assert_close(
        torch.linalg.det(next_rotation),
        torch.ones_like(torch.linalg.det(next_rotation)),
        atol=1e-4,
        rtol=1e-4
    )
    torch.testing.assert_close(
        prediction.motion.predicted_next_invariants,
        prediction.context_sectors.physical_invariants[:, :-1]
    )


    target = prediction.context_sectors.physical_invariants[:, 1:].detach()
    loss = torch.nn.functional.mse_loss(
        prediction.motion.predicted_next_invariants,
        target,
    )
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.invariant_transition.parameters()
    )


def sliced(context):
    return replace(context, 
        position=context.position[:, :-1], 
        rotation_6d=context.rotation_6d[:, :-1], 
        rotation=context.rotation[:, :-1],
        translation_amplitudes=context.translation_amplitudes[:, :-1],
        rotation_amplitudes=context.rotation_amplitudes[:, :-1],
        translation_carrier=context.translation_carrier[:, :-1],
        rotation_carrier=context.rotation_carrier[:, :-1],
        physical_scalars=context.physical_scalars[:, :-1], 
        artifacts=context.artifacts[:, :-1], 
        packed=context.packed[:, :-1]
    )

def test_fixed_layout():
    layout = SE3Layout()
    assert layout.context_artifact_dim == 39
    assert layout.motion_artifact_dim == 42


def test_every_task_packs_to_256_and_masks_inactives_sectors():
    for task in ("translation", "rotation", "combined"):
        context = SE3ContextHead(32, 256, 24, 16, task)(torch.randn(2, 5, 32))
        motion = SE3MotionHead(32, 256, 24, 16, task)(torch.randn(2, 4, 32), sliced(context))
        assert context.packed.shape == (2, 5, 256)
        assert motion.packed.shape == (2, 4, 256)

        if task == "translation":
            assert torch.count_nonzero(context.rotation_carrier) == 0 and torch.count_nonzero(motion.angular_velocity) == 0

        if task == "rotation":
            assert torch.count_nonzero(context.translation_carrier) == 0 and torch.count_nonzero(motion.linear_velocity) == 0

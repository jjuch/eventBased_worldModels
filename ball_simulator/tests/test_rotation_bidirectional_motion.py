import torch

from ball_world_model.models.kinematic_estimator import KinematicStateEstimator
from ball_world_model.models.latent_motion import SharedRotationalCorrector
from ball_world_model.models.rotation import rotation_geodesic_error, so3_exp

def test_rotation_model_shapes_and_diagnostics():
    model = KinematicStateEstimator(
        task="rotation", embedding_dim=64, keypoints=4, motion_dim=48,
        decoder_hidden_dim=64, refinement_hidden_dim=32, refinement_iterations=2,
        angular_velocity_mean=torch.zeros(3), angular_velocity_std=torch.ones(3),
    )
    images = torch.randn(2, 10, 3, 128, 128)
    time = torch.arange(10, dtype=torch.float32).unsqueeze(0).expand(2, -1) * 0.01
    prediction = model(images, time)
    assert prediction.position is None
    assert prediction.rotation_matrix.shape == (2, 10, 3, 3)
    assert prediction.angular_velocity.shape == (2, 10, 3)
    assert prediction.motion.forward_angular_velocity.shape == (2, 9, 3)
    assert prediction.motion.backward_angular_velocity.shape == (2, 9, 3)
    assert len(prediction.motion.rotational_refinement_residuals) == 2


def test_rotational_corrector_starts_as_identity():
    corrector = SharedRotationalCorrector(hidden_dim=32)
    first = so3_exp(torch.randn(4, 3) * 0.2)
    omega = torch.randn(4, 3)
    dt = torch.full((4, 1), 0.01)
    second = so3_exp(omega * dt) @ first
    refined_first, refined_second, refined_omega, residual = corrector(first, second, omega, dt)
    torch.testing.assert_close(refined_first, first)
    torch.testing.assert_close(refined_second, second)
    torch.testing.assert_close(refined_omega, omega)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-5, rtol=1e-5)


def test_rotation_matrices_remain_valid_after_refinement():
    model = KinematicStateEstimator(
        task="rotation", embedding_dim=32, keypoints=4, motion_dim=24,
        decoder_hidden_dim=32, refinement_iterations=1,
    )
    images = torch.randn(2, 10, 3, 64, 64)
    time = torch.arange(10, dtype=torch.float32).unsqueeze(0).expand(2, -1) * 0.01
    matrix = model(images, time).rotation_matrix
    identity = torch.eye(3).expand_as(matrix)
    torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, identity, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones_like(torch.linalg.det(matrix)), atol=1e-4, rtol=1e-4)

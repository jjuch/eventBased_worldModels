import torch

from ball_world_model.models.rotation import(
    quaternion_xyzw_to_matrix,
    rotation_6d_to_matrix,
    rotation_geodesic_error,
)
from ball_world_model.models.state_estimator import StateEstimator

def test_state_estimator_shapes():
    model = StateEstimator(
        embedding_dim=96,
        temporal_depth=2,
        temporal_heads=4,
        maximum_frames=10,
    )
    output = model(torch.randn(2, 10, 3, 128, 128))
    assert output.position.shape == (2, 3)
    assert output.rotation_6d.shape == (2, 6)
    assert output.rotation_matrix.shape == (2, 3, 3)
    assert output.linear_velocity.shape == (2, 3)
    assert output.angular_velocity.shape == (2, 3)


def test_rotation_6d_produces_valid_rotation():
    matrix = rotation_6d_to_matrix(torch.randn(16, 6))
    identity = torch.eye(3).expand(16, -1, -1)
    torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(16), atol=1e-5, rtol=1e-5)


def test_identity_quaternion_has_zero_geodesic_error():
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    matrix = quaternion_xyzw_to_matrix(quaternion)
    error = rotation_geodesic_error(matrix, matrix)
    assert float(error.item()) < 1.0e-3
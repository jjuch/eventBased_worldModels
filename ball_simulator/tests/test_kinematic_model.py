import torch

from ball_world_model.models.rotation import(
    rotation_6d_to_matrix,
    so3_exp,
)
from ball_world_model.models.kinematic_estimator import KinematicStateEstimator

def test_translation_estimator_shapes():
    model = KinematicStateEstimator(
        task="translation", embedding_dim=64, temporal_depth=2, keypoints=4
    )
    output = model(torch.randn(2, 10, 3, 128, 128))
    assert output.position.shape == (2, 10, 3)
    assert output.linear_velocity.shape == (2, 10, 3)
    assert output.rotation_matrix is None
    assert output.angular_velocity is None


def test_rotation_estimator_shapes():
    model = KinematicStateEstimator(
        task="rotation", embedding_dim=64, temporal_depth=2, keypoints=4
    )
    output = model(torch.randn(2, 10, 3, 128, 128))
    assert output.position is None
    assert output.linear_velocity is None
    assert output.rotation_6d.shape == (2, 10, 6)
    assert output.rotation_matrix.shape == (2, 10, 3, 3)
    assert output.angular_velocity.shape == (2, 10, 3)



def test_rotation_outputs_are_valid():
    matrix = rotation_6d_to_matrix(torch.randn(32, 6))
    identity = torch.eye(3).expand(32, -1, -1)
    torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(32), atol=1e-5, rtol=1e-5)


def test_so3_exponential_zero_is_identity():
    matrix = so3_exp(torch.zeros(8, 3))
    expected = torch.eye(3).expand(8, -1, -1)
    torch.testing.assert_close(matrix, expected)
import torch

from ball_world_model.models.observer import SE3Observer, SE3ObserverConfig

def _small_observer() -> SE3Observer:
    return SE3Observer(SE3ObserverConfig(feature_channels=32, descriptor_dim=64, motion_dim=32, keypoints=4, keypoint_dim=8))

def test_observer_output_shapes():
    model = _small_observer()
    images = torch.randn(2, 6, 3, 64, 64)
    time = torch.arange(6).float().mul(0.01).expand(2, -1)
    output = model(images, time)

    assert output.frame_poses.position.shape == (2, 6, 3)
    assert output.frame_poses.rotation.shape == (2, 6, 3, 3)
    assert output.interval_twists.linear.shape == (2, 5, 3)
    assert output.interval_accelerations.angular.shape == (2, 5, 3)
    assert output.interval_dt.shape == (2, 5)
    assert output.current.pose.position.shape == (2, 3)
    assert output.current.twist.angular.shape == (2, 3)

    rotation = output.frame_poses.rotation
    identity = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-5)


def test_current_twist_extrapolates_last_interval_by_half_a_step():
    model = _small_observer().eval()
    images = torch.randn(1, 5, 3, 64, 64)
    output = model(images)
    half_step = 0.5 * output.interval_dt[:, -1:]
    expected = output.interval_twists.linear[:, -1] + half_step * output.interval_accelerations.linear[:, -1]
    assert torch.allclose(output.current.twist.linear, expected)


def test_every_interval_has_its_own_estimate():
    """The temporal mixer must not collapse the window to one shared twist."""
    torch.manual_seed(0)
    model = _small_observer().eval()
    output = model(torch.randn(1, 8, 3, 64, 64))
    assert output.interval_twists.linear.std(dim=1).max() > 0
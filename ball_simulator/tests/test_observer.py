import torch

from ball_world_model.geometry.se3 import compose
from ball_world_model.models.observer import SE3Observer, SE3ObserverConfig


def _small_observer() -> SE3Observer:
    return SE3Observer(
        SE3ObserverConfig(
            feature_channels=32,
            descriptor_dim=64,
            motion_dim=32,
            scene_dim=32,
            keypoints=4,
            keypoint_dim=8,
        )
    )


def test_observer_output_shapes():
    model = _small_observer()
    images = torch.randn(2, 6, 3, 64, 64)
    time = torch.arange(6).float().mul(0.01).expand(2, -1)
    output = model(images, time)

    assert output.object_in_camera.position.shape == (2, 6, 3)
    assert output.object_in_camera.rotation.shape == (2, 6, 3, 3)
    for body in (output.object, output.camera):
        assert body.frame_poses.position.shape == (2, 6, 3)
        assert body.interval_twists.linear.shape == (2, 5, 3)
        assert body.interval_accelerations.angular.shape == (2, 5, 3)
        assert body.current.twist.angular.shape == (2, 3)
    assert output.gravity.shape == (2, 3)
    assert output.interval_dt.shape == (2, 5)

    rotation = output.object_in_camera.rotation
    assert torch.allclose(
        rotation.transpose(-1, -2) @ rotation, torch.eye(3).expand_as(rotation), atol=1e-5
    )


def test_camera_is_identity_at_the_anchor_and_object_pose_is_derived():
    model = _small_observer().eval()
    output = model(torch.randn(2, 5, 3, 64, 64))
    camera = output.camera.frame_poses
    assert torch.equal(camera.position[:, -1], torch.zeros(2, 3))
    assert torch.equal(camera.rotation[:, -1], torch.eye(3).expand(2, 3, 3))
    expected = compose(
        camera.position,
        camera.rotation,
        output.object_in_camera.position,
        output.object_in_camera.rotation,
    )
    assert torch.allclose(output.object.frame_poses.position, expected[0])
    assert torch.allclose(output.object.frame_poses.rotation, expected[1])


def test_current_twist_extrapolates_last_interval_by_half_a_step():
    model = _small_observer().eval()
    output = model(torch.randn(1, 5, 3, 64, 64))
    for body in (output.object, output.camera):
        half_step = 0.5 * output.interval_dt[:, -1:]
        expected = (
            body.interval_twists.linear[:, -1]
            + half_step * body.interval_accelerations.linear[:, -1]
        )
        assert torch.allclose(body.current.twist.linear, expected)


def test_every_interval_has_its_own_estimate():
    """The temporal mixers must not collapse the window to one shared twist."""
    torch.manual_seed(0)
    model = _small_observer().eval()
    output = model(torch.randn(1, 8, 3, 64, 64))
    assert output.object.interval_twists.linear.std(dim=1).max() > 0
    assert output.camera.interval_twists.linear.std(dim=1).max() > 0

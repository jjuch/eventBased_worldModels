import torch
from ball_world_model.models.observer import SE3Observer, SE3ObserverConfig

def test_observer_contract():
    model = SE3Observer(SE3ObserverConfig(feature_channels=32, descriptor_dim=64, temporal_hidden_dim=32, carrier_points=8))
    images = torch.randn(2, 6, 3, 64, 64)
    time = torch.arange(6).float().mul(0.01).expand(2, -1)
    output = model(images, time)
    assert output.pose.position.shape == (2, 3)
    assert output.pose.rotation.shape == (2, 3, 3)
    assert output.twist.linear.shape == (2, 3)
    assert output.acceleration.angular.shape == (2, 3)
    assert output.carrier.shape == (2, 8, 3)
    assert output.carrier_velocity.shape == output.carrier.shape
    assert output.carrier_acceleration.shape == output.carrier.shape
    identity = torch.eye(3).expand(2, -1, -1)
    assert torch.allclose(output.pose.rotation.transpose(-1, -2) @ output.pose.rotation, identity, atol=1e-5)
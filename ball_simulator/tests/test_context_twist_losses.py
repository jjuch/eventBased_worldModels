import torch
from ball_world_model.training.se3_balanced_losses import (
    context_twist_loss,
    interval_twist_variance_loss,
)


def test_context_twist_averages_zero_mean_interval_noise():
    target_v = torch.tensor([[[1.0, 2.0, 3.0]]]).expand(2, 4, -1)
    noise = torch.tensor([[[-0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.1, 0.0, 0.1], [0.1, 0.0, -0.1]]]).expand(2, -1, -1)
    predicted_v = target_v + noise
    loss = context_twist_loss(predicted_v, target_v, torch.ones(3), None, None, None)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1.0e-7, rtol=0.0)


def test_interval_variance_is_zero_for_constant_twist():
    velocity = torch.randn(3, 1, 3).expand(-1, 6, -1)
    omega = torch.randn(3, 1, 3).expand(-1, 6, -1)
    loss = interval_twist_variance_loss(velocity, torch.ones(3), omega, torch.ones(3))
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1.0e-7, rtol=0.0)


def test_interval_variance_penalises_temporal_noise():
    velocity = torch.randn(3, 6, 3)
    loss = interval_twist_variance_loss(velocity, torch.ones(3), None, None)
    assert float(loss) > 0.0
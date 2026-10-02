import torch

from ball_world_model.models.contextual_twist import ContextTwistAggregator


def test_context_twist_attention_is_normalised():
    module = ContextTwistAggregator(32, 16)
    features = torch.randn(4, 9, 32)
    result = module(features)
    assert result.linear_velocity.shape == (4, 3)
    assert result.angular_velocity.shape == (4, 3)
    assert result.attention.shape == (4, 9)
    torch.testing.assert_close(result.attention.sum(dim=1), torch.ones(4))


def test_context_twist_respects_inactive_rotation():
    module = ContextTwistAggregator(8, 8, use_rotation=False)
    result = module(torch.randn(2, 9, 8))
    assert torch.count_nonzero(result.angular_velocity) == 0


def test_context_twist_uses_all_intervals():
    torch.manual_seed(7)
    module = ContextTwistAggregator(4, 8)
    features = torch.randn(1, 9, 4, requires_grad=True)
    result = module(features)
    result.linear_velocity.sum().backward()
    interval_gradient = features.grad.abs().sum(dim=-1)
    assert torch.all(interval_gradient > 0)

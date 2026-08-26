import torch

from ball_world_model.models.kinematic_estimator import KinematicStateEstimator
from ball_world_model.models.latent_motion import RunningDeltaNormaliser


def test_delta_normaliser_preserves_relative_magnitude():
    normaliser = RunningDeltaNormaliser(channels=2)
    normaliser.train()
    difference = torch.randn(8, 9, 2, 4, 4)
    normalised = normaliser(difference)
    assert normalised.shape == difference.shape
    assert torch.isfinite(normalised).all()
    # It is channel-standardized over the batch/time/spatial sample dimensions.
    channel_std = normalised.std(dim=(0, 1, 3, 4), unbiased=False)
    torch.testing.assert_close(channel_std, torch.ones_like(channel_std), atol=2e-3, rtol=2e-3)


def test_translation_model_returns_bidirectional_diagnostics():
    model = KinematicStateEstimator(
        task="translation",
        embedding_dim=64,
        keypoints=4,
        motion_dim=48,
        decoder_hidden_dim=64,
        refinement_hidden_dim=32,
        refinement_iterations=2,
        default_frame_dt=0.01,
    )
    images = torch.randn(2, 10, 3, 128, 128)
    time = torch.arange(10, dtype=torch.float32).unsqueeze(0).expand(2, -1) * 0.01
    prediction = model(images, time)
    assert prediction.position.shape == (2, 10, 3)
    assert prediction.linear_velocity.shape == (2, 10, 3)
    assert prediction.frame_latent.shape == (2, 10, 64)
    assert prediction.motion.forward_motion.shape == (2, 9, 48)
    assert prediction.motion.backward_motion.shape == (2, 9, 48)
    assert prediction.motion.predicted_next_embedding.shape == (2, 9, 64)
    assert prediction.motion.predicted_previous_embedding.shape == (2, 9, 64)
    assert prediction.motion.forward_velocity.shape == (2, 9, 3)
    assert prediction.motion.backward_velocity.shape == (2, 9, 3)
    assert len(prediction.motion.refinement_residuals) == 2


def test_constant_frames_produce_negligible_feature_rate():
    model = KinematicStateEstimator(
        task="translation",
        embedding_dim=32,
        keypoints=4,
        motion_dim=24,
        decoder_hidden_dim=32,
        refinement_iterations=0,
    )
    one = torch.randn(1, 1, 3, 64, 64)
    images = one.expand(1, 10, -1, -1, -1).clone()
    time = torch.arange(10, dtype=torch.float32).unsqueeze(0) * 0.01
    prediction = model(images, time)
    raw_difference = prediction.motion.raw_forward_difference
    feature_rate = prediction.motion.raw_forward_feature_rate
    normalised_rate = prediction.motion.normalised_forward_difference
    assert raw_difference.abs().max() < 1.0e-5
    assert feature_rate.abs().max() < 1.0e-5
    assert torch.isfinite(normalised_rate).all()
    torch.testing.assert_close(normalised_rate, torch.zeros_like(normalised_rate), atol=1e-5, rtol=1e-5)


def test_model_backward_pass_reaches_motion_encoder():
    model = KinematicStateEstimator(
        task="translation",
        embedding_dim=32,
        keypoints=4,
        motion_dim=24,
        decoder_hidden_dim=32,
        refinement_iterations=1,
    )
    images = torch.randn(2, 10, 3, 64, 64)
    time = torch.arange(10, dtype=torch.float32).unsqueeze(0).expand(2, -1) * 0.01
    prediction = model(images, time)
    loss = (
        prediction.position.square().mean()
        + prediction.linear_velocity.square().mean()
        + prediction.motion.predicted_next_embedding.square().mean()
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.motion_encoder.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)

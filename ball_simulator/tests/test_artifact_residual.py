import torch

from ball_world_model.models.artifact_residual import (
    ArtifactResidualDecoder,
    ArtifactResidualEncoder,
    PhysicalFeatureRatePredictor,
    PhysicsAdversary,
    cross_covariance_loss,
    gradient_reverse,
)

def test_residual_roundtrip_shapes():
    rate = torch.randn(2, 4, 128, 16, 16)
    encoder = ArtifactResidualEncoder(128, 32)
    decoder = ArtifactResidualDecoder(32, 128)
    latent = encoder(rate)
    assert latent.shape == (2,4,32) 
    assert decoder(latent).shape == rate.shape


def test_physics_predictor_shape():
    predictor = PhysicalFeatureRatePredictor(128)
    F = torch.randn(2, 4, 128, 16, 16)
    r = torch.randn(2, 4, 6)
    w = torch.randn(2, 4, 3)
    dt = torch.ones(2, 4, 1) * .01
    assert predictor(F, r, w, dt).shape == F.shape

def test_gradient_reversal():
    x = torch.tensor([1.], requires_grad=True)
    (gradient_reverse(x, .5)*2).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([-1.]))


def test_cross_covariance_independent_is_smaller():
    torch.manual_seed(1)
    x = torch.randn(1024,8)
    independent = torch.randn(1024, 8)
    correlated = x +.01 * torch.randn_like(x)
    assert cross_covariance_loss(x, independent) < cross_covariance_loss(x, correlated)


def test_adversary_shapes():
    a = PhysicsAdversary(32)
    omega, relative = a(torch.randn(3, 5, 32), .1)
    assert omega.shape == (3, 5, 3)
    assert relative.shape == (3, 5, 6)
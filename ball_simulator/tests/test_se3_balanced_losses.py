import torch
from ball_world_model.training.se3_balanced_losses import (
    camera_basis,
    dimensionless_twist_loss,
    rms_normalised_loss,
    world_to_camera_vector,
)


def test_twist_loss_is_invariant_to_unit_rescaling():
    predicted_v = torch.tensor([[[0.8, -0.2, 0.4]]])
    target_v = torch.tensor([[[1.0, 0.0, 0.5]]])
    scale_v = torch.tensor([0.5, 0.25, 0.5])
    predicted_w = torch.tensor([[[20.0, -10.0, 5.0]]])
    target_w = torch.tensor([[[25.0, -8.0, 3.0]]])
    scale_w = torch.tensor([10.0, 10.0, 5.0])
    reference = dimensionless_twist_loss(predicted_v, target_v, scale_v, predicted_w, target_w, scale_w)
    converted = dimensionless_twist_loss(
        predicted_v * 100.0, target_v * 100.0, scale_v * 100.0,
        predicted_w * 1.0e-3, target_w * 1.0e-3, scale_w * 1.0e-3,
    )
    torch.testing.assert_close(reference, converted)


def test_auxiliary_loss_is_invariant_to_sector_magnitude():
    target = torch.randn(4, 5, 24, 3)
    prediction = target + 0.1 * torch.randn_like(target)
    first, _ = rms_normalised_loss(prediction, target)
    second, _ = rms_normalised_loss(60.0 * prediction, 60.0 * target)
    torch.testing.assert_close(first, second)


def test_camera_basis_matches_active_camera_geometry():
    location = torch.tensor([-0.0101931936, -2.89531673, 1.60396113])
    target = torch.tensor([-0.0101931936, 0.01482484, 0.65839882])
    basis = camera_basis(location, target)
    world_axes = torch.eye(3)
    camera = world_to_camera_vector(world_axes, basis)
    assert abs(float(camera[0, 0]) - 1.0) < 1.0e-5
    assert abs(float(camera[1, 2]) - 0.9510565) < 1.0e-5
    assert abs(float(camera[2, 1]) - 0.9510565) < 1.0e-5

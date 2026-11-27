"""Anchor-frame targets and the losses on an analytic window: a ballistic, spinning object seen
by a camera that translates with constant acceleration and rotates about a fixed axis."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from ball_world_model.geometry.se3 import compose, inverse, rollout_world, so3_exp
from ball_world_model.models.observer import body_prediction
from ball_world_model.models.state import Acceleration, ObserverPrediction, Pose, Twist
from ball_world_model.training.losses import LossSettings, observer_losses, pose_twist_consistency
from ball_world_model.training.statistics import target_values
from ball_world_model.training.targets import anchor_window_targets

GRAVITY = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
CONTEXT, FUTURE, DT = 8, 5, 0.01


def _quaternions(rotation: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        Rotation.from_matrix(rotation.reshape(-1, 3, 3).numpy()).as_quat()
    ).reshape(*rotation.shape[:-2], 4)


def _body(time, p0, v0, acceleration, r0, omega):
    """Constant acceleration and constant angular velocity about a fixed axis (world frame)."""
    t = time[:, None]
    position = p0 + t * v0 + 0.5 * t**2 * acceleration
    velocity = v0 + t * acceleration
    rotation = so3_exp(t * omega) @ r0
    return position, rotation, velocity, omega.expand(len(time), 3)


def _batch(camera_moves: bool = True) -> dict[str, torch.Tensor]:
    """Two windows: world-frame object and camera states for context and future frames."""
    dtype = torch.float64
    time = torch.arange(CONTEXT + FUTURE, dtype=dtype) * DT
    camera_rotation = torch.as_tensor(
        Rotation.from_euler("xyz", [-110.0, 5.0, 20.0], degrees=True).as_matrix(), dtype=dtype
    )
    windows = []
    for shift in (0.0, 0.3):
        object_states = _body(
            time,
            torch.tensor([0.1 + shift, -0.2, 0.6], dtype=dtype),
            torch.tensor([1.0, 0.5, 2.0], dtype=dtype),
            GRAVITY,
            so3_exp(torch.tensor([0.2, 0.1, -0.3], dtype=dtype)),
            torch.tensor([3.0, -12.0, 20.0], dtype=dtype),
        )
        scale = 1.0 if camera_moves else 0.0
        camera_states = _body(
            time,
            torch.tensor([0.5, -1.8, 1.2], dtype=dtype),
            scale * torch.tensor([0.3, -0.2, 0.1], dtype=dtype),
            scale * torch.tensor([0.5, 0.4, -0.3], dtype=dtype),
            camera_rotation,
            scale * torch.tensor([0.2, -0.3, 0.4], dtype=dtype),
        )
        windows.append((object_states, camera_states))

    batch = {
        "context_time": time[:CONTEXT].expand(2, -1),
        "future_time": time[CONTEXT:].expand(2, -1),
    }
    names = ("position", "quaternion_xyzw", "linear_velocity", "angular_velocity")
    for prefix, index in (("", 0), ("camera_", 1)):
        stacked = [torch.stack([window[index][field] for window in windows]) for field in range(4)]
        stacked[1] = _quaternions(stacked[1])
        for name, value in zip(names, stacked, strict=True):
            batch[f"context_{prefix}{name}"] = value[:, :CONTEXT]
            batch[f"future_{prefix}{name}"] = value[:, CONTEXT:]
    batch["gravity_world"] = GRAVITY.expand(2, 3)
    return batch


def _world(batch, window, prefix=""):
    from ball_world_model.training.targets import quaternion_xyzw_to_matrix

    return (
        batch[f"{window}_{prefix}position"],
        quaternion_xyzw_to_matrix(batch[f"{window}_{prefix}quaternion_xyzw"]),
        batch[f"{window}_{prefix}linear_velocity"],
        batch[f"{window}_{prefix}angular_velocity"],
    )


def test_static_camera_targets_are_world_targets_under_one_constant_transform():
    batch = _batch(camera_moves=False)
    targets = anchor_window_targets(batch)
    position, rotation, velocity, omega = _world(batch, "context")
    camera_position = batch["context_camera_position"][:, -1]
    camera_rotation = _world(batch, "context", "camera_")[1][:, -1]
    to_anchor = camera_rotation.transpose(-1, -2)  # R_A_W for every frame of the window

    expected_position = torch.einsum("bij,btj->bti", to_anchor, position - camera_position[:, None])
    expected_velocity = torch.einsum(
        "bij,btj->bti", to_anchor, 0.5 * (velocity[:, 1:] + velocity[:, :-1])
    )
    torch.testing.assert_close(
        targets.object.frame_position, expected_position.float(), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        targets.object.interval_linear_velocity, expected_velocity.float(), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        targets.object.frame_rotation, (to_anchor[:, None] @ rotation).float(), atol=1e-6, rtol=0
    )
    # A static camera: identity pose in every frame, no motion.
    torch.testing.assert_close(
        targets.camera.frame_rotation, torch.eye(3).expand(2, CONTEXT, 3, 3), atol=1e-6, rtol=0
    )
    assert targets.camera.interval_linear_velocity.abs().max() < 1e-6
    torch.testing.assert_close(
        targets.object_in_camera.position, targets.object.frame_position, atol=1e-6, rtol=0
    )


def test_composition_reproduces_what_the_camera_sees():
    batch = _batch()
    targets = anchor_window_targets(batch)
    object_position, object_rotation, *_ = _world(batch, "context")
    camera_position, camera_rotation, *_ = _world(batch, "context", "camera_")
    expected = compose(*inverse(camera_position, camera_rotation), object_position, object_rotation)
    torch.testing.assert_close(
        targets.object_in_camera.position, expected[0].float(), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        targets.object_in_camera.rotation, expected[1].float(), atol=1e-6, rtol=0
    )
    # Anchor at the last context frame: the camera pose there is identity.
    torch.testing.assert_close(
        targets.camera.frame_position[:, -1], torch.zeros(2, 3), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        targets.camera.frame_rotation[:, -1], torch.eye(3).expand(2, 3, 3), atol=1e-6, rtol=0
    )


def test_interval_means_are_exact_for_ballistic_spin_under_a_moving_camera():
    batch = _batch()
    targets = anchor_window_targets(batch)
    camera_rotation = _world(batch, "context", "camera_")[1][:, -1]
    to_anchor = camera_rotation.transpose(-1, -2)

    def rotate(vector):
        return (
            torch.einsum("bij,j->bi", to_anchor, vector).float()[:, None].expand(2, CONTEXT - 1, 3)
        )

    torch.testing.assert_close(
        targets.object.interval_linear_acceleration, rotate(GRAVITY), atol=1e-4, rtol=0
    )
    torch.testing.assert_close(
        targets.object.interval_angular_velocity,
        rotate(torch.tensor([3.0, -12.0, 20.0], dtype=torch.float64)),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        targets.camera.interval_angular_velocity,
        rotate(torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)),
        atol=1e-5,
        rtol=0,
    )
    # One exact SE(3) step per interval, in the anchor frame.
    for body in (targets.object, targets.camera):
        step = so3_exp(body.interval_dt.unsqueeze(-1) * body.interval_angular_velocity)
        torch.testing.assert_close(
            step @ body.frame_rotation[:, :-1], body.frame_rotation[:, 1:], atol=1e-5, rtol=0
        )
        stepped = (
            body.frame_position[:, :-1]
            + body.interval_dt.unsqueeze(-1) * body.interval_linear_velocity
        )
        torch.testing.assert_close(stepped, body.frame_position[:, 1:], atol=1e-6, rtol=0)


def test_gravity_in_the_anchor_frame():
    targets = anchor_window_targets(_batch())
    torch.testing.assert_close(
        torch.linalg.vector_norm(targets.gravity, dim=-1), torch.full((2,), 9.81)
    )
    # The object's mean acceleration is exactly gravity, in the anchor frame.
    torch.testing.assert_close(
        targets.object.interval_linear_acceleration[:, 0], targets.gravity, atol=1e-4, rtol=0
    )


def test_anchor_frame_rollout_reproduces_ballistic_motion():
    targets = anchor_window_targets(_batch())
    current = targets.object.current
    position, rotation = rollout_world(
        current.position,
        current.rotation,
        current.linear_velocity,
        current.angular_velocity,
        current.linear_acceleration,
        current.angular_acceleration,
        targets.object_future.horizons,
    )
    torch.testing.assert_close(position, targets.object_future.position, atol=1e-5, rtol=0)
    torch.testing.assert_close(rotation, targets.object_future.rotation, atol=1e-5, rtol=0)


def _perfect_prediction(targets) -> ObserverPrediction:
    def body(target):
        return body_prediction(
            Pose(target.frame_position, target.frame_rotation),
            Twist(target.interval_linear_velocity, target.interval_angular_velocity),
            Acceleration(target.interval_linear_acceleration, target.interval_angular_acceleration),
            target.interval_dt,
        )

    return ObserverPrediction(
        object_in_camera=targets.object_in_camera,
        camera=body(targets.camera),
        object=body(targets.object),
        gravity=targets.gravity,
        interval_dt=targets.object.interval_dt,
        keypoint_coordinates=torch.zeros(2, CONTEXT, 4, 2),
    )


def _unit_scales(targets) -> dict[str, torch.Tensor]:
    return {name: torch.ones(3) for name in target_values(targets)}


def test_perfect_predictions_give_zero_for_every_term():
    targets = anchor_window_targets(_batch(), future_steps=FUTURE)
    settings = LossSettings(rollout=1.0, rollout_steps=FUTURE, kinematic=1.0, pose_twist=1.0)
    total, components = observer_losses(
        _perfect_prediction(targets), targets, _unit_scales(targets), settings
    )
    for name, value in components.items():
        assert value < 1e-6, name
    assert total < 1e-5


def test_pose_twist_loss_is_positive_for_a_perturbed_twist():
    targets = anchor_window_targets(_batch())
    prediction = _perfect_prediction(targets)
    body = prediction.object
    perturbed = body_prediction(
        body.frame_poses,
        Twist(
            body.interval_twists.linear + 0.05,
            body.interval_twists.angular + math.radians(5.0) / DT,
        ),
        body.interval_accelerations,
        prediction.interval_dt,
    )
    rotation, translation = pose_twist_consistency(
        perturbed, prediction.interval_dt, torch.ones(3), math.radians(5.0)
    )
    assert rotation > 0.1 and translation > 1e-3


@pytest.mark.parametrize("rotation_kind", ["identity", "random"])
def test_pose_twist_gradients_are_finite(rotation_kind):
    frames = 6
    if rotation_kind == "identity":
        rotations = torch.eye(3).expand(1, frames, 3, 3).clone()
    else:
        rotations = torch.as_tensor(
            Rotation.random(frames, random_state=0).as_matrix(), dtype=torch.float32
        )[None]
    rotations.requires_grad_(True)
    positions = torch.zeros(1, frames, 3, requires_grad=True)
    linear = torch.zeros(1, frames - 1, 3, requires_grad=True)
    angular = torch.zeros(1, frames - 1, 3, requires_grad=True)
    dt = torch.full((1, frames - 1), DT)
    body = body_prediction(
        Pose(positions, rotations),
        Twist(linear, angular),
        Acceleration(torch.zeros(1, frames - 1, 3), torch.zeros(1, frames - 1, 3)),
        dt,
    )
    rotation, translation = pose_twist_consistency(body, dt, torch.ones(3), math.radians(5.0))
    (rotation + translation).backward()
    for tensor in (rotations, positions, linear, angular):
        assert torch.isfinite(tensor.grad).all()
    if rotation_kind == "identity":
        assert rotation.item() == pytest.approx(0.0, abs=1e-10)


def test_target_values_cover_every_standardised_quantity():
    from ball_world_model.models.observer import STANDARDISED_QUANTITIES

    values = target_values(anchor_window_targets(_batch()))
    assert set(values) == set(STANDARDISED_QUANTITIES)
    assert all(value.shape[-1] == 3 for value in values.values())
    assert np.isfinite(torch.cat(list(values.values())).numpy()).all()

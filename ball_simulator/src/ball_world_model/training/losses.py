"""Observer losses for two bodies (camera and object) in the anchor frame.

Every term is dimensionless and O(1) when errors are of typical size:

    position, linear/angular velocity, linear/angular acceleration:
        squared error divided by the training-set std of that quantity and body
    orientation:
        squared angle (chordal surrogate) divided by orientation_scale^2
    pose terms:
        the object's pose relative to the camera (the observable pose) and the camera's pose
        relative to the anchor for every context frame but the last, which is identity by
        construction
    rollout:
        each body's current state is propagated with constant acceleration to the future frames
        (valid in the anchor frame, which is inertial); errors are scaled like the pose terms
    kinematic (twist-acceleration consistency):
        the velocity change between neighbouring intervals must match the accelerations
    gravity:
        direction error 2 (1 - cos angle) / gravity_scale^2, plus the magnitude error divided by
        the RMS std of the object's linear acceleration (gravity is an acceleration)
    pose_twist (pose-twist consistency):
        exp(dt omega_hat[k]) R_hat[k] against R_hat[k+1] and (p_hat[k+1] - p_hat[k]) / dt against
        v_hat[k], per body; the object uses its derived anchor-frame poses, so gradients reach the
        camera pose, the object pose relative to the camera and the twists

Composition consistency needs no loss: the object's anchor-frame pose is *derived* as
T_anchor_camera ∘ T_camera_object, so it holds by construction.

No force law appears anywhere: acceleration and gravity are free outputs.

The total is
    pose * (object + camera pose) + twist * (object + camera twist)
    + acceleration * (object + camera acceleration) + rollout * (object + camera rollout)
    + kinematic * (object + camera kinematic) + gravity * (direction + magnitude)
    + pose_twist * (object + camera pose-twist)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ball_world_model.geometry.se3 import (
    chordal_angle_squared,
    rollout_world,
    rotation_angle,
    so3_exp,
)
from ball_world_model.models.state import BodyPrediction, ObserverPrediction

from .targets import AnchorWindowTargets, BodyTargets, FuturePoses

BODIES = ("object", "camera")


@dataclass(frozen=True)
class LossSettings:
    pose: float = 1.0
    twist: float = 1.0
    acceleration: float = 1.0
    rollout: float = 0.1
    rollout_steps: int = 5
    orientation_scale_deg: float = 5.0
    kinematic: float = 0.0
    gravity: float = 1.0
    gravity_scale_deg: float = 5.0
    pose_twist: float = 0.0

    @property
    def orientation_scale_rad(self) -> float:
        return math.radians(self.orientation_scale_deg)

    @property
    def gravity_scale_rad(self) -> float:
        return math.radians(self.gravity_scale_deg)


def scaled_mse(prediction: Tensor, target: Tensor, scale: Tensor) -> Tensor:
    return ((prediction - target) / scale).square().mean()


def orientation_loss(prediction: Tensor, target: Tensor, scale_rad: float) -> Tensor:
    return chordal_angle_squared(prediction, target).mean() / scale_rad**2


def kinematic_consistency(
    rates: Tensor, accelerations: Tensor, dt: Tensor, velocity_scale: Tensor
) -> Tensor:
    """Mismatch between the velocity change across neighbouring intervals and the accelerations.

    rates [B, I, 3] interval-mean velocities, accelerations [B, I, 3] interval-mean accelerations,
    dt [B, I]. Interval means sit at interval centres, dt_centre = (dt[k] + dt[k+1]) / 2 apart;
    under a linearly varying acceleration the velocity change between them is
    dt_centre * (a[k] + a[k+1]) / 2. The error is a velocity, scaled by the velocity std.

    Assumes the acceleration varies smoothly between neighbouring intervals. That fails across a
    contact, where the velocity jumps; this term will need masking at contacts.
    """
    centre_dt = 0.5 * (dt[:, 1:] + dt[:, :-1]).unsqueeze(-1)
    velocity_change = rates[:, 1:] - rates[:, :-1]
    implied_change = centre_dt * 0.5 * (accelerations[:, 1:] + accelerations[:, :-1])
    return scaled_mse(velocity_change, implied_change, velocity_scale)


def pose_twist_consistency(
    body: BodyPrediction, dt: Tensor, velocity_scale: Tensor, orientation_scale_rad: float
) -> tuple[Tensor, Tensor]:
    """(rotation term, translation term) of one body's pose-twist consistency, dimensionless.

    Rotation: chordal distance between exp(dt omega[k]) R[k] and R[k+1], over orientation_scale^2
    (chordal instead of the geodesic angle: finite gradients at zero error). Translation:
    (p[k+1] - p[k]) / dt against v[k], over the velocity std. Both use interval means, for which
    one step is exact.
    """
    poses, twists = body.frame_poses, body.interval_twists
    step = so3_exp(dt.unsqueeze(-1) * twists.angular)
    rotation = orientation_loss(
        step @ poses.rotation[:, :-1], poses.rotation[:, 1:], orientation_scale_rad
    )
    mean_velocity = (poses.position[:, 1:] - poses.position[:, :-1]) / dt.unsqueeze(-1)
    translation = scaled_mse(mean_velocity, twists.linear, velocity_scale)
    return rotation, translation


def _rollout(
    body: BodyPrediction, future: FuturePoses, position_scale: Tensor, orientation_scale_rad: float
) -> tuple[Tensor, Tensor]:
    state = body.current
    position, rotation = rollout_world(
        state.pose.position,
        state.pose.rotation,
        state.twist.linear,
        state.twist.angular,
        state.acceleration.linear,
        state.acceleration.angular,
        future.horizons,
    )
    return (
        scaled_mse(position, future.position, position_scale),
        orientation_loss(rotation, future.rotation, orientation_scale_rad),
    )


def _motion_terms(
    name: str, body: BodyPrediction, target: BodyTargets, scales: dict[str, Tensor]
) -> dict[str, Tensor]:
    terms = {}
    for kind in ("linear", "angular"):
        terms[f"{name}_{kind}_velocity"] = scaled_mse(
            getattr(body.interval_twists, kind),
            getattr(target, f"interval_{kind}_velocity"),
            scales[f"{name}_{kind}_velocity"],
        )
        terms[f"{name}_{kind}_acceleration"] = scaled_mse(
            getattr(body.interval_accelerations, kind),
            getattr(target, f"interval_{kind}_acceleration"),
            scales[f"{name}_{kind}_acceleration"],
        )
    return terms


def gravity_losses(
    prediction: Tensor, target: Tensor, magnitude_scale: Tensor, scale_rad: float
) -> tuple[Tensor, Tensor]:
    """(direction, magnitude) losses of the gravity vector [B, 3], dimensionless."""
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1)
    direction = (2.0 * (1.0 - cosine)).mean() / scale_rad**2
    magnitude_error = torch.linalg.vector_norm(prediction, dim=-1) - torch.linalg.vector_norm(
        target, dim=-1
    )
    return direction, (magnitude_error / magnitude_scale).square().mean()


def observer_losses(
    prediction: ObserverPrediction,
    targets: AnchorWindowTargets,
    scales: dict[str, Tensor],
    settings: LossSettings,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Total loss and its components; ``targets`` must hold settings.rollout_steps future frames."""
    zero = prediction.gravity.new_zeros(())
    orientation_scale = settings.orientation_scale_rad
    camera_poses = prediction.camera.frame_poses

    components = {
        "object_position": scaled_mse(
            prediction.object_in_camera.position,
            targets.object_in_camera.position,
            scales["object_in_camera_position"],
        ),
        "object_orientation": orientation_loss(
            prediction.object_in_camera.rotation,
            targets.object_in_camera.rotation,
            orientation_scale,
        ),
        "camera_position": scaled_mse(
            camera_poses.position[:, :-1],
            targets.camera.frame_position[:, :-1],
            scales["camera_position"],
        ),
        "camera_orientation": orientation_loss(
            camera_poses.rotation[:, :-1], targets.camera.frame_rotation[:, :-1], orientation_scale
        ),
        **_motion_terms("object", prediction.object, targets.object, scales),
        **_motion_terms("camera", prediction.camera, targets.camera, scales),
    }
    magnitude_scale = scales["object_linear_acceleration"].square().mean().sqrt()
    components["gravity_direction"], components["gravity_magnitude"] = gravity_losses(
        prediction.gravity, targets.gravity, magnitude_scale, settings.gravity_scale_rad
    )

    position_scales = {
        "object": scales["object_in_camera_position"],
        "camera": scales["camera_position"],
    }
    futures = {"object": targets.object_future, "camera": targets.camera_future}
    for name in BODIES:
        body = getattr(prediction, name)
        velocity_scale = (scales[f"{name}_linear_velocity"], scales[f"{name}_angular_velocity"])
        components[f"{name}_rollout_position"], components[f"{name}_rollout_orientation"] = (
            _rollout(body, futures[name], position_scales[name], orientation_scale)
            if settings.rollout > 0
            else (zero, zero)
        )
        components[f"{name}_kinematic_linear"], components[f"{name}_kinematic_angular"] = (
            (
                kinematic_consistency(
                    body.interval_twists.linear,
                    body.interval_accelerations.linear,
                    prediction.interval_dt,
                    velocity_scale[0],
                ),
                kinematic_consistency(
                    body.interval_twists.angular,
                    body.interval_accelerations.angular,
                    prediction.interval_dt,
                    velocity_scale[1],
                ),
            )
            if settings.kinematic > 0
            else (zero, zero)
        )
        components[f"{name}_pose_twist_rotation"], components[f"{name}_pose_twist_translation"] = (
            pose_twist_consistency(
                body, prediction.interval_dt, velocity_scale[0], orientation_scale
            )
            if settings.pose_twist > 0
            else (zero, zero)
        )

    def both(*suffixes: str) -> Tensor:
        return sum(components[f"{name}_{suffix}"] for name in BODIES for suffix in suffixes)

    total = (
        settings.pose * both("position", "orientation")
        + settings.twist * both("linear_velocity", "angular_velocity")
        + settings.acceleration * both("linear_acceleration", "angular_acceleration")
        + settings.rollout * both("rollout_position", "rollout_orientation")
        + settings.kinematic * both("kinematic_linear", "kinematic_angular")
        + settings.gravity * (components["gravity_direction"] + components["gravity_magnitude"])
        + settings.pose_twist * both("pose_twist_rotation", "pose_twist_translation")
    )
    return total, components


@torch.no_grad()
def physical_metrics(
    prediction: ObserverPrediction, targets: AnchorWindowTargets
) -> dict[str, Tensor]:
    """Errors in physical units for logging (anchor frame; object pose relative to the camera)."""

    def rmse(error: Tensor) -> Tensor:
        return error.square().sum(-1).mean().sqrt()

    metrics = {
        "object_position_rmse_m": rmse(
            prediction.object_in_camera.position - targets.object_in_camera.position
        ),
        "object_orientation_error_deg": torch.rad2deg(
            rotation_angle(prediction.object_in_camera.rotation, targets.object_in_camera.rotation)
        ).mean(),
        "camera_position_rmse_m": rmse(
            prediction.camera.frame_poses.position[:, :-1] - targets.camera.frame_position[:, :-1]
        ),
        "gravity_direction_error_deg": torch.rad2deg(
            torch.arccos(
                torch.nn.functional.cosine_similarity(
                    prediction.gravity, targets.gravity, dim=-1
                ).clamp(-1.0, 1.0)
            )
        ).mean(),
    }
    for name in BODIES:
        body, target = getattr(prediction, name), getattr(targets, name)
        metrics[f"{name}_linear_velocity_rmse_mps"] = rmse(
            body.interval_twists.linear - target.interval_linear_velocity
        )
        metrics[f"{name}_angular_velocity_rmse_radps"] = rmse(
            body.interval_twists.angular - target.interval_angular_velocity
        )
        metrics[f"{name}_linear_acceleration_rmse_mps2"] = rmse(
            body.interval_accelerations.linear - target.interval_linear_acceleration
        )
    return metrics

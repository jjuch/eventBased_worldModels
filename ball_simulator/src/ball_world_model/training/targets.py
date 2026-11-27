"""Ground-truth targets for one window, in the window's anchor frame.

The anchor frame is the camera frame (OpenCV axes: x right, y down, z forward) at the last
context frame, frozen. It is an inertial frame, so motion relative to the static scene keeps its
physics when expressed in it. With ``A`` the anchor, ``W`` the world, ``C`` the camera and ``O``
the object:

    T_A_W       = inverse(T_W_C[last context frame])
    T_A_C[k]    = T_A_W T_W_C[k]           (identity at the last context frame)
    T_A_O[k]    = T_A_W T_W_O[k]
    T_C_O[k]    = inverse(T_A_C[k]) T_A_O[k]     (what the camera sees)
    gravity_A   = R_A_W gravity_W

Interval targets are means over each interval, computed from the states without approximation:

    mean linear velocity      = (p[k+1] - p[k]) / dt
    mean angular velocity     = log(R[k+1] R[k]^T) / dt
    mean linear acceleration  = (v[k+1] - v[k]) / dt
    mean angular acceleration = (omega[k+1] - omega[k]) / dt

T_A_W is a constant rigid transform, so these means in the anchor frame are exactly the
world-frame means rotated by R_A_W: one SE(3) step stays exact. The rotation-vector form is exact
when the rotation axis is constant over the interval, and a tight approximation at 100 fps.
Everything is computed in float64 and returned in float32.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ball_world_model.geometry.se3 import compose, inverse, so3_log
from ball_world_model.models.state import Pose


def quaternion_xyzw_to_matrix(q: Tensor) -> Tensor:
    q = torch.nn.functional.normalize(q, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


@dataclass(frozen=True)
class CurrentTargets:
    """Instantaneous state of one body at the last context frame, [B, ...].

    The acceleration is the mean over the last interval, because the dataset stores no
    instantaneous acceleration (identical for constant acceleration).
    """

    position: Tensor
    rotation: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor
    linear_acceleration: Tensor
    angular_acceleration: Tensor


@dataclass(frozen=True)
class BodyTargets:
    """One rigid body over the context frames, in the anchor frame, relative to the static scene.

    frame_position [B, T, 3] m, frame_rotation [B, T, 3, 3] body -> anchor,
    frame_linear_velocity [B, T, 3] m/s, frame_angular_velocity [B, T, 3] rad/s,
    interval_dt [B, I] s, interval means [B, I, 3] in m/s, rad/s, m/s^2 and rad/s^2.
    """

    frame_position: Tensor
    frame_rotation: Tensor
    frame_linear_velocity: Tensor
    frame_angular_velocity: Tensor
    interval_dt: Tensor
    interval_linear_velocity: Tensor
    interval_angular_velocity: Tensor
    interval_linear_acceleration: Tensor
    interval_angular_acceleration: Tensor

    @property
    def current(self) -> CurrentTargets:
        return CurrentTargets(
            position=self.frame_position[:, -1],
            rotation=self.frame_rotation[:, -1],
            linear_velocity=self.frame_linear_velocity[:, -1],
            angular_velocity=self.frame_angular_velocity[:, -1],
            linear_acceleration=self.interval_linear_acceleration[:, -1],
            angular_acceleration=self.interval_angular_acceleration[:, -1],
        )


@dataclass(frozen=True)
class FuturePoses:
    """Poses after the context window, anchor frame: horizons [B, F] s since the last context
    frame, position [B, F, 3] m, rotation [B, F, 3, 3]."""

    horizons: Tensor
    position: Tensor
    rotation: Tensor


@dataclass(frozen=True)
class AnchorWindowTargets:
    """All targets of one window, expressed in its anchor frame.

    object, camera     the two bodies; camera frame poses are identity at the last frame
    object_in_camera   T_C_O for every context frame (the directly observable pose)
    gravity            [B, 3] m/s^2
    object_future, camera_future
                       future poses for rollouts and for drawing future frames
    anchor_in_world    T_W_A [B] (evaluation only)
    """

    object: BodyTargets
    camera: BodyTargets
    object_in_camera: Pose
    gravity: Tensor
    object_future: FuturePoses
    camera_future: FuturePoses
    anchor_in_world: Pose


def _interval_rate(values: Tensor, dt: Tensor) -> Tensor:
    return (values[:, 1:] - values[:, :-1]) / dt.unsqueeze(-1)


def _rotate(rotation: Tensor, vectors: Tensor) -> Tensor:
    """Rotate vectors [B, N, 3] (or [B, 3]) by rotations [B, 3, 3]."""
    if vectors.ndim == 2:
        return torch.einsum("bij,bj->bi", rotation, vectors)
    return torch.einsum("bij,bnj->bni", rotation, vectors)


def body_targets(
    time: Tensor,
    position: Tensor,
    rotation: Tensor,
    linear_velocity: Tensor,
    angular_velocity: Tensor,
) -> BodyTargets:
    """Frame and interval-mean targets of one body from its states, all [B, T, ...] float64."""
    dt = torch.diff(time, dim=1).clamp_min(1.0e-9)
    relative_rotation = rotation[:, 1:] @ rotation[:, :-1].transpose(-1, -2)
    return BodyTargets(
        frame_position=position,
        frame_rotation=rotation,
        frame_linear_velocity=linear_velocity,
        frame_angular_velocity=angular_velocity,
        interval_dt=dt,
        interval_linear_velocity=_interval_rate(position, dt),
        interval_angular_velocity=so3_log(relative_rotation) / dt.unsqueeze(-1),
        interval_linear_acceleration=_interval_rate(linear_velocity, dt),
        interval_angular_acceleration=_interval_rate(angular_velocity, dt),
    )


def _states(
    batch: dict[str, Tensor], window: str, prefix: str
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """World-frame states of one body (prefix "" or "camera_") in "context" or "future", float64."""
    return (
        batch[f"{window}_{prefix}position"].double(),
        quaternion_xyzw_to_matrix(batch[f"{window}_{prefix}quaternion_xyzw"].double()),
        batch[f"{window}_{prefix}linear_velocity"].double(),
        batch[f"{window}_{prefix}angular_velocity"].double(),
    )


def _to_anchor(
    anchor_from_world: tuple[Tensor, Tensor],
    position: Tensor,
    rotation: Tensor,
    linear_velocity: Tensor,
    angular_velocity: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """World-frame states [B, N, ...] of a body -> anchor frame. Velocities only rotate."""
    anchor_position, anchor_rotation = anchor_from_world
    frames = position.shape[1]
    expanded = (
        anchor_position.unsqueeze(1).expand(-1, frames, -1),
        anchor_rotation.unsqueeze(1).expand(-1, frames, -1, -1),
    )
    position, rotation = compose(*expanded, position, rotation)
    return (
        position,
        rotation,
        _rotate(anchor_rotation, linear_velocity),
        _rotate(anchor_rotation, angular_velocity),
    )


def _float(values):
    return type(values)(**{name: value.float() for name, value in vars(values).items()})


def anchor_window_targets(
    batch: dict[str, Tensor], future_steps: int | None = None
) -> AnchorWindowTargets:
    """Anchor-frame targets of a batch from the dataset (world-frame object and camera states).

    ``future_steps`` limits the future poses to the first steps (default: all future frames).
    """
    available = batch["future_time"].shape[1]
    steps = available if future_steps is None else future_steps
    if steps > available:
        raise ValueError(
            f"Requested {steps} future steps but the batch holds {available} future frames."
        )

    context_time = batch["context_time"].double()
    camera_world = _states(batch, "context", "camera_")
    anchor_in_world = (camera_world[0][:, -1], camera_world[1][:, -1])
    anchor_from_world = inverse(*anchor_in_world)

    camera = body_targets(context_time, *_to_anchor(anchor_from_world, *camera_world))
    object_ = body_targets(
        context_time, *_to_anchor(anchor_from_world, *_states(batch, "context", ""))
    )
    camera_inverse = inverse(camera.frame_position, camera.frame_rotation)
    object_in_camera = compose(*camera_inverse, object_.frame_position, object_.frame_rotation)

    horizons = batch["future_time"][:, :steps].double() - context_time[:, -1:]

    def future(prefix: str) -> FuturePoses:
        position, rotation, *_ = _to_anchor(
            anchor_from_world, *(state[:, :steps] for state in _states(batch, "future", prefix))
        )
        return FuturePoses(
            horizons=horizons.float(), position=position.float(), rotation=rotation.float()
        )

    gravity = _rotate(anchor_from_world[1], batch["gravity_world"].double())
    return AnchorWindowTargets(
        object=_float(object_),
        camera=_float(camera),
        object_in_camera=Pose(object_in_camera[0].float(), object_in_camera[1].float()),
        gravity=gravity.float(),
        object_future=future(""),
        camera_future=future("camera_"),
        anchor_in_world=Pose(anchor_in_world[0].float(), anchor_in_world[1].float()),
    )

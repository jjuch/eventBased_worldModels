"""Typed observer outputs.

Shape conventions: B = batch, T = context frames, I = T - 1 intervals between consecutive frames.
Poses, twists and accelerations are expressed in the window's anchor frame (the camera frame at
the last context frame, OpenCV axes) and measured relative to the static scene, except
``object_in_camera``, which is relative to the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ball_world_model.geometry.se3 import matrix_to_rotation_6d


@dataclass(frozen=True)
class Pose:
    """Position [..., 3] in m and body-to-frame rotation [..., 3, 3]."""

    position: Tensor
    rotation: Tensor

    def packed(self) -> Tensor:
        return torch.cat((self.position, matrix_to_rotation_6d(self.rotation)), dim=-1)


@dataclass(frozen=True)
class Twist:
    """Linear velocity [m/s] and angular velocity [rad/s], each [..., 3]."""

    linear: Tensor
    angular: Tensor

    def packed(self) -> Tensor:
        return torch.cat((self.linear, self.angular), dim=-1)


@dataclass(frozen=True)
class Acceleration:
    """Linear [m/s^2] and angular [rad/s^2] acceleration, each [..., 3]."""

    linear: Tensor
    angular: Tensor

    def packed(self) -> Tensor:
        return torch.cat((self.linear, self.angular), dim=-1)


@dataclass(frozen=True)
class KinematicState:
    """Second-order rigid-body state at one instant: g, xi and xi_dot."""

    pose: Pose
    twist: Twist
    acceleration: Acceleration


@dataclass(frozen=True)
class BodyPrediction:
    """One rigid body over a window, anchor frame, relative to the static scene.

    frame_poses             pose at every context frame, [B, T, ...]
    interval_twists         mean twist over every interval, [B, I, 3]
    interval_accelerations  mean acceleration over every interval, [B, I, 3]
    current                 instantaneous state at the last context frame (rollout start)
    """

    frame_poses: Pose
    interval_twists: Twist
    interval_accelerations: Acceleration
    current: KinematicState


@dataclass(frozen=True)
class ObserverPrediction:
    """Everything the observer estimates from one context window.

    object_in_camera      T_camera_object for every context frame, [B, T] (directly observable)
    camera                camera body; frame poses T_anchor_camera, identity at the last frame
    object                object body; frame poses T_anchor_object = T_anchor_camera ∘
                          T_camera_object, derived by composition, never predicted separately
    gravity               gravity vector in the anchor frame, [B, 3] m/s^2
    interval_dt           interval durations in s, [B, I]
    keypoint_coordinates  soft keypoints in image coordinates, [B, T, K, 2] (diagnostics only)
    """

    object_in_camera: Pose
    camera: BodyPrediction
    object: BodyPrediction
    gravity: Tensor
    interval_dt: Tensor
    keypoint_coordinates: Tensor

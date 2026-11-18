"""Typed model outputs; packed vectors are views, never authoritative state."""

from __future__ import annotations

from dataclasses import dataclass
from torch import Tensor

from ball_world_model.geometry.se3 import matrix_to_rotation_6d

@dataclass(frozen=True)
class Pose:
    position: Tensor
    rotation: Tensor

    def packed(self) -> Tensor:
        return __import__("torch").cat((self.position, matrix_to_rotation_6d(self.rotation)), dim=-1)

@dataclass(frozen=True)
class Twist:
    linear: Tensor
    angular: Tensor

    def packed(self) -> Tensor:
        return __import__("torch").cat((self.linear, self.angular), dim=-1)


@dataclass(frozen=True)
class Acceleration:
    linear: Tensor
    angular: Tensor

    def packed(self) -> Tensor:
        return __import__("torch").cat((self.linear, self.angular), dim=-1)


@dataclass(frozen=True)
class ObserverAux:
    feature_maps: Tensor
    frame_descriptors: Tensor
    first_feature_rate: Tensor
    second_feature_rate: Tensor
    twist_attention: Tensor | None = None
    acceleration_attention: Tensor | None = None


@dataclass(frozen=True)
class ObserverPrediction:
    pose: Pose
    twist: Twist
    acceleration: Acceleration
    carrier: Tensor
    carrier_velocity: Tensor
    carrier_acceleration: Tensor
    aux: ObserverAux
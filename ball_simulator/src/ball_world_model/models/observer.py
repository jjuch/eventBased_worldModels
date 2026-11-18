"""Lean causal current-state observer: I[t-h:t] -> (g_t, xi_t, xi_dot_t)."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn

from ball_world_model.geometry.se3 import carrier_acceleration, carrier_state, carrier_velocity, rotation_6d_to_matrix
from .frame_encoder import CoordinateAwareFrameEncoder
from .state import Acceleration, ObserverAux, ObserverPrediction, Pose, Twist
from .temporal_encoder import AccelerationEncoder, TwistEncoder
from .teacher import fibonacci_sphere


@dataclass(frozen=True)
class SE3ObserverConfig:
    feature_channels: int = 128
    descriptor_dim: int = 256
    temporal_hidden_dim: int = 128
    carrier_points: int = 16
    carrier_radius: float = 0.10
    default_dt: float = 0.01
    position_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    linear_velocity_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_velocity_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    angular_velocity_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    linear_acceleration_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    angular_acceleration_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_acceleration_std: tuple[float, float, float] = (1.0, 1.0, 1.0)


class SE3Observer(nn.Module):
    def __init__(self, config: SE3ObserverConfig = SE3ObserverConfig()) -> None:
        super().__init__()
        self.config = config
        self.frame_encoder = CoordinateAwareFrameEncoder(config.feature_channels, config.descriptor_dim)
        self.pose_head = nn.Sequential(
            nn.Linear(config.descriptor_dim, config.descriptor_dim), 
            nn.LayerNorm(config.descriptor_dim), 
            nn.SiLU(),
            nn.Linear(config.descriptor_dim, 9),
        )
        self.twist_encoder = TwistEncoder(config.feature_channels, config.temporal_hidden_dim)
        self.acceleration_encoder = AccelerationEncoder(config.feature_channels, config.temporal_hidden_dim)

        self.register_buffer("templates", fibonacci_sphere(config.carrier_points, config.carrier_radius))

        for name in (
            "position", "linear_velocity", "angular_velocity",
            "linear_acceleration", "angular_acceleration",
        ):
            self.register_buffer(f"{name}_mean", torch.tensor(getattr(config, f"{name}_mean"), dtype=torch.float32))
            self.register_buffer(f"{name}_std", torch.tensor(getattr(config, f"{name}_std"), dtype=torch.float32).clamp_min(1.0e-6))

    @staticmethod
    def _intervals(time: Tensor | None, frames: Tensor, default_dt: float) -> Tensor:
        if time is None:
            time = torch.arange(frames.shape[1], device=frames.device, dtype=frames.dtype) * default_dt
            time = time.unsqueeze(0).expand(frames.shape[0], -1)
        return torch.diff(time.to(frames), dim=1).clamp_min(1.0e-6)


    def forward(self, context_rgb: Tensor, context_time: Tensor | None = None) -> ObserverPrediction:
        if context_rgb.shape[1] < 3:
            raise ValueError("Pose, twist and acceleration observation requires at least three frames.")
        maps, descriptors = self.frame_encoder(context_rgb)
        dt = self._intervals(context_time, context_rgb, self.config.default_dt)

        first_rate = (maps[:, 1:] - maps[:, :-1]) / dt[..., None, None, None]
        centre_dt = 0.5 * (dt[:, 1:] + dt[:, :-1])
        second_rate = (first_rate[:, 1:] - first_rate[:, :-1]) / centre_dt[..., None, None, None]

        # Current pose uses only the current image descriptor.
        pose_raw = self.pose_head(descriptors[:, -1])
        pose = Pose(position=pose_raw[..., :3] * self.position_std + self.position_mean, rotation=rotation_6d_to_matrix(pose_raw[..., 3:]))

        twist_raw, twist_attention = self.twist_encoder(maps, first_rate)
        twist = Twist(
            linear=twist_raw[..., :3] * self.linear_velocity_std + self.linear_velocity_mean,
            angular=twist_raw[..., 3:] * self.angular_velocity_std + self.angular_velocity_mean,
        )

        acceleration_raw, acceleration_attention = self.acceleration_encoder(first_rate, second_rate)
        acceleration = Acceleration(
            linear=acceleration_raw[..., :3] * self.linear_acceleration_std + self.linear_acceleration_mean,
            angular=acceleration_raw[..., 3:] * self.angular_acceleration_std + self.angular_acceleration_mean,
        )

        templates = self.templates.to(context_rgb)
        carrier = carrier_state(pose.position, pose.rotation, templates)
        velocity = carrier_velocity(pose.rotation, twist.linear, twist.angular, templates)
        accel = carrier_acceleration(
            pose.rotation, twist.angular, acceleration.linear, acceleration.angular, templates
        )
        return ObserverPrediction(
            pose=pose, 
            twist=twist, 
            acceleration=acceleration,
            carrier=carrier, 
            carrier_velocity=velocity, 
            carrier_acceleration=accel,
            aux=ObserverAux(maps, descriptors, first_rate, second_rate, twist_attention, acceleration_attention),
        )

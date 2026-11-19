"""SE(3) visual observer: images I[0..T-1] -> pose per frame, twist and acceleration per interval.

Why interval quantities: two consecutive frames reveal the *mean* twist over the interval between them, not the instantaneous twist at either frame. The mean is exactly what makes one SE(3) step exact,

    p[k+1] = p[k] + dt * v_mean,      R[k+1] = exp(dt * omega_mean) R[k]

(the rotation step is exact when the rotation axis is constant over the interval). The instantaneous state at the last frame, needed to start a rollout, follows under constant acceleration as

    xi(t_last) = xi_mean(last interval) + 0.5 * dt * xi_dot_mean(last interval).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ball_world_model.geometry.se3 import rotation_6d_to_matrix
from .frame_encoder import CoordinateAwareFrameEncoder, SpatialKeypointEncoder
from .state import Acceleration, KinematicState, ObserverPrediction, Pose, Twist
from .temporal_encoder import IntervalMotionEncoder, IntervalTemporalMixer

Vector3 = tuple[float, float, float]

# Quantities whose heads predict standardised values; statistics come from training data.
STANDARDISED_QUANTITIES = (
    "position",              # per frame
    "linear_velocity",       # interval mean
    "angular_velocity",      # interval mean
    "linear_acceleration",   # interval mean
    "angular_acceleration",  # interval mean
)


@dataclass(frozen=True)
class SE3ObserverConfig:
    feature_channels: int = 128
    descriptor_dim: int = 256
    motion_dim: int = 128
    temporal_layers: int = 2
    keypoints: int = 8
    keypoint_dim: int = 32
    default_dt: float = 0.01
    predict_acceleration: bool = True
    standardiser_momentum: float = 0.01
    position_mean: Vector3 = (0.0, 0.0, 0.0)
    position_std: Vector3 = (1.0, 1.0, 1.0)
    linear_velocity_mean: Vector3 = (0.0, 0.0, 0.0)
    linear_velocity_std: Vector3 = (1.0, 1.0, 1.0)
    angular_velocity_mean: Vector3 = (0.0, 0.0, 0.0)
    angular_velocity_std: Vector3 = (1.0, 1.0, 1.0)
    linear_acceleration_mean: Vector3 = (0.0, 0.0, 0.0)
    linear_acceleration_std: Vector3 = (1.0, 1.0, 1.0)
    angular_acceleration_mean: Vector3 = (0.0, 0.0, 0.0)
    angular_acceleration_std: Vector3 = (1.0, 1.0, 1.0)


class SE3Observer(nn.Module):
    def __init__(self, config: SE3ObserverConfig = SE3ObserverConfig()) -> None:
        super().__init__()
        self.config = config

        self.frame_encoder = CoordinateAwareFrameEncoder(config.feature_channels)
        self.spatial_encoder = SpatialKeypointEncoder(
            config.feature_channels, config.keypoints, config.keypoint_dim, config.descriptor_dim
        )

        # Position (3) and continuous 6D rotation (6) for every frame.
        self.pose_head = nn.Sequential(
            nn.Linear(config.descriptor_dim, config.descriptor_dim), 
            nn.SiLU(),
            nn.Linear(config.descriptor_dim, 9),
        )
        self.motion_encoder = IntervalMotionEncoder(
            config.feature_channels, config.motion_dim, config.standardiser_momentum
        )
        self.temporal_mixer = IntervalTemporalMixer(config.motion_dim, config.temporal_layers)
        self.twist_head = nn.Linear(config.motion_dim, 6)
        self.acceleration_head = nn.Linear(config.motion_dim, 6) if config.predict_acceleration else None

        for name in STANDARDISED_QUANTITIES:
            mean = torch.tensor(getattr(config, f"{name}_mean"), dtype=torch.float32)
            std = torch.tensor(getattr(config, f"{name}_std"), dtype=torch.float32).clamp_min(1.0e-6)
            self.register_buffer(f"{name}_mean", mean)
            self.register_buffer(f"{name}_std", std)


    def scales(self) -> dict[str, Tensor]:
        """Standard deviations used both to decode the heads and to normalise the losses."""
        return {name: getattr(self, f"{name}_std") for name in STANDARDISED_QUANTITIES}


    def _decode(self, raw: Tensor, name: str) -> Tensor:
        return raw * getattr(self, f"{name}_std") + getattr(self, f"{name}_mean")


    def _interval_durations(self, time: Tensor | None, frames: Tensor) -> Tensor:
        if time is None:
            steps = torch.arange(frames.shape[1], device=frames.device, dtype=torch.float32)
            time = (steps * self.config.default_dt).expand(frames.shape[0], -1)
        return torch.diff(time.float(), dim=1).clamp_min(1.0e-6)


    def forward(self, context_rgb: Tensor, context_time: Tensor | None = None) -> ObserverPrediction:
        """context_rgb [B, T, 3, H, W], context_time [B, T] in s."""
        if context_rgb.shape[1] < 3:
            raise ValueError("At least three frames are needed to observe acceleration.")
        
        maps = self.frame_encoder(context_rgb)
        spatial = self.spatial_encoder(maps)
        dt = self._interval_durations(context_time, context_rgb)

        pose_raw = self.pose_head(spatial.frame_descriptor).float()
        frame_poses = Pose(
            position=self._decode(pose_raw[..., :3], "position"),
            rotation=rotation_6d_to_matrix(pose_raw[..., 3:]),
        )

        motion = self.temporal_mixer(self.motion_encoder(maps, dt))
        twist_raw = self.twist_head(motion).float()
        interval_twists = Twist(
            linear=self._decode(twist_raw[..., :3], "linear_velocity"),
            angular=self._decode(twist_raw[..., 3:], "angular_velocity"),
        )
        if self.acceleration_head is None:
            interval_accelerations = Acceleration(
                torch.zeros_like(interval_twists.linear), torch.zeros_like(interval_twists.angular)
            )
        else:
            acceleration_raw = self.acceleration_head(motion).float()
            interval_accelerations = Acceleration(
                linear=self._decode(acceleration_raw[..., :3], "linear_acceleration"),
                angular=self._decode(acceleration_raw[..., 3:], "angular_acceleration"),
            )

        return ObserverPrediction(
            frame_poses=frame_poses,
            interval_twists=interval_twists,
            interval_accelerations=interval_accelerations,
            interval_dt=dt,
            current=current_state(frame_poses, interval_twists, interval_accelerations, dt),
            keypoint_coordinates=spatial.coordinates,
        )

def current_state(
    frame_poses: Pose, interval_twists: Twist, interval_accelerations: Acceleration, dt: Tensor,
) -> KinematicState:
    """Instantaneous state at the last frame, assuming constant acceleration over the last interval."""
    half_step = 0.5 * dt[:, -1:]
    last_acceleration = Acceleration(
        interval_accelerations.linear[:, -1], interval_accelerations.angular[:, -1]
    )
    return KinematicState(
        pose=Pose(frame_poses.position[:, -1], frame_poses.rotation[:, -1]),
        twist=Twist(
            linear=interval_twists.linear[:, -1] + half_step * last_acceleration.linear,
            angular=interval_twists.angular[:, -1] + half_step * last_acceleration.angular,
        ),
        acceleration=last_acceleration,
    )

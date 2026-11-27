"""SE(3) visual observer of two bodies, camera and object, in the window's anchor frame.

Images I[0..T-1] -> for both bodies a pose per frame and a twist and acceleration per interval,
all relative to the static scene and expressed in the anchor frame (the camera frame at the last
context frame), plus the gravity vector in the anchor frame.

Wiring:

    frames -> CoordinateAwareFrameEncoder -> maps[k]
      maps -> SpatialKeypointEncoder -> pose_head            T_camera_object[k]  (observable)
      maps -> object IntervalMotionEncoder -> mixer --+
      maps -> camera IntervalMotionEncoder -> mixer --+-> camera twist / acceleration heads
                                                      +-> object twist / acceleration heads
                                                          (object heads read both pathways)
      maps[k], maps[T-1] -> anchor pair encoder -> camera_pose_head   T_anchor_camera[k], k < T-1
      maps[T-1] -> SceneDescriptor -> gravity_head           gravity in the anchor frame

    T_anchor_camera[T-1] = identity                      (by construction, not predicted)
    T_anchor_object[k]   = T_anchor_camera[k] ∘ T_camera_object[k]   (derived, never predicted)

The camera pathways pool over the whole image, so the static scene dominates them; the object
pose keeps the soft-keypoint read-out. Object motion relative to the scene is its motion relative
to the camera plus ego-motion, hence the object heads read the camera features too.

Why interval quantities: two consecutive frames reveal the *mean* twist over the interval, which
makes one SE(3) step exact,

    p[k+1] = p[k] + dt * v_mean,      R[k+1] = exp(dt * omega_mean) R[k]

(exact for a constant rotation axis). The instantaneous state at the last frame, which starts a
rollout, follows under constant acceleration as xi(t_last) = xi_mean(last) + 0.5 dt xi_dot(last).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from ball_world_model.geometry.se3 import compose, rotation_6d_to_matrix

from .frame_encoder import CoordinateAwareFrameEncoder, SpatialKeypointEncoder
from .state import Acceleration, BodyPrediction, KinematicState, ObserverPrediction, Pose, Twist
from .temporal_encoder import IntervalMotionEncoder, IntervalTemporalMixer, group_count

# Quantities whose heads predict standardised values; statistics come from training data.
STANDARDISED_QUANTITIES = (
    "object_in_camera_position",  # per frame
    "camera_position",  # per frame, anchor frame
    "object_linear_velocity",  # interval means, anchor frame
    "object_angular_velocity",
    "object_linear_acceleration",
    "object_angular_acceleration",
    "camera_linear_velocity",
    "camera_angular_velocity",
    "camera_linear_acceleration",
    "camera_angular_acceleration",
    "gravity",  # per window, anchor frame
)

class GridAveragePool(nn.Module):
    """AdaptiveAvgPool2d(grid) as two matrix products: [B, C, H, W] -> [B, C, grid, grid].

    The built-in layer has no deterministic CUDA backward.
    """

    def __init__(self, grid: int) -> None:
        super().__init__()
        self.grid = grid

    def _bin_weights(self, size: int, like: Tensor) -> Tensor:
        weights = like.new_zeros(self.grid, size)
        for index in range(self.grid):
            start, end = index * size // self.grid, -(-(index + 1) * size // self.grid)
            weights[index, start:end] = 1.0 / (end - start)
        return weights

    def forward(self, maps: Tensor) -> Tensor:
        rows = self._bin_weights(maps.shape[-2], maps)
        columns = self._bin_weights(maps.shape[-1], maps)
        return rows @ maps @ columns.T


@dataclass(frozen=True)
class SE3ObserverConfig:
    """Architecture sizes and the training-set standardisation.

    standardisation  {quantity: {"mean": [3], "std": [3]}} for STANDARDISED_QUANTITIES; missing
                     quantities use mean 0 and std 1.
    """

    feature_channels: int = 128
    descriptor_dim: int = 256
    motion_dim: int = 128
    scene_dim: int = 256
    temporal_layers: int = 2
    keypoints: int = 8
    keypoint_dim: int = 32
    default_dt: float = 0.01
    standardiser_momentum: float = 0.01
    standardisation: dict[str, dict[str, list[float]]] = field(default_factory=dict)


class SceneDescriptor(nn.Module):
    """Whole-image descriptor that keeps the coarse layout (where floor and walls are).

    A strided convolution followed by a fixed 6 x 6 pooling grid and a linear layer, instead of
    global average pooling, so the gravity head can read the orientation of the scene.
    Input [B, C, H, W] -> [B, scene_dim].
    """

    def __init__(self, feature_channels: int, scene_dim: int, grid: int = 6) -> None:
        super().__init__()
        hidden = 64
        self.network = nn.Sequential(
            nn.Conv2d(feature_channels, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(hidden), hidden),
            nn.SiLU(),
            #nn.AdaptiveAvgPool2d(grid), # No deterministic backward pass, so GridAveragePool is introduced.
            GridAveragePool(grid),
            nn.Flatten(),
            nn.Linear(hidden * grid * grid, scene_dim),
            nn.LayerNorm(scene_dim),
            nn.SiLU(),
        )

    def forward(self, maps: Tensor) -> Tensor:
        return self.network(maps.float())


class SE3Observer(nn.Module):
    def __init__(self, config: SE3ObserverConfig | None = None) -> None:
        super().__init__()
        config = config or SE3ObserverConfig()
        self.config = config
        motion_dim = config.motion_dim

        self.frame_encoder = CoordinateAwareFrameEncoder(config.feature_channels)
        self.spatial_encoder = SpatialKeypointEncoder(
            config.feature_channels, config.keypoints, config.keypoint_dim, config.descriptor_dim
        )
        # Object pose relative to the camera: position (3) and continuous 6D rotation (6).
        self.pose_head = nn.Sequential(
            nn.Linear(config.descriptor_dim, config.descriptor_dim),
            nn.SiLU(),
            nn.Linear(config.descriptor_dim, 9),
        )

        self.object_motion_encoder = IntervalMotionEncoder(
            config.feature_channels, motion_dim, config.standardiser_momentum
        )
        self.object_temporal_mixer = IntervalTemporalMixer(motion_dim, config.temporal_layers)
        self.camera_motion_encoder = IntervalMotionEncoder(
            config.feature_channels, motion_dim, config.standardiser_momentum
        )
        self.camera_temporal_mixer = IntervalTemporalMixer(motion_dim, config.temporal_layers)
        self.object_twist_head = nn.Linear(2 * motion_dim, 6)
        self.object_acceleration_head = nn.Linear(2 * motion_dim, 6)
        self.camera_twist_head = nn.Linear(motion_dim, 6)
        self.camera_acceleration_head = nn.Linear(motion_dim, 6)

        # Camera pose relative to the anchor: each context frame paired with the anchor frame.
        self.anchor_pair_encoder = IntervalMotionEncoder(
            config.feature_channels, motion_dim, config.standardiser_momentum
        )
        self.camera_pose_head = nn.Sequential(
            nn.Linear(motion_dim, motion_dim),
            nn.SiLU(),
            nn.Linear(motion_dim, 9),
        )

        self.scene_descriptor = SceneDescriptor(config.feature_channels, config.scene_dim)
        self.gravity_head = nn.Linear(config.scene_dim, 3)

        for name in STANDARDISED_QUANTITIES:
            entry = config.standardisation.get(name, {"mean": [0.0] * 3, "std": [1.0] * 3})
            self.register_buffer(f"{name}_mean", torch.tensor(entry["mean"], dtype=torch.float32))
            self.register_buffer(
                f"{name}_std", torch.tensor(entry["std"], dtype=torch.float32).clamp_min(1.0e-6)
            )

    def scales(self) -> dict[str, Tensor]:
        """Standard deviations used both to decode the heads and to normalise the losses."""
        return {name: getattr(self, f"{name}_std") for name in STANDARDISED_QUANTITIES}

    def _decode(self, raw: Tensor, name: str) -> Tensor:
        return raw * getattr(self, f"{name}_std") + getattr(self, f"{name}_mean")

    def _times(self, time: Tensor | None, frames: Tensor) -> Tensor:
        if time is None:
            steps = torch.arange(frames.shape[1], device=frames.device, dtype=torch.float32)
            return (steps * self.config.default_dt).expand(frames.shape[0], -1)
        return time.float()

    def _pose(self, raw: Tensor, position_name: str) -> Pose:
        raw = raw.float()
        return Pose(self._decode(raw[..., :3], position_name), rotation_6d_to_matrix(raw[..., 3:]))

    def _motion(self, raw: Tensor, body: str, kind: str) -> tuple[Tensor, Tensor]:
        raw = raw.float()
        return (
            self._decode(raw[..., :3], f"{body}_linear_{kind}"),
            self._decode(raw[..., 3:], f"{body}_angular_{kind}"),
        )

    def forward(
        self, context_rgb: Tensor, context_time: Tensor | None = None
    ) -> ObserverPrediction:
        """context_rgb [B, T, 3, H, W], context_time [B, T] in s."""
        if context_rgb.shape[1] < 3:
            raise ValueError("At least three frames are needed to observe acceleration.")

        maps = self.frame_encoder(context_rgb)
        spatial = self.spatial_encoder(maps)
        time = self._times(context_time, context_rgb)
        dt = torch.diff(time, dim=1).clamp_min(1.0e-6)

        object_in_camera = self._pose(
            self.pose_head(spatial.frame_descriptor), "object_in_camera_position"
        )

        # Camera pose of every earlier frame relative to the anchor; identity at the anchor.
        gap_to_anchor = (time[:, -1:] - time[:, :-1]).clamp_min(1.0e-6)
        anchor = maps[:, -1:].expand(-1, maps.shape[1] - 1, -1, -1, -1)
        pair_features = self.anchor_pair_encoder.encode_pairs(maps[:, :-1], anchor, gap_to_anchor)
        earlier = self._pose(self.camera_pose_head(pair_features), "camera_position")
        batch = context_rgb.shape[0]
        identity = Pose(
            earlier.position.new_zeros(batch, 1, 3),
            torch.eye(3, device=maps.device).expand(batch, 1, 3, 3),
        )
        camera_poses = Pose(
            torch.cat((earlier.position, identity.position), dim=1),
            torch.cat((earlier.rotation, identity.rotation), dim=1),
        )

        camera_motion = self.camera_temporal_mixer(self.camera_motion_encoder(maps, dt))
        object_motion = self.object_temporal_mixer(self.object_motion_encoder(maps, dt))
        object_features = torch.cat((object_motion, camera_motion), dim=-1)

        object_poses = Pose(
            *compose(
                camera_poses.position,
                camera_poses.rotation,
                object_in_camera.position,
                object_in_camera.rotation,
            )
        )
        gravity = self._decode(
            self.gravity_head(self.scene_descriptor(maps[:, -1])).float(), "gravity"
        )

        return ObserverPrediction(
            object_in_camera=object_in_camera,
            camera=body_prediction(
                camera_poses,
                Twist(*self._motion(self.camera_twist_head(camera_motion), "camera", "velocity")),
                Acceleration(
                    *self._motion(
                        self.camera_acceleration_head(camera_motion), "camera", "acceleration"
                    )
                ),
                dt,
            ),
            object=body_prediction(
                object_poses,
                Twist(*self._motion(self.object_twist_head(object_features), "object", "velocity")),
                Acceleration(
                    *self._motion(
                        self.object_acceleration_head(object_features), "object", "acceleration"
                    )
                ),
                dt,
            ),
            gravity=gravity,
            interval_dt=dt,
            keypoint_coordinates=spatial.coordinates,
        )


def current_state(
    frame_poses: Pose,
    interval_twists: Twist,
    interval_accelerations: Acceleration,
    dt: Tensor,
) -> KinematicState:
    """Instantaneous state at the last frame; constant acceleration over the last interval."""
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


def body_prediction(
    frame_poses: Pose,
    interval_twists: Twist,
    interval_accelerations: Acceleration,
    dt: Tensor,
) -> BodyPrediction:
    return BodyPrediction(
        frame_poses=frame_poses,
        interval_twists=interval_twists,
        interval_accelerations=interval_accelerations,
        current=current_state(frame_poses, interval_twists, interval_accelerations, dt),
    )

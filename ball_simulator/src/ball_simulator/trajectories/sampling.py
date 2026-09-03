from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation

from .config import ExperimentConfig, Range
from .models import RigidBodyState, SphereParameters
from .environments import (
    SimulationEnvironment, 
    SingleWallEnvironment, 
    UBoxEnvironment,
    FreeFlightEnvironment,
)


def sample_range(spec: Range, rng: np.random.Generator) -> float:
    if spec.low == spec.high:
        return spec.low
    if spec.scale == "log":
        return float(np.exp(rng.uniform(np.log(spec.low), np.log(spec.high))))
    return float(rng.uniform(spec.low, spec.high))

def sample_angular_velocity(
    spin_range: Range,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.asarray(
        [
            sample_range(spin_range, rng)
            for _ in range(3)
        ],
        dtype=float,
    )

class InitialStateSampler(ABC):
    def __init__(
        self,
        config: ExperimentConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.rng = rng

    
    @abstractmethod
    def sample(
        self,
        params: SphereParameters,
    ) -> RigidBodyState:
        raise NotImplementedError
    

class SingleWallInitialStateSampler(InitialStateSampler):
    def __init__(
            self,
            config: ExperimentConfig,
            environment: SingleWallEnvironment,
            rng: np.random.Generator,
    ) -> None:
        super().__init__(config, rng)
        self.environment = environment

    def sample(
        self, 
        params: SphereParameters,
    ) -> RigidBodyState:
        ranges = self.config.initial_state
        wall = self.environment.surfaces[0]

        # This remains specialized to the x-normal wall
        if not np.allclose(wall.normal, [1.0, 0.0, 0.0]):
            raise NotImplementedError(
                "The single-wall initial-state sampler "
                "currently expects normal=[1, 0, 0]."
            )
        
        gap = sample_range(ranges.wall_gap, self.rng)

        position = wall.point + np.array(
            [
                params.radius + gap,
                sample_range(
                    ranges.lateral_position,
                    self.rng,
                ),
                sample_range(
                    ranges.vertical_position,
                    self.rng,
                ),
            ]
        )

        velocity = np.array(
            [
                -sample_range(
                    ranges.wall_normal_speed,
                    self.rng,
                ),
                sample_range(
                    ranges.tangential_speed,
                    self.rng,
                ),
                sample_range(
                    ranges.tangential_speed,
                    self.rng,
                ),
            ]
        )

        angular_velocity = sample_angular_velocity(
            ranges.spin,
            self.rng,
        )

        quaternion = Rotation.random(
            random_state=self.rng
        ).as_quat(canonical=False)

        return RigidBodyState(
            position=position,
            quaternion_xyzw=quaternion,
            linear_velocity=velocity,
            angular_velocity=angular_velocity,
        )


class UBoxInitialStateSampler(InitialStateSampler):
    def __init__(
        self,
        config: ExperimentConfig,
        environment: UBoxEnvironment,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(config, rng)
        self.environment = environment

    
    def _horizontal_direction(self) -> float:
        mode = (
            self.config
            .u_box_initial_state
            .initial_horizontal_direction
        )

        if mode == "left":
            return -1.0

        if mode == "right":
            return 1.0

        return float(
            self.rng.choice([-1.0, 1.0])
        )

    
    def sample(
        self,
        params: SphereParameters,
    ) -> RigidBodyState:
        ranges = self.config.u_box_initial_state
        box = self.environment.config

        left_x = box.left_wall.point[0]
        right_x = box.right_wall.point[0]
        floor_z = box.floor.point[2]

        usable_left = left_x + params.radius
        usable_right = right_x - params.radius

        if usable_right <= usable_left:
            raise ValueError(
                "The ball diameter is not smaller than "
                "the U-box channel width."
            )

        fraction = sample_range(
            ranges.x_fraction,
            self.rng,
        )

        x_position = (
            usable_left
            + fraction
            * (usable_right - usable_left)
        )

        minimum_height = floor_z + params.radius
        sampled_height = sample_range(
            ranges.height,
            self.rng,
        )

        z_position = max(
            minimum_height + 1.0e-6,
            floor_z + sampled_height,
        )

        position = np.array(
            [
                x_position,
                sample_range(
                    ranges.lateral_position,
                    self.rng,
                ),
                z_position,
            ],
            dtype=float,
        )

        direction = self._horizontal_direction()

        velocity = np.array(
            [
                direction
                * sample_range(
                    ranges.x_speed,
                    self.rng,
                ),
                sample_range(
                    ranges.y_speed,
                    self.rng,
                ),
                sample_range(
                    ranges.z_speed,
                    self.rng,
                ),
            ],
            dtype=float,
        )

        angular_velocity = (
            sample_angular_velocity(
                ranges.spin,
                self.rng,
            )
        )

        quaternion = Rotation.random(
            random_state=self.rng
        ).as_quat(canonical=False)

        return RigidBodyState(
            position=position,
            quaternion_xyzw=quaternion,
            linear_velocity=velocity,
            angular_velocity=angular_velocity,
        )


class FreeFlightInitialStateSampler(InitialStateSampler):
    def __init__(
        self,
        config: ExperimentConfig,
        environment: FreeFlightEnvironment,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(config, rng)
        self.environment = environment


    def _minimum_clearance(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float,
        surface,
    ) -> float:
        """Exact minimum sphere-plane clearance for constant acceleration."""
        duration = float(self.config.simulation.duration)
        gravity = np.asarray(self.config.simulation.gravity, dtype=float)
        relative = position - surface.point
        a = 0.5 * float(np.dot(surface.normal, gravity))
        b = float(np.dot(surface.normal, velocity))
        c = float(np.dot(surface.normal, relative)) - radius
        candidates = [0.0, duration]
        if abs(a) > 1.0e-14:
            stationary_time = -b / (2.0 * a)
            if 0.0 < stationary_time < duration:
                candidates.append(stationary_time)
        return min(a * t * t + b * t + c for t in candidates)

    def _is_safe(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float,
    ) -> bool:
        margin = self.config.free_flight_initial_state.safety_margin
        return all(
            self._minimum_clearance(position, velocity, radius, surface) > margin
            for surface in self.environment.surfaces
        )


    def _sample_position(self) -> np.ndarray:
        ranges = self.config.free_flight_initial_state
        if ranges.mode == "rotation":
            return np.asarray(ranges.fixed_position, dtype=float)
        return np.asarray(
            [
                sample_range(ranges.x_position, self.rng),
                sample_range(ranges.y_position, self.rng),
                sample_range(ranges.z_position, self.rng),
            ],
            dtype=float,
        )

    def _sample_velocity(self) -> np.ndarray:
        ranges = self.config.free_flight_initial_state
        if ranges.mode == "rotation":
            return np.zeros(3, dtype=float)
        return np.asarray(
            [
                sample_range(ranges.x_speed, self.rng),
                sample_range(ranges.y_speed, self.rng),
                sample_range(ranges.z_speed, self.rng),
            ],
            dtype=float,
        )

    def sample(self, params: SphereParameters) -> RigidBodyState:
        ranges = self.config.free_flight_initial_state
        for _ in range(ranges.maximum_sampling_attempts):
            position = self._sample_position()
            velocity = self._sample_velocity()
            if not self._is_safe(position, velocity, params.radius):
                continue

            if ranges.mode == "translation":
                quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
                angular_velocity = np.zeros(3, dtype=float)
            else:
                quaternion = Rotation.random(
                    random_state=self.rng,
                ).as_quat(canonical=False)
                angular_velocity = sample_angular_velocity(ranges.spin, self.rng)

            return RigidBodyState(
                position=position,
                quaternion_xyzw=quaternion,
                linear_velocity=velocity,
                angular_velocity=angular_velocity,
            )

        raise RuntimeError(
            "Could not sample a collision-free free-flight trajectory. "
            "Reduce position/velocity ranges, increase reference-plane distance, "
            "or reduce safety_margin."
        )


class InitialStateSamplerFactory:
    @staticmethod
    def create(
        config: ExperimentConfig,
        environment: SimulationEnvironment,
        rng: np.random.Generator,
    ) -> InitialStateSampler:
        if isinstance(environment, SingleWallEnvironment):
            return SingleWallInitialStateSampler(
                config, environment, rng
            )
        
        if isinstance(environment, UBoxEnvironment):
            return UBoxInitialStateSampler(
                config, environment, rng
            )

        if isinstance(environment, FreeFlightEnvironment):
            return FreeFlightInitialStateSampler(
                config, environment, rng
            )
        
        raise TypeError(f"Unsupported environment type: {type(environment).__name__}.")
    

class ParameterSampler:
    def __init__(self, config: ExperimentConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def sample_parameters(self) -> SphereParameters:
        p = self.config.physics
        k_n = sample_range(p.normal_stiffness, self.rng)
        mass = sample_range(p.mass, self.rng)
        radius = sample_range(p.radius, self.rng)
        k_t = sample_range(p.tangential_stiffness_ratio, self.rng) * k_n
        # Relative to critical damping of the translational modes.
        c_t = sample_range(p.tangential_damping_ratio, self.rng) * 2.0 * np.sqrt(mass * k_t)
        return SphereParameters(
            mass=mass,
            radius=radius,
            normal_stiffness=k_n,
            normal_damping=sample_range(p.normal_damping, self.rng),
            tangential_stiffness=k_t,
            tangential_damping=c_t,
            friction=sample_range(p.friction, self.rng),
            drag_coefficient=sample_range(p.drag_coefficient, self.rng),
            air_density=p.air_density,
            magnus_coefficient=sample_range(p.magnus_coefficient, self.rng),
            rotational_drag=sample_range(p.rotational_drag, self.rng),
        )

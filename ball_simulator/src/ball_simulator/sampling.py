from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .config import ExperimentConfig, Range
from .models import RigidBodyState, SphereParameters


def sample_range(spec: Range, rng: np.random.Generator) -> float:
    if spec.low == spec.high:
        return spec.low
    if spec.scale == "log":
        return float(np.exp(rng.uniform(np.log(spec.low), np.log(spec.high))))
    return float(rng.uniform(spec.low, spec.high))


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

    def sample_initial_state(self, params: SphereParameters) -> RigidBodyState:
        s = self.config.initial_state
        point = np.asarray(self.config.simulation.wall_point)
        normal = np.asarray(self.config.simulation.wall_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        # This sampler assumes the default x-normal wall for intuitive ranges.
        if not np.allclose(normal, [1.0, 0.0, 0.0]):
            raise NotImplementedError("Initial-state sampler currently expects wall_normal=[1,0,0]")
        position = point + np.array([
            params.radius + sample_range(s.wall_gap, self.rng),
            sample_range(s.lateral_position, self.rng),
            sample_range(s.vertical_position, self.rng),
        ])
        velocity = np.array([
            -sample_range(s.wall_normal_speed, self.rng),
            sample_range(s.tangential_speed, self.rng),
            sample_range(s.tangential_speed, self.rng),
        ])
        omega = np.array([sample_range(s.spin, self.rng) for _ in range(3)])
        quaternion = Rotation.random(random_state=self.rng).as_quat(canonical=False)
        return RigidBodyState(position, quaternion, velocity, omega, np.zeros(3))

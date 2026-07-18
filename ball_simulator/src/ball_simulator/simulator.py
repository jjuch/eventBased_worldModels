from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .integrators import Integrator, SemiImplicitEuler
from .models import ContactDiagnostics, ContactMode, RigidBodyState, SphereParameters
from .physics import CompliantPlaneContact, CompositeForceModel, EnvironmentForces
from .trajectory import Trajectory, TrajectoryBuffer


class SphereWallSimulator:
    def __init__(self, config: SimulationConfig, integrator: Integrator | None = None) -> None:
        self.config = config
        self.integrator = integrator or SemiImplicitEuler()
        self.force_model = CompositeForceModel(
            EnvironmentForces(np.asarray(config.gravity)),
            CompliantPlaneContact(np.asarray(config.wall_point), np.asarray(config.wall_normal)),
        )

    def simulate(self, initial_state: RigidBodyState, params: SphereParameters, store_high_rate: bool = False) -> Trajectory:
        state = initial_state.copy()
        obs = TrajectoryBuffer()
        hi = TrajectoryBuffer() if store_high_rate else None
        dt = self.config.internal_dt
        obs_stride = round(self.config.observation_dt / dt)
        high_dt = self.config.high_rate_dt or dt
        high_stride = round(high_dt / dt)
        n_steps = round(self.config.duration / dt)
        empty = ContactDiagnostics(False, ContactMode.FREE, 0.0, np.zeros(3), np.zeros(3), np.zeros(3))
        obs.append(0.0, state, empty)
        if hi is not None:
            hi.append(0.0, state, empty)

        last_contact = empty
        for step in range(1, n_steps + 1):
            last_contact = self.integrator.step(state, params, self.force_model, dt).contact
            t = step * dt
            if step % obs_stride == 0:
                obs.append(t, state, last_contact)
            if hi is not None and step % high_stride == 0:
                hi.append(t, state, last_contact)

        return Trajectory(obs.as_arrays(), params, hi.as_arrays() if hi is not None else None)

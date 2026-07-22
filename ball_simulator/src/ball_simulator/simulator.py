from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .integrators import Integrator, SemiImplicitEuler
from .models import ContactDiagnostics, ContactMode, RigidBodyState, SphereParameters, SimulationContext
from .physics import CompositeForceModel, EnvironmentForces
from .trajectory import Trajectory, TrajectoryBuffer
from .environments import SimulationEnvironment


class BallSimulator:
    def __init__(
        self, 
        config: SimulationConfig, 
        environment: SimulationEnvironment,
        integrator: Integrator | None = None,
    ) -> None:
        self.config = config
        self.environment = environment
        self.integrator = integrator or SemiImplicitEuler()

        self.force_model = CompositeForceModel(
            EnvironmentForces(np.asarray(config.gravity, dtype=float)),
            *environment.make_contact_models(),
        )

    def _create_context(self) -> SimulationContext:
        return SimulationContext.for_surface_ids(
            tuple(
                surface.surface_id
                for surface in self.environment.surfaces
            )
        )
    

    def _empty_contacts(self) -> tuple[ContactDiagnostics, ...]:
        zero = np.zeros(3, dtype=float)
        return tuple(
            ContactDiagnostics(
                surface_id=surface.surface_id,
                active=False,
                mode=ContactMode.FREE,
                penetration=0.0,
                normal_force=zero,
                tangential_force=zero,
                contact_velocity=zero,
                tangential_memory=zero,
            )
            for surface in self.environment.surfaces
        )


    def simulate(
        self, 
        initial_state: RigidBodyState, 
        params: SphereParameters, 
        store_high_rate: bool = False,
    ) -> Trajectory:
        state = initial_state.copy()
        context = self._create_context()

        observations = TrajectoryBuffer(
            surface_ids = self.environment.surface_ids
        )

        high_rate = (
            TrajectoryBuffer(
                surface_ids=self.environment.surface_ids
            ) if store_high_rate else None
        )

        dt = self.config.internal_dt
        observation_stride = round(self.config.observation_dt / dt)

        high_rate_dt = self.config.high_rate_dt or dt
        high_rate_stride = round(high_rate_dt / dt)

        number_of_steps = round(self.config.duration / dt)

        initial_contacts = self._empty_contacts()

        observations.append(time=0.0, state=state, contacts=initial_contacts)

        if high_rate is not None:
            high_rate.append(time=0.0, state=state, contacts=initial_contacts)

        last_contact = initial_contacts

        for step in range(1, number_of_steps + 1):
            result = self.integrator.step(
                state=state, 
                params=params, 
                forces=self.force_model,
                context=context, 
                dt=dt,
                )
            
            last_contact = result.contacts
            t = step * dt
            if step % observation_stride == 0:
                observations.append(time=t, state=state, contacts=last_contact)
            if high_rate is not None and step % high_rate_stride == 0:
                high_rate.append(time=t, state=state, contacts=last_contact)

        return Trajectory(
            observations=observations.as_arrays(),
            parameters=params, 
            environment_kind=self.environment.kind.value,
            surface_ids=self.environment.surface_ids,
            high_rate=(
                high_rate.as_arrays()
                if high_rate is not None
                else None
            ),
        )


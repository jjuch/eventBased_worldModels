from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation

from .models import ForceTorque, RigidBodyState, SphereParameters
from .physics import ForceModel


class Integrator(ABC):
    @abstractmethod
    def step(self, state: RigidBodyState, params: SphereParameters, forces: ForceModel, dt: float) -> ForceTorque:
        raise NotImplementedError


class SemiImplicitEuler(Integrator):
    """Stable, auditable baseline; replaceable without changing simulator or dataset APIs."""

    def step(self, state: RigidBodyState, params: SphereParameters, forces: ForceModel, dt: float) -> ForceTorque:
        result = forces.evaluate(state, params, dt)
        acceleration = result.force / params.mass
        angular_acceleration = result.torque / params.inertia_scalar

        state.linear_velocity += dt * acceleration
        state.angular_velocity += dt * angular_acceleration
        state.position += dt * state.linear_velocity

        # SciPy Rotation uses scalar-last [x,y,z,w]. The incremental active rotation
        # is composed on the left because omega is represented in the world frame.
        increment = Rotation.from_rotvec(state.angular_velocity * dt)
        orientation = Rotation.from_quat(state.quaternion_xyzw)
        state.quaternion_xyzw[:] = (increment * orientation).as_quat(canonical=False)
        state.quaternion_xyzw[:] /= np.linalg.norm(state.quaternion_xyzw)
        return result

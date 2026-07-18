from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .models import ContactDiagnostics, ContactMode, ForceTorque, RigidBodyState, SphereParameters


class ForceModel(ABC):
    @abstractmethod
    def evaluate(self, state: RigidBodyState, params: SphereParameters, dt: float) -> ForceTorque:
        raise NotImplementedError


class EnvironmentForces(ForceModel):
    def __init__(self, gravity: np.ndarray) -> None:
        self.gravity = np.asarray(gravity, dtype=float)

    def evaluate(self, state: RigidBodyState, params: SphereParameters, dt: float) -> ForceTorque:
        del dt
        velocity = state.linear_velocity
        speed = np.linalg.norm(velocity)
        drag = np.zeros(3)
        if speed > 0.0 and params.drag_coefficient > 0.0:
            drag = (-0.5 * params.air_density * params.drag_coefficient
                    * params.cross_section * speed * velocity)
        magnus = params.magnus_coefficient * np.cross(state.angular_velocity, velocity)
        torque = -params.rotational_drag * state.angular_velocity
        empty = np.zeros(3)
        diag = ContactDiagnostics(False, ContactMode.FREE, 0.0, empty, empty, empty)
        return ForceTorque(params.mass * self.gravity + drag + magnus, torque, diag)


class CompliantPlaneContact(ForceModel):
    """Hertz/Hunt-Crossley normal force plus history-dependent tangential friction."""

    def __init__(self, point: np.ndarray, normal: np.ndarray, epsilon: float = 1e-10) -> None:
        self.point = np.asarray(point, dtype=float)
        n = np.asarray(normal, dtype=float)
        self.normal = n / np.linalg.norm(n)
        self.epsilon = epsilon

    def _penetration(self, state: RigidBodyState, radius: float) -> float:
        signed_center_distance = float(np.dot(state.position - self.point, self.normal))
        return max(0.0, radius - signed_center_distance)

    def evaluate(self, state: RigidBodyState, params: SphereParameters, dt: float) -> ForceTorque:
        penetration = self._penetration(state, params.radius)
        zero = np.zeros(3)
        if penetration <= 0.0:
            state.tangential_memory[:] = 0.0
            diag = ContactDiagnostics(False, ContactMode.FREE, 0.0, zero, zero, zero)
            return ForceTorque(zero, zero, diag)

        arm = -params.radius * self.normal
        contact_velocity = state.linear_velocity + np.cross(state.angular_velocity, arm)
        normal_velocity = float(np.dot(contact_velocity, self.normal))
        penetration_rate = -normal_velocity

        elastic = params.normal_stiffness * penetration**1.5
        # Non-adhesive Hunt-Crossley-type dissipation multiplier.
        normal_magnitude = max(0.0, elastic * (1.0 + params.normal_damping * penetration_rate))
        normal_force = normal_magnitude * self.normal

        projector = np.eye(3) - np.outer(self.normal, self.normal)
        tangential_velocity = projector @ contact_velocity
        state.tangential_memory[:] = projector @ (
            state.tangential_memory + dt * tangential_velocity
        )
        trial = (-params.tangential_stiffness * state.tangential_memory
                 - params.tangential_damping * tangential_velocity)
        trial_norm = np.linalg.norm(trial)
        limit = params.friction * normal_magnitude

        if trial_norm <= limit + self.epsilon:
            tangential_force = trial
            mode = ContactMode.STICKING
        else:
            if np.linalg.norm(tangential_velocity) > self.epsilon:
                direction = tangential_velocity / np.linalg.norm(tangential_velocity)
            else:
                direction = -trial / max(trial_norm, self.epsilon)
            tangential_force = -limit * direction
            if params.tangential_stiffness > 0.0:
                state.tangential_memory[:] = -(
                    tangential_force + params.tangential_damping * tangential_velocity
                ) / params.tangential_stiffness
                state.tangential_memory[:] = projector @ state.tangential_memory
            mode = ContactMode.SLIDING

        torque = np.cross(arm, tangential_force)
        diag = ContactDiagnostics(
            True, mode, penetration, normal_force, tangential_force, contact_velocity
        )
        return ForceTorque(normal_force + tangential_force, torque, diag)


class CompositeForceModel(ForceModel):
    def __init__(self, *models: ForceModel) -> None:
        self.models = models

    def evaluate(self, state: RigidBodyState, params: SphereParameters, dt: float) -> ForceTorque:
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        contact = ContactDiagnostics(False, ContactMode.FREE, 0.0, np.zeros(3), np.zeros(3), np.zeros(3))
        for model in self.models:
            result = model.evaluate(state, params, dt)
            total_force += result.force
            total_torque += result.torque
            if result.contact.active:
                contact = result.contact
        return ForceTorque(total_force, total_torque, contact)

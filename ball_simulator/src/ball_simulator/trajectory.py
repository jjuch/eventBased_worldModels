from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import ContactDiagnostics, RigidBodyState, SphereParameters


@dataclass(slots=True)
class TrajectoryBuffer:
    times: list[float] = field(default_factory=list)
    position: list[np.ndarray] = field(default_factory=list)
    quaternion: list[np.ndarray] = field(default_factory=list)
    linear_velocity: list[np.ndarray] = field(default_factory=list)
    angular_velocity: list[np.ndarray] = field(default_factory=list)
    tangential_memory: list[np.ndarray] = field(default_factory=list)
    contact_active: list[bool] = field(default_factory=list)
    contact_mode: list[int] = field(default_factory=list)
    penetration: list[float] = field(default_factory=list)
    normal_force: list[np.ndarray] = field(default_factory=list)
    tangential_force: list[np.ndarray] = field(default_factory=list)
    contact_velocity: list[np.ndarray] = field(default_factory=list)

    def append(self, time: float, state: RigidBodyState, contact: ContactDiagnostics) -> None:
        self.times.append(time)
        self.position.append(state.position.copy())
        self.quaternion.append(state.quaternion_xyzw.copy())
        self.linear_velocity.append(state.linear_velocity.copy())
        self.angular_velocity.append(state.angular_velocity.copy())
        self.tangential_memory.append(state.tangential_memory.copy())
        self.contact_active.append(contact.active)
        self.contact_mode.append(int(contact.mode))
        self.penetration.append(contact.penetration)
        self.normal_force.append(contact.normal_force.copy())
        self.tangential_force.append(contact.tangential_force.copy())
        self.contact_velocity.append(contact.contact_velocity.copy())

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "time": np.asarray(self.times, dtype=np.float64),
            "position": np.asarray(self.position, dtype=np.float64),
            "quaternion_xyzw": np.asarray(self.quaternion, dtype=np.float64),
            "linear_velocity": np.asarray(self.linear_velocity, dtype=np.float64),
            "angular_velocity": np.asarray(self.angular_velocity, dtype=np.float64),
            "tangential_memory": np.asarray(self.tangential_memory, dtype=np.float64),
            "contact_active": np.asarray(self.contact_active, dtype=np.bool_),
            "contact_mode": np.asarray(self.contact_mode, dtype=np.int8),
            "penetration": np.asarray(self.penetration, dtype=np.float64),
            "normal_force": np.asarray(self.normal_force, dtype=np.float64),
            "tangential_force": np.asarray(self.tangential_force, dtype=np.float64),
            "contact_velocity": np.asarray(self.contact_velocity, dtype=np.float64),
        }


@dataclass(frozen=True, slots=True)
class Trajectory:
    observations: dict[str, np.ndarray]
    parameters: SphereParameters
    high_rate: dict[str, np.ndarray] | None = None

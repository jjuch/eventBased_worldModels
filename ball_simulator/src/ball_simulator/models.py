from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]


class ContactMode(IntEnum):
    FREE = 0
    STICKING = 1
    SLIDING = 2


@dataclass(frozen=True, slots=True)
class SphereParameters:
    mass: float
    radius: float
    normal_stiffness: float
    normal_damping: float
    tangential_stiffness: float
    tangential_damping: float
    friction: float
    drag_coefficient: float = 0.0
    air_density: float = 1.225
    magnus_coefficient: float = 0.0
    rotational_drag: float = 0.0

    @property
    def inertia_scalar(self) -> float:
        return 0.4 * self.mass * self.radius**2

    @property
    def cross_section(self) -> float:
        return np.pi * self.radius**2


@dataclass(slots=True)
class RigidBodyState:
    position: Vec
    quaternion_xyzw: Vec
    linear_velocity: Vec
    angular_velocity: Vec
    tangential_memory: Vec

    def copy(self) -> "RigidBodyState":
        return RigidBodyState(*(np.array(x, dtype=float, copy=True) for x in (
            self.position, self.quaternion_xyzw, self.linear_velocity,
            self.angular_velocity, self.tangential_memory
        )))


@dataclass(frozen=True, slots=True)
class ContactDiagnostics:
    active: bool
    mode: ContactMode
    penetration: float
    normal_force: Vec
    tangential_force: Vec
    contact_velocity: Vec


@dataclass(frozen=True, slots=True)
class ForceTorque:
    force: Vec
    torque: Vec
    contact: ContactDiagnostics

from __future__ import annotations

from dataclasses import dataclass, field
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

    def copy(self) -> "RigidBodyState":
        return RigidBodyState(*(np.array(x, dtype=float, copy=True) for x in (
            self.position, self.quaternion_xyzw, self.linear_velocity,
            self.angular_velocity
        )))


@dataclass(frozen=True, slots=True)
class ContactDiagnostics:
    surface_id: str
    active: bool
    mode: ContactMode
    penetration: float
    normal_force: Vec
    tangential_force: Vec
    contact_velocity: Vec
    tangential_memory: Vec


@dataclass(frozen=True, slots=True)
class ForceTorque:
    force: Vec
    torque: Vec
    contacts: tuple[ContactDiagnostics, ...]


@dataclass(slots=True)
class SurfaceContactState:
    tangential_memory: Vec = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    was_active: bool = False

    def reset(self) -> None:
        self.tangential_memory[:] = 0.0
        self.was_active = False


@dataclass(slots=True)
class SimulationContext:
    contacts: dict[str, SurfaceContactState]

    @classmethod
    def for_surface_ids(
        cls,
        surface_ids: list[str] | tuple[str, ...],
    ) -> "SimulationContext":
        return cls(
            contacts={
                surface_id: SurfaceContactState()
                for surface_id in surface_ids
            }
        )

    def contact_state(
            self,
            surface_id: str,
    ) -> SurfaceContactState:
        try:
            return self.contacts[surface_id]
        except KeyError as error:
            raise KeyError(
                f"No contact state registered for surface "
                f"{surface_id!r}."
            ) from error
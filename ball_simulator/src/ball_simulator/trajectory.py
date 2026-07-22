from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import ContactDiagnostics, RigidBodyState, SphereParameters


@dataclass(slots=True)
class TrajectoryBuffer:
    surface_ids: tuple[str, ...]

    times: list[float] = field(default_factory=list)
    position: list[np.ndarray] = field(default_factory=list)
    quaternion: list[np.ndarray] = field(default_factory=list)
    linear_velocity: list[np.ndarray] = field(default_factory=list)
    angular_velocity: list[np.ndarray] = field(default_factory=list)

    contact_active: list[bool] = field(default_factory=list)
    contact_mode: list[int] = field(default_factory=list)
    penetration: list[float] = field(default_factory=list)
    normal_force: list[np.ndarray] = field(default_factory=list)
    tangential_force: list[np.ndarray] = field(default_factory=list)
    contact_velocity: list[np.ndarray] = field(default_factory=list)
    tangential_memory: list[np.ndarray] = field(default_factory=list)

    
    def _contacts_by_surface(
            self,
            contacts: tuple[ContactDiagnostics, ...],
    ) -> dict[str, ContactDiagnostics]:
        contacts_by_surface = {
            contact.surface_id: contact
            for contact in contacts
        }

        expected = set(self.surface_ids)
        received = set(contacts_by_surface)

        if expected != received:
            missing = expected - received
            extra = received - expected

            raise ValueError(
                "Contact diagnostics do not match environment "
                f"surfaces. Missing={missing}, extra={extra}."
            )

        return contacts_by_surface

    def append(
        self, 
        time: float, 
        state: RigidBodyState, 
        contacts: tuple[ContactDiagnostics, ...],
    ) -> None:
        contacts_by_surface = self._contacts_by_surface(contacts)

        ordered = [
            contacts_by_surface[surface_id]
            for surface_id in self.surface_ids
        ]

        self.times.append(time)
        self.position.append(state.position.copy())
        self.quaternion.append(state.quaternion_xyzw.copy())
        self.linear_velocity.append(state.linear_velocity.copy())
        self.angular_velocity.append(state.angular_velocity.copy())

        self.contact_active.append(
            np.asarray(
                [contact.active for contact in ordered], dtype=np.bool_,
            )
        )
        self.contact_mode.append(
            np.asarray(
                [int(contact.mode) for contact in ordered], dtype=np.int8,
            )
        )
        self.penetration.append(
            np.asarray(
                [
                    contact.penetration
                    for contact in ordered
                ], dtype=np.float64,
            )
        )
        self.normal_force.append(
            np.stack(
                [contact.normal_force for contact in ordered]
            )
        )
        self.tangential_force.append(
            np.stack(
                [contact.tangential_force for contact in ordered]
            )
        )
        self.contact_velocity.append(
            np.stack(
                [contact.contact_velocity for contact in ordered]
            )
        )
        self.tangential_memory.append(
            np.stack(
                [contact.tangential_memory for contact in ordered]
            )
        )
        


    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "time": np.asarray(self.times, dtype=np.float64),
            "position": np.asarray(self.position, dtype=np.float64),
            "quaternion_xyzw": np.asarray(self.quaternion, dtype=np.float64),
            "linear_velocity": np.asarray(self.linear_velocity, dtype=np.float64),
            "angular_velocity": np.asarray(self.angular_velocity, dtype=np.float64),
            "contact_active": np.asarray(self.contact_active, dtype=np.bool_),
            "contact_mode": np.asarray(self.contact_mode, dtype=np.int8),
            "penetration": np.asarray(self.penetration, dtype=np.float64),
            "normal_force": np.asarray(self.normal_force, dtype=np.float64),
            "tangential_force": np.asarray(self.tangential_force, dtype=np.float64),
            "contact_velocity": np.asarray(self.contact_velocity, dtype=np.float64),
            "tangential_memory": np.asarray(self.tangential_memory, dtype=np.float64),
        }


@dataclass(frozen=True, slots=True)
class Trajectory:
    observations: dict[str, np.ndarray]
    parameters: SphereParameters
    environment_kind: str
    surface_ids: tuple[str, ...]
    high_rate: dict[str, np.ndarray] | None = None

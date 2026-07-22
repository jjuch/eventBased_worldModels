from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .config import (
    ExperimentConfig,
    PlaneConfig,
    SingleWallEnvironmentConfig,
    UBoxEnvironmentConfig,
)
from .physics import CompliantPlaneContact, ForceModel


class EnvironmentKind(str, Enum):
    SINGLE_WALL = "single-wall"
    U_BOX = "u-box"

@dataclass(frozen=True, slots=True)
class PlaneSurface:
    surface_id: str
    point: np.ndarray
    normal: np.ndarray

    @classmethod
    def from_config(
        cls,
        config: PlaneConfig,
    ) -> "PlaneSurface":
        normal = np.asarray(config.normal, dtype=float)
        normal /= np.linalg.norm(normal)

        return cls(
            surface_id=config.surface_id,
            point=np.asarray(config.point, dtype=float),
            normal=normal,
        )
    

class SimulationEnvironment(ABC):
    @property
    @abstractmethod
    def kind(self) -> EnvironmentKind:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def surfaces(self) -> tuple[PlaneSurface, ...]:
        raise NotImplementedError
    
    def make_contact_models(
        self,
    ) -> tuple[ForceModel, ...]:
        return tuple(
            CompliantPlaneContact(
                surface_id=surface.surface_id,
                point=surface.point,
                normal=surface.normal,
            ) for surface in self.surfaces
        )
    
    @abstractmethod
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError
    
    @property
    def surface_ids(self) -> tuple[str, ...]:
        return tuple(
            surface.surface_id
            for surface in self.surfaces
        )
    

class SingleWallEnvironment(SimulationEnvironment):
    def __init__(
        self,
        config: SingleWallEnvironmentConfig,
    ) -> None:
        self.config = config
        self._surfaces = tuple(
            (PlaneSurface.from_config(config.wall),)
        )

    @property
    def kind(self) -> EnvironmentKind:
        return EnvironmentKind.SINGLE_WALL
    
    @property
    def surfaces(self) -> tuple[PlaneSurface, ...]:
        return self._surfaces
    
    def metadata(self) -> dict[str, object]:
        wall = self._surfaces[0]

        return {
            "kind": self.kind.value,
            "surfaces": {
                wall.surface_id: {
                    "point": wall.point.tolist(),
                    "normal": wall.normal.tolist(),
                }
            },
        }
    
class UBoxEnvironment(SimulationEnvironment):
    def __init__(
        self,
        config: UBoxEnvironmentConfig,
    ) -> None:
        self.config = config
        self._surfaces = tuple(
            PlaneSurface.from_config(plane)
            for plane in (
                config.left_wall,
                config.right_wall,
                config.floor,
            )
        )

    @property
    def kind(self) -> EnvironmentKind:
        return EnvironmentKind.U_BOX

    @property
    def surfaces(self) -> tuple[PlaneSurface, ...]:
        return self._surfaces

    def metadata(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "channel_width": self.config.channel_width,
            "axis": "y",
            "surfaces": {
                surface.surface_id: {
                    "point": surface.point.tolist(),
                    "normal": surface.normal.tolist(),
                }
                for surface in self._surfaces
            },
        }
    

class EnvironmentFactory:
    @staticmethod
    def create(
        kind: EnvironmentKind,
        config: ExperimentConfig,
    ) -> SimulationEnvironment:
        if kind is EnvironmentKind.SINGLE_WALL:
            return SingleWallEnvironment(
                config.environments.single_wall
            )

        if kind is EnvironmentKind.U_BOX:
            return UBoxEnvironment(
                config.environments.u_box
            )

        raise ValueError(
            f"Unsupported environment kind: {kind}"
        )
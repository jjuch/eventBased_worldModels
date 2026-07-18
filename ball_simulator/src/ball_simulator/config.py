from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Range(BaseModel):
    low: float
    high: float
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def valid_range(self) -> "Range":
        if self.high < self.low:
            raise ValueError("high must be at least low")
        if self.scale == "log" and self.low <= 0:
            raise ValueError("log ranges require low > 0")
        return self


class PhysicsRanges(BaseModel):
    mass: Range = Range(low=0.1, high=0.5)
    radius: Range = Range(low=0.04, high=0.08)
    normal_stiffness: Range = Range(low=5e4, high=5e5, scale="log")
    normal_damping: Range = Range(low=0.05, high=0.8)
    tangential_stiffness_ratio: Range = Range(low=0.15, high=0.5)
    tangential_damping_ratio: Range = Range(low=0.05, high=0.5)
    friction: Range = Range(low=0.1, high=1.0)
    drag_coefficient: Range = Range(low=0.0, high=0.0)
    air_density: float = Field(default=1.225, ge=0)
    magnus_coefficient: Range = Range(low=0.0, high=0.0)
    rotational_drag: Range = Range(low=0.0, high=0.0)


class InitialStateRanges(BaseModel):
    wall_gap: Range = Range(low=0.15, high=1.0)
    wall_normal_speed: Range = Range(low=1.0, high=8.0)
    tangential_speed: Range = Range(low=-4.0, high=4.0)
    vertical_position: Range = Range(low=0.5, high=2.0)
    lateral_position: Range = Range(low=-0.5, high=0.5)
    spin: Range = Range(low=-150.0, high=150.0)


class SimulationConfig(BaseModel):
    duration: float = Field(default=1.0, gt=0)
    internal_dt: float = Field(default=2e-5, gt=0)
    observation_dt: float = Field(default=5e-3, gt=0)
    high_rate_dt: float | None = Field(default=None, gt=0)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    wall_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wall_normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    penetration_tolerance: float = Field(default=1e-12, ge=0)

    @model_validator(mode="after")
    def compatible_steps(self) -> "SimulationConfig":
        ratio = self.observation_dt / self.internal_dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("observation_dt must be an integer multiple of internal_dt")
        if self.high_rate_dt is not None:
            ratio = self.high_rate_dt / self.internal_dt
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError("high_rate_dt must be an integer multiple of internal_dt")
        return self


class DatasetConfig(BaseModel):
    trajectories: int = Field(default=100, gt=0)
    seed: int = 42
    output: str = "trajectories.h5"
    compression: Literal["gzip", "lzf", "none"] = "gzip"
    compression_level: int = Field(default=4, ge=0, le=9)
    store_high_rate: bool = False


class ExperimentConfig(BaseModel):
    physics: PhysicsRanges = PhysicsRanges()
    initial_state: InitialStateRanges = InitialStateRanges()
    simulation: SimulationConfig = SimulationConfig()
    dataset: DatasetConfig = DatasetConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)

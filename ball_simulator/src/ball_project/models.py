from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ExperimentInfo(BaseModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    subtype: str = Field(min_length=1)
    mode: Literal["translation", "rotation", "combined"]
    created_at: str
    description: str = ""


class ProjectConfigs(BaseModel):
    trajectories: Path = Path("configs/trajectories.yaml")
    rendering: Path = Path("configs/rendering.yaml")
    data: Path = Path("configs/data.yaml")
    training: Path = Path("configs/training.yaml")
    optimised_rendering: Path = Path("configs/generated/rendering_optimised.yaml")


class ProjectPaths(BaseModel):
    trajectory_dataset: Path = Path("data/trajectories/trajectories.h5")
    rendered: Path = Path("data/rendered")
    manifest: Path = Path("data/manifests/manifest.parquet")
    training_outputs: Path = Path("outputs/training")
    checkpoints: Path = Path("outputs/checkpoints")
    tensorboard: Path = Path("outputs/tensorboard")
    evaluation: Path = Path("outputs/evaluation")
    inspection: Path = Path("outputs/inspection")
    camera_report: Path = Path("reports/camera.json")
    logs: Path = Path("logs")


class RuntimeConfig(BaseModel):
    simulation_workers: int = Field(default=0, ge=0)
    render_workers: int = Field(default=2, ge=1)
    threads_per_blender: int = Field(default=4, ge=0)
    resume_rendering: bool = True
    blender_executable: Path | None = None


class ProjectManifest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    project: ExperimentInfo
    configs: ProjectConfigs = ProjectConfigs()
    paths: ProjectPaths = ProjectPaths()
    runtime: RuntimeConfig = RuntimeConfig()

    @model_validator(mode="after")
    def supported_initial_project(self) -> "ProjectManifest":
        if (self.project.type, self.project.subtype) != ("ball", "free-flight"):
            raise ValueError(
                "This installation currently supports type=ball subtype=free-flight. "
                "The registry is intentionally extensible for future experiment types."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)
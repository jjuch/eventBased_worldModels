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
    training: Path = Path("configs/training.yaml")


class ProjectPaths(BaseModel):
    datasets: Path = Path("datasets")
    runs: Path = Path("runs")
    reports: Path = Path("reports")


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


class DatasetInfo(BaseModel):
    """Contents of ``datasets/<name>/dataset.yaml``.

    reference  dataset whose configs this one was copied from, if any
    camera     ``optimise``: search a camera for this dataset; ``fixed``: keep the camera in
               rendering.yaml (copied from the reference, so a trained observer sees the
               same view)
    """

    name: str = Field(min_length=1)
    created_at: str
    description: str = ""
    reference: str | None = None
    camera: Literal["optimise", "fixed"] = "optimise"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetInfo":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)

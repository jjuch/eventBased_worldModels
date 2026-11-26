from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import ProjectManifest, DatasetInfo

PROJECT_FILE = "project.yaml"
DEFAULT_RUN = "default"
DEFAULT_DATASET = "default"


@dataclass(frozen=True, slots=True)
class DatasetContext:
    """One dataset: its configs, trajectories, renders and manifest, all under ``root``."""

    name: str
    root: Path

    @property
    def info_file(self) -> Path:
        return self.root / "dataset.yaml"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def trajectories_config(self) -> Path:
        return self.configs / "trajectories.yaml"

    @property
    def rendering_config(self) -> Path:
        return self.configs / "rendering.yaml"

    @property
    def data_config(self) -> Path:
        return self.configs / "data.yaml"

    @property
    def trajectories(self) -> Path:
        return self.root / "trajectories.h5"

    @property
    def rendered(self) -> Path:
        return self.root / "rendered"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.parquet"

    @property
    def camera_report(self) -> Path:
        return self.root / "reports" / "camera.json"

    @property
    def inspection(self) -> Path:
        return self.root / "inspection"

    @property
    def effective(self) -> Path:
        return self.root / "effective"

    def info(self) -> DatasetInfo:
        return DatasetInfo.from_yaml(self.info_file)



@dataclass(frozen=True, slots=True)
class RunContext:
    """One training run. Everything run-specific lives under ``root``."""
    name: str
    root: Path

    @property
    def training_config(self) -> Path:
        return self.root / "training.yaml"

    @property
    def effective_training_config(self) -> Path:
        return self.root / "effective" / "training.yaml"

    @property
    def record(self) -> Path:
        return self.root / "run.json"

    @property
    def dataset_snapshot(self) -> Path:
        """Copies of the training dataset's configs, taken when training starts."""
        return self.root / "dataset_snapshot"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evaluations(self) -> Path:
        return self.root / "evaluations"

    @property
    def logs(self) -> Path:
        return self.root / "logs"
    
    def training_dataset_name(self) -> str:
        """Dataset named by the ``dataset:`` entry of this run's training.yaml."""
        value = yaml.safe_load(self.training_config.read_text(encoding="utf-8")) or {}
        name = value.get("dataset")
        if not name:
            raise ValueError(
                f"{self.training_config} has no 'dataset:' entry naming its training dataset."
            )
        return str(name)


def _select(kind: str, name: str | None, names: list[str]) -> str:
    """Resolve an optional name: without one, the project must contain exactly one."""
    if name is None:
        if len(names) != 1:
            available = ", ".join(names) if names else "none yet"
            raise ValueError(
                f"Specify a {kind} with --{kind}. Available: {available}. "
                f"Create one with 'ball_project {kind}s new NAME'."
            )
        return names[0]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"A {kind} name must be one folder name, without path separators.")
    return name



@dataclass(frozen=True, slots=True)
class ProjectContext:
    root: Path
    manifest_path: Path
    manifest: ProjectManifest

    def resolve(self, path: str | Path):
        return (self.root / path).resolve()

    @property
    def runs_directory(self) -> Path:
        return self.resolve(self.manifest.paths.runs)

    @property
    def datasets_directory(self) -> Path:
        return self.resolve(self.manifest.paths.datasets)

    @property
    def reports_directory(self) -> Path:
        return self.resolve(self.manifest.paths.reports)

    def run_names(self) -> list[str]:
        """Names of the existing runs (folders with a training.yaml), sorted."""
        if not self.runs_directory.is_dir():
            return []
        return sorted(
            path.name for path in self.runs_directory.iterdir()
            if (path / "training.yaml").is_file()
        )

    def dataset_names(self) -> list[str]:
        """Names of the existing datasets (folders with a dataset.yaml), sorted."""
        if not self.datasets_directory.is_dir():
            return []
        return sorted(
            path.name for path in self.datasets_directory.iterdir()
            if (path / "dataset.yaml").is_file()
        )

    def run(self, name: str | None = None, *, must_exist: bool = True) -> RunContext:
        """Look up a run. Without ``name`` the project must contain exactly one run."""
        name = _select("run", name, self.run_names())
        run = RunContext(name=name, root=self.runs_directory / name)
        if must_exist and not run.training_config.is_file():
            raise FileNotFoundError(
                f"Run {name!r} does not exist. Create it with 'ball_project runs new {name}'."
            )
        return run

    def dataset(self, name: str | None = None, *, must_exist: bool = True) -> DatasetContext:
        """Look up a dataset. Without ``name`` the project must contain exactly one dataset."""
        name = _select("dataset", name, self.dataset_names())
        dataset = DatasetContext(name=name, root=self.datasets_directory / name)
        if must_exist and not dataset.info_file.is_file():
            raise FileNotFoundError(
                f"Dataset {name!r} does not exist. "
                f"Create it with 'ball_project datasets new {name}'."
            )
        return dataset


def load_project(root: str | Path) -> ProjectContext:
    """Load the project in ``root`` directly, ignoring BALL_PROJECT_ROOT and parent folders."""
    root = Path(root).resolve()
    manifest_path = root / PROJECT_FILE
    return ProjectContext(
        root=root,
        manifest_path=manifest_path,
        manifest=ProjectManifest.from_yaml(manifest_path),
    )


def discover_project(start: str | Path | None = None) -> ProjectContext:
    explicit = os.environ.get("BALL_PROJECT_ROOT")
    current = Path(explicit or start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        manifest_path = candidate / PROJECT_FILE
        if manifest_path.is_file():
            return ProjectContext(
                root=candidate,
                manifest_path=manifest_path,
                manifest=ProjectManifest.from_yaml(manifest_path),
            )
    raise FileNotFoundError(
        "No project.yaml found in the current directory or its parents. "
        "Create a project with 'ball_project create NAME'."
    )
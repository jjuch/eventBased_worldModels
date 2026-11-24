from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .models import ProjectManifest

PROJECT_FILE = "project.yaml"
DEFAULT_RUN = "default"


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
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def logs(self) -> Path:
        return self.root / "logs"
    


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

    def run_names(self) -> list[str]:
        """Names of the existing runs (folders with a training.yaml), sorted."""
        if not self.runs_directory.is_dir():
            return []
        return sorted(
            path.name for path in self.runs_directory.iterdir()
            if (path / "training.yaml").is_file()
        )

    def run(self, name: str | None = None, *, must_exist: bool = True) -> RunContext:
        """Look up a run. Without ``name`` the project must contain exactly one run."""
        if name is None:
            names = self.run_names()
            if len(names) != 1:
                available = ", ".join(names) if names else "none yet"
                raise ValueError(
                    f"Specify a run with --run. Available runs: {available}. "
                    "Create one with 'ball_project runs new NAME'."
                )
            name = names[0]

        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("A run name must be one folder name, without path separators.")
        run = RunContext(name=name, root=self.runs_directory / name)

        if must_exist and not run.training_config.is_file():
            raise FileNotFoundError(
                f"Run {name!r} does not exist. Create it with 'ball_project runs new {name}'." 
            )
        return run


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
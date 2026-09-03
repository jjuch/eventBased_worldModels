from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .models import ProjectManifest

PROJECT_FILE = "project.yaml"


@dataclass(frozen=True, slots=True)
class ProjectContext:
    root: Path
    manifest_path: Path
    manifest: ProjectManifest

    def resolve(self, path: str | Path):
        return (self.root / path).resolve()


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
"""Datasets inside a project: create them and snapshot them for runs.

A dataset is immutable once a run has used it. A new shape, marker layout or physics setting
gets a new dataset, usually copied from a reference so that only the intended settings differ.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from .discovery import DatasetContext, ProjectContext
from .models import DatasetInfo
from .template_registry import template_text

CONFIG_FILES = ("trajectories.yaml", "rendering.yaml", "data.yaml")


def write_template_configs(context: ProjectContext, directory: Path) -> None:
    """Write the project's template trajectories, rendering and data configs into ``directory``."""
    info = context.manifest.project
    templates = {
        "trajectories.yaml": f"trajectories_{info.mode}.yaml",
        "rendering.yaml": f"rendering_{info.mode}.yaml",
        "data.yaml": "data.yaml",
    }
    directory.mkdir(parents=True, exist_ok=True)
    for destination, source in templates.items():
        (directory / destination).write_text(
            template_text(info.type, info.subtype, source), encoding="utf-8"
        )


def create_dataset(
    context: ProjectContext,
    name: str,
    *,
    from_dataset: str | None = None,
    description: str = "",
) -> DatasetContext:
    """Create ``datasets/<name>/`` with configs from a reference dataset or the templates.

    A copied dataset keeps the reference's camera (``camera: fixed``) once that camera has been
    optimised, so models trained on the reference see the same view.
    """
    dataset = context.dataset(name, must_exist=False)
    if dataset.root.exists():
        raise FileExistsError(f"Dataset already exists: {dataset.root}.")

    if from_dataset is not None:
        reference = context.dataset(from_dataset)
        reference_camera_final = (
            reference.camera_report.is_file() or reference.info().camera == "fixed"
        )
        dataset.configs.mkdir(parents=True)
        for file_name in CONFIG_FILES:
            shutil.copy2(reference.configs / file_name, dataset.configs / file_name)
        camera = "fixed" if reference_camera_final else "optimise"
    else:
        write_template_configs(context, dataset.configs)
        camera = "optimise"

    DatasetInfo(
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(),
        description=description,
        reference=from_dataset,
        camera=camera,
    ).to_yaml(dataset.info_file)
    return dataset


def snapshot_dataset(dataset: DatasetContext, destination: Path) -> Path:
    """Copy dataset.yaml and the dataset's configs into ``destination`` (replacing it)."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(dataset.info_file, destination / "dataset.yaml")
    for file_name in CONFIG_FILES:
        shutil.copy2(dataset.configs / file_name, destination / file_name)
    return destination

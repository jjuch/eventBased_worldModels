"""Datasets inside a project: create them and snapshot them for runs.

A dataset is immutable once a run has used it. A new shape, marker layout or physics setting
gets a new dataset, usually copied from a reference so that only the intended settings differ.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .discovery import DatasetContext, ProjectContext
from .models import DatasetInfo
from .template_registry import template_text

CONFIG_FILES = ("trajectories.yaml", "rendering.yaml", "data.yaml")
# Dataset templates besides the project's own mode: name -> files for trajectories, rendering, data.
DATASET_TEMPLATES = {
    "moving-camera": (
        "trajectories_moving_camera.yaml", "rendering_moving_camera.yaml", "data_moving_camera.yaml"
    ),
}


def write_template_configs(
    context: ProjectContext, directory: Path, template: str | None = None
) -> None:
    """Write template trajectories, rendering and data configs into ``directory``.

    Without ``template`` the project's mode templates are used.
    """
    info = context.manifest.project
    if template is None:
        files = (f"trajectories_{info.mode}.yaml", f"rendering_{info.mode}.yaml", "data.yaml")
    elif template in DATASET_TEMPLATES:
        files = DATASET_TEMPLATES[template]
    else:
        available = sorted(DATASET_TEMPLATES)
        raise ValueError(f"Unknown dataset template {template!r}; available: {available}.")
    templates = dict(zip(CONFIG_FILES, files, strict=True))
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
    template: str | None = None,
    description: str = "",
) -> DatasetContext:
    """Create ``datasets/<name>/`` with configs from a reference dataset or the templates.

    A copied dataset keeps the reference's camera (``camera: fixed``) once that camera has been
    optimised, so models trained on the reference see the same view. With both ``template`` and
    ``from_dataset``, the template's configs are used with the reference's camera block.
    """
    dataset = context.dataset(name, must_exist=False)
    if dataset.root.exists():
        raise FileExistsError(f"Dataset already exists: {dataset.root}.")

    camera = "optimise"
    if from_dataset is None or template is not None:
        write_template_configs(context, dataset.configs, template)
    if from_dataset is not None:
        reference = context.dataset(from_dataset)
        reference_camera_final = (
            reference.camera_report.is_file() or reference.info().camera == "fixed"
        )
        if template is None:
            dataset.configs.mkdir(parents=True)
            for file_name in CONFIG_FILES:
                shutil.copy2(reference.configs / file_name, dataset.configs / file_name)
        else:
            _copy_camera(reference.rendering_config, dataset.rendering_config)
        camera = "fixed" if reference_camera_final else "optimise"

    DatasetInfo(
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(),
        description=description,
        reference=from_dataset,
        camera=camera,
    ).to_yaml(dataset.info_file)
    return dataset


def _copy_camera(source: Path, destination: Path) -> None:
    """Replace the camera block of rendering config ``destination`` by the one in ``source``."""
    camera = yaml.safe_load(source.read_text(encoding="utf-8"))["camera"]
    value = yaml.safe_load(destination.read_text(encoding="utf-8"))
    value["camera"] = camera
    destination.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def snapshot_dataset(dataset: DatasetContext, destination: Path) -> Path:
    """Copy dataset.yaml and the dataset's configs into ``destination`` (replacing it)."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(dataset.info_file, destination / "dataset.yaml")
    for file_name in CONFIG_FILES:
        shutil.copy2(dataset.configs / file_name, destination / file_name)
    return destination

from __future__ import annotations

from dataclasses import dataclass

import h5py
import yaml

from ball_simulator.rendering.camera_paths import camera_path_ids

from .discovery import ProjectContext, DatasetContext


@dataclass(frozen=True, slots=True)
class DatasetStatus:
    name: str
    description: str
    reference: str | None
    camera_mode: str
    trajectories: int | None
    camera_ready: bool
    camera_paths: int | None
    rendered_complete: int
    manifest_ready: bool
    camera_summary: str | None


@dataclass(frozen=True, slots=True)
class RunStatus:
    name: str
    training_dataset: str | None
    checkpoints: int
    evaluations: list[str]


@dataclass(frozen=True, slots=True)
class ProjectStatus:
    datasets: list[DatasetStatus]
    runs: list[RunStatus]


def _camera_summary(dataset: DatasetContext) -> str | None:
    path = dataset.rendering_config
    if not path.is_file():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    camera = value.get("camera")
    if not isinstance(camera, dict):
        return None
    return (
        f"location={camera.get('location')}, target={camera.get('target')}, "
        f"focal_length_mm={camera.get('focal_length_mm')}"
    )

def _trajectory_count(dataset: DatasetContext) -> int | None:
    if not dataset.trajectories.is_file():
        return None
    try:
        with h5py.File(dataset.trajectories, "r") as handle:
            return len(handle.get("trajectories", {}))
    except OSError:
        return None


def inspect_dataset(dataset: DatasetContext) -> DatasetStatus:
    info = dataset.info()
    return DatasetStatus(
        name=dataset.name,
        description=info.description,
        reference=info.reference,
        camera_mode=info.camera,
        trajectories=_trajectory_count(dataset),
        camera_ready=info.camera == "fixed" or dataset.camera_report.is_file(),
        camera_paths=(
            len(camera_path_ids(dataset.camera_paths)) if dataset.camera_paths.is_file() else None
        ),
        rendered_complete=sum(1 for _ in dataset.rendered.glob("trajectory_*/_SUCCESS")),
        manifest_ready=dataset.manifest.is_file(),
        camera_summary=_camera_summary(dataset),
    )


def inspect_status(context: ProjectContext) -> ProjectStatus:
    datasets = [inspect_dataset(context.dataset(name)) for name in context.dataset_names()]
    runs = []
    for name in context.run_names():
        run = context.run(name)
        try:
            training_dataset = run.training_dataset_name()
        except ValueError:
            training_dataset = None
        evaluations = (
            sorted(
                path.name for path in run.evaluations.iterdir()
                if (path / "metrics.json").is_file()
            )
            if run.evaluations.is_dir()
            else []
        )
        runs.append(RunStatus(
            name=name,
            training_dataset=training_dataset,
            checkpoints=len(list(run.checkpoints.glob("*.ckpt"))),
            evaluations=evaluations,
        ))
    return ProjectStatus(datasets=datasets, runs=runs)

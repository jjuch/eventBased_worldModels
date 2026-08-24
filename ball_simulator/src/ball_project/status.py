from __future__ import annotations

from dataclasses import dataclass

import h5py
import yaml

from .discovery import ProjectContext

@dataclass(frozen=True, slots=True)
class ProjectStatus:
    trajectories: int | None
    camera_ready: bool
    rendered_complete: int
    manifest_ready: bool
    training_runs: int
    active_rendering_config: str
    camera_summary: str | None


def _camera_summary(context: ProjectContext) -> str | None:
    path = context.resolve(context.manifest.configs.rendering)
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


def inspect_status(context: ProjectContext) -> ProjectStatus:
    dataset = context.resolve(context.manifest.paths.trajectory_dataset)
    trajectory_count = None
    if dataset.is_file():
        try:
            with h5py.File(dataset, "r") as handle:
                trajectory_count = len(handle.get("trajectories", {}))
        except OSError:
            trajectory_count = None
    rendered = context.resolve(context.manifest.paths.rendered)
    rendered_complete = sum(1 for path in rendered.glob("trajectory_*/_SUCCESS"))
    camera_report = context.resolve(context.manifest.paths.camera_report)
    manifest = context.resolve(context.manifest.paths.manifest)
    training = context.resolve(context.manifest.paths.training_outputs)
    runs = len(list(training.glob("**/checkpoints/*.ckpt"))) if training.exists() else 0
    active = context.resolve(context.manifest.configs.rendering)
    return ProjectStatus(
        trajectories=trajectory_count,
        camera_ready=camera_report.is_file(),
        rendered_complete=rendered_complete,
        manifest_ready=manifest.is_file(),
        training_runs=runs,
        active_rendering_config=str(active),
        camera_summary=_camera_summary(context),
    )

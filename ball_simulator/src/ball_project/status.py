from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py

from .discovery import ProjectContext

@dataclass(frozen=True, slots=True)
class ProjectStatus:
    trajectories: int | None
    camera_ready: bool
    rendered_complete: int
    manifest_ready: bool
    training_runs: int


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
    optimized = context.resolve(context.manifest.configs.optimized_rendering)
    manifest = context.resolve(context.manifest.paths.manifest)
    training = context.resolve(context.manifest.paths.training_outputs)
    runs = len(list(training.glob("**/checkpoints/*.ckpt"))) if training.exists() else 0
    return ProjectStatus(
        trajectories=trajectory_count,
        camera_ready=optimized.is_file(),
        rendered_complete=rendered_complete,
        manifest_ready=manifest.is_file(),
        training_runs=runs,
    )

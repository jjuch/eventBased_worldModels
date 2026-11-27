from __future__ import annotations

from .discovery import ProjectContext, DatasetContext
from .status import inspect_dataset
from .stages.ball import (
    build_manifest,
    generate_camera_paths,
    generate_trajectories,
    propose_camera,
    render, 
    train,
)


def prepare_dataset(
    context: ProjectContext, dataset: DatasetContext, *, dry_run: bool = False
) -> None:
    """Run the missing data stages: trajectories, camera (unless fixed), camera paths, renders,
    manifest.

    A trajectory file with no trajectories counts as missing. Trajectories without a camera path
    (no path kept the object in view) are never rendered, so rendering is complete when every
    camera path has a render.
    """
    status = inspect_dataset(dataset)
    if not status.trajectories:  # None (no file) or 0 (empty file)
        generate_trajectories(context, dataset, dry_run=dry_run)
    if not status.camera_ready:
        propose_camera(context, dataset, dry_run=dry_run)
    if status.camera_paths is None:
        generate_camera_paths(context, dataset, dry_run=dry_run)
        status = inspect_dataset(dataset)
    if status.camera_paths is None or status.rendered_complete < status.camera_paths:
        render(context, dataset, dry_run=dry_run)
    if not status.manifest_ready:
        build_manifest(context, dataset, dry_run=dry_run)

def run_pipeline(context: ProjectContext, *, run_name: str | None = None, dry_run: bool = False) -> None:
    """Prepare the run's training dataset, then train the run."""
    run = context.run(run_name)  # fail before any expensive stage if the run is ambiguous
    prepare_dataset(context, context.dataset(run.training_dataset_name()), dry_run=dry_run)
    train(context, run, dry_run=dry_run)

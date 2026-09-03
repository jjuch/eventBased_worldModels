from __future__ import annotations

from .discovery import ProjectContext
from .status import inspect_status
from .stages.ball import (
    build_manifest,
    generate_trajectories,
    propose_camera,
    render, 
    train,
)


def run_pipeline(context: ProjectContext, *, dry_run: bool = False) -> None:
    status = inspect_status(context)
    if status.trajectories is None:
        generate_trajectories(context, dry_run=dry_run)
    if not status.camera_ready:
        propose_camera(context, dry_run=dry_run)
    if status.trajectories is None or status.rendered_complete < status.trajectories:
        render(context, dry_run=dry_run)
    if not status.manifest_ready:
        build_manifest(context, dry_run=dry_run)
    train(context, dry_run=dry_run)

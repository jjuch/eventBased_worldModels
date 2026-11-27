from __future__ import annotations

import subprocess

from ball_project.commands import executable, run_command
from ball_project.discovery import ProjectContext, RunContext, DatasetContext
from ball_project.effective_configs import (
    effective_data_config,
    effective_render_config,
    effective_training_config,
)
from ball_project.render_config import (
    apply_proposed_camera,
    archive_rendering_config,
    camera_work_path,
    load_camera_omtimisation_settings,
)
from ball_project.runs import record_training_dataset

def generate_trajectories(context: ProjectContext, dataset: DatasetContext, *, dry_run: bool = False) -> None:
    """Simulate the dataset's trajectories; a failed run leaves no partial file behind."""
    command = [
        executable("ball_simulator"),
        "generate",
        str(dataset.trajectories_config),
        "--environment",
        "free-flight",
        "--output",
        str(dataset.trajectories),
        "--workers",
        str(context.manifest.runtime.simulation_workers),
    ]
    try:
        run_command(context, command, stage="trajectories", scope=dataset, dry_run=dry_run)
    except subprocess.CalledProcessError:
        dataset.trajaectories.unlink(missing_ok=True)
        raise


def propose_camera(context: ProjectContext, dataset: DatasetContext, *, dry_run: bool = False) -> None:
    if dataset.info().camera == "fixed":
        raise ValueError(
            f"Dataset {dataset.name!r} keeps the camera of its reference dataset "
            f"(camera: fixed in {dataset.info_file}). Set camera: optimise to search a new one; "
            "models trained on another camera then no longer apply."
        )

    base_config = effective_render_config(dataset)
    settings = load_camera_omtimisation_settings(dataset)
    proposal = camera_work_path(dataset)
    command = [
        executable("ball_renderer"),
        "propose-camera",
        str(dataset.trajectories),
        "--config",
        str(base_config),
        "--output",
        str(proposal),
        "--report",
        str(dataset.camera_report),
        "--minimum-diameter-px",
        str(settings.minimum_diameter_px),
        "--target-diameter-px",
        str(settings.target_diameter_px),
    ]

    if dry_run:
        run_command(context, command, stage="camera", scope=dataset, dry_run=True)
        print(
            f"Camera optimisation settings: minimum_diameter_px="
            f"{settings.minimum_diameter_px}, target_diameter_px="
            f"{settings.target_diameter_px}"
        )
        print(
            f"Would archive {base_config} and copy the proposed camera fields back "
            "into that same authoritative file."
        )
        return

    # The proposal is generated separately first. The authoritative config is changed
    # only after the external command has succeeded.
    run_command(context, command, stage="camera", scope=dataset, dry_run=dry_run)
    backup = archive_rendering_config(dataset)
    updated = apply_proposed_camera(dataset, proposal)
    proposal.unlink(missing_ok=True)
    print(f"Updated active rendering config: {updated}")
    print(f"Previous rendering config archived at: {backup}")


def generate_camera_paths(
    context: ProjectContext, dataset: DatasetContext, *, dry_run: bool = False
) -> None:
    """One camera path per trajectory around the (optimised) camera of rendering.yaml."""
    command = [
        executable("ball_renderer"),
        "camera-paths",
        str(dataset.trajectories),
        "--config",
        str(effective_render_config(dataset)),
        "--output",
        str(dataset.camera_paths),
    ]
    run_command(context, command, stage="camera_paths", scope=dataset, dry_run=dry_run)


def render(
    context: ProjectContext,
    dataset: DatasetContext,
    *, 
    one: str | None = None, 
    dry_run: bool = False
) -> None:
    runtime = context.manifest.runtime
    command = [
        executable("ball_renderer"),
        "render-trajectory",
        str(dataset.trajectories),
        "--config",
        str(effective_render_config(dataset)),
        "--output-directory",
        str(dataset.rendered),
        "--camera-paths",
        str(dataset.camera_paths),
        "--workers",
        str(runtime.render_workers),
        "--threads-per-blender",
        str(runtime.threads_per_blender),
    ]
    if one is None:
        command.append("--all")
    else:
        command.append(one)
    if runtime.resume_rendering:
        command.append("--resume")
    if runtime.blender_executable is not None:
        command.extend(["--blender-executable", str(runtime.blender_executable)])
    run_command(context, command, stage="rendering", scope=dataset, dry_run=dry_run)


def build_manifest(
    context: ProjectContext, dataset: DatasetContext, *, dry_run: bool = False) -> None:
    command = [
        executable("ball_world_model"),
        "build-manifest",
        "--config",
        str(effective_data_config(dataset)),
        "--output",
        str(dataset.manifest),
    ]
    run_command(context, command, stage="manifest", scope=dataset, dry_run=dry_run)


def inspect_data(
    context: ProjectContext, dataset: DatasetContext, *, dry_run: bool = False) -> None:
    command = [
        executable("ball_world_model"),
        "inspect-data",
        "--config",
        str(effective_data_config(dataset)),
        "--manifest",
        str(dataset.manifest),
        "--output",
        str(dataset.inspection / "data_batch.png"),
    ]
    run_command(context, command, stage="inspection", scope=dataset, dry_run=dry_run)


def train(context: ProjectContext, run: RunContext, *, dry_run: bool = False) -> None:
    """Train on the dataset named in the run's training.yaml, snapshotting its configs first."""
    dataset = context.dataset(run.training_dataset_name())

    command = [
        executable("ball_world_model"),
        "train-kinematic",
        "--config",
        str(effective_training_config(context, run)),
    ]
    if not dry_run:
        record_training_dataset(run, dataset)
    run_command(context, command, stage="training", scope=run, dry_run=dry_run)

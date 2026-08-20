from __future__ import annotations

from pathlib import Path

from ball_project.commands import executable, run_command
from ball_project.discovery import ProjectContext
from ball_project.effective_configs import (
    effective_data_config,
    effective_render_config,
    effective_training_config,
)

def generate_trajectories(context: ProjectContext, *, dry_run: bool = False) -> None:
    command = [
        executable("ball_simulator"),
        "generate",
        str(context.resolve(context.manifest.configs.trajectories)),
        "--environment",
        "free-flight",
        "--output",
        str(context.resolve(context.manifest.paths.trajectory_dataset)),
        "--workers",
        str(context.manifest.runtime.simulation_workers),
    ]
    run_command(context, command, stage="trajectories", dry_run=dry_run)


def propose_camera(context: ProjectContext, *, dry_run: bool = False) -> None:
    command = [
        executable("ball_renderer"),
        "propose-camera",
        str(context.resolve(context.manifest.paths.trajectory_dataset)),
        "--config",
        str(context.resolve(context.manifest.configs.rendering)),
        "--output",
        str(context.resolve(context.manifest.configs.optimized_rendering)),
        "--report",
        str(context.resolve(context.manifest.paths.camera_report)),
    ]
    if context.manifest.project.mode == "rotation":
        command.extend(["--minimum-diameter-px", "90", "--target-diameter-px", "130"])
    else:
        command.extend(["--minimum-diameter-px", "35", "--target-diameter-px", "70"])
    run_command(context, command, stage="camera", dry_run=dry_run)


def render(context: ProjectContext, *, one: str | None = None, dry_run: bool = False) -> None:
    command = [
        executable("ball_renderer"),
        "render-trajectory",
        str(context.resolve(context.manifest.paths.trajectory_dataset)),
        "--config",
        str(effective_render_config(context)),
        "--output-directory",
        str(context.resolve(context.manifest.paths.rendered)),
        "--workers",
        str(context.manifest.runtime.render_workers),
        "--threads-per-blender",
        str(context.manifest.runtime.threads_per_blender),
    ]
    if one is None:
        command.append("--all")
    else:
        command.append(one)
    if context.manifest.runtime.resume_rendering:
        command.append("--resume")
    if context.manifest.runtime.blender_executable is not None:
        command.extend(["--blender-executable", str(context.manifest.runtime.blender_executable)])
    run_command(context, command, stage="rendering", dry_run=dry_run)


def build_manifest(context: ProjectContext, *, dry_run: bool = False) -> None:
    command = [
        executable("ball_world_model"),
        "build-manifest",
        "--config",
        str(effective_data_config(context)),
        "--output",
        str(context.resolve(context.manifest.paths.manifest)),
    ]
    run_command(context, command, stage="manifest", dry_run=dry_run)


def inspect_data(context: ProjectContext, *, dry_run: bool = False) -> None:
    command = [
        executable("ball_world_model"),
        "inspect-data",
        "--config",
        str(effective_data_config(context)),
        "--manifest",
        str(context.resolve(context.manifest.paths.manifest)),
        "--output",
        str(context.resolve(context.manifest.paths.inspection) / "data_batch.png"),
    ]
    run_command(context, command, stage="inspection", dry_run=dry_run)


def train(context: ProjectContext, *, dry_run: bool = False) -> None:
    command = [
        executable("ball_world_model"),
        "train-kinematic",
        "--config",
        str(effective_training_config(context)),
    ]
    run_command(context, command, stage="training", dry_run=dry_run)

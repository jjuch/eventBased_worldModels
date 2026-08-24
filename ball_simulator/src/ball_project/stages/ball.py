from __future__ import annotations

from ball_project.commands import executable, run_command
from ball_project.discovery import ProjectContext
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
    base_config = effective_render_config(context)
    settings = load_camera_omtimisation_settings(context)
    proposal = camera_work_path(context)
    command = [
        executable("ball_renderer"),
        "propose-camera",
        str(context.resolve(context.manifest.paths.trajectory_dataset)),
        "--config",
        str(base_config),
        "--output",
        str(proposal),
        "--report",
        str(context.resolve(context.manifest.paths.camera_report)),
        "--minimum-diameter-px",
        str(settings.minimum_diameter_px),
        "--target-diameter-px",
        str(settings.target_diameter_px),
    ]

    if dry_run:
        run_command(context, command, stage="camera", dry_run=True)
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
    run_command(context, command, stage="camera", dry_run=dry_run)
    backup = archive_rendering_config(context)
    updated = apply_proposed_camera(context, proposal)
    proposal.unlink(missing_ok=True)
    print(f"Updated active rendering config: {updated}")
    print(f"Previous rendering config archived at: {backup}")


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

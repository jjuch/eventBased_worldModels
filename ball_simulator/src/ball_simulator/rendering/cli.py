from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
import numpy as np

from .config import RenderConfig
from .jobs import prepare_render_job, prepare_all_render_jobs
from .runner import BlenderRunner
from .camera_optimiser import optimise_camera, write_optimised_reder_config

app = typer.Typer(help="Render physically rich sphere-wall trajectories.")

@app.command("blender-info")
def blender_info(
    blender_executable: Annotated[Path | None, typer.Option("--blender-executable")] = None,
) -> None:
    runner = BlenderRunner.discover(blender_executable)
    print(f"[green]Blender:[/green] {runner.executable}")
    print(f"[green]Version:[/green] {runner.version()}")


@app.command("propose-camera")
def propose_camera_command(
    dataset: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, readable=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o"),
    ],
    report: Annotated[
        Path | None,
        typer.Option("--report"),
    ] = None,
    minimum_diameter_px: Annotated[
        float,
        typer.Option("--minimum-diameter-px", min=1.0),
    ] = 35.0,
    target_diameter_px: Annotated[
        float,
        typer.Option("--target-diameter-px", min=1.0),
    ] = 70.0,
) -> None:
    """Propose one fixed camera for the complete dataset and write a render YAML."""
    render_config = RenderConfig.from_yaml(config)
    candidate = optimise_camera(
        dataset,
        render_config,
        minimum_diameter_px=minimum_diameter_px,
        target_diameter_px=target_diameter_px,
    )
    saved = write_optimised_reder_config(config, output, candidate, report)
    print(f"[green]Saved optimized render config:[/green] {saved.resolve()}")
    print(f"  camera location: {candidate.location.tolist()}")
    print(f"  camera target: {candidate.target.tolist()}")
    print(f"  focal length: {candidate.focal_length_mm:.1f} mm")
    print(f"  minimum ball diameter: {candidate.minimum_diameter_px:.1f} px")
    print(f"  median ball diameter: {candidate.median_diameter_px:.1f} px")
    print(f"  minimum image margin: {candidate.minimum_margin_px:.1f} px")
    print(f"  projection conditioning: {candidate.conditioning:.5f}")
    

@app.command("render-trajectory")
def render_trajectory_command(
    dataset: Annotated[
        Path, 
        typer.Argument(
            exists=True, 
            file_okay=True, 
            dir_okay=True, 
            readable=True,
        ),
    ],
    config: Annotated[
        Path, 
        typer.Option(
            "--config", 
            "-c", 
            exists=True, 
            readable=True,
        ),
    ],
    output_directory: Annotated[
        Path, 
        typer.Option(
            "--output-directory", 
            "-o",
        ),
    ] = Path("rendered"),
    trajectory_id: Annotated[
        str | None, 
        typer.Argument(
            help="Trajectory ID. Omit when using --all.",
        ),
    ] = None,
    render_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Render all trajectories in the dataset.",
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help=(
                "Concurrent Blender processes. "
                "Start with 1-3 for one GPU."
            ),
        ),
    ] = 1,
    threads_per_blender: Annotated[
        int,
        typer.Option(
            "--threads-per-blender",
            min=0,
            help=(
                "CPU threads per Blender process. "
                "0 lets Blender decide."
            ),
        ),
    ] = 0,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help=(
                "Skip trajectories with a valid "
                "_SUCCESS marker."
            ),
        ),
    ] = False,
    blender_executable: Annotated[
        Path | None, 
        typer.Option("--blender-executable"),
    ] = None,
    prepare_only: Annotated[
        bool, 
        typer.Option("--prepare-only"),
    ] = False,
) -> None:
    if render_all and trajectory_id is not None:
        raise typer.BadParameter(
            "Provide either a trajectory ID or "
            "--all, not both."
        )

    if not render_all and trajectory_id is None:
        raise typer.BadParameter(
            "Provide a trajectory ID or use --all."
        )

    render_config = RenderConfig.from_yaml(config)

    if render_all:
        jobs = prepare_all_render_jobs(
            dataset=dataset,
            config=render_config,
            output_directory=output_directory,
            resume=resume,
        )
    else:
        job = prepare_render_job(
            dataset=dataset, 
            trajectory_id=trajectory_id, 
            config=render_config, 
            output_directory=output_directory,
        )
        jobs = [job]

    print(f"[green]Prepared {len(jobs):,} render jobs.[/green]")

    if not jobs:
        print(
            "[yellow]Nothing to render. "
            "All requested trajectories are complete.[/yellow]"
        )
        return
    
    if prepare_only:
        return
    
    runner = BlenderRunner.discover(blender_executable)

    print(f"[green]Using:[/green] {runner.version()}")

    progress = Progress(
        TextColumn(
            "[progress.description]{task.description}"
        ),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task(
            "Rendering trajectories",
            total=len(jobs),
        )

        runner.render_many(
            jobs=jobs,
            workers=workers,
            threads_per_blender=threads_per_blender,
            log_directory=output_directory / "_logs",
            on_complete=lambda _: progress.advance(task),
        )

    print(f"[green]Rendered {len(jobs):,} trajectories.[/green]")

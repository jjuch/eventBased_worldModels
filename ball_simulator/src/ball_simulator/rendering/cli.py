from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print
import numpy as np

from .config import RenderConfig
from .jobs import prepare_render_job
from .runner import BlenderRunner

app = typer.Typer(help="Render physically rich sphere-wall trajectories.")

@app.command("blender-info")
def blender_info(
    blender_executable: Annotated[Path | None, typer.Option("--blender-executable")] = None,
) -> None:
    runner = BlenderRunner.discover(blender_executable)
    print(f"[green]Blender:[/green] {runner.executable}")
    print(f"[green]Version:[/green] {runner.version()}")


@app.command("render-trajectory")
def render_trajectory_command(
    dataset: Annotated[Path, typer.Argument(exists=True, readable=True)],
    trajectory_id: Annotated[str, typer.Argument()],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, readable=True)],
    output_directory: Annotated[Path, typer.Option("--output-directory", "-o")] = Path("rendered"),
    blender_executable: Annotated[Path | None, typer.Option("--blender-executable")] = None,
    prepare_only: Annotated[bool, typer.Option("--prepare-only")] = False,
) -> None:
    render_config = RenderConfig.from_yaml(config)
    job = prepare_render_job(dataset, trajectory_id, render_config, output_directory)
    print(f"[green]Prepared render job:[/green] {job}")
    if prepare_only:
        return
    runner = BlenderRunner.discover(blender_executable)
    print(f"[green]Using:[/green] {runner.version()}")
    runner.render(job)
    print(f"[green]Rendered trajectory {trajectory_id}.[/green]")

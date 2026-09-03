from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .creator import create_project
from .discovery import discover_project
from .editor import edit_file
from .menu import run_menu, show_status
from .pipeline import run_pipeline
from .stages.ball import (
    build_manifest,
    generate_trajectories,
    inspect_data, propose_camera, 
    render,
    train,
)
from .stages.evaluation import evaluate_observer
from .derive import derive_project_hard_copy

app = typer.Typer(no_args_is_help=True, help="Create and run reproducible world-model experiment workspaces.")
config_app = typer.Typer(help="Edit project-local scientific configurations.")
app.add_typer(config_app, name="config")


@app.command("create")
def create_command(
    name: Annotated[
        str, 
        typer.Argument(help="Project folder name.")
    ],
    experiment_type: Annotated[
        str, 
        typer.Option("--type")
    ] = "ball",
    subtype: Annotated[
        str, 
        typer.Option("--subtype")
    ] = "free-flight",
    mode: Annotated[
        str, 
        typer.Option("--mode")
    ] = "translation",
    parent: Annotated[
        Path | None, 
        typer.Option("--parent")
    ] = None,
) -> None:
    root = create_project(name, experiment_type=experiment_type, subtype=subtype, mode=mode, parent=parent)
    print(f"[green]Created project:[/green] {root}")
    print(f"Next: cd {root.name} && ball_project run")


@app.command("status")
def status_command() -> None:
    show_status(discover_project())


@app.command("run")
def run_command() -> None:
    run_menu(discover_project())


@app.command("pipeline")
def pipeline_command(
    dry_run: Annotated[
        bool, 
        typer.Option("--dry-run")
    ] = False
) -> None:
    run_pipeline(discover_project(), dry_run=dry_run)


@app.command("trajectories")
def trajectories_command(
    dry_run: Annotated[
        bool, 
        typer.Option("--dry-run")
    ] = False
) -> None:
    generate_trajectories(discover_project(), dry_run=dry_run)


@app.command("camera")
def camera_command(
    dry_run: Annotated[
        bool, 
        typer.Option("--dry-run")
    ] = False
) -> None:
    propose_camera(discover_project(), dry_run=dry_run)


@app.command("render")
def render_command(
    trajectory_id: Annotated[
        str | None, 
        typer.Option("--trajectory-id")
    ] = None,
    dry_run: Annotated[
        bool, 
        typer.Option("--dry-run")
    ] = False,
) -> None:
    render(discover_project(), one=trajectory_id, dry_run=dry_run)


@app.command("build-manifest")
def manifest_command(
    dry_run: Annotated[
        bool, 
        typer.Option("--dry-run")
    ] = False
) -> None:
    build_manifest(discover_project(), dry_run=dry_run)


@app.command("inspect-data")
def inspect_command(
    dry_run: Annotated[
        bool, 
        typer.Option("--dry-run")
    ] = False
) -> None:
    inspect_data(discover_project(), dry_run=dry_run)


@app.command("train")
def train_command(
    dry_run: Annotated[
        bool, 
        typer.Option("--dry-run")
    ] = False
) -> None:
    train(discover_project(), dry_run=dry_run)


@config_app.command("edit")
def edit_command(
    config_name: Annotated[
        str, 
        typer.Argument(help="trajectories, rendering, data, training, or project")
    ],
    editor: Annotated[
        str | None, 
        typer.Option("--editor")
    ] = None,
) -> None:
    context = discover_project()
    mapping = {
        "trajectories": context.manifest.configs.trajectories,
        "rendering": context.manifest.configs.rendering,
        "data": context.manifest.configs.data,
        "training": context.manifest.configs.training,
    }
    path = context.manifest_path if config_name == "project" else context.resolve(mapping[config_name])
    edit_file(path, editor)


def create_entrypoint() -> None:
    app(args=["create", *sys.argv[1:]], prog_name="create_world_model")


def run_entrypoint() -> None:
    app(args=["run", *sys.argv[1:]], prog_name="run_experiment")


@app.command("evaluate")
def evaluate_command(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run"),
    ] = False,
) -> None:
    evaluate_observer(
        discover_project(),
        dry_run=dry_run,
    )

@app.command("derive")
def derive_command(
    name: Annotated[
        str,
        typer.Argument(help="Folder name for the independent derived experiment."),
    ],
    source: Annotated[
        Path,
        typer.Option(
            "--from",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Prepared source project to hard-copy.",
        ),
    ],
    parent: Annotated[
        Path | None,
        typer.Option(
            "--parent",
            file_okay=False,
            dir_okay=True,
            help="Destination parent directory. Defaults to the current directory.",
        ),
    ] = None,
) -> None:
    """Hard-copy prepared trajectories, renders, manifest, configs, and reports."""
    result = derive_project_hard_copy(
        name,
        source=source,
        parent=parent,
    )
    print(f"[green]Created independent derived project:[/green] {result.root}")
    print(f"[green]Source project:[/green] {result.source_root}")
    print(
        f"[green]Copied prepared renders:[/green] "
        f"{result.rendered_trajectories:,} trajectories"
    )
    print(f"[green]Approximate copied size:[/green] {result.copied_gibibytes:.2f} GiB")
    print("Training, evaluation, logs, and runtime records start empty.")
    print(f"Next: cd {result.root.name} && ball_project config edit training")

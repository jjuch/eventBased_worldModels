from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .creator import create_project
from .datasets import create_dataset
from .derive import derive_project_hard_copy
from .discovery import discover_project
from .editor import edit_file
from .menu import run_menu, show_status
from .pipeline import run_pipeline, prepare_dataset
from .report import write_comparison
from .runs import create_run
from .stages.ball import (
    build_manifest,
    generate_trajectories,
    inspect_data, 
    propose_camera, 
    render,
    train,
)
from .stages.evaluation import evaluate_observer
from .status import inspect_status


app = typer.Typer(
    no_args_is_help=True, help="Create and run reproducible world-model experiment workspaces.")
config_app = typer.Typer(help="Edit project-local scientific configurations.")
app.add_typer(config_app, name="config")

runs_app = typer.Typer(help="Create and list training runs.")
app.add_typer(runs_app, name="runs")
datasets_app = typer.Typer(help="Create and list datasets.")
app.add_typer(datasets_app, name="datasets")

RunOption = Annotated[
    str | None,
    typer.Option(
        "--run",
        help="Run name. Optional when the project has exactly one run."
    ),
]
DatasetOption = Annotated[
    str | None,
    typer.Option("--dataset", help="Dataset name. Optional when the project has exactly one."),
]
DryRunOption = Annotated[bool, typer.Option("--dry-run")]


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
    root = create_project(
        name, experiment_type=experiment_type, subtype=subtype, mode=mode, parent=parent
    )
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
    run: RunOption = None,
    dry_run: DryRunOption = False
) -> None:
    """Prepare the run's training dataset, then train the run."""
    run_pipeline(discover_project(), run_name=run, dry_run=dry_run)

@app.command("prepare")
def prepare_command(dataset: DatasetOption = None, dry_run: DryRunOption = False) -> None:
    """Run the missing data stages of a dataset (no training)."""
    context = discover_project()
    prepare_dataset(context, context.dataset(dataset), dry_run=dry_run)


@app.command("trajectories")
def trajectories_command(
    dataset: DatasetOption = None,
    dry_run: DryRunOption = False
) -> None:
    context = discover_project()
    generate_trajectories(context, context.dataset(dataset), dry_run=dry_run)


@app.command("camera")
def camera_command(
    dataset: DatasetOption = None,
    dry_run: DryRunOption = False
) -> None:
    context = discover_project()
    propose_camera(context, context.dataset(dataset), dry_run=dry_run)


@app.command("render")
def render_command(
    dataset: DatasetOption = None,
    trajectory_id: Annotated[
        str | None, 
        typer.Option("--trajectory-id")
    ] = None,
    dry_run: DryRunOption = False,
) -> None:
    context = discover_project()
    render(context, context.dataset(dataset), one=trajectory_id, dry_run=dry_run)


@app.command("build-manifest")
def manifest_command(
    dataset: DatasetOption = None,
    dry_run: DryRunOption = False
) -> None:
    context = discover_project()
    build_manifest(context, context.dataset(dataset), dry_run=dry_run)


@app.command("inspect-data")
def inspect_command(
    dataset: DatasetOption = None,
    dry_run: DryRunOption = False
) -> None:
    context = discover_project()
    inspect_data(context, context.dataset(dataset), dry_run=dry_run)


@app.command("train")
def train_command(
    run: RunOption = None,
    dry_run: DryRunOption = False
) -> None:
    context = discover_project()
    train(context, context.run(run), dry_run=dry_run)


@app.command("evaluate")
def evaluate_command(
    run: RunOption = None,
    datasets: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            help="Dataset to evaluate on; repeat for several. Default: the training dataset.",
        ),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help="train, validation, test or all. Default: test on the training dataset, "
            "all on other datasets.",
        ),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint", exists=True, dir_okay=False,
            help="Checkpoint to evaluate instead of the run's best.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Evaluate despite a camera, resolution or frame-rate mismatch."
        ),
    ] = False,
    dry_run: DryRunOption = False,
) -> None:
    """Evaluate a run; results go to runs/<run>/evaluations/<dataset>/."""
    context = discover_project()
    selected_run = context.run(run)
    targets = [context.dataset(name) for name in datasets] if datasets else [None]
    for dataset in targets:
        output = evaluate_observer(
            context, selected_run, dataset,
            split=split, checkpoint=checkpoint, force=force, dry_run=dry_run,
        )
        print(f"[green]Evaluation:[/green] {output}")

@app.command("report")
def report_command() -> None:
    """Collect all evaluations into reports/comparison.csv (one row per run and dataset)."""
    path, rows = write_comparison(discover_project())
    print(f"[green]Wrote {rows} evaluations to[/green] {path}")


@config_app.command("edit")
def edit_command(
    config_name: Annotated[
        str, 
        typer.Argument(help="trajectories, rendering, data, training, or project")
    ],
    dataset: DatasetOption = None,
    run: RunOption = None,
    editor: Annotated[
        str | None, 
        typer.Option("--editor")
    ] = None,
) -> None:
    """Dataset configs need --dataset (if several); training edits the base or --run's config."""
    context = discover_project()
    if config_name == "project":
        path = context.manifest_path
    elif config_name == "training":
        path = (
            context.run(run).training_config
            if run is not None
            else context.resolve(context.manifest.configs.training)
        )
    elif config_name in ("trajectories", "rendering", "data"):
        path = context.dataset(dataset).configs / f"{config_name}.yaml"
    else:
        raise typer.BadParameter(f"Unknown config: {config_name!r}.")
    edit_file(path, editor)


def create_entrypoint() -> None:
    app(args=["create", *sys.argv[1:]], prog_name="create_world_model")


def run_entrypoint() -> None:
    app(args=["run", *sys.argv[1:]], prog_name="run_experiment")


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


@datasets_app.command("new")
def datasets_new_command(
    name: Annotated[str, typer.Argument(help="Name of the new dataset.")],
    from_dataset: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Copy this dataset's configs (and its final camera) instead of the templates.",
        ),
    ] = None,
    description: Annotated[str, typer.Option("--description")] = "",
) -> None:
    dataset = create_dataset(
        discover_project(), name, from_dataset=from_dataset, description=description
    )
    print(f"[green]Created dataset:[/green] {dataset.root} (camera: {dataset.info().camera})")
    print(f"Next: edit {dataset.configs}, then ball_project prepare --dataset {name}")


@datasets_app.command("list")
def datasets_list_command() -> None:
    for dataset in inspect_status(discover_project()).datasets:
        trajectories = dataset.trajectories if dataset.trajectories is not None else 0
        print(
            f"{dataset.name:30s} trajectories={trajectories:<6d} "
            f"rendered={dataset.rendered_complete:<6d} camera={dataset.camera_mode:<8s} "
            f"reference={dataset.reference or '-'}"
        )


@runs_app.command("new")
def runs_new_command(
    name: Annotated[
        str,
        typer.Argument(help="Name of the new run."),
    ],
    from_run: Annotated[
        str | None,
        typer.Option(
            "--from", 
            help="Copy this run's config instead of configs/training.yaml."
        )
    ] = None,
) -> None:
    run = create_run(discover_project(), name, from_run=from_run)
    print(f"[green]Created run:[/green] {run.root}")
    print(f"Next: edit {run.training_config}, then ball_project train --run {name}")


@runs_app.command("list")
def runs_list_command() -> None:
    for run in inspect_status(discover_project()).runs:
        evaluations = ",".join(run.evaluations) or "-"
        print(
            f"{run.name:30s} dataset={run.training_dataset or '-':<20s} "
            f"checkpoints={run.checkpoints:<3d} evaluations={evaluations}"
        )

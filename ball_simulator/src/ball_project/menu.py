from __future__ import annotations

from rich.console import Console
from rich.prompt import IntPrompt, Prompt

from .datasets import create_dataset
from .discovery import ProjectContext, RunContext, DatasetContext
from .editor import edit_file
from .pipeline import prepare_dataset
from .report import write_comparison
from .runs import create_run
from .stages.ball import (
    build_manifest,
    generate_camera_paths,
    generate_trajectories,
    inspect_data,
    propose_camera,
    render,
    train,
)
from .stages.evaluation import evaluate_observer
from .status import inspect_status

console = Console()


def show_status(
    context: ProjectContext,
    selected_dataset: str | None = None,
    selected_run: str | None = None,
) -> None:
    status = inspect_status(context)
    console.print(f"[bold]Project:[/bold] {context.manifest.project.name}")
    console.print(
        f"[bold]Type:[/bold] {context.manifest.project.type} / "
        f"{context.manifest.project.subtype} / {context.manifest.project.mode}"
    )
    console.print("[bold]Datasets[/bold]" if status.datasets else "Datasets: none yet.")
    for dataset in status.datasets:
        marker = "*" if dataset.name == selected_dataset else " "
        trajectories = dataset.trajectories if dataset.trajectories is not None else "-"
        camera = "fixed" if dataset.camera_mode == "fixed" else (
            "optimised" if dataset.camera_ready else "not optimised"
        )
        reference = f", from {dataset.reference}" if dataset.reference else ""
        console.print(
            f" {marker} {dataset.name}: {trajectories} trajectories, "
            f"{dataset.rendered_complete} rendered, camera {camera}, "
            f"manifest {'ready' if dataset.manifest_ready else 'not built'}{reference}."
        )
    console.print("[bold]Runs[/bold]" if status.runs else "Runs: none yet.")
    for run in status.runs:
        marker = "*" if run.name == selected_run else " "
        evaluations = ", ".join(run.evaluations) if run.evaluations else "none"
        console.print(
            f" {marker} {run.name}: trains on {run.training_dataset}, "
            f"{run.checkpoints} checkpoints, evaluated on: {evaluations}."
        )


def _ask(kind: str, names: list[str]) -> str:
    if not names:
        raise FileNotFoundError(f"No {kind}s yet. Choose 'Create {kind}' first.")

    if len(names) == 1:
        return names[0]
    return Prompt.ask(kind.capitalize(), choices=names, default=names[-1])


def run_menu(context: ProjectContext) -> None:
    datasets, runs = context.dataset_names(), context.run_names()
    selected = {
        "dataset": datasets[0] if len(datasets) == 1 else None,
        "run": runs[0] if len(runs) == 1 else None,
    }

    def current_dataset() -> DatasetContext:
        """The selected dataset. Ask once if none is selected or it no longer exists."""
        if selected["dataset"] not in context.dataset_names():
            selected["dataset"] = _ask("dataset", context.dataset_names())
        return context.dataset(selected["dataset"])

    def current_run() -> RunContext:
        """The selected run. Ask once if none is selected or it no longer exists."""
        if selected["run"] not in context.run_names():
            selected["run"] = _ask("run", context.run_names())
        return context.run(selected["run"])

    def switch(kind: str) -> None:
        names = context.dataset_names() if kind == "dataset" else context.run_names()
        selected[kind] = _ask(kind, names)

    def new_dataset() -> None:
        """The selected dataset. Ask once if none is selected or it no longer exists."""
        if selected["dataset"] not in context.dataset_names():
            selected["dataset"] = _ask("dataset", context.dataset_names())
        return context.dataset(selected["dataset"])


    def new_run() -> None:
        name = Prompt.ask("New run name")
        source = Prompt.ask(
            "Copy config from run (empty = base configs/training.yaml)", default=""
        )
        run = create_run(context, name, from_run=source or None)
        selected["run"] = run.name
        console.print(f"Created {run.root}. Edit its training config before training.")

    def evaluate() -> None:
        output = evaluate_observer(context, current_run(), current_dataset())
        console.print(f"Evaluation written to {output}.")

    def report() -> None:
        path, rows = write_comparison(context)
        console.print(f"Wrote {rows} evaluations to {path}.")

    # Numbered in this order in the menu.
    actions = dict(enumerate([
        ("Generate trajectories (selected dataset)",
            lambda: generate_trajectories(context, current_dataset())),
        ("Propose camera (selected dataset)", lambda: propose_camera(context, current_dataset())),
        ("Generate camera paths (selected dataset)",
         lambda: generate_camera_paths(context, current_dataset())),
        ("Render all trajectories (selected dataset)", lambda: render(context, current_dataset())),
        ("Build manifest (selected dataset)", lambda: build_manifest(context, current_dataset())),
        ("Inspect data (selected dataset)", lambda: inspect_data(context, current_dataset())),
        ("Prepare selected dataset (all missing data stages)",
            lambda: prepare_dataset(context, current_dataset())),
        ("Train selected run", lambda: train(context, current_run())),
        ("Evaluate selected run on selected dataset", evaluate),
        ("Write comparison report", report),
        ("Edit trajectory config (selected dataset)",
             lambda: edit_file(current_dataset().trajectories_config)),
        ("Edit rendering config (selected dataset)",
             lambda: edit_file(current_dataset().rendering_config)),
        ("Edit data config (selected dataset)", lambda: edit_file(current_dataset().data_config)),
        ("Edit selected run's training config", lambda: edit_file(current_run().training_config)),
        ("Edit base training config (template for new runs)",
             lambda: edit_file(context.resolve(context.manifest.configs.training))),
        ("Create dataset", new_dataset),
        ("Create run", new_run),
        ("Switch dataset", lambda: switch("dataset")),
        ("Switch run", lambda: switch("run")),
    ], start=1))

    while True:
        console.rule("Ball World Model Project")
        show_status(context, selected["dataset"], selected["run"])
        console.print()
        for number, (label, _) in actions.items():
            console.print(f"  {number:2d}. {label}")
        console.print("   0. Exit")
        choice = IntPrompt.ask("Choose an action", default=0)
        if choice == 0:
            return
        action = actions.get(choice)
        if action is None:
            console.print("[red]Unknown choice.[/red]")
            continue
        try:
            action[1]()
        except Exception as error:
            console.print(f"[red]{type(error).__name__}: {error}[/red]")

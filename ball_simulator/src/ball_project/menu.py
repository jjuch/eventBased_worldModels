from __future__ import annotations

from rich.console import Console
from rich.prompt import IntPrompt, Prompt

from .discovery import ProjectContext, RunContext
from .runs import create_run
from .editor import edit_file
from .status import inspect_status
from .stages.ball import (
    build_manifest,
    generate_trajectories,
    inspect_data,
    propose_camera,
    render,
    train,
)
from .stages.evaluation import evaluate_observer

console = Console()


def show_status(context: ProjectContext, selected_run: str | None = None) -> None:
    status = inspect_status(context)
    console.print(f"[bold]Project:[/bold] {context.manifest.project.name}")
    console.print(
        f"[bold]Type:[/bold] {context.manifest.project.type} / "
        f"{context.manifest.project.subtype} / {context.manifest.project.mode}"
    )
    console.print(
        f"Trajectories: "
        f"{status.trajectories if status.trajectories is not None else 'not generated'}")
    console.print(f"Camera proposal: {'applied' if status.camera_ready else 'not applied'}")
    console.print(f"Active rendering config: {status.active_rendering_config}")
    if status.camera_summary is not None:
        console.print(f"Active camera: {status.camera_summary}")
    console.print(f"Rendered trajectories: {status.rendered_complete}")
    console.print(f"Manifest: {'ready' if status.manifest_ready else 'not built'}")
    if not status.runs:
        console.print("Runs: none yet.")
    for run in status.runs:
        marker = "*" if run.name == selected_run else " "
        console.print(
            f" {marker} Run {run.name}: {run.checkpoints} checkpoints, "
            f"{'evaluated' if run.evaluated else 'not evaluated'}."
        )


def _ask_run(context: ProjectContext) -> str:
    names = context.run_names()
    if not names:
        raise FileNotFoundError("No runs yet. Choose 'Create run' first.")

    if len(names) == 1:
        return names[0]
    return Prompt.ask("Run", choices=names, default=names[-1])


def run_menu(context: ProjectContext) -> None:
    names = context.run_names()
    selected = {"run": names[0] if len(names) == 1 else None}

    def current_run() -> RunContext:
        """The selected run. Ask once if none is selected or it no longer exists."""
        if selected["run"] not in context.run_names():
            selected["run"] = _ask_run(context)
        return context.run(selected["run"])

    def switch_run() -> None:
        selected["run"] = _ask_run(context)

    def new_run() -> None:
        name = Prompt.ask("New run name")
        source = Prompt.ask(
            "Copy config from run (empty = base configs/training.yaml)", default=""
        )
        run = create_run(context, name, from_run=source or None)
        selected["run"] = run.name
        console.print(f"Created {run.root}. Edit its training config before training.")


    actions = {
        1: ("Generate trajectories", lambda: generate_trajectories(context)),
        2: ("Propose camera", lambda: propose_camera(context)),
        3: ("Render all trajectories", lambda: render(context)),
        4: ("Build manifest", lambda: build_manifest(context)),
        5: ("Inspect data", lambda: inspect_data(context)),
        6: ("Train selected run", lambda: train(context, current_run())),
        7: ("Evaluate selected run", lambda: evaluate_observer(context, current_run())),
        8: ("Edit trajectory config", lambda: edit_file(context.resolve(context.manifest.configs.trajectories))),
        9: ("Edit rendering config", lambda: edit_file(context.resolve(context.manifest.configs.rendering))),
        10: ("Edit data config", lambda: edit_file(context.resolve(context.manifest.configs.data))),
        11: ("Edit base training config (template for new runs)", lambda: edit_file(context.resolve(context.manifest.configs.training))),
        12: ("Create run", new_run),
        13: ("Switch run", switch_run),
        14: ("Edit selected run's training config", lambda: edit_file(current_run().training_config)),
    }

    while True:
        console.rule("Ball World Model Project")
        show_status(context, selected["run"])
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

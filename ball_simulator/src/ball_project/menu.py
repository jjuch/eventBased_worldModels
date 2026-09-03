from __future__ import annotations

from rich.console import Console
from rich.prompt import IntPrompt

from .discovery import ProjectContext
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


def show_status(context: ProjectContext) -> None:
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
    if hasattr(status, "active_rendering_config"):
        console.print(f"Active rendering config: {status.active_rendering_config}")
    if status.camera_summary is not None:
        console.print(f"Active camera: {status.camera_summary}")
    console.print(f"Rendered trajectories: {status.rendered_complete}")
    console.print(f"Manifest: {'ready' if status.manifest_ready else 'not built'}")
    console.print(f"Checkpoints: {status.training_runs}")


def run_menu(context: ProjectContext) -> None:
    actions = {
        1: ("Generate trajectories", lambda: generate_trajectories(context)),
        2: ("Propose camera", lambda: propose_camera(context)),
        3: ("Render all trajectories", lambda: render(context)),
        4: ("Build manifest", lambda: build_manifest(context)),
        5: ("Inspect data", lambda: inspect_data(context)),
        6: ("Train kinematic observer", lambda: train(context)),
        7: ("Evaluate and interprete trained observer", lambda: evaluate_observer(context)),
        8: ("Edit trajectory config", lambda: edit_file(context.resolve(context.manifest.configs.trajectories))),
        9: ("Edit rendering config", lambda: edit_file(context.resolve(context.manifest.configs.rendering))),
        10: ("Edit data config", lambda: edit_file(context.resolve(context.manifest.configs.data))),
        11: ("Edit training config", lambda: edit_file(context.resolve(context.manifest.configs.training))),
    }
    while True:
        console.rule("Ball World Model Project")
        show_status(context)
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

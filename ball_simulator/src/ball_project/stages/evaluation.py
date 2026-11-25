from __future__ import annotations

from pathlib import Path

from ball_project.commands import executable, run_command
from ball_project.discovery import ProjectContext, RunContext
from ball_project.effective_configs import effective_training_config, effective_render_config


def find_best_checkpoint(run: RunContext) -> Path:
    patterns = ("**/best*.ckpt", "**/last.ckpt", "**/*.ckpt")
    for pattern in patterns:
        candidates = sorted(run.root.glob(pattern), key=lambda path: path.stat().st_mtime)
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found under {run.root}. Train run {run.name!r} first."
    )


def evaluate_observer(
    context: ProjectContext, 
    run: RunContext, 
    *, 
    checkpoint: Path | None = None,
    dry_run: bool = False
) -> None:
    """Evaluate on this project's test split; ``checkpoint`` defaults to the run's best."""
    checkpoint = Path(checkpoint) if checkpoint is not None else find_best_checkpoint(run)
    
    command = [
        executable("ball_world_model"),
        "evaluate-kinematic",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(effective_training_config(context, run)),
        "--output",
        str(run.evaluation),
        "--rendering-config",
        str(effective_render_config(context)),
    ]
    run_command(context, command, stage="evaluation", run=run, dry_run=dry_run)
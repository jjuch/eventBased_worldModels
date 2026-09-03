from __future__ import annotations

from pathlib import Path

from ball_project.commands import executable, run_command
from ball_project.discovery import ProjectContext
from ball_project.effective_configs import effective_training_config


def find_best_checkpoint(context: ProjectContext) -> Path:
    root = context.resolve(context.manifest.paths.training_outputs)
    patterns = ("**/best*.ckpt", "**/last.ckpt", "**/*.ckpt")
    for pattern in patterns:
        candidates = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found under {root}. Train the kinematic observer first."
    )


def evaluate_observer(context: ProjectContext, *, dry_run: bool = False) -> None:
    checkpoint = find_best_checkpoint(context)
    command = [
        executable("ball_world_model"),
        "evaluate-kinematic",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(effective_training_config(context)),
        "--output",
        str(context.resolve(context.manifest.paths.evaluation)),
    ]
    run_command(context, command, stage="evaluation", dry_run=dry_run)
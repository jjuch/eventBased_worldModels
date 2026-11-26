from __future__ import annotations

from pathlib import Path

from ball_project.commands import executable, run_command
from ball_project.discovery import ProjectContext, RunContext, DatasetContext
from ball_project.effective_configs import (
    effective_data_config,
    effective_training_config, effective_render_config
)
from ball_project.evaluations import describe_mismatches, plan_evaluation, write_evaluation_record


def find_best_checkpoint(run: RunContext) -> Path:
    patterns = ("**/best*.ckpt", "**/last.ckpt", "**/*.ckpt")
    for pattern in patterns:
        candidates = sorted(run.checkpoints.glob(pattern), key=lambda path: path.stat().st_mtime)
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found under {run.root}. Train run {run.name!r} first."
    )


def evaluate_observer(
    context: ProjectContext, 
    run: RunContext,
    dataset: DatasetContext | None = None,
    *,
    split: str | None = None,
    checkpoint: Path | None = None,
    force: bool = False,
    dry_run: bool = False
) -> Path:
    """Evaluate a run on a dataset (default: its training dataset) into runs/<run>/evaluations/.

    The data is loaded with the run's training data settings. Camera, resolution or frame-rate differences from training stop the evaluation unless ``force`` is set; they are then recorded in evaluation.json together with all other config differences.
    """
    dataset = dataset or context.dataset(run.training_dataset_name())
    plan = plan_evaluation(run, dataset, split)
    if plan.mismatches and not force:
        raise ValueError(
            f"Dataset {dataset.name!r} differs from run {run.name!r}'s training data in settings "
            f"the observer depends on:\n{describe_mismatches(plan)}\n"
            "Pass --force to evaluate anyway; the mismatch is then recorded in evaluation.json."
        )
    checkpoint = Path(checkpoint) if checkpoint is not None else find_best_checkpoint(run)

    effective = plan.output / "effective"
    data_config = effective_data_config(
        dataset,
        settings_from=run.dataset_snapshot / "data.yaml",
        destination=effective / "data.yaml"
    )

    training_config = effective_training_config(
        context, run, data_config=data_config, destination=effective / "training.yaml"
    )
    
    command = [
        executable("ball_world_model"),
        "evaluate-kinematic",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(training_config),
        "--output",
        str(plan.output),
        "--rendering-config",
        str(effective_render_config(dataset)),
        "--split",
        plan.split,
    ]
    if not dry_run:
        write_evaluation_record(plan, checkpoint, forced=bool(plan.mismatches))
    run_command(
        context, command, stage=f"evaluation_{plan.output.name}", scope=run, dry_run=dry_run)
    return plan.output
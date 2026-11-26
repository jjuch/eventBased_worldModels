"""Training runs inside a project."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from .datasets import snapshot_dataset
from .discovery import ProjectContext, RunContext, DatasetContext


def create_run(
    context: ProjectContext,
    name: str,
    *, 
    from_run: str | None = None,
) -> RunContext:
    """Create ``runs/<name>/`` with a full copy of a training config.

    The copy comes from an existing run (``from_run``) or from the project's base ``configs/training.yaml``. The run's own copy is the record of what it used.
    """
    run = context.run(name, must_exist=False)
    if run.root.exists():
        raise FileExistsError(f"Run already exists: {run.root}.")
    source = (
        context.run(from_run).training_config
        if from_run
        else context.resolve(context.manifest.configs.training)
    )

    run.root.mkdir(parents=True)
    shutil.copy2(source, run.training_config)
    record = {
        "name": run.name,
        "kind": "observer",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_run": from_run,
        "config_source": str(source.relative_to(context.root)),
        "training_dataset": None,
    }
    
    run.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return run


def record_training_dataset(run: RunContext, dataset: DatasetContext) -> None:
    """Snapshot the training dataset's configs into the run and note it in run.json."""
    snapshot_dataset(dataset, run.dataset_snapshot)
    record = json.loads(run.record.read_text(encoding="utf-8")) if run.record.is_file() else {}
    record["training_dataset"] = dataset.name
    record["trained_at"] = datetime.now(timezone.utc).isoformat()
    run.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
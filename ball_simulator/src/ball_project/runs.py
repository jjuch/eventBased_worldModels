"""Training runs inside a project."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from .discovery import ProjectContext, RunContext


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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_run": from_run,
        "config_source": str(source.relative_to(context.root)),
    }
    
    (run.root / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return run
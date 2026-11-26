from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .datasets import create_dataset
from .discovery import DEFAULT_DATASET, DEFAULT_RUN, load_project
from .models import ExperimentInfo, ProjectManifest
from .runs import create_run
from .template_registry import definition, template_text

INITIAL_DIRECTORIES = ("configs", ".ball_project")

GITIGNORE = (
    # Heavy or regenerated data; configs, dataset.yaml, training.yaml and run.json stay tracked.
    "datasets/*/trajectories.h5\ndatasets/*/rendered/\ndatasets/*/manifest.parquet\n"
    "datasets/*/inspection/\ndatasets/*/effective/\ndatasets/*/logs/\ndatasets/*/records/\n"
    "datasets/*/reports/*.json\n"
    "runs/*/checkpoints/\nruns/*/tensorboard/\nruns/*/evaluations/\n"
    "runs/*/logs/\nruns/*/records/\nruns/*/effective/\n"
    "logs/\n.ball_project/records/\n"
)


def create_project(
    name: str,
    *,
    experiment_type: str,
    subtype: str,
    mode: str,
    parent: str | Path | None = None,
) -> Path:
    """New project with one dataset and one run, both named ``default``."""
    template = definition(experiment_type, subtype)
    if mode not in template.modes:
        raise ValueError(
            f"Mode must be one of {template.modes}, received {mode!r}."
        )
    
    root = (Path(parent) if parent else Path.cwd()) / name
    if root.exists():
        raise FileExistsError(f"Project path already exists: {root}.")
    root.mkdir(parents=True)

    for directory in INITIAL_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)

    manifest = ProjectManifest(
        project=ExperimentInfo(
            name=name,
            type=experiment_type,
            subtype=subtype,
            mode=mode,
            created_at=datetime.now(timezone.utc).isoformat(),
            description=f"{mode.title()} {subtype} {experiment_type} experiment."
        )
    )
    manifest.to_yaml(root / "project.yaml")
    context = load_project(root)

    create_dataset(context, DEFAULT_DATASET, description="Training data.")
    training = template_text(experiment_type, subtype, f"training_{mode}.yaml")
    context.resolve(manifest.configs.training).write_text(
        f"dataset: {DEFAULT_DATASET} # training dataset a folder under datasets/\n{training}",
        encoding="utf-8",
    )

    create_run(context, DEFAULT_RUN)

    (root / "README.md").write_text(
        template_text(experiment_type, subtype, "PROJECT_README.md").replace(
            "{{PROJECT_NAME}}", name
        ).replace("{{MODE}}", mode),
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(
        GITIGNORE,
        encoding="utf-8",
    )
    (root / ".ball_project" / "state.json").write_text("{}\n", encoding="utf-8")
    return root.resolve()
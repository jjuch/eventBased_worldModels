from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .models import ExperimentInfo, ProjectManifest
from .template_registry import definition, template_text


DIRECTORIES = (
    "configs/generated",
    "data/trajectories",
    "data/rendered",
    "data/manifests",
    "outputs/training",
    "outputs/checkpoints",
    "outputs/tensorboard",
    "outputs/evaluation",
    "outputs/inspection",
    "reports",
    "logs/trajectories",
    "logs/rendering",
    "logs/training",
    ".ball_project/effective",
    ".ball_project/records",
)

def create_project(
    name: str,
    *,
    experiment_type: str,
    subtype: str,
    mode: str,
    parent: str | Path | None = None,
) -> Path:
    template = definition(experiment_type, subtype)
    if mode not in template.modes:
        raise ValueError(
            f"Mode must be one of {template.modes}, received {mode!r}."
        )
    
    root = (Path(parent) if parent else Path.cwd()) / name
    if root.exists():
        raise FileExistsError(f"Project path already exists: {root}.")
    root.mkdir(parents=True)

    for directory in DIRECTORIES:
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

    mapping = {
        "trajectories.yaml": f"trajectories_{mode}.yaml",
        "rendering.yaml": f"rendering_{mode}.yaml",
        "data.yaml": "data.yaml",
        "training.yaml": f"training_{mode}.yaml",
    }

    for destination, source in mapping.items():
        (root / "configs" / destination).write_text(
            template_text(experiment_type, subtype, source), encoding="utf-8"
        )

    (root / "README.md").write_text(
        template_text(experiment_type, subtype, "PROJECT_README.md").replace(
            "{{PROJECT_NAME}}", name
        ).replace("{{MODE}}", mode),
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(
        "data/\noutputs/\nlogs/\nreports/*.json\n.ball_project/effective/\n"
        ".ball_project/records/\nconfigs/generated/\n",
        encoding="utf-8",
    )
    (root / ".ball_project" / "state.json").write_text("{}\n", encoding="utf-8")
    return root.resolve()
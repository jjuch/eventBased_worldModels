from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .discovery import ProjectContext, discover_project, load_project, DEFAULT_RUN
from .runs import create_run


@dataclass(frozen=True, slots=True)
class DeriveResult:
    root: Path
    source_root: Path
    copied_bytes: int
    rendered_trajectories: int
    datasets: int

    @property
    def copied_gibibytes(self) -> float:
        return self.copied_bytes / 1024**3


# Not copied: run-time state, and the manifest, which stores absolute paths into the source
# project and is rebuilt (cheaply) in the derived one.
_SKIPPED_IN_DATASETS = ("logs", "records", "effective", "inspection", "manifest.parquet")

def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

def _rendered_success_count(rendered: Path) -> int:
    return sum(1 for _ in rendered.glob("trajectory_*/_SUCCESS"))


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, copy_function=shutil.copy2, ignore=shutil.ignore_patterns(*_SKIPPED_IN_DATASETS))
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

def _write_manifest_and_lineage(source: ProjectContext, destination: Path, name: str) -> None:
    manifest = source.manifest.model_copy(deep=True)
    manifest.project.name = name
    manifest.project.created_at = datetime.now(timezone.utc).isoformat()
    manifest.to_yaml(destination / "project.yaml")

    state = destination / ".ball_project"
    state.mkdir(parents=True, exist_ok=True)
    (state / "state.json").write_text("{}\n", encoding="utf-8")
    lineage = {
        "kind": "hard-copy",
        "source_project": str(source.root.resolve()),
        "source_manifest": str(source.manifest_path.resolve()),
        "derived_at": datetime.now(timezone.utc).isoformat(),
        "independent_after_creation": True,
    }
    (state / "lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")


def derive_project_hard_copy(
    name: str,
    *,
    source: str | Path,
    parent: str | Path | None = None,
) -> DeriveResult:
    """Create an independent project by hard-copying another project's datasets.

    The destination gets the base configs and every dataset (configs, trajectories, renders,
    camera reports). Manifests are rebuilt on first use. Runs start with a fresh ``default``.
    """
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("name must be one project-folder name, without path separators")

    source_context = discover_project(source)
    destination_parent = Path(parent).expanduser().resolve() if parent else Path.cwd().resolve()
    destination = destination_parent / name
    if destination.exists():
        raise FileExistsError(f"Destination project already exists: {destination}")

    datasets = [source_context.dataset(dataset) for dataset in source_context.dataset_names()]
    rendered_trajectories = sum(_rendered_success_count(dataset.rendered) for dataset in datasets)
    if rendered_trajectories == 0:
        raise FileNotFoundError(
            f"No rendered trajectories found in the datasets of {source_context.root}. "
            "Derive projects after rendering."
        )
    copied_bytes = sum(_directory_size(dataset.root) for dataset in datasets)

    try:
        destination.mkdir(parents=True, exist_ok=False)
        _copy_if_present(source_context.root / "configs", destination / "configs")
        for dataset in datasets:
            _copy_if_present(
                dataset.root,
                destination / source_context.manifest.paths.datasets / dataset.name,
            )
        _write_manifest_and_lineage(source_context, destination, name)
        create_run(load_project(destination), DEFAULT_RUN)

        for file_name in ("README.md", ".gitignore"):
            if (source_context.root / file_name).is_file():
                shutil.copy2(source_context.root / file_name, destination / file_name)
    except BaseException:
        # The destination may be incomplete. Remove it so it is never mistaken for a valid project.
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return DeriveResult(
        root=destination,
        source_root=source_context.root,
        copied_bytes=copied_bytes,
        rendered_trajectories=rendered_trajectories,
        datasets=len(datasets),
    )

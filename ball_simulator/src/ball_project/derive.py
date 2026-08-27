from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .discovery import ProjectContext, discover_project


@dataclass(frozen=True, slots=True)
class DeriveResult:
    root: Path
    source_root: Path
    copied_bytes: int
    rendered_trajectories: int

    @property
    def copied_gibibytes(self) -> float:
        return self.copied_bytes / 1024**3


_EMPTY_DIRECTORIES = (
    "outputs/training",
    "outputs/checkpoints",
    "outputs/tensorboard",
    "outputs/evaluation",
    "outputs/inspection",
    "logs/trajectories",
    "logs/rendering",
    "logs/training",
    ".ball_project/effective",
    ".ball_project/records",
)

def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

def _rendered_success_count(rendered: Path) -> int:
    return sum(1 for _ in rendered.glob("trajectory_*/_SUCCESS"))


def _validate_prepared_source(context: ProjectContext) -> tuple[int, int]:
    root = context.root
    if not (root / "configs").is_dir():
        raise FileNotFoundError(f"Source project has no configs directory: {root / 'configs'}")

    project_asset_paths = (
        context.manifest.paths.trajectory_dataset,
        context.manifest.paths.rendered,
        context.manifest.paths.manifest,
        context.manifest.paths.camera_report,
    )
    absolute_paths = [str(path) for path in project_asset_paths if path.is_absolute()]
    if absolute_paths:
        raise ValueError(
            "Hard-copy derivation requires project-local relative asset paths. "
            f"Absolute paths found: {absolute_paths}"
        )

    trajectory_dataset = context.resolve(context.manifest.paths.trajectory_dataset)
    rendered = context.resolve(context.manifest.paths.rendered)
    manifest = context.resolve(context.manifest.paths.manifest)
    if not trajectory_dataset.is_file():
        raise FileNotFoundError(f"Trajectory dataset does not exist: {trajectory_dataset}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Rendered-data manifest does not exist: {manifest}")

    successes = _rendered_success_count(rendered)
    if successes == 0:
        raise FileNotFoundError(
            f"No completed rendered trajectories were found under {rendered}. "
            "Derive projects after rendering and manifest creation."
        )

    copied_paths = (
        root / "configs",
        trajectory_dataset,
        rendered,
        manifest,
        context.resolve(context.manifest.paths.camera_report),
    )
    return sum(_directory_size(path) for path in copied_paths), successes


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, copy_function=shutil.copy2)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_fresh_runtime_state(root: Path) -> None:
    state_directory = root / ".ball_project"
    state_directory.mkdir(parents=True, exist_ok=True)
    (state_directory / "state.json").write_text("{}\n", encoding="utf-8")


def _rewrite_project_manifest(
    source_context: ProjectContext,
    temporary_root: Path,
    *,
    new_name: str,
) -> None:
    manifest = source_context.manifest.model_copy(deep=True)
    manifest.project.name = new_name
    manifest.project.created_at = datetime.now(timezone.utc).isoformat()

    # `derived_from` is optional metadata. Older ProjectManifest models may not yet define it, so preserve the information in a separate lineage file as well.
    source_manifest_relative = str(source_context.manifest_path.resolve())
    if hasattr(manifest.project, "derived_from"):
        manifest.project.derived_from = source_manifest_relative

    manifest.to_yaml(temporary_root / "project.yaml")
    lineage = {
        "kind": "hard-copy",
        "source_project": str(source_context.root.resolve()),
        "source_manifest": source_manifest_relative,
        "derived_at": datetime.now(timezone.utc).isoformat(),
        "independent_after_creation": True,
    }
    (temporary_root / ".ball_project" / "lineage.json").write_text(
        json.dumps(lineage, indent=2) + "\n",
        encoding="utf-8",
    )

def derive_project_hard_copy(
    name: str,
    *,
    source: str | Path,
    parent: str | Path | None = None,
) -> DeriveResult:
    """
    Create a self-contained experiment by hard-copying prepared simulation/render data.

    The destination receives independent copies of configs, trajectories, renders,manifests, and reports. Training outputs, evaluations, logs, effective configs, and execution records start empty.
    """
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("name must be one project-folder name, without path separators")

    source_context = discover_project(source)
    destination_parent = Path(parent).expanduser().resolve() if parent else Path.cwd().resolve()
    destination = destination_parent / name
    if destination.exists():
        raise FileExistsError(f"Destination project already exists: {destination}")

    copied_bytes, rendered_trajectories = _validate_prepared_source(source_context)

    try:
        destination.mkdir(parents=True, exist_ok=False)

        # Copy only prepared scientific inputs and data. This deliberately avoids
        # training artifacts and stale generated runtime configuration.
        _copy_if_present(source_context.root / "configs", destination / "configs")

        # Preserve project-local path semantics even when a project uses customized
        # relative paths rather than the default data/* layout.
        assets = (
            (
                source_context.resolve(source_context.manifest.paths.trajectory_dataset),
                destination / source_context.manifest.paths.trajectory_dataset,
            ),
            (
                source_context.resolve(source_context.manifest.paths.rendered),
                destination / source_context.manifest.paths.rendered,
            ),
            (
                source_context.resolve(source_context.manifest.paths.manifest),
                destination / source_context.manifest.paths.manifest,
            ),
            (
                source_context.resolve(source_context.manifest.paths.camera_report),
                destination / source_context.manifest.paths.camera_report,
            ),
        )
        for source_asset, destination_asset in assets:
            _copy_if_present(source_asset, destination_asset)

        for directory in _EMPTY_DIRECTORIES:
            (destination / directory).mkdir(parents=True, exist_ok=True)
        _write_fresh_runtime_state(destination)
        _rewrite_project_manifest(source_context, destination, new_name=name)

        source_readme = source_context.root / "README.md"
        if source_readme.is_file():
            shutil.copy2(source_readme, destination / "README.md")
        source_gitignore = source_context.root / ".gitignore"
        if source_gitignore.is_file():
            shutil.copy2(source_gitignore, destination / ".gitignore")

    except BaseException:
        # The destination may be incomplete. Remove it so it is never mistaken for a valid derived project.
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return DeriveResult(
        root=destination,
        source_root=source_context.root,
        copied_bytes=copied_bytes,
        rendered_trajectories=rendered_trajectories,
    )

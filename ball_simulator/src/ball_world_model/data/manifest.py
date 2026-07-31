from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from rich.progress import track

from ball_world_model.config import DataConfig
from .discovery import discover_trajectory_directories, validate_trajectory
from .splits import assign_split

MANIFEST_COLUMNS = (
    "trajectory_id",
    "trajectory_directory",
    "rgb_directory",
    "states_path",
    "metadata_path",
    "frame_count",
    "width",
    "height",
    "fps",
    "duration",
    "environment_kind",
    "split",
)


def build_manifest(config: DataConfig, *, skip_invalid: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    directories = discover_trajectory_directories(config.root)

    if not directories:
        raise ValueError(f"No trajectory_XXXXXXX directories found under {config.root}.")

    for directory in track(directories, description="Validating rendered trajectories"):
        try:
            validation = validate_trajectory(
                directory,
                require_success_marker=config.require_success_marker,
            )
        except Exception as error:
            message = f"{directory}: {error}"
            if not skip_invalid:
                raise RuntimeError(message) from error
            errors.append(message)
            continue

        row = asdict(validation)
        row = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in row.items()
        }
        row["split"] = assign_split(validation.trajectory_id, config.splits)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if frame.empty:
        raise ValueError("No valid rendered trajectories were found.")
    frame = frame.sort_values("trajectory_id").reset_index(drop=True)
    frame.attrs["validatation_errors"] = errors
    return frame


def save_manifest(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)

    return path


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"trajectory_id": str})
    else:
        frame = pd.read_parquet(path)
        frame["trajectory_id"] = frame["trajectory_id"].astype(str).str.zfill(8)

    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}.")

    return frame
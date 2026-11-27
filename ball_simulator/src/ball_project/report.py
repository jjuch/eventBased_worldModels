"""One table of all evaluations in a project: reports/comparison.csv (run x dataset)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .discovery import ProjectContext

COLUMNS = (
    "run",
    "dataset",
    "split",
    "training_dataset",
    "windows",
    "position_norm_rmse_m",
    "linear_velocity_norm_rmse_m_s",
    "angular_velocity_norm_rmse_rad_s",
    "linear_acceleration_norm_rmse_m_s2",
    "angular_acceleration_norm_rmse_rad_s2",
    "position_r2_mean",
    "linear_velocity_r2_mean",
    "angular_velocity_r2_mean",
    "orientation_median_deg",
    "orientation_symmetric_median_deg",
    "rollout_final_position_median_m",
    "rollout_final_orientation_median_deg",
    "gravity_direction_median_deg",
    "camera_linear_velocity_norm_rmse_m_s",
    "camera_angular_velocity_norm_rmse_rad_s",
    "depth_scale",
    "along_ray_bias_m",
    "forced_mismatches",
    "evaluation_directory",
)


def _get(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _mean_defined(values: list[float | None] | None) -> float | None:
    """Mean over components with a defined R² (undefined when the target does not vary)."""
    defined = [value for value in values or [] if value is not None]
    return sum(defined) / len(defined) if defined else None


def _last(values: list[float] | None) -> float | None:
    return values[-1] if values else None


def evaluation_row(run_name: str, directory: Path) -> dict[str, Any]:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    record_path = directory / "evaluation.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    horizon = _get(metrics, "rollout_vs_horizon", "rollout_from_current_state") or {}
    return {
        "run": run_name,
        "dataset": record.get("dataset", directory.name),
        "split": record.get("split", metrics.get("split")),
        "training_dataset": record.get("training_dataset"),
        "windows": metrics.get("windows"),
        "position_norm_rmse_m": _get(metrics, "position", "norm_rmse"),
        "linear_velocity_norm_rmse_m_s": _get(metrics, "linear_velocity", "norm_rmse"),
        "angular_velocity_norm_rmse_rad_s": _get(metrics, "angular_velocity", "norm_rmse"),
        "linear_acceleration_norm_rmse_m_s2": _get(metrics, "linear_acceleration", "norm_rmse"),
        "angular_acceleration_norm_rmse_rad_s2": _get(metrics, "angular_acceleration", "norm_rmse"),
        "position_r2_mean": _mean_defined(_get(metrics, "position", "r2")),
        "linear_velocity_r2_mean": _mean_defined(_get(metrics, "linear_velocity", "r2")),
        "angular_velocity_r2_mean": _mean_defined(_get(metrics, "angular_velocity", "r2")),
        "orientation_median_deg": _get(metrics, "orientation_error_deg", "median"),
        "orientation_symmetric_median_deg": _get(
            metrics, "symmetric_orientation_error_deg", "current", "median"
        ),
        "rollout_final_position_median_m": _last(_get(horizon, "position_error_m", "median")),
        "rollout_final_orientation_median_deg": _last(
            _get(horizon, "orientation_error_deg", "median")
        ),
        "gravity_direction_median_deg": _get(metrics, "gravity_direction_error_deg", "median"),
        "camera_linear_velocity_norm_rmse_m_s": _get(
            metrics, "camera", "linear_velocity", "norm_rmse"
        ),
        "camera_angular_velocity_norm_rmse_rad_s": _get(
            metrics, "camera", "angular_velocity", "norm_rmse"
        ),
        "depth_scale": _get(metrics, "viewing_ray", "depth_scale"),
        "along_ray_bias_m": _get(metrics, "viewing_ray", "along_ray_bias_m"),
        "forced_mismatches": ";".join(sorted(record.get("mismatches", {}))),
        "evaluation_directory": str(directory),
    }


def comparison_rows(context: ProjectContext) -> list[dict[str, Any]]:
    rows = []
    for run_name in context.run_names():
        evaluations = context.run(run_name).evaluations
        if not evaluations.is_dir():
            continue
        for directory in sorted(evaluations.iterdir()):
            if (directory / "metrics.json").is_file():
                rows.append(evaluation_row(run_name, directory))
    return rows


def write_comparison(context: ProjectContext) -> tuple[Path, int]:
    """Write reports/comparison.csv; returns its path and the number of rows."""
    rows = comparison_rows(context)
    path = context.reports_directory / "comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path, len(rows)

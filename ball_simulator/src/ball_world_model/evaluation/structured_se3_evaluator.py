"""Intrinsic SE(3) evaluation shared by all task masks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ball_world_model.models.rotation import (
    matrix_to_rotation_6d,
    quaternion_xyzw_to_matrix,
    rotation_6d_to_matrix,
    rotation_geodesic_error,
)
from .metrics import (
    apply_linear_probe,
    effective_rank,
    fit_linear_probe,
    regression_metrics,
)
from ball_world_model.training.kinematic_module import KinematicObservabilityModule


def _np(value):
    return value.detach().cpu().numpy()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parents.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def _statistics(name: str, unit: str, items: dict) -> dict[str, object]:
    x = np.concatenate(items).reshape(-1)
    return {
        "metric": name,
        "unit": unit,
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "maximum": float(x.max()),
        "count": len(x),
    }


def _limited_batches(loader, maximum_windows= int):
    used = 0
    for batch in loader:
        if used >= maximum_windows:
            break
        batch_size = int(batch["context_rgb"].shape[0])
        keep = min(batch_size, maximum_windows - used)
        if keep < batch_size:
            batch = {
                key: value[:keep] if isinstance(value, torch.Tensor) else value[:keep]
                for key, value in batch.items()
            }
        used += keep
        yield batch


@torch.inference_mode()
def _collect(module, loader, device, maximum_windows: int) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {}

    def add(name: str, value: torch.Tensor) -> None:
        buckets.setdefault(name, []).append(_np(value).reshape(-1, value.shape[-1]))

    for batch in _limited_batches(loader, maximum_windows):
        rgb = batch["context_rgb"].to(device)
        time = batch["context_time"].to(device)
        prediction = module.model(rgb, time)
        context = prediction.context_sectors
        motion = prediction.motion_sectors
        forward = motion.forward_sectors

        target_position = batch["context_position"].to(device)
        target_velocity = batch["context_linear_velocity"].to(device)
        target_rotation = quaternion_xyzw_to_matrix(
            batch["context_quaternion_xyzw"].to(device)
        )
        target_omega = batch["context_angular_velocity"].to(device)

        add("target_position", target_position)
        add("target_velocity", target_velocity[:, :-1])
        add("target_rotation_6d", matrix_to_rotation_6d(target_rotation))
        add("target_omega", target_omega[:, :-1])

        add("canonical_position", context.position)
        add("canonical_rotation_6d", context.rotation_6d)
        add("canonical_velocity", forward.linear_velocity)
        add("canonical_omega", forward.angular_velocity)

        add("translation_carrier", context.translation_carrier.flatten(-2))
        add("rotation_carrier", context.rotation_carrier.flatten(-2))
        add("context_scalars", context.physical_scalars)
        add("context_artifacts", context.artifacts)

        add(
            "translation_tangents",
            forward.translation_carrier_tangents.flatten(-2),
        )
        add(
            "rotation_tangents",
            forward.rotation_carrier_tangents.flatten(-2),
        )
        add("motion_scalars", forward.physical_scalars)
        add("motion_artifacts", forward.artifacts)

    return {name: np.concatenate(parts, axis=0) for name, parts in buckets.items()}


def _regression_rows(
    sector: str,
    quantity: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> list[dict[str, object]]:
    weights = fit_linear_probe(train_x, train_y)
    estimate = apply_linear_probe(test_x, weights)

    rows = []
    for component, axis in enumerate(("x", "y", "z")):
        rows.append(
            {
                "sector": sector,
                "quantity": quantity,
                "component": axis,
                **regression_metrics(test_y[:, component], estimate[:, component]),
            }
        )
    return rows


def _orientation_probe_row(
    sector: str,
    train_x: np.ndarray,
    train_rotation_6d: np.ndarray,
    test_x: np.ndarray,
    test_rotation_6d: np.ndarray,
) -> dict[str, object]:
    weights = fit_linear_probe(train_x, train_rotation_6d)
    estimate_6d = apply_linear_probe(test_x, weights)
    estimate = rotation_6d_to_matrix(torch.from_numpy(estimate_6d).float())
    target = rotation_6d_to_matrix(torch.from_numpy(test_rotation_6d).float())
    error = np.rad2deg(rotation_geodesic_error(estimate, target).numpy())
    return {
        "sector": sector,
        "quantity": "orientation",
        "component": "all",
        "mean_deg": float(error.mean()),
        "median_deg": float(np.median(error)),
        "p95_deg": float(np.quantile(error, 0.95)),
        "maximum_deg": float(error.max()),
        "count": int(error.size),
    }

def evaluate_structured_se3_sectors(
    module,
    train_loader,
    test_loader,
    device,
    output_directory: Path | str,
    *,
    train_maximum: int = 5_000,
    test_maximum: int = 2_000,
) -> dict[str, object]:
    """Evaluate canonical state sectors, carriers, tangents, scalars, and artifacts."""
    output = Path(output_directory) / "sector_analysis"
    output.mkdir(parents=True, exist_ok=True)

    train = _collect(module, train_loader, device, train_maximum)
    test = _collect(module, test_loader, device, test_maximum)
    mask = module.model.context_head.mask

    direct_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []

    if mask.translation:
        for name, target_name, quantity in (
            ("canonical_position", "target_position", "position"),
            ("canonical_velocity", "target_velocity", "linear_velocity"),
        ):
            for component, axis in enumerate(("x", "y", "z")):
                direct_rows.append(
                    {
                        "sector": name,
                        "quantity": quantity,
                        "component": axis,
                        **regression_metrics(
                            test[target_name][:, component],
                            test[name][:, component],
                        ),
                    }
                )

        for sector in ("translation_carrier", "context_scalars", "context_artifacts"):
            probe_rows += _regression_rows(
                sector,
                "position",
                train[sector],
                train["target_position"],
                test[sector],
                test["target_position"],
            )

        for sector in (
            "translation_tangents",
            "motion_scalars",
            "motion_artifacts",
        ):
            probe_rows += _regression_rows(
                sector,
                "linear_velocity",
                train[sector],
                train["target_velocity"],
                test[sector],
                test["target_velocity"],
            )

    if mask.rotation:
        predicted_rotation = rotation_6d_to_matrix(
            torch.from_numpy(test["canonical_rotation_6d"]).float()
        )
        target_rotation = rotation_6d_to_matrix(
            torch.from_numpy(test["target_rotation_6d"]).float()
        )
        orientation_error = np.rad2deg(
            rotation_geodesic_error(predicted_rotation, target_rotation).numpy()
        )
        direct_rows.append(
            {
                "sector": "canonical_rotation_6d",
                "quantity": "orientation",
                "component": "all",
                "mean_deg": float(orientation_error.mean()),
                "median_deg": float(np.median(orientation_error)),
                "p95_deg": float(np.quantile(orientation_error, 0.95)),
                "maximum_deg": float(orientation_error.max()),
                "count": int(orientation_error.size),
            }
        )

        for component, axis in enumerate(("x", "y", "z")):
            direct_rows.append(
                {
                    "sector": "canonical_omega",
                    "quantity": "angular_velocity",
                    "component": axis,
                    **regression_metrics(
                        test["target_omega"][:, component],
                        test["canonical_omega"][:, component],
                    ),
                }
            )

        for sector in ("rotation_carrier", "context_scalars", "context_artifacts"):
            probe_rows.append(
                _orientation_probe_row(
                    sector,
                    train[sector],
                    train["target_rotation_6d"],
                    test[sector],
                    test["target_rotation_6d"],
                )
            )

        for sector in ("rotation_tangents", "motion_scalars", "motion_artifacts"):
            probe_rows += _regression_rows(
                sector,
                "angular_velocity",
                train[sector],
                train["target_omega"],
                test[sector],
                test["target_omega"],
            )

    statistics_rows = []
    target_names = {
        "target_position",
        "target_velocity",
        "target_rotation_6d",
        "target_omega",
    }
    for name, values in test.items():
        if name in target_names:
            continue
        statistics_rows.append(
            {
                "sector": name,
                "dimension": int(values.shape[1]),
                "mean_feature_std": float(values.std(axis=0).mean()),
                "effective_rank": effective_rank(values),
            }
        )

    _write_csv(output / "canonical_state_metrics.csv", direct_rows)
    _write_csv(output / "sector_probes.csv", probe_rows)
    _write_csv(output / "sector_statistics.csv", statistics_rows)

    report = {
        "task": module.hparams.task,
        "canonical_state_metrics": direct_rows,
        "sector_probes": probe_rows,
        "sector_statistics": statistics_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report



@torch.inference_mode()
def evaluate_structured_se3(
    module: KinematicObservabilityModule, 
    loader, 
    device, 
    output_directory: Path | str, 
    maximum_windows: int=2000
) -> dict[str, object]:
    output = Path(output_directory) / "structured_se3"
    translation = []; rotation = []; inv_velocity = []; inv_omega = []
    rt = {h: [] for h in range(1, 10)}; rr = {h: []  for h in range(1, 10)}
    used = 0
    mask = module.model.context_head.mask

    for batch in loader:
        if used >= maximum_windows:
            break

        rgb = batch["context_rgb"].to(device)
        time = batch["context_time"].to(device)
        prediction = module.model(rgb, time)

        k = min(len(rgb), maximum_windows - used)
        used += k

        context = prediction.context_sectors
        motion = prediction.motion

        if mask.translation:
            translation.append(
                torch.linalg.vector_norm(motion.predicted_next_position[:k] - context.position[:k, 1:], dim=-1).cpu().numpy()
            )
            inv_velocity.append(
                torch.linalg.vector_norm(motion.forward_sectors.linear_velocity[:k] + motion.backward_sectors.linear_velocity[:k], dim=-1).cpu().numpy()
            )
        if mask.rotation:
            rotation.append(
                rotation_geodesic_error(motion.predicted_next_rotation[:k], context.rotation[:k, 1:]).cpu().numpy()
            )
            inv_omega.append(torch.linalg.vector_norm(motion.forward_sectors.angular_velocity[:k] + motion.backward_sectors.angular_velocity[:k], dim=-1).cpu().numpy())

    rows = []
    if translation:
        rows += [
            _statistics("translation_one_step", "meter", translation),
            _statistics("linear_inverse", "meter_per_second", inv_velocity),
        ]

    if rotation:
        rows += [
            _statistics("rotation_one_step", "radian", rotation),
            _statistics("angular_inverse", "radian_per_second", inv_omega)
        ]

    _write_csv(output / 'group_validity_and_consistency.csv', rows)

    report = {
        "task": module.hparams.task,
        "metrics": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report
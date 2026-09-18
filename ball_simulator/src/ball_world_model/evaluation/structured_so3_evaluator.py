"""Dedicated diagnostics for StructuredSO3StateEstimator checkpoints."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ball_world_model.models.rotation import rotation_geodesic_error, so3_exp


def _np(value):
    return value.detach().cpu().numpy()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def _effective_rank(value: np.ndarray) -> float:
    value = value.reshape(len(value), -1).astype(np.float64)
    value -= value.mean(0, keepdims=True)
    singular = np.linalg.svd(value, compute_uv=False)
    probability = singular**2
    if probability.sum() <= 0:
        return 0.0
    probability /= probability.sum()
    return float(np.exp(-np.sum(probability * np.log(probability + 1e-12))))


def _regression(train_x, train_y, test_x, test_y, ridge=1e-4):
    train_x = train_x.reshape(len(train_x), -1).astype(np.float64)
    test_x = test_x.reshape(len(test_x), -1).astype(np.float64)
    design = np.column_stack((train_x, np.ones(len(train_x))))

    regulariser =  ridge * np.eye(design.shape[1])
    regulariser[-1, -1] = 0
    weights = np.linalg.solve(design.T @ design + regulariser, design.T @ train_y)
    estimate = np.column_stack((test_x, np.ones(len(test_x)))) @ weights

    rows = []
    for index, axis in enumerate("xyz"):
        error = estimate[:, index] - test_y[:, index]
        total = np.sum((test_y[:, index] - test_y[:, index].mean())**2)
        rows.append({
            "axis": axis,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "r2": float(1 - np.sum(error**2) / total) if total > 0 else float("nan"),
        })
    return rows

def _batches(loader, maximum):
    used = 0
    for batch in loader:
        if used >= maximum:
            return
        keep = min(batch["context_rgb"].shape[0], maximum - used)
        used += keep
        yield {
            key: value[:keep] if isinstance(value, (torch.Tensor, list, tuple)) else value
            for key, value in batch.items()
        }


@torch.inference_mode()
def collect_structured_diagnostics(module, loader, device, maximum=2000) -> list[dict]:
    samples = []
    for batch in _batches(loader, maximum):
        time = batch["context_time"].to(device)
        prediction = module.model(batch["context_rgb"].to(device), time)
        context, motion = prediction.context_sectors, prediction.motion
        dt = torch.diff(time, dim=1).unsqueeze(-1)

        identity = torch.eye(3, device=device, dtype=context.rotation.dtype)
        orthogonality = torch.linalg.matrix_norm(
            context.rotation.transpose(-1, -2) @ context.rotation - identity,
            ord="fro",
            dim=(-2, -1),
        )
        determinant = (torch.linalg.det(context.rotation) - 1).abs()

        one_step = rotation_geodesic_error(
            motion.predicted_next_rotation,
            context.rotation[:, 1:],
        )

        forward_increment = motion.predicted_next_rotation @ context.rotation[:, :-1].transpose(-1, -2)
        backward_increment = motion.predicted_previous_rotation @ context.rotation[:, 1:].transpose(-1, -2)

        inverse = rotation_geodesic_error(
            backward_increment @ forward_increment,
            identity.expand_as(forward_increment),
        )

        horizons = {}
        rolled = context.rotation[:, 0]
        for h in range(1, context.rotation.shape[1]):
            rolled = so3_exp(motion.forward_sectors.omega[:, h-1] * dt[:, h-1]) @ rolled
            horizons[h] = rotation_geodesic_error(rolled, context.rotation[:, h])

        samples.append({
            "orthogonality": _np(orthogonality),
            "determinant": _np(determinant),
            "one_step": _np(one_step),
            "inverse": _np(inverse),
            "horizons": {h: _np(v) for h, v in horizons.items()},
            "rotation6d": _np(context.rotation_6d[:, -1]),
            "carrier": _np(context.carrier[:, -1]),
            "invariants": _np(context.physical_invariants[:, -1]),
            "artifacts": _np(context.artifacts[:, -1]),
            "omega": _np(motion.forward_sectors.omega.mean(1)),
            "tangent": _np(motion.forward_sectors.carrier_tangent.mean(1)),
            "motion_invariants": _np(motion.forward_sectors.physical_invariants.mean(1)),
            "motion_artifacts": _np(motion.forward_sectors.artifacts.mean(1)),
            "target_omega": _np(batch["context_angular_velocity"][:,-1]),        
        })
    return samples


def _join(samples, name) -> np.ndarray:
    return np.concatenate([sample[name] for sample in samples])

def evaluate_structured_so3_latent(
    module,
    train_loader,
    test_loader,
    device,
    output_directory,
    train_maximum=5000,
    test_maximum=2000,
) -> Path:
    output = Path(output_directory) / "structured_so3"
    output_directory.mkdir(parents=True, exist_ok=True)
    train = collect_structured_diagnostics(module, train_loader, device, train_maximum)
    test = collect_structured_diagnostics(module, test_loader, device, test_maximum)

    group_rows = []
    for name, unit in (("orthogonality", "frobenius"), ("determinant", "absolute"), ("one_step", "radian"), ("inverse", "radian")):
        value = _join(test, name).reshape(-1)
        group_rows.append({
            "metric": name, 
            "unit": unit, 
            "mean": float(value.mean()),
            "median": float(np.median(value)), 
            "p95": float(np.quantile(value, .95)),
            "maximum": float(value.max()), 
            "count": len(value),
        })
    _write_csv(output / "group_validity_and_consistency.csv", group_rows)

    horizon_rows = []
    for horizon in sorted(test[0]["horizons"]):
        value = np.concatenate([sample["horizons"][horizon] for sample in test]).reshape(-1)
        horizon_rows.append({
            "horizon_intervals": horizon,
            "mean_deg": float(np.rad2deg(value.mean())),
            "median_deg": float(np.rad2deg(np.median(value))),
            "p95_deg": float(np.rad2deg(np.quantile(value, .95))),
        })
    _write_csv(output / "rollout_by_horizon.csv", horizon_rows)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot([r["horizon_intervals"] for r in horizon_rows], [r["mean_deg"] for r in horizon_rows], marker="o", label="mean")
    axis.plot([r["horizon_intervals"] for r in horizon_rows], [r["p95_deg"] for r in horizon_rows], marker="s", label="p95")
    axis.set_xlabel("Rollout horizon [intervals]"); axis.set_ylabel("Geodesic error [deg]")
    axis.set_title("Intrinsic latent group rollout"); axis.grid(alpha=.25); axis.legend()
    figure.tight_layout()
    figure.savefig(output / "rollout_by_horizon.png", dpi=170)
    plt.close(figure)

    sectors = ["rotation6d", "carrier", "invariants", "artifacts", "omega", "tangent", "motion_invariants", "motion_artifacts"]
    statistic_rows = []
    for name in sectors:
        value = _join(test, name).reshape(len(_join(test, name)), -1)
        statistic_rows.append({
            "sector": name,
            "dimension": value.shape[1],
            "mean_feature_std": float(value.std(0).mean()),
            "effective_rank": _effective_rank(value),
        })
    _write_csv(output / "sector_statistics.csv", statistic_rows)

    probe_rows = []
    target_train, target_test = _join(train, "target_omega"), _join(test, "target_omega")
    for name in ["carrier", "invariants", "artifacts", "omega", "tangent", "motion_invariants", "motion_artifacts"]:
        metrics = _regression(_join(train, name), target_train, _join(test, name), target_test)
        for metric in metrics:
            probe_rows.append({
                "sector": name,
                "target": "angular_velocity",
                **metric,
            })
    _write_csv(output / "sector_omega_probes.csv", probe_rows)

    summary = {
        "group_metrics": group_rows,
        "rollout": horizon_rows,
        "sector_statistics": statistic_rows,
        "sector_omega_probes": probe_rows
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    return output.resolve()
"""Evaluate artifact utility, physical leakage, and post-split bypass."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import torch

from ball_world_model.models.rotation import rotation_geodesic_error

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _ridge(train_x, train_y, test_x, test_y, ridge=1e-4):
    train_x = np.asarray(train_x).reshape(len(train_x),-1)
    test_x = np.asarray(test_x).reshape(len(test_x),-1)
    A = np.column_stack((train_x,np.ones(len(train_x))))
    reg = ridge * np.eye(A.shape[1])
    reg[-1, -1] = 0
    W = np.linalg.solve(A.T @ A + reg, A.T @ train_y)
    pred = np.column_stack((test_x, np.ones(len(test_x)))) @ W
    rows = []
    for i, axis in enumerate("xyz"):
        err = pred[:, i] - test_y[:, i]
        den = ((test_y[:, i] - test_y[:, i].mean())**2).sum()
        rows.append({
            "axis": axis,
            "rmse": float(np.sqrt((err**2).mean())),
            "r2": float(1 - (err**2).sum() / den)
        })
    return rows


@torch.inference_mode()
def collect_artifact_batches(module, loader, device, maximum=5000):
    artifacts = []; omega = []; residual_mse = []; physical_delta = []; used = 0
    for batch in loader:
        if used >= maximum: 
            break

        rgb = batch["context_rgb"].to(device)
        time = batch["context_time"].to(device)
        prediction = module.model(rgb, time)

        motion = prediction.motion
        nu = motion.forward_sectors.artifacts

        keep = min(rgb.shape[0], maximum - used)
        used += keep

        artifacts.append(nu[:keep].mean(1).cpu().numpy())
        omega.append(batch["context_angular_velocity"][:keep, :-1].mean(1).numpy())
        if hasattr(motion, "artifact_residual_reconstruction"):
            residual_mse.append((motion.artifact_residual_reconstruction[:keep] - motion.artifact_residual_target[:keep]).square().mean((1, 2, 3, 4)).cpu().numpy())

        # This is expected to be exactly zero if artifacts are downstream-isolated.
        if hasattr(module.model, "forward_with_zero_artifacts"):
            bypass = module.model.forward_with_zero_artifacts(rgb, time)
            physical_delta.append(torch.linalg.vector_norm(prediction.angular_velocity[:keep] - bypass.angular_velocity[:keep], dim=-1).mean(1).cpu().numpy())
    return {
        "artifacts": np.concatenate(artifacts),
        "omega": np.concatenate(omega),
        "residual_mse": np.concatenate(residual_mse) if residual_mse else np.array([]),
        "physical_delta": np.concatenate(physical_delta) if physical_delta else np.array([])
    }


def evaluate_artifact_disentanglement(module, train_loader, test_loader, device, output_directory, maximum=5000):
    output = Path(output_directory) / "artifact_disentanglement"
    output_directory.mkdir(parents=True, exist_ok=True)
    train = collect_artifact_batches(module, train_loader, device, maximum)
    test = collect_artifact_batches(module, test_loader, device, maximum)

    rows = _ridge(train["artifacts"], train["omega"], test["artifacts"], test["omega"])
    _write_csv(output / "artifact_omega_probe_csv", rows)

    summary = []
    if len(test["residual_mse"]):
        summary.append({
            "metric": "residual_feature_mse", 
            "mean": float(test["residual_mse"].mean()),
            "p95": float(np.quantile(test["residual_mse"], 0.95))
        })

    if len(test["physical_delta"]):
        summary.append({
            "metric": "zero_artifact_omega_delta_radps",
            "mean": float(test["physical_delta"].mean()),
            "p95": float(np.quantile(test["physical_delta"], 0.95))
        })
    _write_csv(output / "artifact_utility_and_bypass.csv", summary)
    return rows, summary


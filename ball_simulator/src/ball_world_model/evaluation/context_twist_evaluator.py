"""Evaluate interval and context-aggregated twists for constant-twist windows."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .metrics import regression_metrics
from .model_loader import denormalised_prediction

def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _limited(loader, maximum: int):
    used = 0
    for batch in loader:
        if used >= maximum:
            break
        size = int(batch["context_rgb"].shape[0])
        keep = min(size, maximum - used)
        yield {key: value[:keep] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        used += keep

def _camera_components(value: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("...j,ij->...i", value, basis)


@torch.inference_mode()
def evaluate_context_twist(
    module,
    loader,
    device: torch.device,
    output: Path,
    maximum_windows: int,
) -> dict[str, object]:
    predicted_v, target_v, predicted_w, target_w = [], [], [], []
    interval_v_std, interval_w_std = [], []

    for batch in _limited(loader, maximum_windows):
        prediction = module.model(
            batch["context_rgb"].to(device),
            batch["context_time"].to(device),
        )
        physical = denormalised_prediction(module, prediction)
        if "linear_velocity" in physical:
            estimate = physical["linear_velocity"][:, :-1]
            target = batch["context_linear_velocity"][:, :-1].to(device)
            predicted_v.append(estimate.mean(1).cpu().numpy())
            target_v.append(target.mean(1).cpu().numpy())
            interval_v_std.append(estimate.std(1, unbiased=False).mean(-1).cpu().numpy())
        if "angular_velocity" in physical:
            estimate = physical["angular_velocity"][:, :-1]
            target = batch["context_angular_velocity"][:, :-1].to(device)
            predicted_w.append(estimate.mean(1).cpu().numpy())
            target_w.append(target.mean(1).cpu().numpy())
            interval_w_std.append(estimate.std(1, unbiased=False).mean(-1).cpu().numpy())

    rows: list[dict[str, object]] = []
    axes = ("x", "y", "z")
    if predicted_v:
        pv, tv = np.concatenate(predicted_v), np.concatenate(target_v)
        for index, axis in enumerate(axes):
            rows.append({"quantity": "context_linear_velocity", "component": axis, **regression_metrics(tv[:, index], pv[:, index])})
        if module.camera_velocity_basis.numel() != 0:
            basis = module.camera_velocity_basis.detach().cpu().numpy()
            pvc, tvc = _camera_components(pv, basis), _camera_components(tv, basis)
            for index, axis in enumerate(("right", "up", "depth")):
                rows.append({"quantity": "context_linear_velocity_camera", "component": axis, **regression_metrics(tvc[:, index], pvc[:, index])})
        rows.append({
            "quantity": "interval_variability",
            "component": "linear_velocity_mean_std_mps",
            "value": float(np.concatenate(interval_v_std).mean()),
        })
    if predicted_w:
        pw, tw = np.concatenate(predicted_w), np.concatenate(target_w)
        for index, axis in enumerate(axes):
            rows.append({"quantity": "context_angular_velocity", "component": axis, **regression_metrics(tw[:, index], pw[:, index])})
        rows.append({
            "quantity": "interval_variability",
            "component": "angular_velocity_mean_std_radps",
            "value": float(np.concatenate(interval_w_std).mean()),
        })

    directory = output / "context_twist"
    _write_csv(directory / "metrics.csv", rows)
    report = {"task": module.hparams.task, "metrics": rows}
    (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

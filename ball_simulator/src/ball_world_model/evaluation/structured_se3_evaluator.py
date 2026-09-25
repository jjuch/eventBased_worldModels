"""Intrinsic SE(3) evaluation shared by all task masks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ball_world_model.models.rotation import rotation_geodesic_error
from ball_world_model.training.kinematic_module import KinematicObservabilityModule


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
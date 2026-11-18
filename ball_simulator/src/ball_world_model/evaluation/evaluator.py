"""Evaluation for the typed causal SE(3) observer."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ball_world_model.config import DataConfig
from ball_world_model.data.dataset import RenderedTrajectoryDataset, build_dataloader
from ball_world_model.geometry.se3 import carrier_state, propagate_world, so3_log
from ball_world_model.training.module import SE3ObserverModule
from ball_world_model.training.targets import current_targets, quaternion_xyzw_to_matrix



@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    maximum_test_windows: int = 2_000
    maximum_probe_train_windows: int = 5_000  # Retained for CLI compatibility.
    trajectory_plots: int = 12
    seed: int = 20260824

def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _safe_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = np.square(prediction - target).sum(axis=0)
    total = np.square(target - target.mean(axis=0)).sum(axis=0)
    return np.where(total > 1.0e-12, 1.0 - residual / total, np.nan)


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    return {
        "rmse": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
        "mae": np.mean(np.abs(error), axis=0).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
        "r2": _safe_r2(prediction, target).tolist(),
        "norm_rmse": float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))),
        "norm_mae": float(np.mean(np.linalg.norm(error, axis=1))),
        "p50_norm_error": float(np.percentile(np.linalg.norm(error, axis=1), 50)),
        "p95_norm_error": float(np.percentile(np.linalg.norm(error, axis=1), 95)),
    }


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _component_scatter(path: Path, prediction: np.ndarray, target: np.ndarray, name: str, unit: str) -> None:
    labels = ("x", "y", "z")
    metrics = _regression_metrics(prediction, target)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, component in zip(axes, range(3), strict=True):
        x, y = target[:, component], prediction[:, component]
        axis.scatter(x, y, s=10, alpha=0.55)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        axis.plot([low, high], [low, high], color="black", linewidth=1.5)
        axis.set_title(f"{name} {labels[component]}: RMSE={metrics['rmse'][component]:.4g}, R2={metrics['r2'][component]:.3f}")
        axis.set_xlabel(f"True [{unit}]")
        axis.set_ylabel(f"Predicted [{unit}]")
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _error_histogram(path: Path, errors: dict[str, tuple[np.ndarray, str]]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (name, (value, unit)) in zip(axes.flat, errors.items(), strict=True):
        axis.hist(value, bins=50, alpha=0.85)
        axis.axvline(np.percentile(value, 50), color="black", linestyle="--", label="median")
        axis.axvline(np.percentile(value, 95), color="red", linestyle=":", label="p95")
        axis.set_title(name)
        axis.set_xlabel(unit)
        axis.set_ylabel("Count")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _error_vs_magnitude(path: Path, magnitude: np.ndarray, error: np.ndarray, x_label: str, y_label: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.scatter(magnitude, error, s=10, alpha=0.45)
    if len(magnitude) >= 20:
        edges = np.quantile(magnitude, np.linspace(0, 1, 11))
        centres, means = [], []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            mask = (magnitude >= left) & (magnitude <= right)
            if mask.any():
                centres.append(magnitude[mask].mean()); means.append(error[mask].mean())
        axis.plot(centres, means, color="red", linewidth=2, marker="o", label="decile mean")
        axis.legend()
    axis.set_xlabel(x_label); axis.set_ylabel(y_label); axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180); plt.close(figure)


def _attention_plot(path: Path, twist: np.ndarray | None, acceleration: np.ndarray | None) -> None:
    if twist is None and acceleration is None:
        return
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if twist is not None:
        axis.plot(np.arange(twist.shape[1]), twist.mean(axis=0), marker="o", label="twist")
    if acceleration is not None:
        axis.plot(np.arange(acceleration.shape[1]), acceleration.mean(axis=0), marker="o", label="acceleration")
    axis.set_xlabel("Temporal interval index"); axis.set_ylabel("Mean attention weight")
    axis.set_title("Temporal evidence used by the observer"); axis.grid(alpha=0.25); axis.legend()
    figure.savefig(path, dpi=180); plt.close(figure)


def _trajectory_plots(output: Path, records: dict[str, np.ndarray], count: int) -> None:
    if count <= 0:
        return
    output.mkdir(parents=True, exist_ok=True)
    identifiers = records["trajectory_id"]
    selected = list(dict.fromkeys(identifiers.tolist()))[:count]
    for identifier in selected:
        mask = identifiers == identifier
        order = np.argsort(records["time"][mask])
        time = records["time"][mask][order]
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for component, label in enumerate("xyz"):
            axes[0, 0].plot(time, records["true_position"][mask][order, component], label=f"true {label}")
            axes[0, 0].plot(time, records["pred_position"][mask][order, component], linestyle="--", label=f"pred {label}")
            axes[0, 1].plot(time, records["true_linear_velocity"][mask][order, component])
            axes[0, 1].plot(time, records["pred_linear_velocity"][mask][order, component], linestyle="--")
            axes[1, 0].plot(time, records["true_angular_velocity"][mask][order, component])
            axes[1, 0].plot(time, records["pred_angular_velocity"][mask][order, component], linestyle="--")
            axes[1, 1].plot(time, records["orientation_error_deg"][mask][order])
        axes[0, 0].set_title("Position [m]"); axes[0, 1].set_title("Linear velocity [m/s]")
        axes[1, 0].set_title("Angular velocity [rad/s]"); axes[1, 1].set_title("Orientation error [deg]")
        for axis in axes.flat: axis.set_xlabel("Time [s]"); axis.grid(alpha=0.25)
        axes[0, 0].legend(ncol=3, fontsize=8)
        figure.suptitle(f"Trajectory {identifier}")
        figure.savefig(output / f"trajectory_{identifier}.png", dpi=170); plt.close(figure)


@torch.inference_mode()
def _collect(module: SE3ObserverModule, loader, device: torch.device, maximum: int) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {}
    seen = 0
    for batch in loader:
        batch_size = len(batch["context_rgb"])
        if seen >= maximum:
            break
        keep = min(batch_size, maximum - seen)
        device_batch = _to_device(batch, device)
        prediction = module.model(device_batch["context_rgb"], device_batch.get("context_time"))
        target = current_targets(device_batch)
        orientation_error = torch.linalg.vector_norm(
            so3_log(target.rotation.transpose(-1, -2) @ prediction.pose.rotation), dim=-1
        )
        carrier_target = carrier_state(target.position, target.rotation, module.model.templates)
        carrier_rmse = torch.sqrt((prediction.carrier - carrier_target).square().mean(dim=(-2, -1)))
        dt = (device_batch["future_time"][:, 0] - device_batch["context_time"][:, -1]).clamp_min(1.0e-6)
        next_p, next_r, next_v, next_w = propagate_world(
            prediction.pose.position, prediction.pose.rotation, prediction.twist.linear,
            prediction.twist.angular, dt, prediction.acceleration.linear, prediction.acceleration.angular,
        )
        true_next_r = quaternion_xyzw_to_matrix(device_batch["future_quaternion_xyzw"][:, 0])
        rollout_orientation = torch.linalg.vector_norm(so3_log(true_next_r.transpose(-1, -2) @ next_r), dim=-1)
        tensors = {
            "pred_position": prediction.pose.position, "true_position": target.position,
            "pred_linear_velocity": prediction.twist.linear, "true_linear_velocity": target.linear_velocity,
            "pred_angular_velocity": prediction.twist.angular, "true_angular_velocity": target.angular_velocity,
            "pred_linear_acceleration": prediction.acceleration.linear, "true_linear_acceleration": target.linear_acceleration,
            "pred_angular_acceleration": prediction.acceleration.angular, "true_angular_acceleration": target.angular_acceleration,
            "orientation_error_deg": torch.rad2deg(orientation_error), "carrier_rmse": carrier_rmse,
            "rollout_position_error": torch.linalg.vector_norm(next_p - device_batch["future_position"][:, 0], dim=-1),
            "rollout_orientation_error_deg": torch.rad2deg(rollout_orientation),
            "rollout_linear_velocity_error": torch.linalg.vector_norm(next_v - device_batch["future_linear_velocity"][:, 0], dim=-1),
            "rollout_angular_velocity_error": torch.linalg.vector_norm(next_w - device_batch["future_angular_velocity"][:, 0], dim=-1),
            "time": device_batch["context_time"][:, -1],
        }
        for key, value in tensors.items():
            values.setdefault(key, []).append(value[:keep].detach().cpu().numpy())
        for key, value in (("twist_attention", prediction.aux.twist_attention), ("acceleration_attention", prediction.aux.acceleration_attention)):
            if value is not None: values.setdefault(key, []).append(value[:keep].detach().cpu().numpy())
        values.setdefault("trajectory_id", []).append(np.asarray(batch["trajectory_id"][:keep], dtype=str))
        seen += keep
    return {key: np.concatenate(parts, axis=0) for key, parts in values.items()}


def _summary(records: dict[str, np.ndarray]) -> dict[str, Any]:
    summary = {
        "windows": int(len(records["true_position"])),
        "position": _regression_metrics(records["pred_position"], records["true_position"]),
        "linear_velocity": _regression_metrics(records["pred_linear_velocity"], records["true_linear_velocity"]),
        "angular_velocity": _regression_metrics(records["pred_angular_velocity"], records["true_angular_velocity"]),
        "linear_acceleration": _regression_metrics(records["pred_linear_acceleration"], records["true_linear_acceleration"]),
        "angular_acceleration": _regression_metrics(records["pred_angular_acceleration"], records["true_angular_acceleration"]),
    }
    for name in ("orientation_error_deg", "carrier_rmse", "rollout_position_error", "rollout_orientation_error_deg", "rollout_linear_velocity_error", "rollout_angular_velocity_error"):
        value = records[name]
        summary[name] = {"mean": float(value.mean()), "median": float(np.median(value)), "p95": float(np.percentile(value, 95)), "maximum": float(value.max())}
    return summary


def _write_window_csv(path: Path, records: dict[str, np.ndarray]) -> None:
    fields = ["trajectory_id", "time", "orientation_error_deg", "carrier_rmse", "rollout_position_error", "rollout_orientation_error_deg", "rollout_linear_velocity_error", "rollout_angular_velocity_error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index in range(len(records["trajectory_id"])):
            writer.writerow({field: records[field][index].item() if np.ndim(records[field][index]) == 0 else records[field][index] for field in fields})


def evaluate_kinematic_observer(
    checkpoint_path: str | Path,
    training_config_path: str | Path,
    output_directory: str | Path,
    settings: EvaluationSettings = EvaluationSettings(),
) -> Path:
    """Evaluate state decoding, SE(3) geometry, temporal evidence, and one-step rollout."""
    torch.manual_seed(settings.seed); np.random.seed(settings.seed)
    checkpoint_path, training_config_path, output = Path(checkpoint_path), Path(training_config_path), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    configuration = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    data_config = DataConfig.from_yaml(Path(configuration["data_config"]))

    manifest = data_config.manifest_path or data_config.root / "manifest.parquet"
    dataset = RenderedTrajectoryDataset(manifest, data_config, split="test")
    loader = build_dataloader(dataset, data_config, shuffle=False)
    module = SE3ObserverModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device).eval()
    records = _collect(module, loader, device, settings.maximum_test_windows)
    summary = _summary(records)
    summary["checkpoint"] = str(checkpoint_path.resolve())
    summary["training_config"] = str(training_config_path.resolve())
    _save_json(output / "metrics.json", summary)
    _write_window_csv(output / "window_errors.csv", records)

    _component_scatter(output / "position_components.png", records["pred_position"], records["true_position"], "Position", "m")
    _component_scatter(output / "linear_velocity_components.png", records["pred_linear_velocity"], records["true_linear_velocity"], "Linear velocity", "m/s")
    _component_scatter(output / "angular_velocity_components.png", records["pred_angular_velocity"], records["true_angular_velocity"], "Angular velocity", "rad/s")
    _component_scatter(output / "linear_acceleration_components.png", records["pred_linear_acceleration"], records["true_linear_acceleration"], "Linear acceleration", "m/s2")
    _component_scatter(output / "angular_acceleration_components.png", records["pred_angular_acceleration"], records["true_angular_acceleration"], "Angular acceleration", "rad/s2")
    _error_histogram(output / "error_distributions.png", {
        "Position norm error": (np.linalg.norm(records["pred_position"]-records["true_position"], axis=1), "m"),
        "Orientation geodesic error": (records["orientation_error_deg"], "deg"),
        "Linear velocity norm error": (np.linalg.norm(records["pred_linear_velocity"]-records["true_linear_velocity"], axis=1), "m/s"),
        "Angular velocity norm error": (np.linalg.norm(records["pred_angular_velocity"]-records["true_angular_velocity"], axis=1), "rad/s"),
        "Linear acceleration norm error": (np.linalg.norm(records["pred_linear_acceleration"]-records["true_linear_acceleration"], axis=1), "m/s2"),
        "Angular acceleration norm error": (np.linalg.norm(records["pred_angular_acceleration"]-records["true_angular_acceleration"], axis=1), "rad/s2"),
    })

    _error_vs_magnitude(output / "linear_velocity_error_vs_speed.png", np.linalg.norm(records["true_linear_velocity"], axis=1), np.linalg.norm(records["pred_linear_velocity"]-records["true_linear_velocity"], axis=1), "True speed [m/s]", "Velocity error [m/s]")
    _error_vs_magnitude(output / "angular_velocity_error_vs_speed.png", np.linalg.norm(records["true_angular_velocity"], axis=1), np.linalg.norm(records["pred_angular_velocity"]-records["true_angular_velocity"], axis=1), "True angular speed [rad/s]", "Angular-velocity error [rad/s]")
    
    _attention_plot(output / "temporal_attention.png", records.get("twist_attention"), records.get("acceleration_attention"))
    _trajectory_plots(output / "trajectories", records, settings.trajectory_plots)
    return output

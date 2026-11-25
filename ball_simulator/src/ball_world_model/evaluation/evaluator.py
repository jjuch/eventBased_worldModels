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
from ball_world_model.data.transforms import denormalise_rgb
from ball_world_model.geometry.se3 import propagate_world, rotation_angle, rollout_world
from ball_world_model.geometry.symmetry import symmetric_rotation_angle
from ball_world_model.models.state import ObserverPrediction
from ball_world_model.training.module import SE3ObserverModule
from ball_world_model.training.targets import quaternion_xyzw_to_matrix, window_targets
from .plots import PinholeCamera, horizon_statistics, rollout_horizon_plot, rollout_overlay_plots, trajectory_window_plots




@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    maximum_test_windows: int = 2_000
    maximum_probe_train_windows: int = 5_000  # Retained for CLI compatibility.
    trajectory_plots: int = 12
    seed: int = 20260824
    ball_radius: float = 0.10 # m, only used to draw the ball in the overlay figures

def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _safe_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = np.square(prediction - target).sum(axis=0)
    total = np.square(target - target.mean(axis=0)).sum(axis=0)
    return np.array([1.0-r/t if t > 1.0e-12 else None for r,t in zip(residual,total,strict=True)], dtype=object)


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    return {
        "rmse": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
        "mae": np.mean(np.abs(error), axis=0).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
        "r2": _safe_r2(prediction, target).tolist(),
        "target_std": target.std(axis=0).tolist(),
        "informative": bool(np.max(target.std(axis=0)) > 1.0e-8),
        "norm_rmse": float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))),
        "norm_mae": float(np.mean(np.linalg.norm(error, axis=1))),
        "p50_norm_error": float(np.percentile(np.linalg.norm(error, axis=1), 50)),
        "p95_norm_error": float(np.percentile(np.linalg.norm(error, axis=1), 95)),
    }

def _json_compatible(value):
    """Convert NumPy and PyTorch values to standard JSON-compatible types."""

    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]

    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()

        return _json_compatible(
            value.detach().cpu().tolist()
        )

    if isinstance(value, Path):
        return str(value)

    return value

def _save_json(path: Path, value: Any) -> None:
    compatible_value = _json_compatible(value)
    path.write_text(json.dumps(
        compatible_value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

def _format_metric(
    value: float | None,
    format_spec: str,
    *,
    unavailable: str = "undefined",
) -> str:
    """Format a metric while handling undefined values."""

    if value is None:
        return unavailable

    return format(value, format_spec)

def _component_scatter(path: Path, prediction: np.ndarray, target: np.ndarray, name: str, unit: str) -> None:
    labels = ("x", "y", "z")
    metrics = _regression_metrics(prediction, target)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, component in zip(axes, range(3), strict=True):
        x, y = target[:, component], prediction[:, component]
        axis.scatter(x, y, s=10, alpha=0.55)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        axis.plot([low, high], [low, high], color="black", linewidth=1.5)
        rmse_text = _format_metric(metrics["rmse"][component], ".4g")
        r2_text = _format_metric(metrics["r2"][component], ".3f")
        axis.set_title(f"{name} {labels[component]}: RMSE={rmse_text}, R²={r2_text}")
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


def _rollout_from_first_frame(prediction: ObserverPrediction, horizons: torch.Tensor, use_acceleration: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Open-loop rollout from the frame-0 estimate. horizons [B, H] in s since frame 0.

    Without acceleration the first interval-mean twist is held constant (the
    constant-velocity baseline).
    """
    poses, twists = prediction.frame_poses, prediction.interval_twists
    accelerations, half_step = prediction.interval_accelerations, 0.5 * prediction.interval_dt[:, :1]
    if use_acceleration:
        # The first interval mean sits half a step after frame 0; step back to the instant of frame 0.
        linear = twists.linear[:, 0] - half_step * accelerations.linear[:, 0]
        angular = twists.angular[:, 0] - half_step * accelerations.angular[:, 0]
        linear_acceleration, angular_acceleration = accelerations.linear[:, 0], accelerations.angular[:, 0]
    else:
        linear, angular = twists.linear[:, 0], twists.angular[:, 0]
        linear_acceleration, angular_acceleration = torch.zeros_like(linear), torch.zeros_like(angular)
    return rollout_world(
        poses.position[:, 0], poses.rotation[:, 0], linear, angular, linear_acceleration, angular_acceleration, horizons,
    )


def _twist_difference_acceleration(rates: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Acceleration implied by neighbouring interval-mean velocities, [B, I-1, 3] in m/s² or rad/s².

    Baseline for the acceleration head: it needs no acceleration branch at all.
    Interval means sit at interval midpoints, so the time between two neighbouring means is the average of the two interval durations. The result estimates the acceleration at the inner frames 1..T-2.
    """
    midpoint_spacing = 0.5 * (dt[:, 1:] + dt[:, :-1]).unsqueeze(-1)
    return (rates[:, 1:] - rates[:, :-1]) / midpoint_spacing


def _sequence_tensors(prediction: ObserverPrediction, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Whole-window estimates and both rollouts, for the window and rollout figures."""
    targets = window_targets(batch)
    context_time, future_time = batch["context_time"].float(), batch["future_time"].float()
    future_position = batch["future_position"].float()
    future_rotation = quaternion_xyzw_to_matrix(batch["future_quaternion_xyzw"].float())
    state = prediction.current
 
    current_horizon = future_time - context_time[:, -1:]
    current_position, current_rotation = rollout_world(
        state.pose.position, state.pose.rotation, state.twist.linear, state.twist.angular,
        state.acceleration.linear, state.acceleration.angular, current_horizon,
    )

    zero = torch.zeros_like(state.twist.linear)
    current_position_no_acceleration, current_rotation_no_acceleration = rollout_world(
        state.pose.position, state.pose.rotation, state.twist.linear, state.twist.angular,
        zero, zero, current_horizon,
    )

    first_horizon = torch.cat((context_time[:, 1:], future_time), dim=1) - context_time[:, :1]
    first_position, first_rotation = _rollout_from_first_frame(prediction, first_horizon)

    first_position_no_acceleration, first_rotation_no_acceleration = _rollout_from_first_frame(
        prediction, first_horizon, use_acceleration=False
    )

    true_position_after_first = torch.cat((targets.frame_position[:, 1:], future_position), dim=1)
    true_rotation_after_first = torch.cat((targets.frame_rotation[:, 1:], future_rotation), dim=1)
 
    return {
        "frame_time": context_time,
        "future_time": future_time,
        "interval_time": 0.5 * (context_time[:, 1:] + context_time[:, :-1]),
        "true_frame_position": targets.frame_position,
        "pred_frame_position": prediction.frame_poses.position,
        "true_frame_rotation": targets.frame_rotation,
        "pred_frame_rotation": prediction.frame_poses.rotation,
        "frame_orientation_error_deg": torch.rad2deg(rotation_angle(targets.frame_rotation, prediction.frame_poses.rotation)),
        "true_frame_linear_velocity": targets.frame_linear_velocity,
        "true_frame_angular_velocity": targets.frame_angular_velocity,
        "true_interval_linear_velocity": targets.interval_linear_velocity,
        "pred_interval_linear_velocity": prediction.interval_twists.linear,
        "true_interval_angular_velocity": targets.interval_angular_velocity,
        "pred_interval_angular_velocity": prediction.interval_twists.angular,
        "true_interval_linear_acceleration": targets.interval_linear_acceleration,
        "pred_interval_linear_acceleration": prediction.interval_accelerations.linear,
        "true_interval_angular_acceleration": targets.interval_angular_acceleration,
        "pred_interval_angular_acceleration": prediction.interval_accelerations.angular,
        # Inner frames 1..T-2: true acceleration (mean of the neighbouring interval means) and the acceleration implied by the predicted velocities, as a baseline for the acceleration head.
        "true_inner_linear_acceleration": 0.5 * (targets.interval_linear_acceleration[:, 1:] + targets.interval_linear_acceleration[:, :-1]),
        "pred_inner_linear_acceleration_from_twist": _twist_difference_acceleration(prediction.interval_twists.linear, prediction.interval_dt),
        "pred_inner_linear_acceleration_from_head": 0.5 * (prediction.interval_accelerations.linear[:, 1:] + prediction.interval_accelerations.linear[:, :-1]),
        "true_inner_angular_acceleration": 0.5 * (targets.interval_angular_acceleration[:, 1:] + targets.interval_angular_acceleration[:, :-1]),
        "pred_inner_angular_acceleration_from_twist": _twist_difference_acceleration(prediction.interval_twists.angular, prediction.interval_dt),
        "pred_inner_angular_acceleration_from_head": 0.5 * (prediction.interval_accelerations.angular[:, 1:] + prediction.interval_accelerations.angular[:, :-1]),
        "true_future_position": future_position,
        "true_future_rotation": future_rotation,
        "rollout_current_horizon": current_horizon,
        "rollout_current_position": current_position,
        "rollout_current_rotation": current_rotation,
        "rollout_current_position_error": torch.linalg.vector_norm(current_position - future_position, dim=-1),
        "rollout_current_orientation_error_deg": torch.rad2deg(rotation_angle(future_rotation, current_rotation)),
        "rollout_current_no_acceleration_position_error": torch.linalg.vector_norm(current_position_no_acceleration - future_position, dim=-1),
        "rollout_current_no_acceleration_orientation_error_deg": torch.rad2deg(rotation_angle(future_rotation, current_rotation_no_acceleration)),
        "rollout_first_horizon": first_horizon,
        "rollout_first_position": first_position,
        "rollout_first_rotation": first_rotation,
        "rollout_first_position_error": torch.linalg.vector_norm(first_position - true_position_after_first, dim=-1),
        "rollout_first_orientation_error_deg": torch.rad2deg(rotation_angle(true_rotation_after_first, first_rotation)),
        "rollout_first_no_acceleration_position_error": torch.linalg.vector_norm(first_position_no_acceleration - true_position_after_first, dim=-1),
        "rollout_first_no_acceleration_orientation_error_deg": torch.rad2deg(rotation_angle(true_rotation_after_first, first_rotation_no_acceleration)),
    }

@torch.inference_mode()
def _collect(module: SE3ObserverModule, loader, device: torch.device, maximum: int, image_windows: int = 0, normalisation: str = "imagenet") -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Run the observer on up to ``maximum`` test windows.
 
    Returns per-window records (NumPy, one row per window) and, for the first
    ``image_windows`` windows, the context + future frames as uint8 [M, T+F, H, W, 3].
    """
    values: dict[str, list[np.ndarray]] = {}
    images: list[np.ndarray] = []
    seen = 0
    for batch in loader:
        batch_size = len(batch["context_rgb"])
        if seen >= maximum:
            break
        keep = min(batch_size, maximum - seen)
        device_batch = _to_device(batch, device)

        prediction = module.model(device_batch["context_rgb"], device_batch["context_time"])
        state, target = prediction.current, window_targets(device_batch).current
        orientation_error = rotation_angle(target.rotation, state.pose.rotation)

        # One-step rollout from the current state to the first future frame.
        dt = (device_batch["future_time"][:, 0] - device_batch["context_time"][:, -1]).clamp_min(1.0e-6)
        next_p, next_r, next_v, next_w = propagate_world(
            state.pose.position, state.pose.rotation,
            state.twist.linear, state.twist.angular,
            dt,
            state.acceleration.linear, state.acceleration.angular,
        )
        true_next_r = quaternion_xyzw_to_matrix(device_batch["future_quaternion_xyzw"][:, 0])
        
        tensors = {
            "pred_position": state.pose.position, "true_position": target.position,
            "pred_linear_velocity": state.twist.linear, "true_linear_velocity": target.linear_velocity,
            "pred_angular_velocity": state.twist.angular, "true_angular_velocity": target.angular_velocity,
            "pred_linear_acceleration": state.acceleration.linear, "true_linear_acceleration": target.linear_acceleration,
            "pred_angular_acceleration": state.acceleration.angular, "true_angular_acceleration": target.angular_acceleration,
            "orientation_error_deg": torch.rad2deg(orientation_error),
            "rollout_position_error": torch.linalg.vector_norm(next_p - device_batch["future_position"][:, 0], dim=-1),
            "rollout_orientation_error_deg": torch.rad2deg(rotation_angle(true_next_r, next_r)),
            "rollout_linear_velocity_error": torch.linalg.vector_norm(next_v - device_batch["future_linear_velocity"][:, 0], dim=-1),
            "rollout_angular_velocity_error": torch.linalg.vector_norm(next_w - device_batch["future_angular_velocity"][:, 0], dim=-1),
            "time": device_batch["context_time"][:, -1],
            **_sequence_tensors(prediction, device_batch),
        }

        for key, value in tensors.items():
            values.setdefault(key, []).append(value[:keep].detach().float().cpu().numpy())

        values.setdefault("trajectory_id", []).append(np.asarray(batch["trajectory_id"][:keep], dtype=str))

        stored = sum(len(part) for part in images)
        if stored < image_windows:
            take = min(keep, image_windows - stored)
            frames = torch.cat((batch["context_rgb"][:take], batch["future_rgb"][:take]), dim=1)
            frames = denormalise_rgb(frames, normalisation) # [take, T+F, 3, H, W] in [0, 1]
            images.append((frames.permute(0, 1, 3, 4, 2) * 255).round().to(torch.uint8).numpy())

        seen += keep

    records = {key: np.concatenate(parts, axis=0) for key, parts in values.items()}
    return records, (np.concatenate(images, axis=0) if images else None)


def _summary(records: dict[str, np.ndarray]) -> dict[str, Any]:
    summary = {
        "windows": int(len(records["true_position"])),
        "position": _regression_metrics(records["pred_position"], records["true_position"]),
        "linear_velocity": _regression_metrics(records["pred_linear_velocity"], records["true_linear_velocity"]),
        "angular_velocity": _regression_metrics(records["pred_angular_velocity"], records["true_angular_velocity"]),
        "linear_acceleration": _regression_metrics(records["pred_linear_acceleration"], records["true_linear_acceleration"]),
        "angular_acceleration": _regression_metrics(records["pred_angular_acceleration"], records["true_angular_acceleration"]),
    }

    for name in ("orientation_error_deg", "rollout_position_error", "rollout_orientation_error_deg", "rollout_linear_velocity_error", "rollout_angular_velocity_error"):
        value = records[name]
        summary[name] = {"mean": float(value.mean()), "median": float(np.median(value)), "p95": float(np.percentile(value, 95)), "maximum": float(value.max())}

    summary["rollout_vs_horizon"] = horizon_statistics(records)
    summary["interval_acceleration"] = _interval_acceleration_summary(records)
    return summary


def _interval_acceleration_summary(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """Acceleration over every interval, and at inner frames head vs. differenced velocities."""

    def flat(name: str) -> np.ndarray:
        return records[name].reshape(-1, 3)

    summary = {}
    for kind in {"linear", "angular"}:
        summary[kind] = {
            "head_all_intervals": _regression_metrics(
                flat(f"pred_interval_{kind}_acceleration"), flat(f"true_interval_{kind}_acceleration")
            ),
            "head_inner_frames": _regression_metrics(
                flat(f"pred_inner_{kind}_acceleration_from_head"), flat(f"true_inner_{kind}_acceleration")
            ),
            "velocity_difference_inner_frames": _regression_metrics(
                flat(f"pred_inner_{kind}_acceleration_from_twist"), flat(f"true_inner_{kind}_acceleration")
            ),
        }

    return summary


# Written next to the renders by ball_simulator.trajectories.shapes.write_appearance.
APPEARANCE_FILE = "appearance.json"


def _describe(value: np.ndarray) -> dict[str, float]:
    return {"mean": float(value.mean()), "median": float(np.median(value)),
            "p95": float(np.percentile(value, 95)), "maximum": float(value.max())}


def _symmetric_orientation_summary(records: dict[str, np.ndarray], symmetries: np.ndarray) -> dict[str, Any]:
    """Orientation errors in deg that ignore rotations under which the body looks identical.

    symmetries [G, 3, 3] body-frame rotations S; R and R S cannot be told apart in the images.
    """
    group = torch.from_numpy(symmetries)

    def error_deg(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        angle = symmetric_rotation_angle(torch.from_numpy(true), torch.from_numpy(predicted), group)
        return np.rad2deg(angle.numpy())

    true_after_first = np.concatenate(
        (records["true_frame_rotation"][:, 1:], records["true_future_rotation"]), axis=1
    )
    frames = error_deg(records["true_frame_rotation"], records["pred_frame_rotation"])
    rollout_first = error_deg(true_after_first, records["rollout_first_rotation"])
    rollout_current = error_deg(records["true_future_rotation"], records["rollout_current_rotation"])
    return {
        "symmetry_rotations": int(len(symmetries)),
        "current": _describe(frames[:, -1]),
        "all_frames": _describe(frames),
        "rollout_from_first_frame_median_vs_horizon": np.median(rollout_first, axis=0).tolist(),
        "rollout_from_current_state_median_vs_horizon": np.median(rollout_current, axis=0).tolist(),
    }



def _write_window_csv(path: Path, records: dict[str, np.ndarray]) -> None:
    fields = ["trajectory_id", "time", "orientation_error_deg", "rollout_position_error", "rollout_orientation_error_deg", "rollout_linear_velocity_error", "rollout_angular_velocity_error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index in range(len(records["trajectory_id"])):
            writer.writerow({field: records[field][index].item() if np.ndim(records[field][index]) == 0 else records[field][index] for field in fields})


def evaluate_kinematic_observer(
    checkpoint_path: str | Path,
    training_config_path: str | Path,
    output_directory: str | Path,
    settings: EvaluationSettings = EvaluationSettings(),
    rendering_config_path: str | Path | None = None,
) -> Path:
    """Evaluate the current-state estimate, whole-window estimates and rollouts.
 
    With ``rendering_config_path`` (the project's rendering.yaml, which holds the camera) the rollouts are also drawn on the rendered frames.
    """
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
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
    
    image_windows = settings.trajectory_plots if rendering_config_path is not None else 0
    records, images = _collect(
        module, loader, device, settings.maximum_test_windows, image_windows, data_config.image.normalization
    )
    
    summary = _summary(records)
    summary["checkpoint"] = str(checkpoint_path.resolve())
    summary["training_config"] = str(training_config_path.resolve())

    appearance_path = Path(data_config.root) / APPEARANCE_FILE
    if appearance_path.is_file():
        symmetries = json.loads(appearance_path.read_text(encoding="utf-8"))["symmetry_rotations"]
        if symmetries is None:
            summary["orientation_observable"] = False
        elif len(symmetries) > 1:
            summary["symmetric_orientation_error_deg"] = _symmetric_orientation_summary(
                records, np.asarray(symmetries, dtype=np.float32)
            )
            
    _save_json(output / "metrics.json", summary)
    _write_window_csv(output / "window_errors.csv", records)

    # Generate plots
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

    for kind, unit in (("linear", "m/s²"), ("angular", "rad/s²")):
        _component_scatter(
            output / f"interval_{kind}_acceleration_head.png",
            records[f"pred_interval_{kind}_acceleration"].reshape(-1, 3),
            records[f"true_interval_{kind}_acceleration"].reshape(-1, 3),
            f"{kind.capitalize()} acceleration, head, every interval", unit,
        )
        _component_scatter(
            output / f"inner_{kind}_acceleration_from_velocity_difference.png",
            records[f"pred_inner_{kind}_acceleration_from_twist"].reshape(-1, 3),
            records[f"true_inner_{kind}_acceleration"].reshape(-1, 3),
            f"{kind.capitalize()} acceleration, differenced velocities", unit,
        )

    trajectory_window_plots(output / "trajectories", records, settings.trajectory_plots)

    rollout_horizon_plot(output / "rollout_error_vs_horizon.png", records)
    
    if images is not None:
        camera = PinholeCamera.from_rendering_config(rendering_config_path)
        rollout_overlay_plots(output / "rollout_overlays", records, images, camera, settings.trajectory_plots, settings.ball_radius)
    return output
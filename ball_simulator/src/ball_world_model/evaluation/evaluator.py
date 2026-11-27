"""Evaluation of the two-body SE(3) observer in the anchor frame.

Every window is evaluated in its own anchor frame (the camera frame at the last context frame).
Object quantities keep the record and metric names of the fixed-camera evaluation; camera,
gravity, viewing-ray and per-camera-mode results are added. With a fixed camera the anchor frame
is the camera frame for every window, so norms and angles are directly comparable with
world-frame results; per-component R² refers to camera axes instead of world axes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ball_simulator.rendering.camera_axes import CameraIntrinsics
from ball_world_model.config import DataConfig
from ball_world_model.data.dataset import RenderedTrajectoryDataset, build_dataloader
from ball_world_model.data.transforms import denormalise_rgb
from ball_world_model.geometry.se3 import propagate_world, rollout_world, rotation_angle
from ball_world_model.geometry.symmetry import symmetric_rotation_angle
from ball_world_model.models.state import BodyPrediction, ObserverPrediction
from ball_world_model.training.module import SE3ObserverModule
from ball_world_model.training.targets import AnchorWindowTargets, anchor_window_targets

from .plots import (
    camera_mode_plot,
    horizon_statistics,
    rollout_horizon_plot,
    rollout_overlay_plots,
    trajectory_window_plots,
)


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    maximum_test_windows: int = 2_000
    trajectory_plots: int = 12
    seed: int = 20260824
    ball_radius: float = 0.10  # m, only used to draw the ball in the overlay figures


def _safe_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = np.square(prediction - target).sum(axis=0)
    total = np.square(target - target.mean(axis=0)).sum(axis=0)
    return np.array(
        [1.0 - r / t if t > 1.0e-12 else None for r, t in zip(residual, total, strict=True)],
        dtype=object,
    )


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
        return {str(key): _json_compatible(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]

    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()

        return _json_compatible(value.detach().cpu().tolist())

    if isinstance(value, Path):
        return str(value)

    return value


def _save_json(path: Path, value: Any) -> None:
    compatible_value = _json_compatible(value)
    path.write_text(
        json.dumps(compatible_value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


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


def _component_scatter(
    path: Path, prediction: np.ndarray, target: np.ndarray, name: str, unit: str
) -> None:
    labels = ("x", "y", "z")
    metrics = _regression_metrics(prediction, target)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    figure.suptitle(name)
    for axis, component in zip(axes, range(3), strict=True):
        x, y = target[:, component], prediction[:, component]
        axis.scatter(x, y, s=10, alpha=0.55)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        axis.plot([low, high], [low, high], color="black", linewidth=1.5)
        rmse_text = _format_metric(metrics["rmse"][component], ".4g")
        r2_text = _format_metric(metrics["r2"][component], ".3f")
        axis.set_title(f"{labels[component]}: RMSE={rmse_text}, R²={r2_text}")
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


def _error_vs_magnitude(
    path: Path, magnitude: np.ndarray, error: np.ndarray, x_label: str, y_label: str
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.scatter(magnitude, error, s=10, alpha=0.45)
    if len(magnitude) >= 20:
        edges = np.quantile(magnitude, np.linspace(0, 1, 11))
        centres, means = [], []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            mask = (magnitude >= left) & (magnitude <= right)
            if mask.any():
                centres.append(magnitude[mask].mean())
                means.append(error[mask].mean())
        axis.plot(centres, means, color="red", linewidth=2, marker="o", label="decile mean")
        axis.legend()
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _rollout_from_first_frame(
    body: BodyPrediction,
    dt: torch.Tensor,
    horizons: torch.Tensor,
    acceleration_model: str = "predicted",
    gravity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Open-loop rollout from the frame-0 estimate, anchor frame. horizons [B, H] s since frame 0.

    acceleration_model:
        "predicted"  learned linear and angular acceleration of the first interval,
        "gravity"    the predicted gravity vector [B, 3] as linear acceleration, no angular one,
        "none"       constant twist (the first interval-mean twist is held).
    """
    poses, twists, accelerations = (
        body.frame_poses,
        body.interval_twists,
        body.interval_accelerations,
    )
    half_step = 0.5 * dt[:, :1]
    if acceleration_model == "predicted":
        # The first interval mean sits half a step after frame 0; step back to frame 0.
        linear = twists.linear[:, 0] - half_step * accelerations.linear[:, 0]
        angular = twists.angular[:, 0] - half_step * accelerations.angular[:, 0]
        linear_acceleration, angular_acceleration = (
            accelerations.linear[:, 0],
            accelerations.angular[:, 0],
        )
    elif acceleration_model == "gravity":
        linear = twists.linear[:, 0] - half_step * gravity
        angular = twists.angular[:, 0]
        linear_acceleration, angular_acceleration = gravity, torch.zeros_like(angular)
    elif acceleration_model == "none":
        linear, angular = twists.linear[:, 0], twists.angular[:, 0]
        linear_acceleration, angular_acceleration = (
            torch.zeros_like(linear),
            torch.zeros_like(angular),
        )
    else:
        raise ValueError(f"Unknown acceleration_model: {acceleration_model!r}.")
    return rollout_world(
        poses.position[:, 0],
        poses.rotation[:, 0],
        linear,
        angular,
        linear_acceleration,
        angular_acceleration,
        horizons,
    )


def _twist_difference_acceleration(rates: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Acceleration implied by neighbouring interval-mean velocities, [B, I-1, 3].

    Baseline for the acceleration head: interval means sit at interval midpoints, so the time
    between two neighbouring means is the average of the two durations. Estimates the
    acceleration at the inner frames 1..T-2.
    """
    midpoint_spacing = 0.5 * (dt[:, 1:] + dt[:, :-1]).unsqueeze(-1)
    return (rates[:, 1:] - rates[:, :-1]) / midpoint_spacing


def _angle_deg(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(rotation_angle(first, second))


def _distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(first - second, dim=-1)


def _window_tensors(
    prediction: ObserverPrediction, targets: AnchorWindowTargets, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Per-window records of one batch; object keys keep their fixed-camera names (anchor frame)."""
    obj, camera = prediction.object, prediction.camera
    true_obj, true_camera = targets.object, targets.camera
    dt = prediction.interval_dt
    context_time, future_time = batch["context_time"].float(), batch["future_time"].float()
    current, current_target = obj.current, true_obj.current
    future = targets.object_future
    horizons = future.horizons
    zero = torch.zeros_like(current.twist.linear)

    def rollout(linear, angular, linear_acceleration, angular_acceleration):
        return rollout_world(
            current.pose.position,
            current.pose.rotation,
            linear,
            angular,
            linear_acceleration,
            angular_acceleration,
            horizons,
        )

    current_position, current_rotation = rollout(
        current.twist.linear,
        current.twist.angular,
        current.acceleration.linear,
        current.acceleration.angular,
    )
    still_position, still_rotation = rollout(
        current.twist.linear, current.twist.angular, zero, zero
    )
    # Gravity-only baseline with the *predicted* gravity: last interval mean plus half a step.
    half_step = 0.5 * dt[:, -1:]
    gravity_position, gravity_rotation = rollout(
        obj.interval_twists.linear[:, -1] + half_step * prediction.gravity,
        obj.interval_twists.angular[:, -1],
        prediction.gravity,
        zero,
    )

    first_horizon = torch.cat((context_time[:, 1:], future_time), dim=1) - context_time[:, :1]
    first_position, first_rotation = _rollout_from_first_frame(obj, dt, first_horizon)
    first_still_position, first_still_rotation = _rollout_from_first_frame(
        obj, dt, first_horizon, "none"
    )
    first_gravity_position, first_gravity_rotation = _rollout_from_first_frame(
        obj, dt, first_horizon, "gravity", prediction.gravity
    )
    true_after_first_position = torch.cat((true_obj.frame_position[:, 1:], future.position), dim=1)
    true_after_first_rotation = torch.cat((true_obj.frame_rotation[:, 1:], future.rotation), dim=1)

    # One step from the current state to the first future frame, with future velocities in the
    # anchor frame (velocities only rotate).
    to_anchor = targets.anchor_in_world.rotation.transpose(-1, -2)
    future_velocity = torch.einsum(
        "bij,bj->bi", to_anchor, batch["future_linear_velocity"][:, 0].float()
    )
    future_angular = torch.einsum(
        "bij,bj->bi", to_anchor, batch["future_angular_velocity"][:, 0].float()
    )
    next_position, next_rotation, next_velocity, next_angular = propagate_world(
        current.pose.position,
        current.pose.rotation,
        current.twist.linear,
        current.twist.angular,
        horizons[:, 0],
        current.acceleration.linear,
        current.acceleration.angular,
    )

    records = {
        "time": context_time[:, -1],
        "frame_time": context_time,
        "future_time": future_time,
        "interval_time": 0.5 * (context_time[:, 1:] + context_time[:, :-1]),
        # Object, current state at the last context frame (anchor frame).
        "pred_position": current.pose.position,
        "true_position": current_target.position,
        "pred_linear_velocity": current.twist.linear,
        "true_linear_velocity": current_target.linear_velocity,
        "pred_angular_velocity": current.twist.angular,
        "true_angular_velocity": current_target.angular_velocity,
        "pred_linear_acceleration": current.acceleration.linear,
        "true_linear_acceleration": current_target.linear_acceleration,
        "pred_angular_acceleration": current.acceleration.angular,
        "true_angular_acceleration": current_target.angular_acceleration,
        "orientation_error_deg": _angle_deg(current_target.rotation, current.pose.rotation),
        "rollout_position_error": _distance(next_position, future.position[:, 0]),
        "rollout_orientation_error_deg": _angle_deg(future.rotation[:, 0], next_rotation),
        "rollout_linear_velocity_error": _distance(next_velocity, future_velocity),
        "rollout_angular_velocity_error": _distance(next_angular, future_angular),
        # Object, whole window (anchor frame).
        "true_frame_position": true_obj.frame_position,
        "pred_frame_position": obj.frame_poses.position,
        "true_frame_rotation": true_obj.frame_rotation,
        "pred_frame_rotation": obj.frame_poses.rotation,
        "frame_orientation_error_deg": _angle_deg(
            true_obj.frame_rotation, obj.frame_poses.rotation
        ),
        "true_frame_linear_velocity": true_obj.frame_linear_velocity,
        "true_frame_angular_velocity": true_obj.frame_angular_velocity,
        "true_future_position": future.position,
        "true_future_rotation": future.rotation,
        # Object relative to the camera: the directly observed pose, for viewing-ray errors.
        "true_object_in_camera_position": targets.object_in_camera.position,
        "pred_object_in_camera_position": prediction.object_in_camera.position,
        "true_object_in_camera_rotation": targets.object_in_camera.rotation,
        "pred_object_in_camera_rotation": prediction.object_in_camera.rotation,
        # Camera relative to the anchor; true future poses draw future frames (own motion is given).
        "true_camera_frame_position": true_camera.frame_position,
        "pred_camera_frame_position": camera.frame_poses.position,
        "true_camera_frame_rotation": true_camera.frame_rotation,
        "pred_camera_frame_rotation": camera.frame_poses.rotation,
        "camera_frame_orientation_error_deg": _angle_deg(
            true_camera.frame_rotation, camera.frame_poses.rotation
        ),
        "true_future_camera_position": targets.camera_future.position,
        "true_future_camera_rotation": targets.camera_future.rotation,
        # Gravity in the anchor frame.
        "pred_gravity": prediction.gravity,
        "true_gravity": targets.gravity,
        "gravity_direction_error_deg": torch.rad2deg(
            torch.arccos(
                torch.nn.functional.cosine_similarity(
                    prediction.gravity, targets.gravity, dim=-1
                ).clamp(-1.0, 1.0)
            )
        ),
        # Rollouts of the object (anchor frame), with the baselines.
        "rollout_current_horizon": horizons,
        "rollout_current_position": current_position,
        "rollout_current_rotation": current_rotation,
        "rollout_current_position_error": _distance(current_position, future.position),
        "rollout_current_orientation_error_deg": _angle_deg(future.rotation, current_rotation),
        "rollout_current_no_acceleration_position_error": _distance(
            still_position, future.position
        ),
        "rollout_current_no_acceleration_orientation_error_deg": _angle_deg(
            future.rotation, still_rotation
        ),
        "rollout_current_gravity_only_position_error": _distance(gravity_position, future.position),
        "rollout_current_gravity_only_orientation_error_deg": _angle_deg(
            future.rotation, gravity_rotation
        ),
        "rollout_first_horizon": first_horizon,
        "rollout_first_position": first_position,
        "rollout_first_rotation": first_rotation,
        "rollout_first_position_error": _distance(first_position, true_after_first_position),
        "rollout_first_orientation_error_deg": _angle_deg(
            true_after_first_rotation, first_rotation
        ),
        "rollout_first_no_acceleration_position_error": _distance(
            first_still_position, true_after_first_position
        ),
        "rollout_first_no_acceleration_orientation_error_deg": _angle_deg(
            true_after_first_rotation, first_still_rotation
        ),
        "rollout_first_gravity_only_position_error": _distance(
            first_gravity_position, true_after_first_position
        ),
        "rollout_first_gravity_only_orientation_error_deg": _angle_deg(
            true_after_first_rotation, first_gravity_rotation
        ),
    }
    for body_name, body, target in (("", obj, true_obj), ("camera_", camera, true_camera)):
        for kind in ("linear", "angular"):
            records[f"true_{body_name}interval_{kind}_velocity"] = getattr(
                target, f"interval_{kind}_velocity"
            )
            records[f"pred_{body_name}interval_{kind}_velocity"] = getattr(
                body.interval_twists, kind
            )
            records[f"true_{body_name}interval_{kind}_acceleration"] = getattr(
                target, f"interval_{kind}_acceleration"
            )
            records[f"pred_{body_name}interval_{kind}_acceleration"] = getattr(
                body.interval_accelerations, kind
            )
    for kind in ("linear", "angular"):
        true_acceleration = getattr(true_obj, f"interval_{kind}_acceleration")
        predicted_acceleration = getattr(obj.interval_accelerations, kind)
        records[f"true_inner_{kind}_acceleration"] = 0.5 * (
            true_acceleration[:, 1:] + true_acceleration[:, :-1]
        )
        records[f"pred_inner_{kind}_acceleration_from_head"] = 0.5 * (
            predicted_acceleration[:, 1:] + predicted_acceleration[:, :-1]
        )
        records[f"pred_inner_{kind}_acceleration_from_twist"] = _twist_difference_acceleration(
            getattr(obj.interval_twists, kind), dt
        )
    return records


@torch.inference_mode()
def _collect(
    module: SE3ObserverModule,
    loader,
    device: torch.device,
    maximum: int,
    image_windows: int = 0,
    normalisation: str = "imagenet",
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Run the observer on up to ``maximum`` windows.

    Returns per-window records (NumPy, one row per window) and, for the first
    ``image_windows`` windows, the context + future frames as uint8 [M, T+F, H, W, 3].
    """
    values: dict[str, list[np.ndarray]] = {}
    images: list[np.ndarray] = []
    seen = 0
    for batch in loader:
        if seen >= maximum:
            break
        keep = min(len(batch["context_rgb"]), maximum - seen)
        device_batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        prediction = module.model(device_batch["context_rgb"], device_batch["context_time"])
        targets = anchor_window_targets(device_batch)
        for key, value in _window_tensors(prediction, targets, device_batch).items():
            values.setdefault(key, []).append(value[:keep].detach().float().cpu().numpy())
        values.setdefault("trajectory_id", []).append(
            np.asarray(batch["trajectory_id"][:keep], dtype=str)
        )
        values.setdefault("camera_mode", []).append(
            np.asarray(batch["camera_mode"][:keep], dtype=str)
        )

        stored = sum(len(part) for part in images)
        if stored < image_windows:
            take = min(keep, image_windows - stored)
            frames = torch.cat((batch["context_rgb"][:take], batch["future_rgb"][:take]), dim=1)
            frames = denormalise_rgb(frames, normalisation)  # [take, T+F, 3, H, W] in [0, 1]
            images.append((frames.permute(0, 1, 3, 4, 2) * 255).round().to(torch.uint8).numpy())
        seen += keep

    records = {key: np.concatenate(parts, axis=0) for key, parts in values.items()}
    return records, (np.concatenate(images, axis=0) if images else None)


def _describe(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(value.mean()),
        "median": float(np.median(value)),
        "p95": float(np.percentile(value, 95)),
        "maximum": float(value.max()),
    }


def viewing_ray_errors(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Split position errors of camera-frame positions [..., 3] along and across the true ray.

    along   signed error along the unit ray from the camera centre to the true position (m);
            positive = estimated too far away
    across  error perpendicular to the ray (m), the part the image constrains directly
    """
    ray = target / np.linalg.norm(target, axis=-1, keepdims=True)
    error = prediction - target
    along = np.sum(error * ray, axis=-1)
    across = np.linalg.norm(error - along[..., None] * ray, axis=-1)
    return {"along": along, "across": across}


def depth_scale(prediction: np.ndarray, target: np.ndarray) -> float:
    """One least-squares scale s minimising |s p_hat - p|^2 over camera-frame positions [N, 3].

    A monocular observer can only get distance up to the object's (assumed) size; s != 1 is a
    depth-scale error, which scaling along the rays removes.
    """
    return float(np.sum(prediction * target) / np.sum(prediction * prediction))


def _viewing_ray_summary(records: dict[str, np.ndarray]) -> dict[str, Any]:
    prediction = records["pred_object_in_camera_position"].reshape(-1, 3)
    target = records["true_object_in_camera_position"].reshape(-1, 3)
    errors = viewing_ray_errors(prediction, target)
    scale = depth_scale(prediction, target)
    return {
        "along_ray_m": _describe(np.abs(errors["along"])),
        "along_ray_bias_m": float(errors["along"].mean()),
        "across_ray_m": _describe(errors["across"]),
        "depth_scale": scale,
        "position_error_m": _describe(np.linalg.norm(prediction - target, axis=-1)),
        "scale_aligned_position_error_m": _describe(
            np.linalg.norm(scale * prediction - target, axis=-1)
        ),
    }


def _camera_mode_summary(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """Object accuracy per camera motion mode: it must not depend on how the camera moves."""
    summary = {}
    for mode in sorted(set(records["camera_mode"].tolist())):
        mask = records["camera_mode"] == mode
        summary[mode] = {
            "windows": int(mask.sum()),
            "position_error_m": _describe(
                np.linalg.norm(
                    records["pred_position"][mask] - records["true_position"][mask], axis=-1
                )
            ),
            "orientation_error_deg": _describe(records["orientation_error_deg"][mask]),
            "linear_velocity": _regression_metrics(
                records["pred_linear_velocity"][mask], records["true_linear_velocity"][mask]
            ),
            "angular_velocity": _regression_metrics(
                records["pred_angular_velocity"][mask], records["true_angular_velocity"][mask]
            ),
            "linear_acceleration": _regression_metrics(
                records["pred_linear_acceleration"][mask], records["true_linear_acceleration"][mask]
            ),
            "rollout_final_position_error_m": _describe(
                records["rollout_current_position_error"][mask][:, -1]
            ),
        }
    return summary


def _camera_summary(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """Camera relative to the anchor: frame poses (all but the anchor frame) and interval motion."""

    def frames(name: str) -> np.ndarray:
        return records[name][:, :-1].reshape(-1, 3)

    summary = {
        "position": _regression_metrics(
            frames("pred_camera_frame_position"), frames("true_camera_frame_position")
        ),
        "orientation_error_deg": _describe(records["camera_frame_orientation_error_deg"][:, :-1]),
    }
    for kind in ("linear", "angular"):
        for quantity in ("velocity", "acceleration"):
            summary[f"{kind}_{quantity}"] = _regression_metrics(
                records[f"pred_camera_interval_{kind}_{quantity}"].reshape(-1, 3),
                records[f"true_camera_interval_{kind}_{quantity}"].reshape(-1, 3),
            )
    return summary


def _summary(records: dict[str, np.ndarray]) -> dict[str, Any]:
    summary = {
        "windows": int(len(records["true_position"])),
        "frame": "anchor (camera frame at the last context frame, OpenCV axes)",
        "position": _regression_metrics(records["pred_position"], records["true_position"]),
        "linear_velocity": _regression_metrics(
            records["pred_linear_velocity"], records["true_linear_velocity"]
        ),
        "angular_velocity": _regression_metrics(
            records["pred_angular_velocity"], records["true_angular_velocity"]
        ),
        "linear_acceleration": _regression_metrics(
            records["pred_linear_acceleration"], records["true_linear_acceleration"]
        ),
        "angular_acceleration": _regression_metrics(
            records["pred_angular_acceleration"], records["true_angular_acceleration"]
        ),
    }
    for name in (
        "orientation_error_deg",
        "rollout_position_error",
        "rollout_orientation_error_deg",
        "rollout_linear_velocity_error",
        "rollout_angular_velocity_error",
    ):
        summary[name] = _describe(records[name])

    summary["rollout_vs_horizon"] = horizon_statistics(records)
    summary["interval_acceleration"] = _interval_acceleration_summary(records)
    summary["camera"] = _camera_summary(records)
    summary["gravity_direction_error_deg"] = _describe(records["gravity_direction_error_deg"])
    summary["gravity_magnitude_error"] = _describe(
        np.abs(
            np.linalg.norm(records["pred_gravity"], axis=-1)
            - np.linalg.norm(records["true_gravity"], axis=-1)
        )
    )
    summary["viewing_ray"] = _viewing_ray_summary(records)
    summary["by_camera_mode"] = _camera_mode_summary(records)
    return summary


def _interval_acceleration_summary(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """Object acceleration over every interval; at inner frames head vs differenced velocities."""

    def flat(name: str) -> np.ndarray:
        return records[name].reshape(-1, 3)

    summary = {}
    for kind in ("linear", "angular"):
        summary[kind] = {
            "head_all_intervals": _regression_metrics(
                flat(f"pred_interval_{kind}_acceleration"),
                flat(f"true_interval_{kind}_acceleration"),
            ),
            "head_inner_frames": _regression_metrics(
                flat(f"pred_inner_{kind}_acceleration_from_head"),
                flat(f"true_inner_{kind}_acceleration"),
            ),
            "velocity_difference_inner_frames": _regression_metrics(
                flat(f"pred_inner_{kind}_acceleration_from_twist"),
                flat(f"true_inner_{kind}_acceleration"),
            ),
        }
    return summary


# Written next to the renders by ball_simulator.trajectories.shapes.write_appearance.
APPEARANCE_FILE = "appearance.json"


def _symmetric_orientation_summary(
    records: dict[str, np.ndarray], symmetries: np.ndarray
) -> dict[str, Any]:
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
    rollout_current = error_deg(
        records["true_future_rotation"], records["rollout_current_rotation"]
    )
    return {
        "symmetry_rotations": int(len(symmetries)),
        "current": _describe(frames[:, -1]),
        "all_frames": _describe(frames),
        "rollout_from_first_frame_median_vs_horizon": np.median(rollout_first, axis=0).tolist(),
        "rollout_from_current_state_median_vs_horizon": np.median(rollout_current, axis=0).tolist(),
    }


def _write_window_csv(path: Path, records: dict[str, np.ndarray]) -> None:
    fields = [
        "trajectory_id",
        "camera_mode",
        "time",
        "orientation_error_deg",
        "rollout_position_error",
        "rollout_orientation_error_deg",
        "rollout_linear_velocity_error",
        "rollout_angular_velocity_error",
        "gravity_direction_error_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(records["trajectory_id"])):
            writer.writerow({field: records[field][index].item() for field in fields})


def evaluate_kinematic_observer(
    checkpoint_path: str | Path,
    training_config_path: str | Path,
    output_directory: str | Path,
    settings: EvaluationSettings | None = None,
    rendering_config_path: str | Path | None = None,
    split: str = "test",
) -> Path:
    """Evaluate both bodies, gravity and the object rollouts, per window in its anchor frame.

    ``split`` is train, validation, test, or all (every trajectory, for held-out datasets).
    With ``rendering_config_path`` (the dataset's rendering.yaml, for the intrinsics) estimates
    and rollouts are also drawn on the rendered frames, using the per-frame camera poses.
    """
    settings = settings or EvaluationSettings()
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    checkpoint_path, training_config_path = Path(checkpoint_path), Path(training_config_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    configuration = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    data_config = DataConfig.from_yaml(Path(configuration["data_config"]))
    manifest = data_config.manifest_path or data_config.root / "manifest.parquet"
    dataset = RenderedTrajectoryDataset(manifest, data_config, split=split)
    loader = build_dataloader(dataset, data_config, shuffle=False)
    module = SE3ObserverModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device).eval()

    image_windows = settings.trajectory_plots if rendering_config_path is not None else 0
    records, images = _collect(
        module,
        loader,
        device,
        settings.maximum_test_windows,
        image_windows,
        data_config.image.normalization,
    )

    summary = _summary(records)
    summary["checkpoint"] = str(checkpoint_path.resolve())
    summary["training_config"] = str(training_config_path.resolve())
    summary["split"] = split

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

    for name, unit in (
        ("position", "m"),
        ("linear_velocity", "m/s"),
        ("angular_velocity", "rad/s"),
        ("linear_acceleration", "m/s²"),
        ("angular_acceleration", "rad/s²"),
    ):
        _component_scatter(
            output / f"{name}_components.png",
            records[f"pred_{name}"],
            records[f"true_{name}"],
            f"Object {name.replace('_', ' ')}, anchor frame",
            unit,
        )
    for kind, unit in (("linear", "m/s"), ("angular", "rad/s")):
        _component_scatter(
            output / f"camera_{kind}_velocity_components.png",
            records[f"pred_camera_interval_{kind}_velocity"].reshape(-1, 3),
            records[f"true_camera_interval_{kind}_velocity"].reshape(-1, 3),
            f"Camera {kind} velocity, every interval, anchor frame",
            unit,
        )
    _component_scatter(
        output / "gravity_components.png",
        records["pred_gravity"],
        records["true_gravity"],
        "Gravity, anchor frame",
        "m/s²",
    )

    ray = viewing_ray_errors(
        records["pred_object_in_camera_position"], records["true_object_in_camera_position"]
    )
    _error_histogram(
        output / "error_distributions.png",
        {
            "Object position norm error": (
                np.linalg.norm(records["pred_position"] - records["true_position"], axis=1),
                "m",
            ),
            "Object orientation error": (records["orientation_error_deg"], "deg"),
            "Along viewing ray (signed)": (ray["along"].reshape(-1), "m"),
            "Across viewing ray": (ray["across"].reshape(-1), "m"),
            "Gravity direction error": (records["gravity_direction_error_deg"], "deg"),
            "Camera orientation error": (
                records["camera_frame_orientation_error_deg"][:, :-1].reshape(-1),
                "deg",
            ),
        },
    )
    _error_vs_magnitude(
        output / "linear_velocity_error_vs_speed.png",
        np.linalg.norm(records["true_linear_velocity"], axis=1),
        np.linalg.norm(records["pred_linear_velocity"] - records["true_linear_velocity"], axis=1),
        "True speed [m/s]",
        "Velocity error [m/s]",
    )
    for kind, unit in (("linear", "m/s²"), ("angular", "rad/s²")):
        _component_scatter(
            output / f"interval_{kind}_acceleration_head.png",
            records[f"pred_interval_{kind}_acceleration"].reshape(-1, 3),
            records[f"true_interval_{kind}_acceleration"].reshape(-1, 3),
            f"Object {kind} acceleration, head, every interval",
            unit,
        )
        _component_scatter(
            output / f"inner_{kind}_acceleration_from_velocity_difference.png",
            records[f"pred_inner_{kind}_acceleration_from_twist"].reshape(-1, 3),
            records[f"true_inner_{kind}_acceleration"].reshape(-1, 3),
            f"Object {kind} acceleration, differenced velocities",
            unit,
        )

    camera_mode_plot(output / "object_error_by_camera_mode.png", records)
    trajectory_window_plots(output / "trajectories", records, settings.trajectory_plots)
    rollout_horizon_plot(output / "rollout_error_vs_horizon.png", records)
    if images is not None:
        render = yaml.safe_load(Path(rendering_config_path).read_text(encoding="utf-8"))
        rollout_overlay_plots(
            output / "rollout_overlays",
            records,
            images,
            CameraIntrinsics.from_render_config(render),
            settings.trajectory_plots,
            settings.ball_radius,
        )
    return output

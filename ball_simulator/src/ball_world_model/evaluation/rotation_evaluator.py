from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import torch

from ball_world_model.models.rotation import quaternion_xyzw_to_matrix
from .metrics import apply_linear_probe, effective_rank, fit_linear_probe, regression_metrics
from .model_loader import denormalised_prediction
from .structured_so3_evaluator import evaluate_structured_so3_latent


def _numpy(value):
    return value.detach().cpu().numpy()


def _batches(loader, maximum):
    consumed = 0
    for batch in loader:
        if consumed >= maximum:
            return
        size = batch["context_rgb"].shape[0]
        keep = min(size, maximum - consumed)
        if keep < size:
            batch = {
                key: value[:keep] if isinstance(value, (torch.Tensor, list, tuple)) else value
                for key, value in batch.items()
            }
        consumed += keep
        yield batch


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _orientation_error(target, predicted):
    relative = np.swapaxes(predicted, -1, -2) @ target
    return np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=-1)

def _matrix_to_6d(matrix):
    return np.swapaxes(matrix[..., :, :2], -1, -2).reshape(matrix.shape[:-2] + (6,))

def _six_to_matrix(value):
    first, second = value[..., :3], value[..., 3:]
    first = first / np.clip(np.linalg.norm(first, axis=-2, keepdims=True), 1e-12, None)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.clip(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12, None)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


@torch.inference_mode()
def _collect(module, loader, device, maximum):
    records = []
    for batch in _batches(loader, maximum):
        images = batch["context_rgb"].to(device)
        time = batch["context_time"].to(device)
        prediction = module.model(images, time)
        decoded = denormalised_prediction(module, prediction)
        target_rotation = _numpy(
            quaternion_xyzw_to_matrix(batch["context_quaternion_xyzw"].to(device))
        )
        for index in range(len(images)):
            records.append(
                {
                    "trajectory_id": batch["trajectory_id"][index],
                    "start_frame": int(batch["start_frame"][index]),
                    "time": _numpy(time[index]),
                    "target_rotation": target_rotation[index],
                    "predicted_rotation": _numpy(decoded["rotation_matrix"][index]),
                    "target_omega": _numpy(batch["context_angular_velocity"][index]),
                    "predicted_omega": _numpy(decoded["angular_velocity"][index]),
                }
            )
    return records


def _scatter(target, predicted, output):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for component, label in enumerate(("x", "y", "z")):
        truth, estimate = target[:, component], predicted[:, component]
        low, high = min(truth.min(), estimate.min()), max(truth.max(), estimate.max())
        axes[component].scatter(truth, estimate, s=7, alpha=0.3)
        axes[component].plot([low, high], [low, high], color="black")
        metric = regression_metrics(truth, estimate)
        axes[component].set_title(
            f"omega {label}: RMSE={metric['rmse']:.4g}, R2={metric['r2']:.3f}"
        )
        axes[component].set_xlabel("True [rad/s]")
        axes[component].set_ylabel("Predicted [rad/s]")
        axes[component].grid(alpha=0.2)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _trajectory_plot(record, output, title):
    time = record["time"]
    target_rotvec = Rotation.from_matrix(record["target_rotation"]).as_rotvec()
    predicted_rotvec = Rotation.from_matrix(record["predicted_rotation"]).as_rotvec()
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for component, label in enumerate(("x", "y", "z")):
        axes[0].plot(time, target_rotvec[:, component], label=f"true {label}")
        axes[0].plot(time, predicted_rotvec[:, component], "--", label=f"pred {label}")
        axes[1].plot(time, record["target_omega"][:, component], label=f"true {label}")
        axes[1].plot(time, record["predicted_omega"][:, component], "--", label=f"pred {label}")
    axes[0].set_ylabel("Orientation rotvec [rad]")
    axes[1].set_ylabel("Angular velocity [rad/s]")
    orientation = np.rad2deg(
        _orientation_error(record["target_rotation"], record["predicted_rotation"])
    )
    omega = np.linalg.norm(record["predicted_omega"] - record["target_omega"], axis=-1)
    axes[2].plot(time, orientation, label="orientation error [deg]")
    axes[2].plot(time, omega, label="omega error [rad/s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncols=3)
    axes[2].set_xlabel("Time [s]")
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _aggregate(records, output):
    target_rotation = np.concatenate([record["target_rotation"] for record in records])
    predicted_rotation = np.concatenate([record["predicted_rotation"] for record in records])
    error = _orientation_error(target_rotation, predicted_rotation)
    target_omega = np.concatenate([record["target_omega"] for record in records])
    predicted_omega = np.concatenate([record["predicted_omega"] for record in records])
    _scatter(target_omega, predicted_omega, output / "angular_velocity_component_scatter.png")
    orientation = {
        "mean_deg": float(np.rad2deg(error.mean())),
        "median_deg": float(np.rad2deg(np.median(error))),
        "p95_deg": float(np.rad2deg(np.quantile(error, 0.95))),
        "maximum_deg": float(np.rad2deg(error.max())),
        "count": int(error.size),
    }
    rows: list[dict[str, object]] = []
    for metric_name, metric_value in orientation.items():
        rows.append(
            {
                "quantity": "orientation",
                "component": "all",
                "metric": metric_name,
                "value": metric_value,
            }
        )

    omega_summary = {}
    for component, axis in enumerate(("x", "y", "z")):
        metrics = regression_metrics(target_omega[:, component], predicted_omega[:, component])
        omega_summary[axis] = metrics
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "quantity": "angular_velocity", 
                    "component": axis,
                    "metric": metric_name,
                    "value": metric_value,
                })

    _write_csv(output / "aggregate_metrics.csv", rows)
    return {"orientation": orientation, "angular_velocity": omega_summary}

def _trajectory_reports(records, output, count, seed):
    scored = []
    for record in records:
        score = _orientation_error(
            record["target_rotation"], record["predicted_rotation"]
        ).mean()
        score += np.linalg.norm(
            record["predicted_omega"] - record["target_omega"], axis=-1
        ).mean()
        scored.append((float(score), record))
    scored.sort(key=lambda item: item[0])
    selected = []
    if scored:
        selected += [("best", scored[0][1]), ("median", scored[len(scored)//2][1]), ("worst", scored[-1][1])]
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(scored), min(count, len(scored)), replace=False)
        selected += [(f"random_{n:02d}", scored[i][1]) for n, i in enumerate(indices)]
    for label, record in selected:
        _trajectory_plot(
            record,
            output / f"{label}_{record['trajectory_id']}_{record['start_frame']:06d}.png",
            f"{label}: trajectory {record['trajectory_id']}, start {record['start_frame']}",
        )


@torch.inference_mode()
def _interventions(module, loader, device, maximum, seed):
    generator = torch.Generator().manual_seed(seed)
    estimates = {name: [] for name in ("forward", "reversed", "repeated_last", "shuffled")}
    motions = {name: [] for name in estimates}
    for batch in _batches(loader, maximum):
        images = batch["context_rgb"]
        time = batch["context_time"] - batch["context_time"][:, :1]
        permutation = torch.randperm(images.shape[1], generator=generator)
        variants = {
            "forward": images,
            "reversed": torch.flip(images, dims=(1,)),
            "repeated_last": images[:, -1:].expand_as(images),
            "shuffled": images[:, permutation],
        }
        for name, variant in variants.items():
            prediction = module.model(variant.to(device), time.to(device))
            decoded = denormalised_prediction(module, prediction)
            estimates[name].append(_numpy(decoded["angular_velocity"][:, :-1]))
            motions[name].append(_numpy(prediction.motion.forward_motion))
    values = {name: np.concatenate(items).reshape(-1, 3) for name, items in estimates.items()}
    latents = {name: np.concatenate(items).reshape(-1, items[0].shape[-1]) for name, items in motions.items()}
    forward, forward_latent = values["forward"], latents["forward"]
    return [
        {
            "intervention": name,
            "mean_angular_speed": float(np.linalg.norm(value, axis=-1).mean()),
            "omega_delta_from_forward": float(np.linalg.norm(value - forward, axis=-1).mean()),
            "omega_reversal_error": float(np.linalg.norm(value + forward, axis=-1).mean()),
            "motion_delta_from_forward": float(np.linalg.norm(latents[name] - forward_latent, axis=-1).mean()),
            "count": len(value),
        }
        for name, value in values.items()
    ]


@torch.inference_mode()
def _representations(module, loader, device, maximum):
    features = {name: [] for name in (
        "content_last", "content_difference", "spatial_map_last_mean",
        "spatial_feature_rate_mean", "motion_forward_mean", "motion_backward_mean",
        "predicted_next_last",
    )}
    orientation, omega = [], []
    for batch in _batches(loader, maximum):
        prediction = module.model(
            batch["context_rgb"].to(device), batch["context_time"].to(device)
        )
        content, maps, motion = prediction.frame_latent, prediction.feature_maps, prediction.motion
        features["content_last"].append(_numpy(content[:, -1]))
        features["content_difference"].append(_numpy(content[:, -1] - content[:, 0]))
        features["spatial_map_last_mean"].append(_numpy(maps[:, -1].mean(dim=(-1, -2))))
        features["spatial_feature_rate_mean"].append(_numpy(motion.normalised_forward_difference.mean(dim=(1, 3, 4))))
        features["motion_forward_mean"].append(_numpy(motion.forward_motion.mean(dim=1)))
        features["motion_backward_mean"].append(_numpy(motion.backward_motion.mean(dim=1)))
        features["predicted_next_last"].append(_numpy(motion.predicted_next_embedding[:, -1]))
        matrix = quaternion_xyzw_to_matrix(batch["context_quaternion_xyzw"][:, -1])
        orientation.append(_matrix_to_6d(_numpy(matrix)))
        omega.append(_numpy(batch["context_angular_velocity"][:, -1]))
    joined = {name: np.concatenate(value) for name, value in features.items()}
    targets = {"orientation": np.concatenate(orientation), "angular_velocity": np.concatenate(omega)}
    statistics = [
        {
            "representation": name,
            "mean_feature_std": float(value.std(axis=0).mean()),
            "effective_rank": effective_rank(value),
        }
        for name, value in joined.items()
    ]
    return joined, targets, statistics


def _probes(module, train_loader, test_loader, device, train_maximum, test_maximum, output):
    train_features, train_targets, _ = _representations(module, train_loader, device, train_maximum)
    test_features, test_targets, statistics = _representations(module, test_loader, device, test_maximum)
    rows = []
    for representation in train_features:
        orientation_weights = fit_linear_probe(train_features[representation], train_targets["orientation"])
        orientation_estimate = apply_linear_probe(test_features[representation], orientation_weights)
        error = _orientation_error(_six_to_matrix(test_targets["orientation"]), _six_to_matrix(orientation_estimate))
        rows.append(
            {
                "representation": representation,
                "quantity": "orientation",
                "score": float(np.rad2deg(error.mean())),
                "mean_deg": float(np.rad2deg(error.mean())),
                "median_deg": float(np.rad2deg(np.median(error))),
                "p95_deg": float(np.rad2deg(np.quantile(error, 0.95))),
            }
        )
        omega_weights = fit_linear_probe(train_features[representation], train_targets["angular_velocity"])
        estimate = apply_linear_probe(test_features[representation], omega_weights)
        metrics = [regression_metrics(test_targets["angular_velocity"][:, i], estimate[:, i]) for i in range(3)]
        rows.append(
            {
                "representation": representation,
                "quantity": "angular_velocity",
                "score": float(np.mean([metric["r2"] for metric in metrics])),
                "r2_x": metrics[0]["r2"], "r2_y": metrics[1]["r2"], "r2_z": metrics[2]["r2"],
                "rmse_x": metrics[0]["rmse"], "rmse_y": metrics[1]["rmse"], "rmse_z": metrics[2]["rmse"],
            }
        )
    _write_csv(output / "layerwise_linear_probes.csv", rows)
    _write_csv(output / "representation_statistics.csv", statistics)
    return rows, statistics


def evaluate_loaded_rotation_observer(
    *, module, train_loader, test_loader, device, checkpoint_path,
    training_config_path, output, settings,
):
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    records = _collect(module, test_loader, device, settings.maximum_test_windows)
    aggregate = _aggregate(records, output)
    _trajectory_reports(records, output / "trajectories", settings.trajectory_plots, settings.seed)
    intervention_rows = _interventions(module, test_loader, device, settings.maximum_test_windows, settings.seed)
    _write_csv(output / "interventions.csv", intervention_rows)
    probes, statistics = _probes(
        module, train_loader, test_loader, device,
        settings.maximum_probe_train_windows, settings.maximum_test_windows,
        output / "probes",
    )
    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "training_config": str(Path(training_config_path).resolve()),
        "task": "rotation",
        "device": str(device),
        "test_windows": len(records),
        "aggregate": aggregate,
        "interventions": intervention_rows,
        "probes": probes,
        "representation_statistics": statistics,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # test SO3 latent if available
    if hasattr(module.model, "context_head"):
        evaluate_structured_so3_latent(
            module,
            train_loader,
            test_loader,
            device,
            output,
            train_maximum=settings.maximum_probe_train_windows,
            test_maximum=settings.maximum_test_windows,
        )
    return output.resolve()
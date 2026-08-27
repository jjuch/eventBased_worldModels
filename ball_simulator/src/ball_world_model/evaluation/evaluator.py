from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from ball_world_model.config import DataConfig
from ball_world_model.data.dataset import RenderedTrajectoryDataset, build_dataloader
from ball_world_model.models.rotation import (
    quaternion_xyzw_to_matrix,
    rotation_geodesic_error,
)
from .metrics import (
    apply_linear_probe,
    fit_linear_probe,
    regression_metrics,
    effective_rank,
)
from .model_loader import denormalised_prediction, load_kinematic_module
from .plots import component_scatter, error_vs_speed, probe_plot, trajectory_plot


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    maximum_test_windows: int = 2_000
    maximum_probe_train_windows: int = 5_000
    trajectory_plots: int = 12
    seed: int = 20260824


def _configuration(training_config_path: Path):
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    data_config = DataConfig.from_yaml(Path(training_config["data_config"]))
    manifest = data_config.manifest_path or data_config.root / "manifest.parquet"
    return training_config, data_config, manifest


def _limit_loader(loader, maximum: int):
    consumed = 0
    for batch in loader:
        if consumed >= maximum:
            break
        batch_size = batch["context_rgb"].shape[0]
        if consumed + batch_size > maximum:
            keep = maximum - consumed
            batch = {
                key: value[:keep] if isinstance(value, torch.Tensor) else value[:keep]
                for key, value in batch.items()
            }
            batch_size = keep
        consumed += batch_size
        yield batch


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@torch.inference_mode()
def collect_predictions(module, loader, device, maximum_windows: int):
    records = []
    for batch in _limit_loader(loader, maximum_windows):
        images = batch["context_rgb"].to(device)
        time = batch["context_time"].to(device)
        prediction = module.model(images, time)
        decoded = denormalised_prediction(module, prediction)
        for index in range(len(images)):
            records.append(
                {
                    "trajectory_id": batch["trajectory_id"][index],
                    "start_frame": int(batch["start_frame"][index]),
                    "time": _numpy(time[index]),
                    "target_position": _numpy(batch["context_position"][index]),
                    "target_velocity": _numpy(batch["context_linear_velocity"][index]),
                    "target_quaternion": _numpy(batch["context_quaternion_xyzw"][index]),
                    "predicted_position": _numpy(decoded["position"][index]),
                    "predicted_velocity": _numpy(decoded["linear_velocity"][index])
                }
            )
    return records

def aggregate_report(records: list[dict], output: Path) -> dict:
    rows = []
    summary = {}
    for quantity, target_key, prediction_key, unit in (
        ("position", "target_position", "predicted_position", "m"),
        ("velocity", "target_velocity", "predicted_velocity", "m/s"),
    ):
        available = [record for record in records if prediction_key in record]
        if not available:
            continue
        target = np.concatenate([record[target_key] for record in available], axis=0)
        prediction = np.concatenate([record[prediction_key] for record in available], axis=0)
        component_scatter(
            target,
            prediction,
            quantity=quantity.title(),
            unit=unit,
            output=output / f"{quantity}_component_scatter.png",
        )
        components = {}
        for component, label in enumerate(("x", "y", "z")):
            metrics = regression_metrics(target[:, component], prediction[:, component])
            components[label] = metrics
            rows.append({"quantity": quantity, "component": label, **metrics})
        summary[quantity] = components
    _csv(output / "aggregate_metrics.csv", rows)
    error_vs_speed(records, output / "error_vs_speed.png")
    return summary


def trajectory_reports(records: list[dict], output: Path, count: int, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scored = []
    for record in records:
        errors = []
        if "predicted_position" in record:
            errors.append(
                np.mean(np.linalg.norm(record["predicted_position"] - record["target_position"], axis=-1))
            )
        if "predicted_linear_velocity" in record:
            errors.append(
                np.mean(
                    np.linalg.norm(record["predicted_velocity"] - record["target_velocity"], axis=-1)
                )
            )
        scored.append((float(np.mean(errors)), record))
    scored.sort(key=lambda item: item[0])

    selected = []
    if scored:
        selected.extend(
            [
                ("best", scored[0][1]),
                ("median", scored[len(scored) // 2][1]),
                ("worst", scored[-1][1]),
            ]
        )
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(scored), size=min(count, len(scored)), replace=False)
        selected.extend((f"random_{number:02d}", scored[index][1]) for number, index in enumerate(indices))

    for label, record in selected:
        trajectory_plot(
            record,
            time=record["time"],
            output=output / f"{label}_{record['trajectory_id']}_{record['start_frame']:06d}.png",
            title=f"{label}: trajectory {record['trajectory_id']}, start {record['start_frame']}",
        )


@torch.inference_mode()
def intervention_report(module, loader, device, maximum_windows: int, seed: int) -> list[dict]:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    accumulators: dict[str, list[np.ndarray]] = {
        "forward": [],
        "reversed": [],
        "repeated_last": [],
        "shuffled": [],
    }
    motion_latents: dict[str, list[np.ndarray]]
    for batch in _limit_loader(loader, maximum_windows):
        images = batch["context_rgb"]
        relative_time = batch["context_time"] - batch["context_time"][:, :1]
        permutation = torch.randperm(images.shape[1], generator=rng)
        variants = {
            "forward": images,
            "reversed": torch.flip(images, dims=(1,)),
            "repeated_last": images[:, -1:].expand_as(images),
            "shuffled": images[:, permutation],
        }
        for name, variant in variants.items():
            prediction = module.model(variant.to(device), relative_time.to(device))
            decoded = denormalised_prediction(module, prediction)
            if "linear_velocity" in decoded:
                # Mean over time is robust to the changed endpoint in reversed sequences.
                accumulators[name].append(_numpy(decoded["linear_velocity"][:, :-1]))
            motion_latents[name].append(_numpy(prediction.motion.forward_motion))

    values = {name: np.concatenate(items).reshape(-1, 3) for name, items in accumulators.items()}
    latents = {
        name: np.concatenate(items).reshape(-1, items[0].shape[-1])
        for name, items in motion_latents.items()
    }

    forward = values["forward"]
    forward_motion = latents["forward"]
    rows = []
    for name in values:
        estimate = values[name]
        motion = latents[name]
        rows.append(
            {
                "intervention": name,
                "mean_speed": float(np.mean(np.linalg.norm(estimate, axis=-1))),
                "velocity_delta_from_forward": float(np.mean(np.linalg.norm(estimate - forward, axis=-1))),
                "velocity_reversal_error": float(np.mean(np.linalg.norm(estimate + forward, axis=-1))),
                "motion_delta_from_forward": float(np.mean(np.linalg.norm(motion - forward_motion, axis=-1))),
                "count": len(estimate),

            }
        )
    return rows


@torch.inference_mode()
def extract_representations(module, loader, device, maximum_windows: int):
    features: dict[str, list[np.ndarray]] = {
        "content_last": [],
        "content_difference": [],
        "spatial_map_last_mean": [],
        "spatial_feature_rate_mean": [],
        "motion_forward_mean": [],
        "motion_backward_mean": [],
        "predicted_next_last": [],
    }
    targets = {"position": [], "velocity": []}
    diagnostics = {"representation": [], "mean_std": [], "effective_rank": []}

    for batch in _limit_loader(loader, maximum_windows):
        images = batch["context_rgb"].to(device)
        time = batch["context_time"].to(device)
        prediction = module.model(images, time)
        maps = prediction.feature_maps
        content = prediction.frame_latent
        motion = prediction.motion

        features["content_last"].append(_numpy(content[:, -1]))
        features["content_difference"].append(_numpy(content[:, -1] - content[:, 0]))
        features["spatial_map_last_mean"].append(_numpy(maps[:, -1].mean(dim=(-1, -2))))
        normalised_difference = getattr(
            motion,
            "normalised_forward_difference",
            getattr(motion, "normalised_forward_difference", None),
        )
        if normalised_difference is None:
            raise AttributeError(
                "MotionDiagnostics exposes neither normalised_forward_difference "
                "nor normalized_forward_difference."
            )
        features["spatial_feature_rate_mean"].append(
            _numpy(normalised_difference.mean(dim=(1, 3, 4)))
        )
        features["motion_forward_mean"].append(_numpy(motion.forward_motion.mean(dim=1)))
        features["motion_backward_mean"].append(_numpy(motion.backward_motion.mean(dim=1)))
        features["predicted_next_last"].append(_numpy(motion.predicted_next_embedding[:, -1]))
        targets["position"].append(_numpy(batch["context_position"][:, -1]))
        targets["velocity"].append(_numpy(batch["context_linear_velocity"][:, -1]))

    joined = {name: np.concatenate(values) for name, values in features.items()}
    joined_targets = {name: np.concatenate(values) for name, values in targets.items()}
    for name, value in joined.items():
        diagnostics["representation"].append(name)
        diagnostics["mean_std"].append(float(value.std(axis=0).mean()))
        diagnostics["effective_rank"].append(effective_rank(value))
    return joined, joined_targets, diagnostics


def probe_report(module, train_loader, test_loader, device, train_maximum: int, test_maximum: int, output: Path):
    train_features, train_targets, _ = extract_representations(module, train_loader, device, train_maximum)
    test_features, test_targets, diagnostics = extract_representations(module, test_loader, device, test_maximum)
    rows = []
    for representation in train_features:
        for quantity in ("position", "velocity"):
            weights = fit_linear_probe(train_features[representation], train_targets[quantity])
            estimate = apply_linear_probe(test_features[representation], weights)
            row = {"representation": representation, "quantity": quantity}
            r2_values, rmse_values = [], []

            for component, label in enumerate(("x", "y", "z")):
                metrics = regression_metrics(
                    test_targets[quantity][:, component], estimate[:, component]
                )
                row[f"r2_{label}"] = metrics["r2"]
                row[f"rmse_{label}"] = metrics["rmse"]
                r2_values.append(metrics["r2"])
                rmse_values.append(metrics["rmse"])
            row["r2_mean"] = float(np.nanmean(r2_values))
            row["rmse_mean"] = float(np.mean(rmse_values))
            rows.append(row)
    _csv(output / "layerwise_linear_probes.csv", rows)

    activation_rows = [
        {
            "representation": name,
            "mean_feature_std": std,
            "effective_rank": rank,
        }
        for name, std, rank in zip(
            diagnostics["representation"], diagnostics["mean_std"], diagnostics["effective_rank"]
        )
    ]
    _csv(output / "representation_statistics.csv", activation_rows)
    probe_plot(rows, output / "layerwise_linear_probes.png")
    return rows, activation_rows


def evaluate_kinematic_observer(
    *,
    checkpoint_path: str | Path,
    training_config_path: str | Path,
    output_directory: str | Path,
    settings: EvaluationSettings = EvaluationSettings(),
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _, data_config, manifest = _configuration(Path(training_config_path))
    test_data = RenderedTrajectoryDataset(manifest, data_config, split="test")
    train_data = RenderedTrajectoryDataset(manifest, data_config, split="train")
    test_loader = build_dataloader(test_data, data_config, shuffle=False)
    train_loader = build_dataloader(train_data, data_config, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_kinematic_module(checkpoint_path, device=device)

    records = collect_predictions(module, test_loader, device, settings.maximum_test_windows)
    summary = aggregate_report(records, output)
    trajectory_reports(
        records,
        output / "trajectories",
        settings.trajectory_plots,
        settings.seed,
    )
    interventions = intervention_report(
        module,
        test_loader,
        device,
        settings.maximum_test_windows,
        settings.seed,
    )
    _csv(output / "interventions.csv", interventions)
    probes, representation_statistics = probe_report(
        module,
        train_loader,
        test_loader,
        device,
        settings.maximum_probe_train_windows,
        settings.maximum_test_windows,
        output / "probes",
    )

    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "training_config": str(Path(training_config_path).resolve()),
        "test_windows": len(records),
        "device": str(device),
        "aggregate": summary,
        "interventions": interventions,
        "probes": probes,
        "representation_statistics": representation_statistics,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output.resolve()
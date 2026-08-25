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
)
from .model_loader import denormalised_prediction, load_kinematic_module
from .plots import save_component_scatter, save_probe_plot, save_trajectory_plot


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
        prediction = module.model(images)
        decoded = denormalised_prediction(module, prediction)
        size = images.shape[0]
        for index in range(size):
            record = {
                "trajectory_id": batch["trajectory_id"][index],
                "start_frame": int(batch["start_frame"][index]),
                "time": _numpy(batch["context_time"][index]),
                "target_position": _numpy(batch["context_position"][index]),
                "target_velocity": _numpy(batch["context_linear_velocity"][index]),
                "target_quaternion": _numpy(batch["context_quaternion_xyzw"][index]),
            }
            for key, value in decoded.items():
                record[f"predicted_{key}"] = _numpy(value[index])
            records.append(record)
    return records

def aggregate_report(records, output: Path):
    rows = []
    summary = {}
    for quantity, target_key, prediction_key, unit in (
        ("position", "target_position", "predicted_position", "m"),
        ("velocity", "target_velocity", "predicted_linear_velocity", "m/s"),
    ):
        available = [record for record in records if prediction_key in record]
        if not available:
            continue
        target = np.concatenate([record[target_key] for record in available], axis=0)
        prediction = np.concatenate([record[prediction_key] for record in available], axis=0)
        save_component_scatter(
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
    return summary


def trajectory_reports(records, output: Path, count: int, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    scored = []
    for record in records:
        errors = []
        if "predicted_position" in record:
            errors.append(
                np.mean(np.linalg.norm(record["prediction_position"] - record["target_position"], axis=-1))
            )
        if "predicted_linear_velocity" in record:
            errors.append(
                np.mean(
                    np.linalg.norm(record["predicted_linear_velocity"] - record["target_velocity"], axis=-1)
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
        save_trajectory_plot(
            time=record["time"],
            target_position=(record["target_position"] if "predicted_position" in record else None),
            predicted_position=record.get("predicted_position"),
            target_velocity=(
                record["target_velocity"] if "predicted_linear_velocity" in record else None
            ),
            predicted_velocity=record.get("predicted_linear_velocity"),
            output=output / f"{label}_{record['trajectory_id']}_{record['start_frame']:06d}.png",
            title=f"{label}: trajectory {record['trajectory_id']}, start {record['start_frame']}",
        )


@torch.inference_mode()
def intervention_report(module, loader, device, maximum_windows: int, seed: int):
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    accumulators = {
        "forward": [],
        "reversed": [],
        "repeated_last": [],
        "shuffled": [],
    }
    for batch in _limit_loader(loader, maximum_windows):
        images = batch["context_rgb"]
        variants = {
            "forward": images,
            "reversed": torch.flip(images, dims=(1,)),
            "repeated_last": images[:, -1].expand_as(images),
            "shuffled": images[:, torch.randperm(images.shape[1], generator=rng)],
        }
        for name, variant in variants.items():
            prediction = module.model(variant.to(device))
            decoded = denormalised_prediction(module, prediction)
            if "linear_velocity" in decoded:
                # Mean over time is robust to the changed endpoint in reversed sequences.
                accumulators[name].append(_numpy(decoded["linear_velocity"].mean(dim=1)))

    if not accumulators["forward"]:
        return []
    values = {name: np.concatenate(items) for name, items in accumulators.items()}
    forward = values["forward"]
    rows = []
    for name, estimate in values.items():
        rows.append(
            {
                "intervention": name,
                "mean_speed": float(np.mean(np.linalg.norm(estimate, axis=-1))),
                "mean_delta_from_forward": float(
                    np.mean(np.linalg.norm(estimate - forward, axis=-1))
                ),
                "mean_reversal_error": float(
                    np.mean(np.linalg.norm(estimate + forward, axis=-1)) # negative velocity expected
                ),
                "vx": float(estimate[:, 0].mean()),
                "vy": float(estimate[:, 1].mean()),
                "vz": float(estimate[:, 2].mean()),
                "count": len(estimate),
            }
        )
    return rows


@torch.inference_mode()
def extract_features(module, loader, device, maximum_windows: int):
    features = {
        "frame_last": [],
        "frame_sequence_mean": [],
        "frame_difference": [],
        "temporal_last": [],
    }
    targets = {"position": [], "velocity": []}
    consumed = 0
    for batch in _limit_loader(loader, maximum_windows):
        images = batch["context_rgb"].to(device)
        frame = module.model.frame_encoder(images)
        temporal = module.model.temporal_encoder(frame)
        features["frame_last"].append(_numpy(frame[:, -1]))
        features["frame_sequence_mean"].append(_numpy(frame.mean(dim=1)))
        features["frame_difference"].append(_numpy(frame[:, -1] - frame[:, 0]))
        features["temporal_last"].append(_numpy(temporal[:, -1]))
        targets["position"].append(_numpy(batch["context_position"][:, -1]))
        targets["velocity"].append(_numpy(batch["context_linear_velocity"][:, -1]))
        consumed += len(images)
    return (
        {name: np.concatenate(value) for name, value in features.items()},
        {name: np.concatenate(value) for name, value in targets.items()},
    )


def probe_report(module, train_loader, test_loader, device, train_maximum, test_maximum, output):
    train_features, train_targets = extract_features(module, train_loader, device, train_maximum)
    test_features, test_targets = extract_features(module, test_loader, device, test_maximum)
    rows = []
    for quantity in ("position", "velocity"):
        mean_estimate = np.broadcast_to(
            train_targets[quantity].mean(axis=0),
            test_targets[quantity].shape,
        )
        component_r2 = []
        component_rmse = []
        mean_row = {"layer": "mean_state_baseline", "quantity": quantity}
        for component, label in enumerate(("x", "y", "z")):
            metrics = regression_metrics(
                test_targets[quantity][:, component], mean_estimate[:, component]
            )
            mean_row[f"r2_{label}"] = metrics["r2"]
            mean_row[f"rmse_{label}"] = metrics["rmse"]
            component_r2.append(metrics["r2"])
            component_rmse.append(metrics["rmse"])
        mean_row["r2_mean"] = float(np.nanmean(component_r2))
        mean_row["rmse_mean"] = float(np.mean(component_rmse))
        rows.append(mean_row)

    for layer in train_features:
        for quantity in ("position", "velocity"):
            weights = fit_linear_probe(train_features[layer], train_targets[quantity])
            estimate = apply_linear_probe(test_features[layer], weights)
            component_r2 = []
            component_rmse = []
            row = {"layer": layer, "quantity": quantity}
            for component, label in enumerate(("x", "y", "z")):
                metrics = regression_metrics(test_targets[quantity][:, component], estimate[:, component])
                row[f"r2_{label}"] = metrics["r2"]
                row[f"rmse_{label}"] = metrics["rmse"]
                component_r2.append(metrics["r2"])
                component_rmse.append(metrics["rmse"])
            row["r2_mean"] = float(np.nanmean(component_r2))
            row["rmse_mean"] = float(np.mean(component_rmse))
            rows.append(row)
    _csv(output / "layerwise_linear_probes.csv", rows)
    save_probe_plot(rows, output / "layerwise_linear_probes.png")
    return rows


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
    probes = probe_report(
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
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output.resolve()
"""Frozen post-hoc linear and nonlinear velocity decoders."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import (
    apply_linear_probe,
    fit_linear_probe,
    regression_metrics,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _limited_batches(loader, maximum_windows: int):
    used = 0
    for batch in loader:
        if used >= maximum_windows:
            break
        batch_size = int(batch["context_rgb"].shape[0])
        keep = min(batch_size, maximum_windows - used)
        if keep < batch_size:
            batch = {
                key: value[:keep] if isinstance(value, torch.Tensor) else value[:keep]
                for key, value in batch.items()
            }
        used += keep
        yield batch

@torch.inference_mode()
def _collect(module, loader, device, maximum_windows: int):
    features: dict[str, list[np.ndarray]] = {
        "content_difference": [],
        "motion_forward": [],
        "context_pair": [],
        "context_pair_motion": [],
    }
    targets: list[np.ndarray] = []

    for batch in _limited_batches(loader, maximum_windows):
        rgb = batch["context_rgb"].to(device)
        time = batch["context_time"].to(device)
        prediction = module.model(rgb, time)
        context = prediction.frame_latent
        motion = prediction.motion.forward_motion
        difference = context[:, 1:] - context[:, :-1]
        first = context[:, :-1]
        second = context[:, 1:]

        representations = {
            "content_difference": difference,
            "motion_forward": motion,
            "context_pair": torch.cat((first, second, difference), dim=-1),
            "context_pair_motion": torch.cat(
                (first, second, difference, motion),
                dim=-1,
            ),
        }
        for name, value in representations.items():
            features[name].append(value.detach().cpu().numpy().reshape(-1, value.shape[-1]))
        targets.append(
            batch["context_linear_velocity"][:, :-1]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 3)
        )

    return (
        {name: np.concatenate(parts, axis=0) for name, parts in features.items()},
        np.concatenate(targets, axis=0),
    )


class _VelocityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _metric_rows(
    representation: str,
    decoder: str,
    target: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "representation": representation,
            "decoder": decoder,
            "component": axis,
            **regression_metrics(target[:, component], prediction[:, component]),
        }
        for component, axis in enumerate(("x", "y", "z"))
    ]


def _fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    device,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True) + 1.0e-6
    y_mean = train_y.mean(axis=0, keepdims=True)
    y_std = train_y.std(axis=0, keepdims=True) + 1.0e-6

    train_x_n = torch.from_numpy(((train_x - x_mean) / x_std).astype(np.float32))
    train_y_n = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32))
    test_x_n = torch.from_numpy(((test_x - x_mean) / x_std).astype(np.float32))

    loader = DataLoader(
        TensorDataset(train_x_n, train_y_n),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = _VelocityMLP(train_x.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    history = []

    model.train()
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(x_batch)
            loss = torch.mean((prediction - y_batch) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(x_batch)
            count += len(x_batch)
        history.append({"epoch": epoch + 1, "normalised_mse": total / max(count, 1)})

    model.eval()
    estimates = []
    with torch.inference_mode():
        for start in range(0, len(test_x_n), batch_size):
            estimates.append(model(test_x_n[start : start + batch_size].to(device)).cpu())
    estimate_n = torch.cat(estimates, dim=0).numpy()
    estimate = estimate_n * y_std + y_mean
    state = {
        "model_state_dict": model.cpu().state_dict(),
        "input_mean": torch.from_numpy(x_mean.astype(np.float32)),
        "input_std": torch.from_numpy(x_std.astype(np.float32)),
        "target_mean": torch.from_numpy(y_mean.astype(np.float32)),
        "target_std": torch.from_numpy(y_std.astype(np.float32)),
        "input_dim": train_x.shape[1],
        "hidden_dim": hidden_dim,
    }
    return estimate, history, state


def evaluate_posthoc_velocity_decoders(
    module,
    train_loader,
    test_loader,
    device,
    output_directory: Path | str,
    *,
    train_maximum: int = 5_000,
    test_maximum: int = 2_000,
    epochs: int = 40,
    hidden_dim: int = 128,
    batch_size: int = 2_048,
    learning_rate: float = 1.0e-3,
    seed: int = 20260928,
) -> dict[str, object]:
    """Fit frozen ridge and MLP decoders without updating the observer."""
    output = Path(output_directory) / "posthoc_velocity"
    output.mkdir(parents=True, exist_ok=True)

    train_features, train_target = _collect(module, train_loader, device, train_maximum)
    test_features, test_target = _collect(module, test_loader, device, test_maximum)

    rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}

    for index, representation in enumerate(train_features):
        train_x = train_features[representation]
        test_x = test_features[representation]

        ridge_weights = fit_linear_probe(train_x, train_target)
        ridge_estimate = apply_linear_probe(test_x, ridge_weights)
        rows += _metric_rows(representation, "ridge", test_target, ridge_estimate)

        mlp_estimate, history, state = _fit_mlp(
            train_x,
            train_target,
            test_x,
            device=device,
            epochs=epochs,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + index,
        )
        rows += _metric_rows(representation, "mlp", test_target, mlp_estimate)
        histories[representation] = history
        torch.save(state, output / f"{representation}_mlp.pt")
        np.save(output / f"{representation}_test_prediction.npy", mlp_estimate)

    _write_csv(output / "metrics.csv", rows)
    for representation, history in histories.items():
        _write_csv(output / f"{representation}_training.csv", history)

    report = {
        "task": module.hparams.task,
        "observer_frozen": True,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "metrics": rows,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

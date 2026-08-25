from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_trajectory_plot(
    *,
    time: np.ndarray,
    target_position: np.ndarray | None,
    predicted_position: np.ndarray | None,
    target_velocity: np.ndarray | None,
    predicted_velocity: np.ndarray | None,
    output: str | Path,
    title: str,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = int(target_position is not None) + int(target_velocity is not None) + 1
    figure, axes = plt.subplots(rows, 1, figsize=(11, 3.3 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes)
    row = 0
    labels = ("x", "y", "z")

    if target_position is not None:
        axis = axes[row]
        for component, label in enumerate(labels):
            axis.plot(time, target_position[:, component], label=f"true {label}")
            axis.plot(
                time,
                predicted_position[:, component],
                linestyle="--",
                label=f"pred {label}",
            )
        axis.set_ylabel("Position [m]")
        axis.grid(alpha=0.25)
        axis.legend(ncols=3)
        row += 1

    if target_velocity is not None:
        axis = axes[row]
        for component, label in enumerate(labels):
            axis.plot(time, target_velocity[:, component], label=f"true {label}")
            axis.plot(
                time,
                predicted_velocity[:, component],
                linestyle="--",
                label=f"pred {label}",
            )
        axis.set_ylabel("Velocity [m/s]")
        axis.grid(alpha=0.25)
        axis.legend(ncols=3)
        row += 1

    axis = axes[row]
    if target_position is not None:
        position_error = np.linalg.norm(predicted_position - target_position, axis=-1)
        axis.plot(time, position_error, label="position error [m]")
    if target_velocity is not None:
        velocity_error = np.linalg.norm(predicted_velocity - target_velocity, axis=-1)
        axis.plot(time, velocity_error, label="velocity error [m/s]")
    if predicted_position is not None and predicted_velocity is not None and len(time) > 1:
        dt = np.diff(time)[:, None]
        consistency = np.linalg.norm(
            predicted_position[1:] - predicted_position[:-1] - dt * predicted_velocity[:-1],
            axis=-1,
        )
        axis.plot(time[1:], consistency, label="kinematic consistency [m]")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Error norm")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.suptitle(title)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output


def save_component_scatter(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    quantity: str,
    unit: str,
    output: str | Path,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    labels = ("x", "y", "z")
    for component, axis in enumerate(axes):
        truth = target[:, component]
        estimate = prediction[:, component]
        low = min(float(truth.min()), float(estimate.min()))
        high = max(float(truth.max()), float(estimate.max()))
        axis.scatter(truth, estimate, s=7, alpha=0.35)
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        residual = estimate - truth
        total = np.sum((truth - truth.mean()) ** 2)
        r2 = 1.0 - np.sum(residual**2) / total if total > 0 else np.nan
        axis.set_title(
            f"{quantity} {labels[component]}\n"
            f"RMSE={np.sqrt(np.mean(residual**2)):.4g}, R2={r2:.3f}"
        )
        axis.set_xlabel(f"True [{unit}]")
        axis.set_ylabel(f"Predicted [{unit}]")
        axis.grid(alpha=0.2)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output


def save_probe_plot(rows: list[dict[str, object]], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted({str(row["layer"]) for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for quantity, axis in zip(("position", "velocity"), axes):
        subset = [row for row in rows if row["quantity"] == quantity]
        values = {str(row["layer"]): float(row["r2_mean"] or np.nan) for row in subset}
        axis.bar(layers, [values.get(layer, np.nan) for layer in layers])
        axis.set_title(f"Linear probe: {quantity}")
        axis.set_ylabel("Mean component R2")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output

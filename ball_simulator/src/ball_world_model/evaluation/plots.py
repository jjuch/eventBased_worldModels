from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def trajectory_plot(record: dict, output: Path, title: str) -> None:
    time = record["time"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for component, axis_name in enumerate(("x", "y", "z")):
        axes[0].plot(time, record["target_position"][:, component], label=f"true {axis_name}")
        axes[0].plot(time, record["predicted_position"][:, component], "--", label=f"pred {axis_name}")
        axes[1].plot(time, record["target_velocity"][:, component], label=f"true {axis_name}")
        axes[1].plot(time, record["predicted_velocity"][:, component], "--", label=f"pred {axis_name}")
    axes[0].set_ylabel("Position [m]")
    axes[1].set_ylabel("Velocity [m/s]")
    position_error = np.linalg.norm(record["predicted_position"] - record["target_position"], axis=-1)
    velocity_error = np.linalg.norm(record["predicted_velocity"] - record["target_velocity"], axis=-1)
    axes[2].plot(time, position_error, label="position error [m]")
    axes[2].plot(time, velocity_error, label="velocity error [m/s]")
    if len(time) > 1:
        dt = np.diff(time)[:, None]
        residual = np.linalg.norm(
            record["predicted_position"][1:]
            - record["predicted_position"][:-1]
            - dt * record["predicted_velocity"][:-1],
            axis=-1,
        )
        axes[2].plot(time[1:], residual, label="kinematic residual [m]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncols=3)
    axes[2].set_xlabel("Time [s]")
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def component_scatter(target: np.ndarray, prediction: np.ndarray, quantity: str, unit: str, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for component, label in enumerate(("x", "y", "z")):
        truth = target[:, component]
        estimate = prediction[:, component]
        low = min(float(truth.min()), float(estimate.min()))
        high = max(float(truth.max()), float(estimate.max()))
        axes[component].scatter(truth, estimate, s=7, alpha=0.3)
        axes[component].plot([low, high], [low, high], color="black")
        residual = estimate - truth
        total = np.sum((truth - truth.mean()) ** 2)
        r2 = 1.0 - np.sum(residual**2) / total if total > 0 else np.nan
        axes[component].set_title(
            f"{quantity} {label}: RMSE={np.sqrt(np.mean(residual**2)):.4g}, R2={r2:.3f}"
        )
        axes[component].set_xlabel(f"True [{unit}]")
        axes[component].set_ylabel(f"Predicted [{unit}]")
        axes[component].grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def probe_plot(rows: list[dict], output: Path) -> None:
    layers = [row["representation"] for row in rows if row["quantity"] == "velocity"]
    position = {row["representation"]: row["r2_mean"] for row in rows if row["quantity"] == "position"}
    velocity = {row["representation"]: row["r2_mean"] for row in rows if row["quantity"] == "velocity"}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(layers, [position.get(layer, np.nan) for layer in layers])
    axes[1].bar(layers, [velocity.get(layer, np.nan) for layer in layers])
    axes[0].set_title("Linear probe: position")
    axes[1].set_title("Linear probe: velocity")
    for axis in axes:
        axis.set_ylabel("Mean component R2")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def error_vs_speed(records: list[dict], output: Path) -> None:
    speed, position_error, velocity_error = [], [], []
    for record in records:
        speed.append(np.linalg.norm(record["target_velocity"], axis=-1))
        position_error.append(np.linalg.norm(record["predicted_position"] - record["target_position"], axis=-1))
        velocity_error.append(np.linalg.norm(record["predicted_velocity"] - record["target_velocity"], axis=-1))
    speed = np.concatenate(speed)
    position_error = np.concatenate(position_error)
    velocity_error = np.concatenate(velocity_error)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].scatter(speed, position_error, s=5, alpha=0.2)
    axes[1].scatter(speed, velocity_error, s=5, alpha=0.2)
    axes[0].set_ylabel("Position error [m]")
    axes[1].set_ylabel("Velocity error [m/s]")
    for axis in axes:
        axis.set_xlabel("True speed [m/s]")
        axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)

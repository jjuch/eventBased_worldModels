from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ball_world_model.config import DataConfig
from .transforms import denormalise_rgb


def save_batch_inspection(
    batch: dict[str, object],
    config: DataConfig,
    output: str | Path,
    *,
    batch_index: int = 0,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    context = batch["context_rgb"][batch_index]
    future = batch["future_rgb"][batch_index]
    context_time = batch["context_time"][batch_index].numpy()
    future_time = batch["future_time"][batch_index].numpy()
    context_position = batch["context_position"][batch_index].numpy()
    future_position = batch["future_position"][batch_index].numpy()
    context_velocity = batch["context_linear_velocity"][batch_index].numpy()
    future_velocity = batch["future_linear_velocity"][batch_index].numpy()
    context_omega = batch["context_angular_velocity"][batch_index].numpy()
    future_omega = batch["future_angular_velocity"][batch_index].numpy()

    context_show = list(range(len(context)))
    future_show = np.linspace(0, len(future) - 1, min(5, len(future)), dtype=int).tolist()
    columns = max(len(context_show), len(future_show))

    figure = plt.figure(figsize=(2.0 * columns, 10), constrained_layout=True)
    grid = figure.add_gridspec(5, columns)

    for column, frame_index in enumerate(context_show):
        axis = figure.add_subplot(grid[0, column])
        image = denormalise_rgb(context[frame_index], config.image.normalization)
        axis.imshow(image.permute(1, 2, 0))
        axis.set_title(f"C {frame_index}\n{context_time[frame_index]:.3f}s")
        axis.axis("off")

    for column in range(columns):
        axis = figure.add_subplot(grid[1, column])
        if column < len(future_show):
            frame_index = future_show[column]
            image = denormalise_rgb(future[frame_index], config.image.normalization)
            axis.imshow(image.permute(1, 2, 0))
            axis.set_title(f"F {frame_index}\n{future_time[frame_index]:.3f}s")
        axis.axis("off")


    full_time = np.concatenate([context_time, future_time])
    labels = ("x", "y", "z")
    quantities = (
        (np.concatenate([context_position, future_position]), "Position [m]"),
        (np.concatenate([context_velocity, future_velocity]), "Velocity [m/s]"),
        (np.concatenate([context_omega, future_omega]), "Angular velocity [rad/s]"),
    )
    for row_index, (values, title) in enumerate(quantities, start=2):
        axis = figure.add_subplot(grid[row_index, :])
        for component, label in enumerate(labels):
            axis.plot(full_time, values[:, component], label=label)
        axis.axvline(context_time[-1], color="black", linestyle="--", label="context end")
        axis.set_title(title)
        axis.set_xlabel("Time [s]")
        axis.grid(alpha=0.25)
        axis.legend(ncols=4)

    trajectory_id = batch["trajectory_id"][batch_index]
    start_frame = int(batch["start_frame"][batch_index])
    figure.suptitle(f"Trajectory {trajectory_id}, start frame {start_frame}", fontsize=15)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def describe_batch(batch: dict[str, object]) -> str:
    lines = []
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            lines.append(f"{key:30s} shape={tuple(value.shape)!s:20s} dtype={value.dtype}")
        elif isinstance(value, list):
            lines.append(f"{key:30s} list[{len(value)}]")
        else:
            lines.append(f"{key:30s} {type(value).__name__}")
    return "\n".join(lines)
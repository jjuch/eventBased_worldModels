from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class InitialConditionData:
    trajectory_ids: list[str]
    position: FloatArray
    linear_velocity: FloatArray
    angular_velocity: FloatArray
    parameter_names: list[str]
    parameters: FloatArray

    @property
    def linear_speed(self) -> FloatArray:
        return np.linalg.norm(self.linear_velocity, axis=1)

    @property
    def angular_speed(self) -> FloatArray:
        return np.linalg.norm(self.angular_velocity, axis=1)

    @property
    def tangential_speed(self) -> FloatArray:
        return np.linalg.norm(self.linear_velocity[:, 1:3], axis=1)


@dataclass(frozen=True, slots=True)
class TrajectoryData:
    trajectory_id: str
    time: FloatArray
    position: FloatArray
    quaternion_xyzw: FloatArray
    linear_velocity: FloatArray
    angular_velocity: FloatArray
    tangential_memory: FloatArray
    contact_active: NDArray[np.bool_]
    contact_mode: NDArray[np.int8]
    penetration: FloatArray
    normal_force: FloatArray
    tangential_force: FloatArray
    contact_velocity: FloatArray
    parameters: dict[str, float]

    @property
    def linear_speed(self) -> FloatArray:
        return np.linalg.norm(self.linear_velocity, axis=1)

    @property
    def angular_speed(self) -> FloatArray:
        return np.linalg.norm(self.angular_velocity, axis=1)

    @property
    def normal_force_magnitude(self) -> FloatArray:
        return np.linalg.norm(self.normal_force, axis=1)

    @property
    def tangential_force_magnitude(self) -> FloatArray:
        return np.linalg.norm(self.tangential_force, axis=1)


def natural_trajectory_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name), name
    except ValueError:
        return 0, name


def list_trajectory_ids(path: str | Path) -> list:
    with h5py.File(path, "r") as handle:
        if "trajectories" not in handle:
            raise ValueError(
                f"{path} does not contain a '/trajectories' group."
            )

        return sorted(
            handle["trajectories"].keys(),
            key=natural_trajectory_sort_key,
        )


def resolve_trajectory_id(
    available_ids: Sequence[str],
    requested_id: str | int,
) -> str:
    requested = str(requested_id)

    if requested in available_ids:
        return requested

    try:
        padded = f"{int(requested):08d}"
    except ValueError as error:
        raise KeyError(
            f"Invalid trajectory ID {requested_id!r}."
        ) from error

    if padded in available_ids:
        return padded

    raise KeyError(
        f"Trajectory {requested_id!r} was not found. "
        f"Available range: {available_ids[0]} to {available_ids[-1]}."
    )


def load_initial_conditions(
    path: str | Path,
) -> InitialConditionData:
    trajectory_ids: list[str] = []
    position: list[FloatArray] = []
    linear_velocity: list[FloatArray] = []
    angular_velocity: list[FloatArray] = []
    parameter_rows: list[list[float]] = []

    with h5py.File(path, "r") as handle:
        root = handle["trajectories"]
        ids = sorted(root.keys(), key=natural_trajectory_sort_key)

        if not ids:
            raise ValueError("The dataset contains no trajectories.")

        first_parameters = root[ids[0]]["parameters"].attrs
        parameter_names = sorted(first_parameters.keys())

        for trajectory_id in ids:
            group = root[trajectory_id]
            observations = group["observations"]

            # Only the first observation is read. This remains efficient
            # even when trajectories or high-rate datasets are very large.
            position.append(
                np.asarray(observations["position"][0], dtype=np.float64)
            )
            linear_velocity.append(
                np.asarray(
                    observations["linear_velocity"][0],
                    dtype=np.float64,
                )
            )
            angular_velocity.append(
                np.asarray(
                    observations["angular_velocity"][0],
                    dtype=np.float64,
                )
            )

            attrs = group["parameters"].attrs
            parameter_rows.append(
                [float(attrs[name]) for name in parameter_names]
            )
            trajectory_ids.append(trajectory_id)

    return InitialConditionData(
        trajectory_ids=trajectory_ids,
        position=np.asarray(position),
        linear_velocity=np.asarray(linear_velocity),
        angular_velocity=np.asarray(angular_velocity),
        parameter_names=parameter_names,
        parameters=np.asarray(parameter_rows),
    )


def load_trajectory(
    path: str | Path,
    trajectory_id: str | int,
    use_high_rate: bool = False,
) -> TrajectoryData:
    with h5py.File(path, "r") as handle:
        root = handle["trajectories"]
        available_ids = sorted(
            root.keys(),
            key=natural_trajectory_sort_key,
        )
        resolved_id = resolve_trajectory_id(
            available_ids,
            trajectory_id,
        )

        group = root[resolved_id]

        data_group_name = (
            "high_rate"
            if use_high_rate and "high_rate" in group
            else "observations"
        )
        observations = group[data_group_name]

        required = [
            "time",
            "position",
            "quaternion_xyzw",
            "linear_velocity",
            "angular_velocity",
            "tangential_memory",
            "contact_active",
            "contact_mode",
            "penetration",
            "normal_force",
            "tangential_force",
            "contact_velocity",
        ]

        missing = [name for name in required if name not in observations]
        if missing:
            raise ValueError(
                f"Trajectory {resolved_id} is missing fields: {missing}"
            )

        arrays = {
            name: np.asarray(observations[name][:])
            for name in required
        }

        parameters = {
            name: float(value)
            for name, value in group["parameters"].attrs.items()
        }

    return TrajectoryData(
        trajectory_id=resolved_id,
        time=np.asarray(arrays["time"], dtype=np.float64),
        position=np.asarray(arrays["position"], dtype=np.float64),
        quaternion_xyzw=np.asarray(
            arrays["quaternion_xyzw"],
            dtype=np.float64,
        ),
        linear_velocity=np.asarray(
            arrays["linear_velocity"],
            dtype=np.float64,
        ),
        angular_velocity=np.asarray(
            arrays["angular_velocity"],
            dtype=np.float64,
        ),
        tangential_memory=np.asarray(
            arrays["tangential_memory"],
            dtype=np.float64,
        ),
        contact_active=np.asarray(
            arrays["contact_active"],
            dtype=np.bool_,
        ),
        contact_mode=np.asarray(
            arrays["contact_mode"],
            dtype=np.int8,
        ),
        penetration=np.asarray(
            arrays["penetration"],
            dtype=np.float64,
        ),
        normal_force=np.asarray(
            arrays["normal_force"],
            dtype=np.float64,
        ),
        tangential_force=np.asarray(
            arrays["tangential_force"],
            dtype=np.float64,
        ),
        contact_velocity=np.asarray(
            arrays["contact_velocity"],
            dtype=np.float64,
        ),
        parameters=parameters,
    )


def load_environment_metadata(
    path: str | Path,
) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if "metadata_json" not in handle.attrs:
            raise ImportError("File is corrupt: no metadata found.")
        
        metadata = json.loads(handle.attrs["metadata_json"])

    if "environment_geometry" in metadata:
        return metadata["environment_geometry"]
    else:
        raise ImportError("File is corrupt: no environment geometry found.")



def robust_limits(
    values: FloatArray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return 0.0, 1.0

    lower, upper = np.percentile(
        finite,
        [lower_percentile, upper_percentile],
    )

    if np.isclose(lower, upper):
        margin = max(abs(lower) * 0.1, 1.0e-9)
        return lower - margin, upper + margin

    return float(lower), float(upper)


def apply_equal_3d_axes(
    axes: Iterable,
    points: FloatArray,
    wall_x: float,
    radius: float,
) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)

    minimum[0] = min(minimum[0], wall_x - 0.05 * radius)
    maximum[0] = max(maximum[0], wall_x + 2.0 * radius)

    spans = maximum - minimum
    largest_span = max(float(np.max(spans)), 2.0 * radius)
    centers = 0.5 * (minimum + maximum)

    for axis in axes:
        axis.set_xlim(
            centers[0] - largest_span / 2.0,
            centers[0] + largest_span / 2.0,
        )
        axis.set_ylim(
            centers[1] - largest_span / 2.0,
            centers[1] + largest_span / 2.0,
        )
        axis.set_zlim(
            centers[2] - largest_span / 2.0,
            centers[2] + largest_span / 2.0,
        )
        axis.set_box_aspect((1.0, 1.0, 1.0))


def plot_initial_condition_spread(
    data: InitialConditionData,
    output: str | Path | None = None,
    show: bool = True,
    max_scatter_points: int = 10_000,
    random_seed: int = 0,
) -> Figure:
    number_of_trajectories = len(data.trajectory_ids)

    rng = np.random.default_rng(random_seed)
    if number_of_trajectories > max_scatter_points:
        indices = np.sort(
            rng.choice(
                number_of_trajectories,
                max_scatter_points,
                replace=False,
            )
        )
    else:
        indices = np.arange(number_of_trajectories)

    position = data.position[indices]
    velocity = data.linear_velocity[indices]
    angular_velocity = data.angular_velocity[indices]

    linear_speed = np.linalg.norm(velocity, axis=1)
    angular_speed = np.linalg.norm(angular_velocity, axis=1)

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)

    axis_position = figure.add_subplot(grid[0, 0])
    axis_velocity = figure.add_subplot(grid[0, 1])
    axis_spin = figure.add_subplot(grid[0, 2], projection="3d")
    axis_position_hist = figure.add_subplot(grid[1, 0])
    axis_speed_hist = figure.add_subplot(grid[1, 1])
    axis_angular_hist = figure.add_subplot(grid[1, 2])

    position_plot = axis_position.scatter(
        position[:, 1],
        position[:, 2],
        c=position[:, 0],
        s=10,
        alpha=0.55,
        cmap="viridis",
        edgecolors="none",
        rasterized=True,
    )
    figure.colorbar(
        position_plot,
        ax=axis_position,
        label="Initial wall-normal position x [m]",
    )
    axis_position.set(
        title="Initial position in wall plane",
        xlabel="Lateral position y [m]",
        ylabel="Vertical position z [m]",
    )
    axis_position.grid(alpha=0.25)

    velocity_plot = axis_velocity.scatter(
        -velocity[:, 0],
        np.linalg.norm(velocity[:, 1:3], axis=1),
        c=linear_speed,
        s=10,
        alpha=0.55,
        cmap="plasma",
        edgecolors="none",
        rasterized=True,
    )
    figure.colorbar(
        velocity_plot,
        ax=axis_velocity,
        label="Initial linear speed [m/s]",
    )
    axis_velocity.set(
        title="Impact-speed coverage",
        xlabel="Incoming normal speed -vx [m/s]",
        ylabel="Tangential speed sqrt(vy² + vz²) [m/s]",
    )
    axis_velocity.grid(alpha=0.25)

    spin_plot = axis_spin.scatter(
        angular_velocity[:, 0],
        angular_velocity[:, 1],
        angular_velocity[:, 2],
        c=angular_speed,
        s=10,
        alpha=0.55,
        cmap="magma",
        depthshade=False,
    )
    figure.colorbar(
        spin_plot,
        ax=axis_spin,
        shrink=0.75,
        label="Initial angular speed [rad/s]",
    )
    axis_spin.set(
        title="Initial angular-velocity coverage",
        xlabel="ωx [rad/s]",
        ylabel="ωy [rad/s]",
        zlabel="ωz [rad/s]",
    )

    axis_position_hist.hist(
        data.position[:, 0],
        bins="auto",
        alpha=0.8,
        color="tab:blue",
    )
    axis_position_hist.set(
        title="Distance from wall",
        xlabel="Initial x position [m]",
        ylabel="Trajectory count",
    )
    axis_position_hist.grid(alpha=0.25)

    axis_speed_hist.hist(
        data.linear_speed,
        bins="auto",
        alpha=0.8,
        color="tab:orange",
    )
    axis_speed_hist.set(
        title="Linear-speed distribution",
        xlabel="Initial |v| [m/s]",
        ylabel="Trajectory count",
    )
    axis_speed_hist.grid(alpha=0.25)

    axis_angular_hist.hist(
        data.angular_speed,
        bins="auto",
        alpha=0.8,
        color="tab:red",
    )
    axis_angular_hist.set(
        title="Angular-speed distribution",
        xlabel="Initial |ω| [rad/s]",
        ylabel="Trajectory count",
    )
    axis_angular_hist.grid(alpha=0.25)

    figure.suptitle(
        "Initial-condition coverage "
        f"({number_of_trajectories:,} trajectories)",
        fontsize=16,
    )

    if output is not None:
        figure.savefig(output, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return figure


def trajectory_segments(points: FloatArray) -> FloatArray:
    if len(points) < 2:
        raise ValueError("A trajectory must contain at least two points.")

    return np.stack([points[:-1], points[1:]], axis=1)


def add_colored_trajectory(
    axis,
    position: FloatArray,
    values: FloatArray,
    label: str,
    cmap: str,
    linewidth: float = 3.0,
) -> Line3DCollection:
    segments = trajectory_segments(position)

    segment_values = 0.5 * (values[:-1] + values[1:])
    lower, upper = robust_limits(segment_values)
    normalization = colors.Normalize(vmin=lower, vmax=upper)

    collection = Line3DCollection(
        segments,
        cmap=cmap,
        norm=normalization,
        linewidth=linewidth,
    )
    collection.set_array(segment_values)

    axis.add_collection3d(collection)
    axis.figure.colorbar(
        collection,
        ax=axis,
        shrink=0.72,
        pad=0.08,
        label=label,
    )
    return collection


def add_wall(
    axis,
    position: FloatArray,
    wall_x: float,
    radius: float,
) -> None:
    y_min, y_max = robust_limits(position[:, 1], 0.0, 100.0)
    z_min, z_max = robust_limits(position[:, 2], 0.0, 100.0)

    y_margin = max(0.1 * (y_max - y_min), radius)
    z_margin = max(0.1 * (z_max - z_min), radius)

    y = np.linspace(y_min - y_margin, y_max + y_margin, 2)
    z = np.linspace(z_min - z_margin, z_max + z_margin, 2)
    y_grid, z_grid = np.meshgrid(y, z)
    x_grid = np.full_like(y_grid, wall_x)

    axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color="gray",
        alpha=0.22,
        edgecolor="gray",
        linewidth=0.5,
    )


def mark_contact_points(
    axis,
    trajectory: TrajectoryData,
) -> None:
    mask = trajectory.contact_active

    if not np.any(mask):
        return

    positions = trajectory.position[mask]
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="black",
        marker="o",
        s=16,
        alpha=0.8,
        label="Contact observations",
        depthshade=False,
    )


def plot_trajectory_3d(
    trajectory: TrajectoryData,
    wall_x: float = 0.0,
    output: str | Path | None = None,
    show: bool = True,
) -> Figure:
    position = trajectory.position
    radius = trajectory.parameters["radius"]

    figure = plt.figure(figsize=(16, 8), constrained_layout=True)
    velocity_axis = figure.add_subplot(1, 2, 1, projection="3d")
    angular_axis = figure.add_subplot(1, 2, 2, projection="3d")

    axes = (velocity_axis, angular_axis)

    add_colored_trajectory(
        velocity_axis,
        position,
        trajectory.linear_speed,
        label="Linear speed |v| [m/s]",
        cmap="viridis",
    )
    velocity_axis.set_title("Trajectory colored by linear speed")

    add_colored_trajectory(
        angular_axis,
        position,
        trajectory.angular_speed,
        label="Angular speed |ω| [rad/s]",
        cmap="magma",
    )
    angular_axis.set_title("Trajectory colored by angular speed")

    for axis in axes:
        add_wall(axis, position, wall_x, radius)
        mark_contact_points(axis, trajectory)

        axis.scatter(
            *position[0],
            color="green",
            marker="o",
            s=60,
            label="Start",
            depthshade=False,
        )
        axis.scatter(
            *position[-1],
            color="red",
            marker="X",
            s=70,
            label="End",
            depthshade=False,
        )

        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.view_init(elev=22.0, azim=-60.0)
        axis.legend(loc="upper right")

    apply_equal_3d_axes(axes, position, wall_x, radius)

    contact_count = int(np.count_nonzero(trajectory.contact_active))
    figure.suptitle(
        f"Trajectory {trajectory.trajectory_id} — "
        f"{contact_count} recorded contact samples",
        fontsize=16,
    )

    if output is not None:
        figure.savefig(output, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return figure


def compute_mechanical_energy(
    trajectory: TrajectoryData,
    gravity: float = 9.81,
) -> dict[str, FloatArray]:
    mass = trajectory.parameters["mass"]
    radius = trajectory.parameters["radius"]
    inertia = 0.4 * mass * radius**2

    translational = (
        0.5
        * mass
        * np.sum(trajectory.linear_velocity**2, axis=1)
    )
    rotational = (
        0.5
        * inertia
        * np.sum(trajectory.angular_velocity**2, axis=1)
    )
    gravitational = mass * gravity * trajectory.position[:, 2]

    if "normal_stiffness" in trajectory.parameters:
        elastic_normal = (
            0.4
            * trajectory.parameters["normal_stiffness"]
            * np.maximum(trajectory.penetration, 0.0) ** 2.5
        )
    else:
        elastic_normal = np.zeros_like(trajectory.time)

    if "tangential_stiffness" in trajectory.parameters:
        elastic_tangential = (
            0.5
            * trajectory.parameters["tangential_stiffness"]
            * np.sum(trajectory.tangential_memory**2, axis=1)
        )
    else:
        elastic_tangential = np.zeros_like(trajectory.time)

    total = (
        translational
        + rotational
        + gravitational
        + elastic_normal
        + elastic_tangential
    )

    return {
        "translational": translational,
        "rotational": rotational,
        "gravitational": gravitational,
        "normal_elastic": elastic_normal,
        "tangential_elastic": elastic_tangential,
        "total": total,
    }


def add_contact_shading(
    axis,
    time: FloatArray,
    contact_active: NDArray[np.bool_],
) -> None:
    if len(time) < 2 or not np.any(contact_active):
        return

    active_indices = np.flatnonzero(contact_active)
    start = active_indices[0]
    previous = active_indices[0]

    for current in active_indices[1:]:
        if current != previous + 1:
            axis.axvspan(
                time[start],
                time[previous],
                color="tab:red",
                alpha=0.10,
            )
            start = current
        previous = current

    axis.axvspan(
        time[start],
        time[previous],
        color="tab:red",
        alpha=0.10,
    )


def plot_trajectory_diagnostics(
    trajectory: TrajectoryData,
    output: str | Path | None = None,
    show: bool = True,
) -> Figure:
    time = trajectory.time
    energy = compute_mechanical_energy(trajectory)

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(15, 12),
        sharex=True,
        constrained_layout=True,
    )

    velocity_axis = axes[0, 0]
    angular_axis = axes[0, 1]
    contact_axis = axes[1, 0]
    penetration_axis = axes[1, 1]
    energy_axis = axes[2, 0]
    mode_axis = axes[2, 1]

    labels = ("x", "y", "z")

    for component, label in enumerate(labels):
        velocity_axis.plot(
            time,
            trajectory.linear_velocity[:, component],
            label=f"v{label}",
        )
        angular_axis.plot(
            time,
            trajectory.angular_velocity[:, component],
            label=f"ω{label}",
        )

    velocity_axis.set_ylabel("Linear velocity [m/s]")
    velocity_axis.set_title("Linear-velocity components")
    velocity_axis.legend()
    velocity_axis.grid(alpha=0.25)

    angular_axis.set_ylabel("Angular velocity [rad/s]")
    angular_axis.set_title("Angular-velocity components")
    angular_axis.legend()
    angular_axis.grid(alpha=0.25)

    contact_axis.plot(
        time,
        trajectory.normal_force_magnitude,
        label="Normal force",
    )
    contact_axis.plot(
        time,
        trajectory.tangential_force_magnitude,
        label="Tangential force",
    )
    contact_axis.set_ylabel("Force [N]")
    contact_axis.set_title("Contact forces")
    contact_axis.legend()
    contact_axis.grid(alpha=0.25)

    penetration_axis.plot(
        time,
        1_000.0 * trajectory.penetration,
        color="tab:purple",
    )
    penetration_axis.set_ylabel("Penetration [mm]")
    penetration_axis.set_title("Compliant penetration")
    penetration_axis.grid(alpha=0.25)

    energy_axis.plot(
        time,
        energy["translational"],
        label="Translational",
    )
    energy_axis.plot(
        time,
        energy["rotational"],
        label="Rotational",
    )
    energy_axis.plot(
        time,
        energy["gravitational"],
        label="Gravitational",
    )
    energy_axis.plot(
        time,
        energy["normal_elastic"],
        label="Contact elastic",
    )
    energy_axis.plot(
        time,
        energy["total"],
        color="black",
        linewidth=2.0,
        label="Total",
    )
    energy_axis.set(
        xlabel="Time [s]",
        ylabel="Energy [J]",
        title="Mechanical-energy accounting",
    )
    energy_axis.legend(ncols=2)
    energy_axis.grid(alpha=0.25)

    mode_axis.step(
        time,
        trajectory.contact_mode,
        where="post",
        color="tab:brown",
    )
    mode_axis.set(
        xlabel="Time [s]",
        ylabel="Contact mode",
        title="Contact-mode history",
        yticks=[0, 1, 2],
        yticklabels=["Free", "Sticking", "Sliding"],
    )
    mode_axis.grid(alpha=0.25)

    for axis in axes.flat:
        add_contact_shading(
            axis,
            time,
            trajectory.contact_active,
        )

    figure.suptitle(
        f"Physics diagnostics — trajectory "
        f"{trajectory.trajectory_id}",
        fontsize=16,
    )

    if output is not None:
        figure.savefig(output, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return figure
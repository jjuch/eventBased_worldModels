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

from .dataset_geometry import _decode_attribute, EnvironmentGeometry, PlaneGeometry


FloatArray = NDArray[np.float64]


SURFACE_COLORS = {
    "wall": "gray",
    "left_wall": "tab:blue",
    "right_wall": "tab:orange",
    "floor": "saddlebrown",
}

SURFACE_MARKERS = {
    "wall": "o",
    "left_wall": "<",
    "right_wall": ">",
    "floor": "v",
}

CONTACT_SHADING_COLORS = {
    "wall": "gray",
    "left_wall": "tab:blue",
    "right_wall": "tab:orange",
    "floor": "tab:green",
}

@dataclass(frozen=True, slots=True)
class InitialConditionData:
    trajectory_ids: list[str]
    position: FloatArray
    linear_velocity: FloatArray
    angular_velocity: FloatArray
    radii: FloatArray
    parameter_names: list[str]
    parameters: FloatArray

    @property
    def linear_speed(self) -> FloatArray:
        return np.linalg.norm(self.linear_velocity, axis=1)

    @property
    def angular_speed(self) -> FloatArray:
        return np.linalg.norm(self.angular_velocity, axis=1)

    # @property
    # def tangential_speed(self) -> FloatArray:
    #     return np.linalg.norm(self.linear_velocity[:, 1:3], axis=1)


@dataclass(frozen=True, slots=True)
class TrajectoryData:
    trajectory_id: str
    surface_ids: tuple[str, ...]
    time: FloatArray
    position: FloatArray
    quaternion_xyzw: FloatArray
    linear_velocity: FloatArray
    angular_velocity: FloatArray
    contact_active: NDArray[np.bool_]
    contact_mode: NDArray[np.int8]
    penetration: FloatArray
    normal_force: FloatArray
    tangential_force: FloatArray
    contact_velocity: FloatArray
    tangential_memory: FloatArray
    parameters: dict[str, float]

    @property
    def linear_speed(self) -> FloatArray:
        return np.linalg.norm(self.linear_velocity, axis=1)

    @property
    def angular_speed(self) -> FloatArray:
        return np.linalg.norm(self.angular_velocity, axis=1)

    @property
    def any_contact_active(self) -> NDArray[np.bool_]:
        return np.any(self.contact_active, axis=1)

    @property
    def any_contact_count(self) -> NDArray[np.int64]:
        return np.sum(self.contact_active, axis=1)

    @property
    def total_normal_force(self) -> FloatArray:
        return np.sum(self.normal_force, axis=1)

    @property
    def total_tangential_force(self) -> FloatArray:
        return np.sum(self.tangential_force, axis=1)

    @property
    def total_normal_force_magnitude(self) -> FloatArray:
        return np.linalg.norm(self.total_normal_force, axis=1)

    @property
    def total_tangential_force_magnitude(self) -> FloatArray:
        return np.linalg.norm(self.total_tangential_force, axis=1)

    def surface_index(self, surface_id: str) -> int:
        try:
            return self.surface_ids.index(surface_id)
        except ValueError as error:
            raise KeyError(
                f"Unknown trajectory surface "
                f"{surface_id!r}."
            ) from error

    def surface_contact_active(self, surface_id: str) -> NDArray[np.bool_]:
        return self.contact_active[:, self.surface_index(surface_id)]

    def surface_penetration(self, surface_id: str) -> FloatArray:
        return self.penetration[:, self.surface_index(surface_id)]

    def surface_normal_force(self, surface_id: str) -> FloatArray:
        return self.normal_force[:, self.surface_index(surface_id)]


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
    radii: list[FloatArray] = []
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
            radii.append(
                float(
                    group["parameters"].attrs["radius"]
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
        radii=np.asarray(radii),
        parameter_names=parameter_names,
        parameters=np.asarray(parameter_rows),
    )


def validate_trajectory_shapes(trajectory: TrajectoryData) -> None:
    number_of_samples = len(trajectory.time)
    number_of_surfaces = len(trajectory.surface_ids)

    expected_shapes = {
        "position": (number_of_samples, 3),
        "quaternion_xyzw": (number_of_samples, 4),
        "linear_velocity": (number_of_samples, 3),
        "angular_velocity": (number_of_samples, 3),
        "contact_active": (number_of_samples, number_of_surfaces),
        "contact_mode": (number_of_samples, number_of_surfaces),
        "penetration": (number_of_samples, number_of_surfaces),
        "normal_force": (number_of_samples, number_of_surfaces, 3),
        "tangential_force": (number_of_samples, number_of_surfaces, 3),
        "contact_velocity": (number_of_samples, number_of_surfaces, 3),
        "tangential_memory": (number_of_samples, number_of_surfaces, 3),
    }

    for name, expected in expected_shapes.items():
        actual = getattr(trajectory, name).shape
        if actual != expected:
            raise ValueError(
                f"Trajectory {trajectory.trajectory_id}: "
                f"{name} has shape {actual}; "
                f"expected {expected}."
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

        if "surface_ids_json" not in group.attrs:
            raise ValueError(
                f"Trajectory {resolved_id} has no "
                "'surface_ids_json' attribute."
            )

        surface_ids = tuple(
            json.loads(
                _decode_attribute(group.attrs["surface_ids_json"])
            )
        )

        parameters = {
            name: float(value)
            for name, value in group["parameters"].attrs.items()
        }

        trajectory = TrajectoryData(
            trajectory_id=resolved_id,
            surface_ids=surface_ids,
            time=np.asarray(
                observations["time"][:], 
                dtype=float,
            ),
            position=np.asarray(
                observations["position"][:],
                dtype=float,
            ),
            quaternion_xyzw=np.asarray(
                observations["quaternion_xyzw"][:],
                dtype=float,
            ),
            linear_velocity=np.asarray(
                observations["linear_velocity"][:],
                dtype=float,
            ),
            angular_velocity=np.asarray(
                observations["angular_velocity"][:],
                dtype=float,
            ),
            contact_active=np.asarray(
                observations["contact_active"][:],
                dtype=bool,
            ),
            contact_mode=np.asarray(
                observations["contact_mode"][:],
                dtype=np.int8,
            ),
            penetration=np.asarray(
                observations["penetration"][:],
                dtype=float,
            ),
            normal_force=np.asarray(
                observations["normal_force"][:],
                dtype=float,
            ),
            tangential_force=np.asarray(
                observations["tangential_force"][:],
                dtype=float,
            ),
            contact_velocity=np.asarray(
                observations["contact_velocity"][:],
                dtype=float,
            ),
            tangential_memory=np.asarray(
                observations["tangential_memory"][:],
                dtype=float,
            ),
            parameters=parameters,
        )
    validate_trajectory_shapes(trajectory)
    return trajectory


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


@dataclass(frozen=True, slots=True)
class InitialGeometryFeatures:
    target_surface_ids: tuple[str, ...]
    target_clearance: FloatArray
    incoming_normal_speed: FloatArray
    tangential_speed: FloatArray
    floor_clearance: FloatArray

def determine_initial_target_surface(
    position: FloatArray,
    velocity: FloatArray,
    radius: float,
    geometry: EnvironmentGeometry,
) -> tuple[str, float, float, float]:
    candidates: list[tuple[float, str, float, float]] = []

    for surface in geometry.surfaces:
        normal_velocity = float(
            np.dot(velocity, surface.normal)
        )

        # The ball approaches the admissible plane when its velocity points opposite to the inward normal.
        approach_speed = normal_velocity

        if approach_speed <= 0.0:
            continue

        clearance = float(
            surface.sphere_clearance(
                position[None, :], radius
            )[0]
        )

        time_to_plane = (
            clearance / approach_speed
            if clearance >= 0.0 else 0.0
        )

        tangential_velocity = (
            velocity
            - normal_velocity * surface.normal
        )

        candidates.append(
            (
                time_to_plane,
                surface.surface_id,
                approach_speed,
                float(np.linalg.norm(tangential_velocity)),
            )
        )

    if not candidates:
        return (
            "none",
            float("nan"),
            0.0,
            float(np.linalg.norm(velocity)),
        )

    _, surface_id, normal_speed, tangential_speed = min(
        candidates, key=lambda candidate: candidate[0],
    )

    surface = geometry.surface(surface_id)

    clearance = float(
        surface.sphere_clearance(
            position[None, :],
            radius,
        )[0]
    )

    return (
        surface_id,
        clearance,
        normal_speed,
        tangential_speed,
    )

def compute_initial_geometry_features(
        data: InitialConditionData,
        geometry: EnvironmentGeometry,
) -> InitialGeometryFeatures:
    target_surface_ids: list[str] = []
    target_clearance: list[float] = []
    incoming_normal_speed: list[float] = []
    tangential_speed: list[float] = []
    floor_clearance: list[float] = []

    floor = next(
        (surface for surface in geometry.surfaces if surface.surface_id == "floor"),
        None,
    )

    for position, velocity, radius in zip(data.position, data.linear_velocity, data.radii, strict=True):
        (
            surface_id,
            clearance,
            normal_speed,
            tangent_speed,
        ) = determine_initial_target_surface(
            position, velocity, radius, geometry
        )

        target_surface_ids.append(surface_id)
        target_clearance.append(clearance)
        incoming_normal_speed.append(normal_speed)
        tangential_speed.append(tangent_speed)

        if floor is None:
            floor_clearance.append(float("nan"))
        else:
            floor_clearance.append(
                float(
                    floor.sphere_clearance(
                        position[None, :], radius
                    )[0]
                )
            )

    return InitialGeometryFeatures(
        target_surface_ids=tuple(target_surface_ids),
        target_clearance=np.asarray(target_clearance),
        incoming_normal_speed=np.asarray(incoming_normal_speed),
        tangential_speed=np.asarray(tangential_speed),
        floor_clearance=np.asarray(floor_clearance),
    )



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


def apply_equal_3d_axes_to_geometry(
    axes: Iterable,
    points: FloatArray,
    geometry: EnvironmentGeometry,
    radius: float,
) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)

    for surface in geometry.surfaces:
        dominant_axis = int(
            np.argmax(np.abs(surface.normal))
        )
        coordinate = surface.point[dominant_axis]

        minimum[dominant_axis] = min(
            minimum[dominant_axis], coordinate - radius
        )
        maximum[dominant_axis] = max(
            maximum[dominant_axis], coordinate + radius
        )

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

def draw_environment_cross_section(
    axis,
    geometry: EnvironmentGeometry,
) -> None:
    for surface in geometry.surfaces:
        normal = surface.normal
        dominant_axis = int(
            np.argmax(np.abs(normal))
        )

        if dominant_axis == 0:
            axis.axvline(
                surface.point[0],
                color=SURFACE_COLORS.get(
                    surface.surface_id,
                    "gray",
                ),
                linestyle="--",
                linewidth=2.0,
                label=surface.surface_id,
            )

        elif dominant_axis == 2:
            axis.axhline(
                surface.point[2],
                color=SURFACE_COLORS.get(
                    surface.surface_id,
                    "gray",
                ),
                linestyle="--",
                linewidth=2.0,
                label=surface.surface_id,
            )

    axis.legend(
        loc="best",
        fontsize="small",
    )

def plot_initial_condition_spread(
    data: InitialConditionData,
    geometry: EnvironmentGeometry,
    output: str | Path | None = None,
    show: bool = True,
    max_scatter_points: int = 10_000,
    random_seed: int = 0,
) -> Figure:
    features = compute_initial_geometry_features(
        data,
        geometry,
    )

    number_of_trajectories = len(
        data.trajectory_ids
    )

    rng = np.random.default_rng(
        random_seed
    )

    if number_of_trajectories > max_scatter_points:
        indices = np.sort(
            rng.choice(
                number_of_trajectories,
                max_scatter_points,
                replace=False,
            )
        )
    else:
        indices = np.arange(
            number_of_trajectories
        )

    figure, axes = plt.subplots(2, 3,
        figsize=(17, 10),
        constrained_layout=True,
    )

    cross_section_axis = axes[0, 0]
    lateral_axis = axes[0, 1]
    velocity_axis = axes[0, 2]
    impact_axis = axes[1, 0]
    spin_axis = axes[1, 1]
    histogram_axis = axes[1, 2]

    selected_position = data.position[indices]
    selected_velocity = data.linear_velocity[
        indices
    ]

    cross_section = cross_section_axis.scatter(
        selected_position[:, 0],
        selected_position[:, 2],
        c=data.linear_speed[indices],
        cmap="viridis",
        s=12,
        alpha=0.6,
        edgecolors="none",
    )

    figure.colorbar(
        cross_section,
        ax=cross_section_axis,
        label="Initial linear speed [m/s]",
    )

    draw_environment_cross_section(
        cross_section_axis,
        geometry,
    )

    cross_section_axis.set(
        title="Initial positions in channel cross-section",
        xlabel="x [m]",
        ylabel="z [m]",
    )
    cross_section_axis.grid(alpha=0.25)

    lateral_plot = lateral_axis.scatter(
        selected_position[:, 1],
        selected_position[:, 2],
        c=selected_position[:, 0],
        cmap="plasma",
        s=12,
        alpha=0.6,
        edgecolors="none",
    )

    figure.colorbar(
        lateral_plot,
        ax=lateral_axis,
        label="Initial x position [m]",
    )

    lateral_axis.set(
        title="Initial positions along unbounded direction",
        xlabel="y [m]",
        ylabel="z [m]",
    )
    lateral_axis.grid(alpha=0.25)

    velocity_plot = velocity_axis.scatter(
        selected_velocity[:, 0],
        selected_velocity[:, 2],
        c=np.abs(selected_velocity[:, 1]),
        cmap="cividis",
        s=12,
        alpha=0.6,
        edgecolors="none",
    )

    figure.colorbar(
        velocity_plot,
        ax=velocity_axis,
        label="|vy| [m/s]",
    )

    velocity_axis.set(
        title="Initial velocity coverage",
        xlabel="vx [m/s]",
        ylabel="vz [m/s]",
    )
    velocity_axis.grid(alpha=0.25)

    target_labels = sorted(
        set(features.target_surface_ids)
    )

    target_colors = {
        surface_id: color
        for surface_id, color
        in zip(
            target_labels,
            plt.get_cmap("tab10").colors,
            strict=False,
        )
    }

    for surface_id in target_labels:
        mask = np.asarray(
            [
                target == surface_id
                for target
                in features.target_surface_ids
            ]
        )

        impact_axis.scatter(
            features.incoming_normal_speed[
                mask
            ],
            features.tangential_speed[mask],
            s=14,
            alpha=0.6,
            label=surface_id,
            color=target_colors[
                surface_id
            ],
            edgecolors="none",
        )

    impact_axis.set(
        title="Predicted first-surface approach",
        xlabel="Incoming normal speed [m/s]",
        ylabel="Tangential speed [m/s]",
    )
    impact_axis.legend()
    impact_axis.grid(alpha=0.25)

    spin_plot = spin_axis.scatter(
        data.angular_velocity[
            indices,
            0,
        ],
        data.angular_velocity[
            indices,
            2,
        ],
        c=data.angular_speed[indices],
        cmap="magma",
        s=12,
        alpha=0.6,
        edgecolors="none",
    )

    figure.colorbar(
        spin_plot,
        ax=spin_axis,
        label="Initial |ω| [rad/s]",
    )

    spin_axis.set(
        title="Initial spin coverage",
        xlabel="ωx [rad/s]",
        ylabel="ωz [rad/s]",
    )
    spin_axis.grid(alpha=0.25)

    histogram_axis.hist(
        data.linear_speed,
        bins="auto",
        alpha=0.55,
        label="|v| [m/s]",
    )

    histogram_axis.hist(
        data.angular_speed,
        bins="auto",
        alpha=0.55,
        label="|ω| [rad/s]",
    )

    histogram_axis.set(
        title="Initial speed distributions",
        xlabel="Magnitude",
        ylabel="Trajectory count",
    )
    histogram_axis.legend()
    histogram_axis.grid(alpha=0.25)

    figure.suptitle(
        f"Initial-condition coverage — "
        f"{geometry.kind} — "
        f"{number_of_trajectories:,} trajectories",
        fontsize=16,
    )

    if output is not None:
        figure.savefig(
            output,
            dpi=180,
            bbox_inches="tight",
        )

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



def add_environment_surfaces(
    axis,
    geometry: EnvironmentGeometry,
    position: FloatArray,
    radius: float,
) -> None:
    minimum = np.min(position, axis=0)
    maximum = np.max(position, axis=0)

    margin = np.maximum(
        0.15 * (maximum - minimum),
        2.0 * radius,
    )

    minimum -= margin
    maximum += margin

    for surface in geometry.surfaces:
        add_plane_surface(
            axis, surface, minimum, maximum
        )


def add_plane_surface(
    axis,
    surface: PlaneGeometry,
    minimum: FloatArray,
    maximum: FloatArray,
) -> None:
    normal = surface.normal
    dominant_axis = int(
        np.argmax(np.abs(normal))
    )

    color = SURFACE_COLORS.get(
        surface.surface_id, "gray"
    )

    if dominant_axis == 0:
        y = np.linspace(minimum[1], maximum[1], 2)
        z = np.linspace(minimum[2], maximum[2],2)
        y_grid, z_grid = np.meshgrid(y, z)
        x_grid = np.full_like(y_grid, surface.point[0])
    elif dominant_axis == 1:
        x = np.linspace(minimum[0], maximum[0], 2)
        z = np.linspace(minimum[2], maximum[2],2)
        x_grid, z_grid = np.meshgrid(x, z)
        y_grid = np.full_like(x_grid, surface.point[1])
    else:
        x = np.linspace(minimum[0], maximum[0], 2)
        y = np.linspace(minimum[1], maximum[1],2)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.full_like(x_grid, surface.point[2])

    axis.plot_surface(
        x_grid, y_grid, z_grid,
        color=color,
        alpha=0.18,
        edgecolor=color,
        linewidth=0.5,
    )


def mark_contact_points(
    axis,
    trajectory: TrajectoryData,
) -> None:
    for surface_index, surface_id in enumerate(trajectory.surface_ids):
        mask = trajectory.contact_active[:, surface_index]

        if not np.any(mask):
            continue

        positions = trajectory.position[mask]
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=SURFACE_COLORS.get(
                surface_id, "black",
            ),
            marker=SURFACE_MARKERS.get(
                surface_id, "o",
            ),
            s=24,
            alpha=0.9,
            label=f"Contact: {surface_id}",
            depthshade=False,
        )


def plot_trajectory_3d(
    trajectory: TrajectoryData,
    geometry: EnvironmentGeometry,
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
        add_environment_surfaces(axis, geometry, position, radius)
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
        axis.legend(loc="upper right", fontsize="small")

    apply_equal_3d_axes_to_geometry(axes, position, geometry, radius)

    surface_counts = {
        surface_id: int(
            np.count_nonzero(trajectory.contact_active[:, surface_index])
        )
        for surface_index, surface_id in enumerate(trajectory.surface_ids)
    }

    contact_summary = ", ".join(
        f"{name}: {count}"
        for name, count in surface_counts.items()
    )

    figure.suptitle(
        f"Trajectory {trajectory.trajectory_id} — "
        f"{geometry.kind} - {contact_summary}",
        fontsize=15,
    )

    if output is not None:
        figure.savefig(output, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return figure


def compute_mechanical_energy(
    trajectory: TrajectoryData,
    gravity_vector: float,
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
    gravitational = -mass * trajectory.position @ np.asarray(
        gravity_vector, dtype=float,
    )

    normal_stiffness = trajectory.parameters["normal_stiffness"]
    normal_elastic_by_surface = (
        0.4 * normal_stiffness * np.maximum(
            trajectory.penetration, 0.0,
        ) ** 2.5
    )
    normal_elastic = np.sum(normal_elastic_by_surface, axis=1)

    tangential_stiffness = trajectory.parameters["tangential_stiffness"]
    tangential_elastic_by_surface = (
        0.5 * tangential_stiffness * np.sum(
            trajectory.tangential_memory**2,
            axis=2
        )
    )
    tangential_elastic = np.sum(tangential_elastic_by_surface, axis=1)

    total = (
        translational
        + rotational
        + gravitational
        + normal_elastic
        + tangential_elastic
    )

    return {
        "translational": translational,
        "rotational": rotational,
        "gravitational": gravitational,
        "normal_elastic": normal_elastic,
        "tangential_elastic": tangential_elastic,
        "total": total,
    }


def contiguous_true_regions(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)

    return list(zip(starts, stops, strict=True))


def add_surface_contact_shading(
    axis,
    trajectory: TrajectoryData,
) -> None:
    for surface_index, surface_id in enumerate(trajectory.surface_ids):
        mask = trajectory.contact_active[:, surface_index]

        for start, stop in contiguous_true_regions(mask):
            first_time = trajectory.time[start]
            last_index = min(stop, len(trajectory.time) - 1)
            last_time = trajectory.time[last_index]

            axis.axvspan(
                first_time,
                last_time,
                color=CONTACT_SHADING_COLORS.get(
                    surface_id,
                    "gray",
                ),
                alpha=0.08,
            )

def plot_trajectory_diagnostics(
    trajectory: TrajectoryData,
    gravity_vector: FloatArray,
    output: str | Path | None = None,
    show: bool = True,
) -> Figure:
    time = trajectory.time
    energy = compute_mechanical_energy(trajectory, gravity_vector)

    figure, axes = plt.subplots(4, 2,
        figsize=(16, 15),
        sharex=True,
        constrained_layout=True,
    )

    velocity_axis = axes[0, 0]
    angular_axis = axes[0, 1]
    normal_force_axis = axes[1, 0]
    tangential_force_axis = axes[1, 1]
    penetration_axis = axes[2, 0]
    mode_axis = axes[2, 1]
    energy_axis = axes[3, 0]
    active_axis = axes[3, 1]
    

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

    for surface_index, surface_id in enumerate(trajectory.surface_ids):
        normal_magnitude = np.linalg.norm(
            trajectory.normal_force[:, surface_index, :], axis=1
        )
        tangential_magnitude = np.linalg.norm(
            trajectory.tangential_force[:, surface_index, :], axis=1
        )
        normal_force_axis.plot(
            time, 
            normal_magnitude, 
            label=surface_id,
            )
        tangential_force_axis.plot(
            time, 
            tangential_magnitude, 
            label=surface_id,
            )
        penetration_axis.plot(
            time,
            1_000.0 * trajectory.penetration[:, surface_index],
            label=surface_id,
        )
        mode_axis.step(
            time,
            trajectory.contact_mode[:, surface_index] + 3 * surface_index,
            where="post",
            label=surface_id,
        )
        active_axis.step(
            time,
            trajectory.contact_active[:, surface_index].astype(float) + 1.25 * surface_index,
            where="post",
            label=surface_id
        )

    velocity_axis.set_ylabel("Velocity [m/s]")
    velocity_axis.set_title("Linear-velocity components")
    velocity_axis.legend()
    velocity_axis.grid(alpha=0.25)

    angular_axis.set_ylabel("Angular velocity [rad/s]")
    angular_axis.set_title("Angular-velocity components")
    angular_axis.legend()
    angular_axis.grid(alpha=0.25)

    normal_force_axis.set_ylabel("Force [N]")
    normal_force_axis.set_title("Normal contact force by surface")
    normal_force_axis.legend()
    normal_force_axis.grid(alpha=0.25)

    tangential_force_axis.set_ylabel("Force [N]")
    tangential_force_axis.set_title("Tangential contact force by surface")
    tangential_force_axis.legend()
    tangential_force_axis.grid(alpha=0.25)

    penetration_axis.set_ylabel("Penetration [mm]")
    penetration_axis.set_title("Compliant penetration")
    penetration_axis.legend()
    penetration_axis.grid(alpha=0.25)

    mode_axis.set_ylabel("Offset contact mode")
    mode_axis.set_title("Contact mode by surface")
    mode_axis.legend()
    mode_axis.grid(alpha=0.25)

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
        label="Normal elastic",
    )
    energy_axis.plot(
        time,
        energy["tangential_elastic"],
        label="Tangential elastic",
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
    energy_axis.legend(ncols=2, fontsize="small")
    energy_axis.grid(alpha=0.25)

    active_axis.set(
        title="Active surfaces",
        xlabel="Time [s]",
        ylabel="Offset activity indicator"
    )
    active_axis.legend()
    active_axis.grid(alpha=0.25)

    for axis in axes.flat:
        add_surface_contact_shading(
            axis,
            trajectory,
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
"""Evaluation figures for whole windows and rollouts.

Every function takes the ``records`` dictionary built by ``evaluator._collect``:
NumPy arrays with one leading row per evaluated window. Shapes use N windows, T context frames, I = T - 1 intervals, F future frames and H = T - 1 + F horizons of the rollout that starts at the first context frame.

Two rollouts are shown:
    rollout from current state: starts at the last context frame (what a world
        model would do), evaluated on the F future frames.
    rollout from first frame:   starts at the estimate for frame 0 and runs open
        loop through the rest of the window and the future frames. It is longer
        (H steps), so it shows how estimation errors grow with the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

COMPONENTS = ("x", "y", "z")
COMPONENT_COLOURS = ("tab:blue", "tab:orange", "tab:green")
# Marker directions and colours as rendered (ball_simulator/rendering/blender/render_trajectory.py).
MARKER_DIRECTIONS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3.0)
MARKER_COLOURS = ("#e60d40", "#0d40f2", "#0dcc33", "#f2f20d")

STYLE_TRUE = dict(linestyle="-", linewidth=1.6)
STYLE_ESTIMATE = dict(linestyle="none", marker="o", markersize=4)
STYLE_ROLLOUT_CURRENT = dict(linestyle="--", marker="x", markersize=5, linewidth=1.4)
STYLE_ROLLOUT_FIRST = dict(linestyle=":", linewidth=1.6)


# --------------------------------------------------------------------------------------
# Rollout error versus horizon
# --------------------------------------------------------------------------------------

def horizon_statistics(records: dict[str, np.ndarray]) -> dict[str, dict[str, list[float]]]:
    """Median, quartiles and p95 of rollout errors per horizon step, for both rollouts."""

    def describe(errors: np.ndarray) -> dict[str, list[float]]:
        return {
            "median": np.median(errors, axis=0).tolist(),
            "p25": np.percentile(errors, 25, axis=0).tolist(),
            "p75": np.percentile(errors, 75, axis=0).tolist(),
            "p95": np.percentile(errors, 95, axis=0).tolist(),
        }

    first_horizon = records["rollout_first_horizon"].mean(axis=0)
    current_horizon = records["rollout_current_horizon"].mean(axis=0)
    return {
        "rollout_from_first_frame": {
            "horizon_s": first_horizon.tolist(),
            "position_error_m": describe(records["rollout_first_position_error"]),
            "orientation_error_deg": describe(records["rollout_first_orientation_error_deg"]),
        },
        "rollout_from_current_state": {
            "horizon_s": current_horizon.tolist(),
            "position_error_m": describe(records["rollout_current_position_error"]),
            "orientation_error_deg": describe(records["rollout_current_orientation_error_deg"]),
        },
    }


def rollout_horizon_plot(path: Path, records: dict[str, np.ndarray]) -> None:
    """Rollout position and orientation error against the prediction horizon.

    The dashed grey line is the median error of the per-frame estimates: the
    accuracy obtained by simply observing every frame. A rollout that stays close
    to it adds little error of its own.
    """
    statistics = horizon_statistics(records)
    context_frames = records["frame_time"].shape[1]
    observation_floor = {
        "position_error_m": np.median(
            np.linalg.norm(records["pred_frame_position"] - records["true_frame_position"], axis=-1)
        ),
        "orientation_error_deg": np.median(records["frame_orientation_error_deg"]),
    }
    panels = (("position_error_m", "Position error [mm]", 1e3), ("orientation_error_deg", "Orientation error [deg]", 1.0))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, (key, label, scale) in zip(axes, panels, strict=True):
        for name, colour, text in (
            ("rollout_from_first_frame", "tab:purple", "from first frame"),
            ("rollout_from_current_state", "tab:red", "from current state"),
        ):
            entry = statistics[name]
            steps = np.arange(1, len(entry["horizon_s"]) + 1)
            median = np.asarray(entry[key]["median"]) * scale
            axis.plot(steps, median, marker="o", color=colour, label=f"{text}: median")
            axis.fill_between(
                steps, np.asarray(entry[key]["p25"]) * scale, np.asarray(entry[key]["p75"]) * scale,
                color=colour, alpha=0.2, label=f"{text}: 25-75 %",
            )
            axis.plot(steps, np.asarray(entry[key]["p95"]) * scale, linestyle=":", color=colour, label=f"{text}: p95")
        axis.axhline(observation_floor[key] * scale, color="grey", linestyle="--", label="per-frame estimate (median)")
        axis.axvline(context_frames - 0.5, color="black", linewidth=0.8)
        axis.annotate("first-frame rollout leaves\nthe observed window", (context_frames - 0.4, axis.get_ylim()[1] * 0.9),
                      fontsize=8, va="top")
        axis.set_xlabel("Horizon [frames of 10 ms]")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Open-loop rollout error vs horizon ({len(records['frame_time'])} test windows)")
    figure.savefig(path, dpi=170)
    plt.close(figure)


# --------------------------------------------------------------------------------------
# Per-window trajectory figures
# --------------------------------------------------------------------------------------

def _style_legend(axis) -> None:
    handles = [
        plt.Line2D([], [], color="black", **STYLE_TRUE, label="true"),
        plt.Line2D([], [], color="black", **STYLE_ESTIMATE, label="estimate per frame / interval"),
        plt.Line2D([], [], color="black", marker="*", markersize=10, linestyle="none", label="current-state estimate"),
        plt.Line2D([], [], color="black", **STYLE_ROLLOUT_CURRENT, label="rollout from current state"),
        plt.Line2D([], [], color="black", **STYLE_ROLLOUT_FIRST, label="rollout from first frame"),
    ]
    handles += [plt.Line2D([], [], color=c, linewidth=3, label=n) for n, c in zip(COMPONENTS, COMPONENT_COLOURS)]
    axis.legend(handles=handles, fontsize=7, loc="best")


def _velocity_panel(axis, index: int, records: dict[str, np.ndarray], name: str, unit: str) -> None:
    frame_time, interval_time = records["frame_time"][index], records["interval_time"][index]
    for component, colour in enumerate(COMPONENT_COLOURS):
        axis.plot(frame_time, records[f"true_frame_{name}"][index, :, component], color=colour, **STYLE_TRUE)
        axis.plot(interval_time, records[f"pred_interval_{name}"][index, :, component], color=colour, **STYLE_ESTIMATE)
        axis.plot(frame_time[-1], records[f"pred_{name}"][index, component], color=colour, marker="*",
                  markersize=11, markeredgecolor="black", linestyle="none")
    axis.set_title(f"{name.replace('_', ' ').capitalize()} [{unit}]  (estimates are interval means)")


def _window_figure(path: Path, index: int, records: dict[str, np.ndarray]) -> None:
    frame_time, future_time = records["frame_time"][index], records["future_time"][index]
    all_time = np.concatenate((frame_time, future_time))
    rollout_first_time = all_time[1:]
    true_position = np.concatenate((records["true_frame_position"][index], records["true_future_position"][index]))

    figure = plt.figure(figsize=(18, 9.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    position_axis = figure.add_subplot(grid[0, 0])
    linear_axis = figure.add_subplot(grid[0, 1])
    angular_axis = figure.add_subplot(grid[0, 2])
    position_error_axis = figure.add_subplot(grid[1, 0])
    orientation_error_axis = figure.add_subplot(grid[1, 1])
    path_axis = figure.add_subplot(grid[1, 2], projection="3d")

    # Position components.
    for component, colour in enumerate(COMPONENT_COLOURS):
        position_axis.plot(all_time, true_position[:, component], color=colour, **STYLE_TRUE)
        position_axis.plot(frame_time, records["pred_frame_position"][index, :, component], color=colour, **STYLE_ESTIMATE)
        position_axis.plot(future_time, records["rollout_current_position"][index, :, component], color=colour, **STYLE_ROLLOUT_CURRENT)
        position_axis.plot(rollout_first_time, records["rollout_first_position"][index, :, component], color=colour, **STYLE_ROLLOUT_FIRST)
    position_axis.set_title("Position [m]")
    _style_legend(position_axis)

    _velocity_panel(linear_axis, index, records, "linear_velocity", "m/s")
    _velocity_panel(angular_axis, index, records, "angular_velocity", "rad/s")

    # Error norms over time.
    frame_position_error = np.linalg.norm(records["pred_frame_position"][index] - records["true_frame_position"][index], axis=-1)
    position_error_axis.plot(frame_time, 1e3 * frame_position_error, color="black", **STYLE_ESTIMATE, label="estimate per frame")
    position_error_axis.plot(future_time, 1e3 * records["rollout_current_position_error"][index], color="tab:red",
                             **STYLE_ROLLOUT_CURRENT, label="rollout from current state")
    position_error_axis.plot(rollout_first_time, 1e3 * records["rollout_first_position_error"][index], color="tab:purple",
                             **STYLE_ROLLOUT_FIRST, label="rollout from first frame")
    position_error_axis.set_title("Position error [mm]")
    position_error_axis.legend(fontsize=8)

    orientation_error_axis.plot(frame_time, records["frame_orientation_error_deg"][index], color="black", **STYLE_ESTIMATE,
                                label="estimate per frame")
    orientation_error_axis.plot(future_time, records["rollout_current_orientation_error_deg"][index], color="tab:red",
                                **STYLE_ROLLOUT_CURRENT, label="rollout from current state")
    orientation_error_axis.plot(rollout_first_time, records["rollout_first_orientation_error_deg"][index], color="tab:purple",
                                **STYLE_ROLLOUT_FIRST, label="rollout from first frame")
    orientation_error_axis.set_title("Orientation error [deg]")
    orientation_error_axis.legend(fontsize=8)

    for axis in (position_axis, linear_axis, angular_axis, position_error_axis, orientation_error_axis):
        axis.axvspan(frame_time[-1], all_time[-1], color="grey", alpha=0.12, linewidth=0)
        axis.set_xlabel("Time [s]  (shaded: future frames, not observed)")
        axis.grid(alpha=0.25)

    # 3-D path.
    path_axis.plot(*true_position.T, color="black", **STYLE_TRUE, label="true")
    path_axis.plot(*records["pred_frame_position"][index].T, color="tab:blue", **STYLE_ESTIMATE, label="estimate per frame")
    path_axis.plot(*records["rollout_current_position"][index].T, color="tab:red", **STYLE_ROLLOUT_CURRENT, label="rollout from current")
    path_axis.plot(*records["rollout_first_position"][index].T, color="tab:purple", **STYLE_ROLLOUT_FIRST, label="rollout from first")
    path_axis.set_xlabel("x [m]"); path_axis.set_ylabel("y [m]"); path_axis.set_zlabel("z [m]")
    path_axis.set_title("Path in world frame")
    path_axis.legend(fontsize=7)

    figure.suptitle(f"Trajectory {records['trajectory_id'][index]}, window ending at t = {frame_time[-1]:.3f} s")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def trajectory_window_plots(output: Path, records: dict[str, np.ndarray], count: int) -> None:
    """One figure per window for the first ``count`` windows."""
    if count <= 0:
        return
    output.mkdir(parents=True, exist_ok=True)
    for index in range(min(count, len(records["frame_time"]))):
        name = f"window_{records['trajectory_id'][index]}_t{records['frame_time'][index, -1]:.3f}.png"
        _window_figure(output / name, index, records)


# --------------------------------------------------------------------------------------
# Rollouts drawn on the rendered frames
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PinholeCamera:
    """The Blender camera used for rendering: look-at from location to target, world z up.

    Blender's default sensor fit uses the sensor width for square or landscape
    images, so one focal length in pixels applies to both image axes.
    """

    location: np.ndarray
    target: np.ndarray
    focal_length_mm: float
    sensor_width_mm: float

    @classmethod
    def from_rendering_config(cls, path: str | Path) -> "PinholeCamera":
        camera = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["camera"]
        return cls(
            location=np.asarray(camera["location"], dtype=float),
            target=np.asarray(camera["target"], dtype=float),
            focal_length_mm=float(camera["focal_length_mm"]),
            sensor_width_mm=float(camera["sensor_width_mm"]),
        )

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = self.target - self.location
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right = right / np.linalg.norm(right)
        return right, np.cross(right, forward), forward

    def project(self, points: np.ndarray, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points [..., 3] -> pixel column, pixel row (0 at top) and depth in m."""
        right, up, forward = self._basis()
        relative = np.asarray(points, dtype=float) - self.location
        depth = relative @ forward
        focal_pixels = image_width * self.focal_length_mm / self.sensor_width_mm
        column = image_width / 2.0 + focal_pixels * (relative @ right) / depth
        row = image_height / 2.0 - focal_pixels * (relative @ up) / depth
        return column, row, depth

    def focal_pixels(self, image_width: int) -> float:
        return image_width * self.focal_length_mm / self.sensor_width_mm


def _draw_ball(axis, camera: PinholeCamera, image_shape, position, rotation, radius, *, edge, linestyle, markers, filled) -> None:
    """Outline of the ball and its camera-facing markers."""
    height, width = image_shape
    column, row, depth = camera.project(position, width, height)
    pixel_radius = camera.focal_pixels(width) * radius / depth
    axis.add_patch(plt.Circle((column - 0.5, row - 0.5), pixel_radius, fill=False, edgecolor=edge,
                              linestyle=linestyle, linewidth=1.6))
    if not markers:
        return
    towards_camera = camera.location - position
    for direction, colour in zip(MARKER_DIRECTIONS, MARKER_COLOURS, strict=True):
        surface_offset = radius * (rotation @ direction)
        if surface_offset @ towards_camera <= 0.0:  # marker on the far side of the ball
            continue
        marker_column, marker_row, _ = camera.project(position + surface_offset, width, height)
        axis.plot(marker_column - 0.5, marker_row - 0.5, marker="o", markersize=7,
                  markerfacecolor=colour if filled else "none", markeredgecolor=colour if not filled else "white",
                  markeredgewidth=1.8 if not filled else 0.8, linestyle="none")


def rollout_overlay_plots(
    output: Path,
    records: dict[str, np.ndarray],
    images: np.ndarray,
    camera: PinholeCamera,
    count: int,
    ball_radius: float,
) -> None:
    """Draw estimates and rollouts on the rendered frames, cropped around the true ball.

    images: uint8 [M, T + F, H, W, 3] for the first M windows (context then future frames).
    Truth: black outline with filled markers. Per-frame estimate (context) and
    rollout from the current state (future): hollow markers and an orange or red
    outline. Rollout from the first frame: purple dotted outline.
    """
    output.mkdir(parents=True, exist_ok=True)
    context_frames = records["frame_time"].shape[1]
    total_frames = images.shape[1]
    shown = sorted(set(np.linspace(0, context_frames - 1, 4).round().astype(int).tolist()) | set(range(context_frames, total_frames)))

    for index in range(min(count, len(images))):
        true_position = np.concatenate((records["true_frame_position"][index], records["true_future_position"][index]))
        true_rotation = np.concatenate((records["true_frame_rotation"][index], records["true_future_rotation"][index]))
        figure, axes = plt.subplots(1, len(shown), figsize=(2.6 * len(shown), 3.2), constrained_layout=True)
        for axis, frame in zip(np.atleast_1d(axes), shown, strict=True):
            image = images[index, frame]
            height, width = image.shape[:2]
            axis.imshow(image)
            _draw_ball(axis, camera, (height, width), true_position[frame], true_rotation[frame], ball_radius,
                       edge="black", linestyle="-", markers=True, filled=True)
            if frame < context_frames:
                estimate_position = records["pred_frame_position"][index, frame]
                estimate_rotation = records["pred_frame_rotation"][index, frame]
                edge, label = "tab:orange", f"context frame {frame}"
            else:
                future_index = frame - context_frames
                estimate_position = records["rollout_current_position"][index, future_index]
                estimate_rotation = records["rollout_current_rotation"][index, future_index]
                edge, label = "tab:red", f"future +{future_index + 1}"
            _draw_ball(axis, camera, (height, width), estimate_position, estimate_rotation, ball_radius,
                       edge=edge, linestyle="--", markers=True, filled=False)
            if frame > 0:
                _draw_ball(axis, camera, (height, width), records["rollout_first_position"][index, frame - 1],
                           records["rollout_first_rotation"][index, frame - 1], ball_radius,
                           edge="tab:purple", linestyle=":", markers=False, filled=False)

            # Crop around the true ball: three ball diameters wide.
            column, row, depth = camera.project(true_position[frame], width, height)
            half = max(24.0, 3.0 * camera.focal_pixels(width) * ball_radius / depth)
            axis.set_xlim(column - half, column + half)
            axis.set_ylim(row + half, row - half)
            axis.set_title(label, fontsize=9)
            axis.axis("off")
        handles = [
            plt.Line2D([], [], color="black", linestyle="-", label="true"),
            plt.Line2D([], [], color="tab:orange", linestyle="--", label="estimate (context)"),
            plt.Line2D([], [], color="tab:red", linestyle="--", label="rollout from current (future)"),
            plt.Line2D([], [], color="tab:purple", linestyle=":", label="rollout from first frame"),
        ]
        figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08),
                      facecolor="lightgrey")
        figure.suptitle(f"Trajectory {records['trajectory_id'][index]}: filled markers = truth, hollow = estimate", fontsize=10)
        figure.savefig(output / f"overlay_{records['trajectory_id'][index]}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ball_simulator.trajectories.dataset_geometry import load_environment_geometry
from ball_simulator.trajectories.visualization import load_trajectory, list_trajectory_ids
from ball_simulator.trajectories.shapes import (
    build_shape,
    marker_layout,
    render_body_description,
    shape_config_from_dataset,
    write_appearance,
)

from scipy.spatial.transform import Rotation

from .camera_axes import flip_camera_axes
from .camera_paths import camera_path_ids, load_camera_path
from .config import RenderConfig
from .resampling import resample_states

def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions [N, 4] lie in the same hemisphere (q and -q are the
    same rotation), which keeps keyframe interpolation between frames on the short arc."""
    result = quaternions.copy()
    for index in range(1, len(result)):
        if result[index] @ result[index - 1] < 0.0:
            result[index] = -result[index]
    return result


def prepare_render_job(
    dataset: str | Path,
    trajectory_id: str | int,
    config: RenderConfig,
    output_directory: str | Path,
    camera_paths: str | Path,
    job_directory: str | Path | None = None,
) -> Path:
    """Create a dependency-light JSON+NPZ job consumable by Blender's Python.

    Object and camera states are resampled to the frame times. The camera arrays stay in OpenCV
    axes (the dataset convention); Blender receives its own quaternion in
    ``camera_rotation_blender_wxyz`` and never converts axes itself.
    """
    dataset = Path(dataset).resolve()
    output_directory = Path(output_directory).resolve()
    job_directory = Path(job_directory or output_directory / "_jobs").resolve()
    job_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    trajectory = load_trajectory(
        dataset, trajectory_id, use_high_rate=config.source == "high_rate"
    )
    geometry = load_environment_geometry(dataset)
    states = resample_states(
        trajectory.time,
        trajectory.position,
        trajectory.quaternion_xyzw,
        trajectory.linear_velocity,
        trajectory.angular_velocity,
        config.frame_dt,
    )
    resolved_id = trajectory.trajectory_id
    camera = load_camera_path(camera_paths, resolved_id)
    camera_states = resample_states(
        camera.time,
        camera.position,
        camera.quaternion_xyzw,
        camera.linear_velocity,
        camera.angular_velocity,
        config.frame_dt,
    )
    if not np.allclose(camera_states["time"], states["time"]):
        raise ValueError(f"Camera and object time bases differ for trajectory {resolved_id}.")
    for name in ("position", "quaternion_xyzw", "linear_velocity", "angular_velocity"):
        states[f"camera_{name}"] = camera_states[name]
    camera_rotation = Rotation.from_quat(camera_states["quaternion_xyzw"]).as_matrix()
    blender_rotation = flip_camera_axes(camera_rotation)
    states["camera_rotation_blender_wxyz"] = _continuous_quaternions(
        np.roll(Rotation.from_matrix(blender_rotation).as_quat(canonical=False), 1, axis=-1)
    )
    state_path = job_directory / f"trajectory_{resolved_id}.npz"
    np.savez_compressed(state_path, **states)

    ball = config.ball
    shape = build_shape(shape_config_from_dataset(dataset), float(trajectory.parameters["radius"]))
    markers = marker_layout(
        shape, ball.marker_placement if ball.markers_enabled else "none", ball.marker_count
    )
    write_appearance(output_directory, shape, markers)

    surfaces = [
        {
            "surface_id": surface.surface_id,
            "point": surface.point.tolist(),
            "normal": surface.normal.tolist(),
        }
        for surface in geometry.surfaces
    ]
    job = {
        "schema_version": "1.0",
        "dataset": str(dataset),
        "trajectory_id": resolved_id,
        "camera_mode": camera.mode,
        "state_npz": str(state_path),
        "output_directory": str(output_directory / f"trajectory_{resolved_id}"),
        "body": render_body_description(
            shape, markers, ball.marker_angular_radius_degree, ball.marker_surface_offset,
        ),
        "environment": {
            "kind": geometry.kind,
            "surface_ids": list(geometry.surface_ids),
            "surfaces": surfaces,
            "channel_width": geometry.channel_width,
            "unbounded_axis": 0, #geometry.unbounded_axis, TODO
        },
        "render": config.model_dump(mode="json"),
    }
    job_path = job_directory / f"trajectory_{resolved_id}.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job_path


def prepared_output_directory(
    output_directory: str | Path,
    trajectory_id: str,
) -> Path:
    return Path(output_directory).resolve() / f"trajectory_{trajectory_id}"


def render_is_complete(
    output_directory: str | Path,
    trajectory_id: str,
) -> bool:
    return (
        prepared_output_directory(
            output_directory,
            trajectory_id,
        ) / "_SUCCESS"
    ).is_file()


def prepare_all_render_jobs(
    dataset: str | Path,
    config: RenderConfig,
    output_directory: str | Path,
    camera_paths: str | Path,
    *,
    resume: bool = False,
) -> list[Path]:
    """Jobs for every trajectory that has a camera path (trajectories without one were skipped
    because no camera path kept the object in view)."""
    with_camera = set(camera_path_ids(camera_paths))
    trajectory_ids = [
        trajectory_id for trajectory_id in list_trajectory_ids(dataset)
        if str(trajectory_id) in with_camera
    ]

    jobs: list[Path] = []

    for trajectory_id in trajectory_ids:
        if (
            resume
            and render_is_complete(
                output_directory,
                trajectory_id,
            )
        ):
            continue

        jobs.append(
            prepare_render_job(
                dataset=dataset,
                trajectory_id=trajectory_id,
                config=config,
                output_directory=output_directory,
                camera_paths=camera_paths,
            )
        )

    return jobs

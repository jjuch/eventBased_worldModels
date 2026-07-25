from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ball_simulator.trajectories.dataset_geometry import load_environment_geometry
from ball_simulator.trajectories.visualization import load_trajectory

from .config import RenderConfig
from .resampling import resample_states

def prepare_render_job(
    dataset: str | Path,
    trajectory_id: str | int,
    config: RenderConfig,
    output_directory: str | Path,
    job_directory: str | Path | None = None,
) -> Path:
    """Create a dependency-light JSON+NPZ job consumable by Blender's Python."""
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
    state_path = job_directory / f"trajectory_{resolved_id}.npz"
    np.savez_compressed(state_path, **states)

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
        "state_npz": str(state_path),
        "output_directory": str(output_directory / f"trajectory_{resolved_id}"),
        "radius": float(trajectory.parameters["radius"]),
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

from pathlib import Path

import h5py
import numpy as np

from ball_simulator.rendering.camera_optimiser import optimise_camera, project_spheres
from ball_simulator.rendering.config import RenderConfig

def _dataset(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        trajectories = handle.create_group("trajectories")
        for index, x in enumerate(np.linspace(-0.3, 0.3, 5)):
            group = trajectories.create_group(f"{index:08d}")
            observations = group.create_group("observations")
            positions = np.column_stack(
                (
                    np.linspace(x, x + 0.1, 20),
                    np.linspace(-0.2, 0.2, 20),
                    np.linspace(0.45, 0.85, 20),
                )
            )
            observations.create_dataset("position", data=positions)
            parameters = group.create_group("parameters")
            parameters.attrs["radius"] = 0.10


def test_camera_proposal_contains_all_spheres(tmp_path):
    dataset = tmp_path / "camera.h5"
    _dataset(dataset)
    config = RenderConfig(width=256, height=256)
    candidate = optimise_camera(
        dataset,
        config,
        minimum_diameter_px=20.0,
        target_diameter_px=60.0,
    )
    with h5py.File(dataset, "r") as handle:
        positions = np.concatenate(
            [np.asarray(group["observations/position"]) for group in handle["trajectories"].values()]
        )
    radii = np.full(len(positions), 0.10)
    u, v, diameter, depth = project_spheres(
        positions,
        radii,
        location=candidate.location,
        target=candidate.target,
        focal_length_mm=candidate.focal_length_mm,
        sensor_width_mm=config.camera.sensor_width_mm,
        width=config.width,
        height=config.height,
    )
    radius = diameter / 2.0
    assert np.all(depth > 0.0)
    assert np.all(u - radius > 0.0)
    assert np.all(u + radius < config.width)
    assert np.all(v - radius > 0.0)
    assert np.all(v + radius < config.height)
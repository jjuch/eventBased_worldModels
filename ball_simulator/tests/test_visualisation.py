import matplotlib

matplotlib.use("Agg")

import h5py
import numpy as np

import os
import tempfile

from ball_simulator.visualization import (
    load_initial_conditions,
    load_trajectory,
    plot_initial_condition_spread, plot_trajectory_3d,
    plot_trajectory_diagnostics
)
from ball_simulator.dataset_geometry import EnvironmentGeometry, PlaneGeometry

def test_load_initial_conditions(smoke_dataset):
    data = load_initial_conditions(smoke_dataset)

    assert data.position.ndim == 2
    assert data.position.shape[1] == 3
    assert data.linear_velocity.shape == data.position.shape
    assert data.angular_velocity.shape == data.position.shape


def test_load_trajectory_accepts_short_id(smoke_dataset):
    trajectory = load_trajectory(smoke_dataset, 0)

    assert trajectory.trajectory_id == "00000000"
    assert trajectory.position.shape[1] == 3
    assert trajectory.linear_velocity.shape == trajectory.position.shape


def test_visualizations_can_be_saved(smoke_dataset, tmp_path
):
    initial_data = load_initial_conditions(smoke_dataset)
    trajectory = load_trajectory(smoke_dataset, 0)

    initial_output = tmp_path / "initial.png"
    trajectory_output = tmp_path / "trajectory.png"
    diagnostic_output = tmp_path / "diagnostics.png"
    geometry = EnvironmentGeometry(
            kind="single-wall",
            surfaces=[PlaneGeometry(
                surface_id="test",
                point=np.asarray([0, 0, 0]),
                normal=np.asarray([1, 0, 0])
            )]
        )
    gravity_vector = np.asarray([0, 0, -1])

    plot_initial_condition_spread(
        initial_data,
        geometry=geometry,
        output=initial_output,
        show=False,
    )
    plot_trajectory_3d(
        trajectory,
        geometry=geometry,
        output=trajectory_output,
        show=False,
    )
    plot_trajectory_diagnostics(
        trajectory,
        gravity_vector=gravity_vector,
        output=diagnostic_output,
        show=False,
    )

    assert initial_output.exists()
    assert trajectory_output.exists()
    assert diagnostic_output.exists()
from ball_simulator.regime_inspection import (
    TrajectorySummary,
    _contact_episode_count,
    select_stratified_regimes,
)
from ball_simulator.dataset_geometry import load_environment_geometry, load_root_metadata, EnvironmentGeometry, PlaneGeometry
from ball_simulator.visualization import load_trajectory, load_initial_conditions, plot_initial_condition_spread
from ball_simulator.regime_inspection import plot_trajectory_3d, plot_trajectory_diagnostics

import numpy as np
import pytest


def test_contact_episode_count():
    assert _contact_episode_count(np.array([0, 1, 1, 0, 1, 0], dtype=bool)) == 2
    assert _contact_episode_count(np.zeros(5, dtype=bool)) == 0


def make_summary(index: int) -> TrajectorySummary:
    x = float(index + 1)
    return TrajectorySummary(
        trajectory_id=f"{index:08d}",
        initial_target_surface=str(x), 
        initial_normal_speed=x,
        initial_tangential_speed=11.0-x, 
        initial_linear_speed=12.0,
        initial_angular_speed=10.0*x, 
        incidence_cosine=x/11.0,
        friction=0.05*x, 
        normal_stiffness=1e4*x, 
        normal_damping=0.05*x,
        mass=0.2, 
        radius=0.05, 
        contact_episodes=2 if index == 9 else 1,
        total_surface_episodes=x,
        maximum_simultaneous_contacts=x,
        simultaneous_contact_samples=x,
        contacted_surfaces=(x-1, x),
        surface_epsiode_counts=(x-1, x),
        has_sticking=index % 2 == 0, 
        has_sliding=index % 2 == 1,
        maximum_penetration=1e-4*x, 
        peak_normal_force=100.0*x,
        final_linear_speed=5.0, 
        final_angular_speed=20.0,
    )


def test_selector_finds_key_regimes_without_reuse():
    geometry = EnvironmentGeometry(
        kind="single-wall",
        surfaces=[PlaneGeometry(
            surface_id="test",
            point=np.asarray([0, 0, 0]),
            normal=np.asarray([1, 0, 0])
        )]
    )
    selected = select_stratified_regimes(
        [make_summary(i) for i in range(20)],
        geometry=geometry)
    names = {item.regime for item in selected}
    ids = [item.trajectory_id for item in selected]
    assert "representative" in names
    assert "fast-impact" in names
    assert "grazing-impact" in names
    assert "repeated-contact" in names
    assert len(ids) == len(set(ids))


def test_u_box_geometry_loaded(
    u_box_dataset,
):
    geometry = load_environment_geometry(
        u_box_dataset
    )

    assert geometry.kind == "u-box"
    assert geometry.surface_ids == (
        "left_wall",
        "right_wall",
        "floor",
    )


def test_single_wall_trajectory_shapes(
    single_wall_dataset,
):
    trajectory = load_trajectory(
        single_wall_dataset,
        0,
    )

    number_of_samples = len(
        trajectory.time
    )

    assert trajectory.contact_active.shape == (
        number_of_samples,
        1,
    )
    assert trajectory.normal_force.shape == (
        number_of_samples,
        1,
        3,
    )

def test_u_box_trajectory_shapes(
    u_box_dataset,
):
    trajectory = load_trajectory(
        u_box_dataset,
        0,
    )

    number_of_samples = len(
        trajectory.time
    )

    assert trajectory.contact_active.shape == (
        number_of_samples,
        3,
    )
    assert trajectory.normal_force.shape == (
        number_of_samples,
        3,
        3,
    )


@pytest.mark.parametrize(
    "dataset_fixture",
    [
        "single_wall_dataset",
        "u_box_dataset",
    ],
)
def test_all_visualizations_render(
    dataset_fixture,
    request,
    tmp_path,
):
    dataset = request.getfixturevalue(
        dataset_fixture
    )

    geometry = load_environment_geometry(
        dataset
    )
    metadata = load_root_metadata(
        dataset
    )
    initial_conditions = (
        load_initial_conditions(dataset)
    )
    trajectory = load_trajectory(
        dataset,
        0,
    )

    initial_output = (
        tmp_path / "initial.png"
    )
    trajectory_output = (
        tmp_path / "trajectory.png"
    )
    diagnostics_output = (
        tmp_path / "diagnostics.png"
    )

    plot_initial_condition_spread(
        data=initial_conditions,
        geometry=geometry,
        output=initial_output,
        show=False,
    )

    plot_trajectory_3d(
        trajectory=trajectory,
        geometry=geometry,
        output=trajectory_output,
        show=False,
    )

    plot_trajectory_diagnostics(
        trajectory=trajectory,
        gravity_vector=np.asarray(
            metadata["simulation"]["gravity"],
            dtype=float,
        ),
        output=diagnostics_output,
        show=False,
    )

    assert initial_output.exists()
    assert trajectory_output.exists()
    assert diagnostics_output.exists()
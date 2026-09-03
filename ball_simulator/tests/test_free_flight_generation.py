from pathlib import Path

import numpy as np

from ball_simulator.trajectories.config import ExperimentConfig
from ball_simulator.trajectories.environments import EnvironmentFactory, EnvironmentKind
from ball_simulator.trajectories.sampling import InitialStateSamplerFactory, ParameterSampler
from ball_simulator.trajectories.simulator import BallSimulator


CONFIG_DIRECTORY = Path(__file__).parents[1] / "configs" / "kinematic"


def _sample(config_name: str):
    config = ExperimentConfig.from_yaml(CONFIG_DIRECTORY / config_name)
    rng = np.random.default_rng(123)
    environment = EnvironmentFactory.create(EnvironmentKind.FREE_FLIGHT, config)
    parameters = ParameterSampler(config, rng).sample_parameters()
    state = InitialStateSamplerFactory.create(config, environment, rng).sample(parameters)
    trajectory = BallSimulator(config.simulation, environment).simulate(state, parameters)
    return config, parameters, state, trajectory


def test_translation_has_fixed_ball_and_no_rotation_or_contact():
    _, parameters, state, trajectory = _sample("translation_free_flight.yaml")
    assert parameters.mass == 0.20
    assert parameters.radius == 0.10
    np.testing.assert_allclose(state.quaternion_xyzw, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(state.angular_velocity, 0.0)
    assert not np.any(trajectory.observations["contact_active"])
    np.testing.assert_allclose(
        trajectory.observations["linear_velocity"],
        np.broadcast_to(state.linear_velocity, trajectory.observations["linear_velocity"].shape),
        atol=1.0e-10,
    )


def test_rotation_stays_in_place_and_has_no_contact():
    _, parameters, state, trajectory = _sample("rotation_in_place.yaml")
    np.testing.assert_allclose(state.position, [0.0, 0.0, 0.65])
    np.testing.assert_allclose(state.linear_velocity, 0.0)
    assert np.linalg.norm(state.angular_velocity) > 0.0
    assert parameters.radius == 0.10
    assert not np.any(trajectory.observations["contact_active"])
    np.testing.assert_allclose(
        trajectory.observations["position"],
        np.broadcast_to(state.position, trajectory.observations["position"].shape),
        atol=1.0e-10,
    )


def test_combined_varies_translation_and_rotation_without_contact():
    _, _, state, trajectory = _sample("combined_free_flight.yaml")
    assert np.linalg.norm(state.linear_velocity) > 0.0
    assert np.linalg.norm(state.angular_velocity) > 0.0
    assert not np.any(trajectory.observations["contact_active"])

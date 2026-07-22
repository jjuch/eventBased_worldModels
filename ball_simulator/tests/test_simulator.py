import numpy as np

from ball_simulator.config import SimulationConfig
from ball_simulator.models import RigidBodyState, SphereParameters
from ball_simulator.simulator import BallSimulator
from ball_simulator.environments import EnvironmentFactory, EnvironmentKind, ExperimentConfig


def params(friction=0.0, damping=0.2):
    return SphereParameters(0.2, 0.05, 1e5, damping, 2e4, 10.0, friction)


def state(x=0.2, vx=-2.0, vy=0.0, omega=None):
    return RigidBodyState(
        np.array([x, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([vx, vy, 0.0]), np.zeros(3) if omega is None else np.asarray(omega, float),
    )


def test_free_flight_matches_ballistics():
    cfg = SimulationConfig(duration=0.1, internal_dt=1e-4, observation_dt=0.01)
    exp_cfg = ExperimentConfig()
    environment = EnvironmentFactory.create(
        EnvironmentKind.SINGLE_WALL,
        exp_cfg,
    )

    tr = BallSimulator(cfg, environment).simulate(state(x=10.0, vx=1.0), params())
    t = tr.observations["time"][-1]
    expected_z = 1.0 - 0.5 * 9.81 * t**2
    assert abs(tr.observations["position"][-1, 2] - expected_z) < 2e-4


def test_frictionless_impact_does_not_change_spin_or_tangential_speed():
    cfg = SimulationConfig(duration=0.15, internal_dt=1e-5, observation_dt=0.001,
                           gravity=(0.0, 0.0, 0.0))
    exp_cfg = ExperimentConfig()
    environment = EnvironmentFactory.create(
        EnvironmentKind.SINGLE_WALL,
        exp_cfg,
    )
    initial = state(x=0.12, vx=-2.0, vy=1.5, omega=[2.0, 3.0, 4.0])
    tr = BallSimulator(cfg, environment).simulate(initial, params(friction=0.0))
    assert np.isclose(tr.observations["linear_velocity"][-1, 1], 1.5, atol=1e-8)
    assert np.allclose(tr.observations["angular_velocity"][-1], [2, 3, 4], atol=1e-8)
    assert tr.observations["linear_velocity"][-1, 0] > 0.0


def test_quaternion_stays_normalized():
    cfg = SimulationConfig(duration=0.05, internal_dt=1e-4, observation_dt=0.005,
                           gravity=(0.0, 0.0, 0.0))
    exp_cfg = ExperimentConfig()
    environment = EnvironmentFactory.create(
        EnvironmentKind.SINGLE_WALL,
        exp_cfg,
    )
    tr = BallSimulator(cfg, environment).simulate(state(x=10.0, vx=0.0, omega=[5, -3, 2]), params())
    norms = np.linalg.norm(tr.observations["quaternion_xyzw"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_ball_bounces_on_u_box_floor():
    cfg = SimulationConfig(duration=1.0, internal_dt=1e-4, observation_dt=0.005,
                           gravity=(0.0, 0.0, -9.81))
    exp_cfg = ExperimentConfig(default_environment='u-box')
    environment = EnvironmentFactory.create(
        EnvironmentKind.U_BOX,
        exp_cfg,
    )

    simulator = BallSimulator(
        cfg,
        environment,
    )

    initial = RigidBodyState(
        position=np.array([1.0, 0.0, 0.5]),
        quaternion_xyzw=np.array(
            [0.0, 0.0, 0.0, 1.0]
        ),
        linear_velocity=np.array(
            [0.0, 0.0, -1.0]
        ),
        angular_velocity=np.zeros(3),
    )

    trajectory = simulator.simulate(
        initial,
        params(),
    )

    surface_ids = trajectory.surface_ids
    floor_index = surface_ids.index("floor")

    assert np.any(
        trajectory.observations[
            "contact_active"
        ][:, floor_index]
    )
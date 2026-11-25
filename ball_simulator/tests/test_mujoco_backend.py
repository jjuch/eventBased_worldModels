"""MuJoCo backend against closed-form motion, conservation laws and the native simulator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from ball_simulator.trajectories.backends import create_simulator  # noqa: E402
from ball_simulator.trajectories.config import ExperimentConfig  # noqa: E402
from ball_simulator.trajectories.environments import (  # noqa: E402
    EnvironmentFactory,
    EnvironmentKind,
)
from ball_simulator.trajectories.models import RigidBodyState, SphereParameters  # noqa: E402
from ball_simulator.trajectories.mujoco_backend import (  # noqa: E402
    BODY_GEOM,
    MujocoSimulator,
    _write_state,
)
from ball_simulator.trajectories.shapes import build_shape  # noqa: E402


def _config(
    *,
    backend: str = "mujoco",
    shape: dict | None = None,
    gravity=(0.0, 0.0, -9.81),
    duration: float = 0.3,
    environment: str = "free-flight",
) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "default_environment": environment,
            "shape": shape or {"kind": "sphere"},
            "simulation": {
                "duration": duration,
                "internal_dt": 1.0e-3,
                "observation_dt": 1.0e-2,
                "gravity": list(gravity),
                "backend": backend,
            },
        }
    )


def _parameters(radius: float = 0.1, mass: float = 0.2, friction: float = 0.5) -> SphereParameters:
    return SphereParameters(
        mass=mass,
        radius=radius,
        normal_stiffness=1.5e5,
        normal_damping=0.3,
        tangential_stiffness=4.5e4,
        tangential_damping=10.0,
        friction=friction,
    )


def _state(
    position, velocity, angular_velocity, quaternion_xyzw=(0.0, 0.0, 0.0, 1.0)
) -> RigidBodyState:
    return RigidBodyState(
        position=np.asarray(position, dtype=float),
        quaternion_xyzw=np.asarray(quaternion_xyzw, dtype=float),
        linear_velocity=np.asarray(velocity, dtype=float),
        angular_velocity=np.asarray(angular_velocity, dtype=float),
    )


def _simulate(config: ExperimentConfig, state: RigidBodyState, params: SphereParameters):
    environment = EnvironmentFactory.create(EnvironmentKind(config.default_environment), config)
    return create_simulator(config, environment).simulate(state, params).observations


def test_free_flight_matches_closed_form():
    gravity = np.array([0.0, 0.0, -9.81])
    position, velocity = np.array([0.1, -0.2, 0.5]), np.array([1.0, 0.5, 2.0])
    angular_velocity = np.array([12.0, -7.0, 20.0])
    initial_rotation = Rotation.from_rotvec([0.3, -0.2, 0.1])
    state = _state(position, velocity, angular_velocity, initial_rotation.as_quat())

    observations = _simulate(_config(gravity=gravity), state, _parameters())

    time = observations["time"][:, None]
    expected_position = position + velocity * time + 0.5 * gravity * time**2
    np.testing.assert_allclose(observations["position"], expected_position, atol=1e-9)
    np.testing.assert_allclose(
        observations["linear_velocity"], velocity + gravity * time, atol=1e-9
    )
    # A sphere spins at constant world-frame angular velocity: R(t) = exp(t ω) R(0).
    expected_rotation = Rotation.from_rotvec(time * angular_velocity) * initial_rotation
    error = expected_rotation.inv() * Rotation.from_quat(observations["quaternion_xyzw"])
    assert np.max(error.magnitude()) < 1e-6
    np.testing.assert_allclose(
        observations["angular_velocity"],
        np.broadcast_to(angular_velocity, time.shape[:1] + (3,)),
        atol=1e-9,
    )


def test_sphere_free_flight_agrees_with_native_simulator():
    state = _state([0.0, 0.0, 0.6], [0.8, -0.4, 1.5], [10.0, 5.0, -8.0])
    native = _simulate(_config(backend="native"), state, _parameters())
    mujoco_run = _simulate(_config(backend="mujoco"), state, _parameters())
    # The native semi-implicit Euler step is first order: ~0.5 g dt t ≈ 1.5 mm after 0.3 s.
    np.testing.assert_allclose(mujoco_run["position"], native["position"], atol=2e-3)
    np.testing.assert_allclose(
        mujoco_run["angular_velocity"], native["angular_velocity"], atol=1e-9
    )
    relative = Rotation.from_quat(native["quaternion_xyzw"]).inv() * Rotation.from_quat(
        mujoco_run["quaternion_xyzw"]
    )
    assert np.max(relative.magnitude()) < 1e-6


def test_torque_free_box_conserves_angular_momentum_and_precesses():
    config = _config(
        shape={"kind": "box", "dimensions": (0.08, 0.14, 0.22)}, gravity=(0, 0, 0), duration=1.0
    )
    params = _parameters()
    state = _state(
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [6.0, 1.0, 8.0],
        Rotation.from_rotvec([0.4, 0.1, -0.3]).as_quat(),
    )
    observations = _simulate(config, state, params)

    inertia = build_shape(config.shape, params.radius).inertia(params.mass)
    rotations = Rotation.from_quat(observations["quaternion_xyzw"]).as_matrix()
    omega = observations["angular_velocity"]
    momentum = np.einsum("tij,jk,tlk,tl->ti", rotations, inertia, rotations, omega)
    energy = 0.5 * np.einsum("ti,tij,jk,tlk,tl->t", omega, rotations, inertia, rotations, omega)
    # RK4 at dt = 1 ms drifts by ~1e-5 relative over 1 s.
    np.testing.assert_allclose(
        momentum, np.broadcast_to(momentum[0], momentum.shape), rtol=1e-4, atol=1e-9
    )
    np.testing.assert_allclose(energy, energy[0], rtol=1e-4)
    # An asymmetric body precesses: the world-frame angular velocity is not constant.
    assert np.max(np.linalg.norm(omega - omega[0], axis=1)) > 0.5


def test_mesh_vertices_follow_the_body_pose():
    config = _config(shape={"kind": "pyramid", "dimensions": (0.12, 0.16, 0.20)}, gravity=(0, 0, 0))
    params = _parameters()
    environment = EnvironmentFactory.create(EnvironmentKind.FREE_FLIGHT, config)
    simulator = MujocoSimulator(config, environment)
    model, data, shape = simulator.build(params)
    state = _state(
        [0.3, -0.1, 0.7], [0, 0, 0], [0, 0, 0], Rotation.from_rotvec([0.5, -1.0, 0.3]).as_quat()
    )
    _write_state(data, state)
    mujoco.mj_forward(model, data)

    geom = model.geom(BODY_GEOM)
    mesh = model.mesh(geom.dataid[0])
    mesh_vertices = model.mesh_vert[mesh.vertadr[0] : mesh.vertadr[0] + mesh.vertnum[0]]
    world_mujoco = data.geom_xpos[geom.id] + mesh_vertices @ data.geom_xmat[geom.id].reshape(3, 3).T
    world_expected = state.position + Rotation.from_quat(state.quaternion_xyzw).apply(
        shape.vertices
    )
    distances = np.linalg.norm(world_expected[:, None] - world_mujoco[None], axis=-1)
    # MuJoCo stores mesh vertices in float32: agreement to ~10 nm, far below a pixel.
    assert np.max(distances.min(axis=1)) < 1e-6


def test_resting_sphere_contact_force_balances_weight():
    config = _config(environment="u-box", duration=0.5)
    params = _parameters()
    observations = _simulate(
        config, _state([1.0, 0.0, params.radius + 1e-3], [0, 0, 0], [0, 0, 0]), params
    )
    floor = config.environments.u_box.floor.surface_id
    index = ("left_wall", "right_wall", "floor").index(floor)
    assert observations["contact_active"][-1, index]
    np.testing.assert_allclose(
        observations["normal_force"][-1, index],
        [0.0, 0.0, params.mass * 9.81],
        rtol=0.02,
        atol=1e-3,
    )
    assert observations["position"][-1, 2] > params.radius - 2e-3


def test_cube_bounces_off_wall():
    config = _config(
        environment="single-wall",
        shape={"kind": "cube", "dimensions": (0.16, 0.16, 0.16)},
        gravity=(0, 0, 0),
        duration=0.4,
    )
    params = _parameters()
    initial = _state(
        [0.4, 0.0, 1.0],
        [-2.0, 0.3, 0.0],
        [0.0, 3.0, 5.0],
        Rotation.from_rotvec([0.2, 0.4, 0.1]).as_quat(),
    )
    observations = _simulate(config, initial, params)
    assert observations["contact_active"][:, 0].any()
    assert observations["linear_velocity"][-1, 0] > 0.0
    # The body never passes through the wall (vertices stay at x > -tolerance).
    shape = build_shape(config.shape, params.radius)
    for position, quaternion in zip(
        observations["position"], observations["quaternion_xyzw"], strict=True
    ):
        world = position + Rotation.from_quat(quaternion).apply(shape.vertices)
        assert world[:, 0].min() > -5e-3

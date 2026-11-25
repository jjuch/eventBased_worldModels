"""MuJoCo dynamics backend with the same inputs and Trajectory output as BallSimulator.

MuJoCo provides what the native simulator lacks for non-spherical bodies: full inertia tensors
(torque-free precession), convex mesh collisions and multi-point contact.

Conventions at the boundary:
- The body frame is the shape frame from ``shapes.py``, so the free-joint position is the
  centre of mass, as in the native simulator.
- A MuJoCo free joint stores the angular velocity in the body frame. It is converted to and from
  the world frame here, so datasets keep world-frame angular velocities.
- MuJoCo quaternions are wxyz; datasets store xyzw.
- Contact uses MuJoCo's soft-contact model (solref/solimp). Only ``friction`` and ``mass`` of
  SphereParameters are used; the native stiffness and damping parameters are not mapped.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .config import ExperimentConfig
from .environments import PlaneSurface, SimulationEnvironment
from .models import ContactDiagnostics, ContactMode, RigidBodyState, SphereParameters
from .physics import EnvironmentForces
from .shapes import RigidShape, build_shape
from .trajectory import Trajectory, TrajectoryBuffer

BODY_NAME = "body"
BODY_GEOM = "body_geom"
SURFACE_PREFIX = "surface_"
# A contact counts as sliding when friction is within this fraction of the Coulomb limit.
SLIDING_FRACTION = 0.99


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in np.ravel(values))


def _plane_quaternion_wxyz(normal: np.ndarray) -> np.ndarray:
    """Rotation taking a MuJoCo plane's local +z axis onto ``normal``, as wxyz."""
    rotation, _ = Rotation.align_vectors([normal], [[0.0, 0.0, 1.0]])
    return np.roll(rotation.as_quat(), 1)


def build_model_xml(
    shape: RigidShape,
    params: SphereParameters,
    environment: SimulationEnvironment,
    config: ExperimentConfig,
) -> str:
    """MJCF for one free body and the environment's planes."""
    simulation, settings = config.simulation, config.simulation.mujoco
    friction = (params.friction, settings.torsional_friction, settings.rolling_friction)
    solref = (settings.contact_time_constant, settings.contact_damping_ratio)
    contact = (
        f'friction="{_numbers(friction)}" solref="{_numbers(solref)}" '
        f'solimp="{_numbers(settings.solimp)}" condim="{settings.contact_dimension}"'
    )
    inertia = shape.inertia(params.mass)
    full_inertia = (
        inertia[0, 0],
        inertia[1, 1],
        inertia[2, 2],
        inertia[0, 1],
        inertia[0, 2],
        inertia[1, 2],
    )

    if shape.is_sphere:
        assets = ""
        body_geom = f'type="sphere" size="{_numbers(shape.sphere_radius)}"'
    else:
        assets = (
            f'<asset><mesh name="body_mesh" vertex="{_numbers(shape.vertices)}" '
            f'face="{_numbers(shape.triangles())}"/></asset>'
        )
        body_geom = 'type="mesh" mesh="body_mesh"'

    planes = "\n    ".join(
        f'<geom name="{SURFACE_PREFIX}{surface.surface_id}" type="plane" size="0 0 1" '
        f'pos="{_numbers(surface.point)}" '
        f'quat="{_numbers(_plane_quaternion_wxyz(surface.normal))}" {contact}/>'
        for surface in environment.surfaces
    )
    return f"""<mujoco model="rigid_body">
  <option timestep="{_numbers(simulation.internal_dt)}" integrator="{settings.integrator}"
          gravity="{_numbers(simulation.gravity)}"/>
  {assets}
  <worldbody>
    {planes}
    <body name="{BODY_NAME}">
      <freejoint name="body_joint"/>
      <inertial pos="0 0 0" mass="{_numbers(params.mass)}" fullinertia="{_numbers(full_inertia)}"/>
      <geom name="{BODY_GEOM}" {body_geom} {contact}/>
    </body>
  </worldbody>
</mujoco>"""


def _write_state(data: mujoco.MjData, state: RigidBodyState) -> None:
    rotation = Rotation.from_quat(state.quaternion_xyzw)
    data.qpos[:3] = state.position
    data.qpos[3:7] = np.roll(rotation.as_quat(), 1)
    data.qvel[:3] = state.linear_velocity
    data.qvel[3:6] = rotation.inv().apply(state.angular_velocity)


def _read_state(data: mujoco.MjData) -> RigidBodyState:
    quaternion_xyzw = np.roll(data.qpos[3:7], -1)
    quaternion_xyzw = quaternion_xyzw / np.linalg.norm(quaternion_xyzw)
    return RigidBodyState(
        position=data.qpos[:3].copy(),
        quaternion_xyzw=quaternion_xyzw,
        linear_velocity=data.qvel[:3].copy(),
        angular_velocity=Rotation.from_quat(quaternion_xyzw).apply(data.qvel[3:6]),
    )


def _inactive(surface_id: str) -> ContactDiagnostics:
    zero = np.zeros(3)
    return ContactDiagnostics(
        surface_id=surface_id,
        active=False,
        mode=ContactMode.FREE,
        penetration=0.0,
        normal_force=zero,
        tangential_force=zero,
        contact_velocity=zero,
        tangential_memory=zero,
    )


def contact_diagnostics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_geom: int,
    surfaces_by_geom: dict[int, PlaneSurface],
    state: RigidBodyState,
    friction: float,
) -> tuple[ContactDiagnostics, ...]:
    """Per-surface contact summary in the dataset's format, from MuJoCo's contact list.

    All contact points with one surface are combined: forces (N, acting on the body, world
    frame) are summed, penetration (m) is the deepest point, and the contact velocity (m/s) is
    that of the body point at the mean contact position. ``tangential_memory`` has no MuJoCo
    equivalent and is zero.
    """
    forces: dict[str, list[np.ndarray]] = {}
    depths: dict[str, list[float]] = {}
    positions: dict[str, list[np.ndarray]] = {}
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if body_geom not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom1 if contact.geom2 == body_geom else contact.geom2
        surface = surfaces_by_geom.get(int(other))
        if surface is None:
            continue
        # Contact-frame force; the frame's first row is the normal pointing from geom1 to geom2,
        # and the force acts on geom2.
        mujoco.mj_contactForce(model, data, index, wrench)
        force_on_geom2 = contact.frame.reshape(3, 3).T @ wrench[:3]
        force = force_on_geom2 if contact.geom2 == body_geom else -force_on_geom2
        forces.setdefault(surface.surface_id, []).append(force)
        depths.setdefault(surface.surface_id, []).append(-float(contact.dist))
        positions.setdefault(surface.surface_id, []).append(contact.pos.copy())

    diagnostics = []
    for surface in surfaces_by_geom.values():
        if surface.surface_id not in forces:
            diagnostics.append(_inactive(surface.surface_id))
            continue
        total = np.sum(forces[surface.surface_id], axis=0)
        normal_force = float(total @ surface.normal) * surface.normal
        tangential_force = total - normal_force
        lever = np.mean(positions[surface.surface_id], axis=0) - state.position
        contact_velocity = state.linear_velocity + np.cross(state.angular_velocity, lever)
        sliding = np.linalg.norm(tangential_force) >= SLIDING_FRACTION * friction * np.linalg.norm(
            normal_force
        )
        diagnostics.append(
            ContactDiagnostics(
                surface_id=surface.surface_id,
                active=True,
                mode=ContactMode.SLIDING if sliding else ContactMode.STICKING,
                penetration=max(0.0, max(depths[surface.surface_id])),
                normal_force=normal_force,
                tangential_force=tangential_force,
                contact_velocity=contact_velocity,
                tangential_memory=np.zeros(3),
            )
        )
    return tuple(diagnostics)


class MujocoSimulator:
    """Drop-in replacement for BallSimulator that integrates the dynamics with MuJoCo."""

    def __init__(self, config: ExperimentConfig, environment: SimulationEnvironment) -> None:
        self.config = config
        self.environment = environment
        # Drag, Magnus and rotational drag only; MuJoCo applies gravity itself.
        self.aerodynamics = EnvironmentForces(np.zeros(3))

    def _has_aerodynamics(self, params: SphereParameters) -> bool:
        return any(
            value != 0.0
            for value in (
                params.drag_coefficient,
                params.magnus_coefficient,
                params.rotational_drag,
            )
        )

    def build(self, params: SphereParameters) -> tuple[mujoco.MjModel, mujoco.MjData, RigidShape]:
        """Compile the model for one trajectory (mass, size and friction vary per trajectory)."""
        shape = build_shape(self.config.shape, params.radius)
        xml = build_model_xml(shape, params, self.environment, self.config)
        model = mujoco.MjModel.from_xml_string(xml)
        return model, mujoco.MjData(model), shape

    def simulate(
        self,
        initial_state: RigidBodyState,
        params: SphereParameters,
        store_high_rate: bool = False,
    ) -> Trajectory:
        simulation = self.config.simulation
        model, data, _ = self.build(params)
        body = model.body(BODY_NAME).id
        body_geom = model.geom(BODY_GEOM).id
        surfaces_by_geom = {
            model.geom(f"{SURFACE_PREFIX}{surface.surface_id}").id: surface
            for surface in self.environment.surfaces
        }
        aerodynamics = self._has_aerodynamics(params)

        _write_state(data, initial_state)
        mujoco.mj_forward(model, data)

        surface_ids = self.environment.surface_ids
        observations = TrajectoryBuffer(surface_ids=surface_ids)
        high_rate = TrajectoryBuffer(surface_ids=surface_ids) if store_high_rate else None

        dt = simulation.internal_dt
        observation_stride = round(simulation.observation_dt / dt)
        high_rate_stride = round((simulation.high_rate_dt or dt) / dt)
        number_of_steps = round(simulation.duration / dt)

        def record(buffer: TrajectoryBuffer, time: float) -> None:
            state = _read_state(data)
            contacts = contact_diagnostics(
                model, data, body_geom, surfaces_by_geom, state, params.friction
            )
            buffer.append(time=time, state=state, contacts=contacts)

        record(observations, 0.0)
        if high_rate is not None:
            record(high_rate, 0.0)

        for step in range(1, number_of_steps + 1):
            if aerodynamics:
                wrench = self.aerodynamics.evaluate(_read_state(data), params, None, dt)
                data.xfrc_applied[body, :3] = wrench.force
                data.xfrc_applied[body, 3:] = wrench.torque
            mujoco.mj_step(model, data)

            on_observation = step % observation_stride == 0
            on_high_rate = high_rate is not None and step % high_rate_stride == 0
            if on_observation or on_high_rate:
                # Recompute contacts for the new state; mj_step leaves those of the previous one.
                mujoco.mj_forward(model, data)
            if on_observation:
                record(observations, step * dt)
            if on_high_rate:
                record(high_rate, step * dt)

        return Trajectory(
            observations=observations.as_arrays(),
            parameters=params,
            environment_kind=self.environment.kind.value,
            surface_ids=surface_ids,
            high_rate=high_rate.as_arrays() if high_rate is not None else None,
        )

"""Select the dynamics backend configured in ``simulation.backend``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import ExperimentConfig
from .environments import SimulationEnvironment
from .simulator import BallSimulator

if TYPE_CHECKING:
    from .mujoco_backend import MujocoSimulator


def create_simulator(
    config: ExperimentConfig, environment: SimulationEnvironment
) -> BallSimulator | MujocoSimulator:
    """Native sphere simulator or MuJoCo; both return the same Trajectory format."""
    if config.simulation.backend == "mujoco":
        from .mujoco_backend import MujocoSimulator

        return MujocoSimulator(config, environment)
    return BallSimulator(config.simulation, environment)

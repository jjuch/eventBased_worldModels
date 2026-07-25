from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import trange

from .config import ExperimentConfig
from .environments import EnvironmentFactory, EnvironmentKind
from .sampling import InitialStateSamplerFactory, ParameterSampler
from .simulator import BallSimulator
from .storage import HDF5TrajectoryWriter


class DatasetGenerator:
    def __init__(
        self, 
        config: ExperimentConfig,
        environment_kind: EnvironmentKind | None = None,
    ) -> None:
        self.config = config
        self.environment_kind = (
            environment_kind or EnvironmentKind(config.default_environment)
        )
        self.rng = np.random.default_rng(config.dataset.seed)
        self.environment = EnvironmentFactory.create(
            self.environment_kind,
            config,
        )
        self.parameter_sampler = ParameterSampler(config, self.rng)
        self.initial_state_sampler = (
            InitialStateSamplerFactory.create(
                config, self.environment, self.rng,
            )
        )
        self.simulator = BallSimulator(config.simulation, self.environment)


    def generate(self, output: str | Path | None = None) -> Path:
        path = Path(output or self.config.dataset.output)

        dataset_config = self.config.dataset
        metadata = self.config.model_dump(mode="json")
        metadata["selected_environment"] = self.environment_kind.value
        metadata["environment_geometry"] = self.environment.metadata()

        with HDF5TrajectoryWriter(
            path, 
            dataset_config.compression, 
            dataset_config.compression_level,
            metadata=metadata,
        ) as writer:
            for i in trange(
                dataset_config.trajectories,
                desc=(
                    "Generating "
                    f"{self.environment_kind.value} " 
                    "trajectories"
                )
            ):
                params = self.parameter_sampler.sample_parameters()
                initial_state = self.initial_state_sampler.sample(params)
                trajectory = self.simulator.simulate(
                    initial_state=initial_state,
                    params=params,
                    store_high_rate=dataset_config.store_high_rate,
                )
                writer.write(i, trajectory)
        return path

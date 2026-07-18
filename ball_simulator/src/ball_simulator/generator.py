from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import trange

from .config import ExperimentConfig
from .sampling import ParameterSampler
from .simulator import SphereWallSimulator
from .storage import HDF5TrajectoryWriter


class DatasetGenerator:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.dataset.seed)
        self.sampler = ParameterSampler(config, self.rng)
        self.simulator = SphereWallSimulator(config.simulation)

    def generate(self, output: str | Path | None = None) -> Path:
        path = Path(output or self.config.dataset.output)
        d = self.config.dataset
        with HDF5TrajectoryWriter(path, d.compression, d.compression_level) as writer:
            for i in trange(d.trajectories, desc="Generating trajectories"):
                params = self.sampler.sample_parameters()
                initial = self.sampler.sample_initial_state(params)
                trajectory = self.simulator.simulate(initial, params, d.store_high_rate)
                writer.write(i, trajectory)
        return path

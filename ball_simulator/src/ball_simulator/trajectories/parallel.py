from __future__ import annotations

import os
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from multiprocessing import get_context
from pathlib import Path
from typing import Iterator

import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from .config import ExperimentConfig
from .environments import (
    EnvironmentFactory,
    EnvironmentKind,
    SimulationEnvironment,
)
from .sampling import InitialStateSamplerFactory, ParameterSampler
from .simulator import BallSimulator
from .storage import HDF5TrajectoryWriter
from .trajectory import Trajectory

_WORKER_CONFIG: ExperimentConfig | None = None
_WORKER_ENVIRONMENT: SimulationEnvironment | None = None
_WORKER_SIMULATOR: BallSimulator | None = None 

def recommended_simulation_workers() -> int:
    """Conservative automatic CPU worker count."""
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1

    return max(1, cpu_count - 1)


def _initialize_simulation_worker(
    serialized_config: dict[str, object],
    environment_value: str,
) -> None:
    """
    Initialize process-local immutable objects.
    This executes once in each worker process rather than once per trajectory.
    """
    global _WORKER_CONFIG
    global _WORKER_ENVIRONMENT
    global _WORKER_SIMULATOR

    _WORKER_CONFIG = ExperimentConfig.model_validate(serialized_config)
    environment_kind = EnvironmentKind(environment_value)
    _WORKER_ENVIRONMENT = EnvironmentFactory.create(environment_kind, _WORKER_CONFIG)
    _WORKER_SIMULATOR = BallSimulator(
        _WORKER_CONFIG.simulation,
        _WORKER_ENVIRONMENT,
    )


def _simulate_trajectory_worker(
    trajectory_index: int,
) -> tuple[int, Trajectory]:
    """
    Generate one trajectory with an index-specific deterministic seed.
    The output is independent of worker count and scheduling order.
    """
    if (
        _WORKER_CONFIG is None
        or _WORKER_ENVIRONMENT is None
        or _WORKER_SIMULATOR is None
    ):
        raise RuntimeError("Simulation worker was not initialised.")

    base_seed = _WORKER_CONFIG.dataset.seed

    # Deterministic independent random stream for every trajectory.
    seed_sequence = np.random.SeedSequence([base_seed, trajectory_index])
    rng = np.random.default_rng(seed_sequence)

    parameter_sampler = ParameterSampler(
        _WORKER_CONFIG, rng,
    )

    initial_state_sampler = (
        InitialStateSamplerFactory.create(
            _WORKER_CONFIG,
            _WORKER_ENVIRONMENT,
            rng,
        )
    )

    parameters = (
        parameter_sampler.sample_parameters()
    )

    initial_state = initial_state_sampler.sample(parameters)

    trajectory = _WORKER_SIMULATOR.simulate(
        initial_state=initial_state,
        params=parameters,
        store_high_rate=(
            _WORKER_CONFIG.dataset.store_high_rate
        )
    )

    return trajectory_index, trajectory


def _dataset_metadata(
    config: ExperimentConfig,
    environment_kind: EnvironmentKind,
    environment: SimulationEnvironment,
) -> dict[str, object]:
    metadata = config.model_dump(mode="json")
    metadata["selected_environment"] = environment_kind.value
    metadata["environment_geometry"] = environment.metadata()
    metadata["generation"] = {
        "parallel": True,
        "seed_strategy": (
            "numpy.random.SeedSequence([base_seed, trajectory_index])"
        ),
    }

    return metadata


def generate_parallel_dataset(
    config: ExperimentConfig,
    environment_kind: EnvironmentKind,
    output: str | Path,
    workers: int,
    pending_multiplier: int = 2,
) -> Path:
    """
    Simulate trajectories concurrently and write them from the parent.
    Only the parent process touches the output HDF5 file.
    """
    output = Path(output)

    if workers == 0: 
        workers = recommended_simulation_workers()

    if workers < 1:
        raise ValueError("workers must be at leat 1, or 0 for auto.")

    environment = EnvironmentFactory.create(
        environment_kind, config,
    )
    metadata = _dataset_metadata(
        config,
        environment_kind,
        environment,
    )

    total = config.dataset.trajectories

    if total <= 0:
        raise ValueError("Dataset must contain at least one trajectory.")

    # Use spawn explicitly for consistent Windows/UNIX behaviour.
    multiprocessing_context = get_context("spawn")
    maximum_pending = max(workers, workers * pending_multiplier)

    with HDF5TrajectoryWriter(
        output,
        config.dataset.compression,
        config.dataset.compression_level,
        metadata=metadata,
    ) as writer:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing_context,
            initializer=_initialize_simulation_worker,
            initargs=(
                config.model_dump(mode="json"),
                environment_kind.value,
            ),
        ) as executor:
            pending: dict[
                Future[tuple[int, Trajectory]],
                int,
            ] = {}

            next_index = 0

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )

            with progress:
                task = progress.add_task(
                    (
                        f"Generating "
                        f"{environment_kind.value}"
                    ),
                    total=total,
                )

                # Submit only a bounded number of jobs.
                while(
                    next_index < total
                    and len(pending) < maximum_pending
                ):
                    future = executor.submit(
                        _simulate_trajectory_worker,
                        next_index,
                    )
                    pending[future] = next_index
                    next_index += 1

                while pending:
                    completed, _ = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )

                    for future in completed:
                        submitted_index = pending.pop(future)

                        try:
                            trajectory_index, trajectory = future.result()
                        except Exception as error:
                            raise RuntimeError(
                                "Trajectory generation failed "
                                f"for index {submitted_index:08d}."
                            ) from error

                        writer.write(trajectory_index, trajectory)

                        progress.advance(task)

                        if next_index < total:
                            replacement = executor.submit(
                                _simulate_trajectory_worker,
                                next_index,
                            )
                            pending[replacement] = next_index
                            next_index += 1

    return output
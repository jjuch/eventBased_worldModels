from __future__ import annotations

from pathlib import Path

import yaml

from ball_simulator.trajectories.config import ExperimentConfig

from .discovery import ProjectContext, RunContext, DatasetContext


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return value


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
    return path

def effective_data_config(
    dataset: DatasetContext,
    *,
    settings_from: Path | None = None,
    destination: Path | None = None,
) -> Path:
    """data.yaml with the dataset's render folder, manifest and gravity injected.

    Gravity comes from the dataset's trajectories.yaml, the single source of the simulated physics.

    ``destination`` is the file to write; by default ``<dataset>/effective/data.yaml``.
    ``settings_from`` takes image, timing and loader settings from another data.yaml (a run's training snapshot), so a run is evaluated on this dataset exactly as it was trained.
    """
    source = settings_from or dataset.data_config
    value = _read(source)
    value["root"] = str(dataset.rendered)
    value["manifest_path"] = str(dataset.manifest)
    simulation = ExperimentConfig.from_yaml(dataset.trajectories_config).simulation
    value["gravity_world"] = list(simulation.gravity)
    return _write(destination or dataset.effective / "data.yaml", value)


def effective_training_config(
    context: ProjectContext,
    run: RunContext,
    *,
    data_config: Path | None = None,
    destination: Path | None = None,
) -> Path:
    """Run's training.yaml with the data config and run directory injected.

    The project-level ``dataset:`` entry is replaced by the path of that dataset's effective
    data config (or ``data_config`` when given).
    """
    value = _read(run.training_config)
    dataset_name = value.pop("dataset", None) or run.training_dataset_name()
    if data_config is None:
        data_config = effective_data_config(context.dataset(dataset_name))
    value["data_config"] = str(data_config)
    value.setdefault("training", {})["output_directory"] = str(run.root)
    location = destination or run.effective_training_config
    return _write(location, value)


def effective_render_config(dataset: DatasetContext) -> Path:
    path = dataset.rendering_config
    if not path.is_file():
        raise FileNotFoundError(f"Rendering configuration does not exist: {path}.")
    return path
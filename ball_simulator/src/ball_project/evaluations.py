"""Plan the evaluation of a run on any dataset of the project.

The data is read with the settings the run was trained with (image size, frame timing,
normalisation, from the run's dataset snapshot); only the data location changes. Rendering
settings that define what the observer has learned (camera, resolution, frame rate) must match
training; every other difference (shape, markers, lighting, physics) is recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .discovery import DatasetContext, RunContext
from .models import DatasetInfo

SPLITS = ("train", "validation", "test", "all")

# Rendering settings the observer depends on: a single fixed camera and frame geometry.
MUST_MATCH = (
    "width",
    "height",
    "fps",
    "frame_dt",
    "camera.location",
    "camera.target",
    "camera.focal_length_mm",
    "camera.sensor_width_mm",
)


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """What to evaluate and how it differs from the run's training data.

    mismatches   MUST_MATCH settings that differ: {setting: {"training": .., "evaluation": ..}}
    differences  all other config settings that differ, keyed "<file>:<setting>"
    """

    run: RunContext
    dataset: DatasetContext
    training_dataset: str
    split: str
    output: Path
    mismatches: dict[str, dict[str, Any]]
    differences: dict[str, dict[str, Any]]


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return value


def flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested mapping as {"a.b.c": leaf}; lists are leaves."""
    flat: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            flat.update(flatten(item, f"{name}."))
        else:
            flat[name] = item
    return flat


def _compare(first: dict[str, Any], second: dict[str, Any], keys) -> dict[str, dict[str, Any]]:
    return {
        key: {"training": first.get(key), "evaluation": second.get(key)}
        for key in keys
        if first.get(key) != second.get(key)
    }


def plan_evaluation(
    run: RunContext, dataset: DatasetContext, split: str | None = None
) -> EvaluationPlan:
    """Default split: ``test`` on the training dataset, ``all`` on any other dataset."""
    snapshot = run.dataset_snapshot
    if not (snapshot / "dataset.yaml").is_file():
        raise FileNotFoundError(
            f"Run {run.name!r} has no training-data snapshot in {snapshot}. "
            "It is written when training starts; train the run first."
        )
    training_dataset = DatasetInfo.from_yaml(snapshot / "dataset.yaml").name
    default_split = "test" if dataset.name == training_dataset else "all"
    split = split or default_split
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, received {split!r}.")
    folder = dataset.name if split == default_split else f"{dataset.name}_{split}"

    training_rendering = flatten(_read_yaml(snapshot / "rendering.yaml"))
    evaluation_rendering = flatten(_read_yaml(dataset.rendering_config))
    mismatches = _compare(training_rendering, evaluation_rendering, MUST_MATCH)

    differences: dict[str, dict[str, Any]] = {}
    for file_name in ("trajectories.yaml", "rendering.yaml"):
        training = flatten(_read_yaml(snapshot / file_name))
        evaluation = flatten(_read_yaml(dataset.configs / file_name))
        keys = sorted((training.keys() | evaluation.keys()) - set(MUST_MATCH))
        for key, values in _compare(training, evaluation, keys).items():
            differences[f"{file_name}:{key}"] = values

    return EvaluationPlan(
        run=run,
        dataset=dataset,
        training_dataset=training_dataset,
        split=split,
        output=run.evaluations / folder,
        mismatches=mismatches,
        differences=differences,
    )


def describe_mismatches(plan: EvaluationPlan) -> str:
    lines = [
        f"  {key}: training={values['training']!r}, evaluation={values['evaluation']!r}"
        for key, values in plan.mismatches.items()
    ]
    return "\n".join(lines)


def write_evaluation_record(plan: EvaluationPlan, checkpoint: Path, forced: bool) -> Path:
    """evaluation.json next to the metrics: which run, checkpoint and data, and what differed."""
    plan.output.mkdir(parents=True, exist_ok=True)
    record = {
        "run": plan.run.name,
        "checkpoint": str(checkpoint),
        "training_dataset": plan.training_dataset,
        "dataset": plan.dataset.name,
        "split": plan.split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "forced": forced,
        "mismatches": plan.mismatches,
        "differences": plan.differences,
    }
    path = plan.output / "evaluation.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path

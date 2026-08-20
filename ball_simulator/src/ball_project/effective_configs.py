from __future__ import annotations

from pathlib import Path

import yaml

from .discovery import ProjectContext


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

def effective_data_config(context: ProjectContext) -> Path:
    source = context.resolve(context.manifest.configs.data)
    value = _read(source)
    value["root"] = str(context.resolve(context.manifest.paths.rendered))
    value["manifest_path"] = str(context.resolve(context.manifest.paths.manifest))
    return _write(context.root / ".ball_project/effective/data.yaml", value)

def effective_training_config(context: ProjectContext) -> Path:
    source = context.resolve(context.manifest.configs.training)
    value = _read(source)
    value["data_config"] = str(effective_data_config(context))
    value.setdefault("training", {})["output_directory"] = str(
        context.resolve(context.manifest.paths.training_outputs)
    )
    return _write(context.root / ".ball_project/effective/training.yaml", value)

def effective_render_config(context: ProjectContext) -> Path:
    optimised = context.resolve(context.manifest.configs.optimised_rendering)
    if optimised.is_file():
        return optimised
    return context.resolve(context.manifest.configs.rendering)
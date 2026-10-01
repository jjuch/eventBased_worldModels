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

def _active_camera(context: ProjectContext) -> tuple[list[float], list[float]]:
    """Read the camera actually used for rendering this project.

    Camera optimisation ultimately updates the project's active rendering
    configuration. Training diagnostics must therefore derive their camera
    basis from that effective rendering configuration, never from a template.
    """
    rendering_path = effective_render_config(context)
    rendering = _read(rendering_path)
    camera = rendering.get("camera")
    if not isinstance(camera, dict):
        raise ValueError(
            f"Rendering configuration has no camera mapping: {rendering_path}"
        )

    location = camera.get("location")
    target = camera.get("target")
    if not (
        isinstance(location, list)
        and isinstance(target, list)
        and len(location) == 3
        and len(target) == 3
    ):
        raise ValueError(
            "The active rendering camera must define three-dimensional "
            f"location and target vectors: {rendering_path}"
        )
    return [float(value) for value in location], [float(value) for value in target]


def effective_training_config(context: ProjectContext) -> Path:
    source = context.resolve(context.manifest.configs.training)
    value = _read(source)
    value["data_config"] = str(effective_data_config(context))
    value.setdefault("training", {})["output_directory"] = str(
        context.resolve(context.manifest.paths.training_outputs)
    )

    # Inject the camera selected after trajectory generation and camera
    # optimisation. Keeping this in the generated effective config makes the
    # checkpoint self-describing and guarantees that camera-frame diagnostics
    # use the same view as the rendered dataset.
    location, target = _active_camera(context)
    model = value.setdefault("model", {})
    model["camera_location"] = location
    model["camera_target"] = target

    return _write(context.root / ".ball_project/effective/training.yaml", value)

def effective_render_config(context: ProjectContext) -> Path:
    path = context.resolve(context.manifest.configs.rendering)
    if not path.is_file():
        raise FileNotFoundError(f"Rendering configuration does not exist: {path}.")
    return path
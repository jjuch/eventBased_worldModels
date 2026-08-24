from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .discovery import ProjectContext


CAMERA_KEYS = (
    "location",
    "target",
    "focal_length_mm",
    "sensor_width_mm",
)

def rendering_config_path(context: ProjectContext) -> Path:
    return context.resolve(context.manifest.configs.rendering)


def camera_work_path(context: ProjectContext) -> Path:
    """Temporary output used by external camera proposal command."""
    path = context.root / ".ball_project/effective/rendering_camera_proposal.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def archive_rendering_config(context: ProjectContext) -> Path:
    source = rendering_config_path(context)
    if not source.is_file():
        raise FileNotFoundError(f"Rendering configuration does not exist: {source}")
    history = context.root / "configs/history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = history / f"rendering_before_camera_{stamp}.yaml"
    shutil.copy2(source, destination)
    return destination


def apply_proposed_camera(
    context: ProjectContext,
    proposal_path: str | Path,
) -> Path:
    """
    Copy only camera parameters from a generated proposal into rendering.yaml.

    Lighting, output, marker, resolution and all other manually edited setting remain exactly as they were in the authoritative base configuration.
    """
    authoritative = rendering_config_path(context)
    proposal_path = Path(proposal_path)
    current = yaml.safe_load(authoritative.read_text(encoding="utf-8")) or {}
    proposal = yaml.safe_load(proposal_path.read_text(encoding="utf-8")) or {}
    if not isinstance(current, dict) or not isinstance(proposal, dict):
        raise ValueError("Rendering configuration must contain YAML mappings.")
    if not isinstance(proposal.get("camera"), dict):
        raise ValueError(f"Camera proposal contains no camera mapping: {proposal_path}.")

    current_camera = current.setdefault("camera", {})
    for key in CAMERA_KEYS:
        if key in proposal["camera"]:
            current_camera[key] = proposal["camera"][key]

    authoritative.write_text(
        yaml.safe_dump(current, sort_keys=False),
        encoding="utf-8",
    )
    return authoritative
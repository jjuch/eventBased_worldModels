from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TemplateDefinition:
    experiment_type: str
    subtype: str
    modes: tuple[str, ...]
    resource_directory: str


_REGISTRY = {
    ("ball", "free-flight"): TemplateDefinition(
        experiment_type="ball",
        subtype="free-flight",
        modes=("translation", "rotation", "combined"),
        resource_directory="templates/ball/free-flight",
    )
}


def definition(experiment_type: str, subtype: str) -> TemplateDefinition:
    try:
        return _REGISTRY[(experiment_type, subtype)]
    except KeyError as error:
        available = ", ".join(f"{key[0]}/{key[1]}" for key in sorted(_REGISTRY))
        raise ValueError(
            f"Unsupported project type/subtype: {experiment_type}/{subtype}. "
            f"Available: {available}."
        ) from error


def template_text(experiment_type: str, subtype: str, filename: str) -> str:
    item = files("ball_project").joinpath(
        "templates", experiment_type, subtype, filename
    )
    return item.read_text(encoding="utf-8")
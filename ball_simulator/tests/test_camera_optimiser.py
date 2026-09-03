from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import h5py
import numpy as np

from ball_simulator.rendering.camera_optimiser import optimise_camera, project_spheres
from ball_simulator.rendering.config import RenderConfig
from ball_project.creator import create_project
from ball_project.discovery import discover_project
from ball_project.render_config import (
    CameraOptimisationSettings,
    apply_proposed_camera,
    load_camera_omtimisation_settings,
)

def _dataset(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        trajectories = handle.create_group("trajectories")
        for index, x in enumerate(np.linspace(-0.3, 0.3, 5)):
            group = trajectories.create_group(f"{index:08d}")
            observations = group.create_group("observations")
            positions = np.column_stack(
                (
                    np.linspace(x, x + 0.1, 20),
                    np.linspace(-0.2, 0.2, 20),
                    np.linspace(0.45, 0.85, 20),
                )
            )
            observations.create_dataset("position", data=positions)
            parameters = group.create_group("parameters")
            parameters.attrs["radius"] = 0.10


def test_camera_proposal_contains_all_spheres(tmp_path):
    dataset = tmp_path / "camera.h5"
    _dataset(dataset)
    config = RenderConfig(width=256, height=256)
    candidate = optimise_camera(
        dataset,
        config,
        minimum_diameter_px=20.0,
        target_diameter_px=60.0,
    )
    with h5py.File(dataset, "r") as handle:
        positions = np.concatenate(
            [np.asarray(group["observations/position"]) for group in handle["trajectories"].values()]
        )
    radii = np.full(len(positions), 0.10)
    u, v, diameter, depth = project_spheres(
        positions,
        radii,
        location=candidate.location,
        target=candidate.target,
        focal_length_mm=candidate.focal_length_mm,
        sensor_width_mm=config.camera.sensor_width_mm,
        width=config.width,
        height=config.height,
    )
    radius = diameter / 2.0
    assert np.all(depth > 0.0)
    assert np.all(u - radius > 0.0)
    assert np.all(u + radius < config.width)
    assert np.all(v - radius > 0.0)
    assert np.all(v + radius < config.height)


@pytest.mark.parametrize(
    ("mode", "minimum", "target"),
    [
        ("translation", 20.0, 70.0),
        ("rotation", 90.0, 130.0),
        ("combined", 20.0, 70.0),
    ],
)
def test_project_templates_define_camera_optimisation_settings(
    tmp_path,
    mode,
    minimum,
    target,
):
    root = create_project(
        f"{mode}_trial",
        experiment_type="ball",
        subtype="free-flight",
        mode=mode,
        parent=tmp_path,
    )
    settings = load_camera_omtimisation_settings(discover_project(root))
    assert settings.minimum_diameter_px == minimum
    assert settings.target_diameter_px == target


def test_user_changes_are_loaded_from_rendering_yaml(tmp_path):
    root = create_project(
        "trial",
        experiment_type="ball",
        subtype="free-flight",
        mode="translation",
        parent=tmp_path,
    )
    rendering = root / "configs/rendering.yaml"
    value = yaml.safe_load(rendering.read_text(encoding="utf-8"))
    value["camera_optimisation_settings"]["minimum_diameter_px"] = 22.0
    value["camera_optimisation_settings"]["target_diameter_px"] = 55.0
    rendering.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    settings = load_camera_omtimisation_settings(discover_project(root))
    assert settings.minimum_diameter_px == 22.0
    assert settings.target_diameter_px == 55.0


def test_camera_application_preserves_optimisation_settings(tmp_path):
    root = create_project(
        "trial",
        experiment_type="ball",
        subtype="free-flight",
        mode="translation",
        parent=tmp_path,
    )
    context = discover_project(root)
    authoritative = root / "configs/rendering.yaml"
    before = yaml.safe_load(authoritative.read_text(encoding="utf-8"))
    proposal = root / ".ball_project/effective/proposal.yaml"
    proposed = dict(before)
    proposed["camera"] = {
        **before["camera"],
        "location": [1.0, 2.0, 3.0],
        "target": [0.0, 0.0, 0.5],
        "focal_length_mm": 50.0,
    }
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(yaml.safe_dump(proposed, sort_keys=False), encoding="utf-8")

    apply_proposed_camera(context, proposal)
    after = yaml.safe_load(authoritative.read_text(encoding="utf-8"))
    assert (
        after["camera_optimisation_settings"]
        == before["camera_optimisation_settings"]
    )


def test_invalid_diameter_order_is_rejected():
    with pytest.raises(ValidationError):
        CameraOptimisationSettings(
            minimum_diameter_px=80.0,
            target_diameter_px=60.0,
        )

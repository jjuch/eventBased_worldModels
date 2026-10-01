from pathlib import Path
from types import SimpleNamespace

import yaml

from ball_project.effective_configs import effective_training_config


class _Context:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = SimpleNamespace(
            configs=SimpleNamespace(
                data=Path("configs/data.yaml"),
                training=Path("configs/training.yaml"),
                rendering=Path("configs/rendering.yaml"),
            ),
            paths=SimpleNamespace(
                rendered=Path("data/rendered"),
                manifest=Path("data/manifest.parquet"),
                training_outputs=Path("outputs/training"),
            ),
        )

    def resolve(self, value: Path) -> Path:
        return self.root / value


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_effective_training_uses_active_optimised_camera(tmp_path: Path):
    context = _Context(tmp_path)
    location = [-0.3, -4.2, 1.7]
    target = [0.1, 0.2, 0.8]
    _write_yaml(tmp_path / "configs/data.yaml", {"root": "placeholder"})
    _write_yaml(
        tmp_path / "configs/training.yaml",
        {"model": {"task": "combined"}, "training": {}},
    )
    _write_yaml(
        tmp_path / "configs/rendering.yaml",
        {"camera": {"location": location, "target": target}},
    )

    effective = effective_training_config(context)
    configured = yaml.safe_load(effective.read_text(encoding="utf-8"))

    assert configured["model"]["camera_location"] == location
    assert configured["model"]["camera_target"] == target

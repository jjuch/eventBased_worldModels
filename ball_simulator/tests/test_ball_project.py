from pathlib import Path

import yaml

from ball_project.creator import create_project
from ball_project.discovery import discover_project
from ball_project.effective_configs import effective_data_config, effective_training_config


def test_create_translation_project(tmp_path):
    root = create_project(
        "translation_trial",
        experiment_type="ball",
        subtype="free-flight",
        mode="translation",
        parent=tmp_path,
    )
    assert (root / "project.yaml").is_file()
    assert (root / "configs/trajectories.yaml").is_file()
    assert (root / "configs/rendering.yaml").is_file()
    assert (root / "configs/data.yaml").is_file()
    assert (root / "configs/training.yaml").is_file()
    context = discover_project(root / "outputs" / "inspection")
    assert context.root == root
    assert context.manifest.project.mode == "translation"


def test_effective_configs_inject_only_central_paths(tmp_path):
    root = create_project(
        "rotation_trial",
        experiment_type="ball",
        subtype="free-flight",
        mode="rotation",
        parent=tmp_path,
    )
    context = discover_project(root)
    original_data = (root / "configs/data.yaml").read_text(encoding="utf-8")
    original_training = (root / "configs/training.yaml").read_text(encoding="utf-8")

    data_path = effective_data_config(context)
    training_path = effective_training_config(context)
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))

    assert Path(data["root"]).is_absolute()
    assert Path(data["manifest_path"]).is_absolute()
    assert Path(training["data_config"]) == data_path
    assert Path(training["training"]["output_directory"]).is_absolute()
    assert (root / "configs/data.yaml").read_text(encoding="utf-8") == original_data
    assert (root / "configs/training.yaml").read_text(encoding="utf-8") == original_training


def test_project_templates_do_not_share_mutable_config_files(tmp_path):
    first = create_project(
        "first",
        experiment_type="ball",
        subtype="free-flight",
        mode="translation",
        parent=tmp_path,
    )
    second = create_project(
        "second",
        experiment_type="ball",
        subtype="free-flight",
        mode="translation",
        parent=tmp_path,
    )
    first_config = first / "configs/trajectories.yaml"
    first_config.write_text(first_config.read_text() + "\n# local edit\n", encoding="utf-8")
    assert "# local edit" not in (second / "configs/trajectories.yaml").read_text()

from pathlib import Path
import sys

import pytest
import yaml

from ball_project.commands import run_command
from ball_project.creator import create_project
from ball_project.discovery import discover_project, load_project
from ball_project.effective_configs import effective_data_config, effective_training_config
from ball_project.runs import create_run


def _project(tmp_path, mode="translation", name="trial"):
    return create_project(name, experiment_type="ball", subtype="free-flight",
                          mode=mode, parent=tmp_path)


def test_create_project_has_default_dataset_and_run(tmp_path):
    root = _project(tmp_path)
    assert (root / "project.yaml").is_file()
    assert (root / "configs/training.yaml").is_file()
    for name in ("trajectories.yaml", "rendering.yaml", "data.yaml"):
        assert (root / "datasets/default/configs" / name).is_file()
    assert (root / "datasets/default/dataset.yaml").is_file()
    context = discover_project(root / "runs" / "default")
    assert context.root == root
    assert context.manifest.project.mode == "translation"
    assert context.run().training_dataset_name() == "default"


def test_effective_configs_inject_only_central_paths(tmp_path):
    root = _project(tmp_path, mode="rotation")
    context = discover_project(root)
    dataset = context.dataset()
    original_data = dataset.data_config.read_text(encoding="utf-8")
    original_training = context.run().training_config.read_text(encoding="utf-8")

    data_path = effective_data_config(dataset)
    training_path = effective_training_config(context, context.run())
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))

    assert Path(data["root"]) == dataset.rendered
    assert Path(data["manifest_path"]) == dataset.manifest
    assert Path(training["data_config"]) == data_path
    assert "dataset" not in training  # project-level key, not part of the training schema
    assert Path(training["training"]["output_directory"]) == context.run().root
    assert dataset.data_config.read_text(encoding="utf-8") == original_data
    assert context.run().training_config.read_text(encoding="utf-8") == original_training


def test_project_templates_do_not_share_mutable_config_files(tmp_path):
    first = _project(tmp_path, name="first")
    second = _project(tmp_path, name="second")
    first_config = first / "datasets/default/configs/trajectories.yaml"
    first_config.write_text(first_config.read_text() + "\n# local edit\n", encoding="utf-8")
    assert "# local edit" not in (
        second / "datasets/default/configs/trajectories.yaml"
    ).read_text()


def test_run_command_logs_inside_its_scope(tmp_path):
    context = load_project(_project(tmp_path))
    run, dataset = context.run(), context.dataset()
    run_command(context, [sys.executable, "-c", "print('x')"], stage="training", scope=run)
    run_command(context, [sys.executable, "-c", "print('x')"], stage="rendering", scope=dataset)
    assert list((run.logs / "training").glob("*.log"))
    assert list((run.root / "records").glob("*.json"))
    assert list((dataset.root / "logs" / "rendering").glob("*.log"))


def test_run_lookup_requires_name_when_ambiguous(tmp_path):
    context = load_project(_project(tmp_path))
    assert context.run().name == "default"
    create_run(context, "second", from_run="default")
    with pytest.raises(ValueError):
        context.run()
    assert context.run("second").training_config.is_file()

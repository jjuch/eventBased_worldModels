"""Datasets, training snapshots, evaluation planning and the comparison report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from ball_project.creator import create_project
from ball_project.datasets import create_dataset
from ball_project.discovery import load_project
from ball_project.evaluations import plan_evaluation
from ball_project.report import write_comparison
from ball_project.runs import record_training_dataset
from ball_project.stages.ball import propose_camera, train
from ball_project.stages.evaluation import evaluate_observer
from ball_simulator.trajectories.config import ExperimentConfig


def _context(tmp_path):
    root = create_project("trial", experiment_type="ball", subtype="free-flight",
                          mode="combined", parent=tmp_path)
    return load_project(root)


def _edit_yaml(path: Path, edit) -> None:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _trained(context):
    """Default run with a training snapshot, as written when training starts."""
    run = context.run()
    record_training_dataset(run, context.dataset("default"))
    return run


def test_copied_dataset_keeps_optimised_camera(tmp_path):
    context = _context(tmp_path)
    reference = context.dataset("default")
    assert create_dataset(context, "early", from_dataset="default").info().camera == "optimise"

    reference.camera_report.parent.mkdir(parents=True)
    reference.camera_report.write_text("{}\n", encoding="utf-8")
    cube = create_dataset(context, "cube", from_dataset="default", description="cube, 4 markers")
    assert cube.info().camera == "fixed"
    assert cube.info().reference == "default"
    assert cube.rendering_config.read_text() == reference.rendering_config.read_text()
    with pytest.raises(ValueError, match="camera"):
        propose_camera(context, cube, dry_run=True)
    with pytest.raises(FileExistsError):
        create_dataset(context, "cube", from_dataset="default")


def test_dataset_lookup_requires_name_when_ambiguous(tmp_path):
    context = _context(tmp_path)
    assert context.dataset().name == "default"
    create_dataset(context, "cube", from_dataset="default")
    with pytest.raises(ValueError):
        context.dataset()


def test_training_snapshot_is_written_only_for_real_runs(tmp_path):
    context = _context(tmp_path)
    run = context.run()
    train(context, run, dry_run=True)
    assert not run.dataset_snapshot.exists()

    record_training_dataset(run, context.dataset("default"))
    assert (run.dataset_snapshot / "rendering.yaml").is_file()
    assert json.loads(run.record.read_text())["training_dataset"] == "default"


def test_evaluation_needs_a_trained_run(tmp_path):
    context = _context(tmp_path)
    with pytest.raises(FileNotFoundError, match="snapshot"):
        plan_evaluation(context.run(), context.dataset("default"))


def test_default_split_and_output_folder(tmp_path):
    context = _context(tmp_path)
    run = _trained(context)
    create_dataset(context, "cube", from_dataset="default")

    own = plan_evaluation(run, context.dataset("default"))
    other = plan_evaluation(run, context.dataset("cube"))
    explicit = plan_evaluation(run, context.dataset("cube"), split="test")
    assert (own.split, own.output.name) == ("test", "default")
    assert (other.split, other.output.name) == ("all", "cube")
    assert (explicit.split, explicit.output.name) == ("test", "cube_test")
    with pytest.raises(ValueError):
        plan_evaluation(run, context.dataset("cube"), split="everything")


def test_shape_change_is_recorded_and_camera_change_is_a_mismatch(tmp_path):
    context = _context(tmp_path)
    run = _trained(context)
    cube = create_dataset(context, "cube", from_dataset="default")
    _edit_yaml(cube.trajectories_config, lambda value: value.update(
        shape={"kind": "cube", "dimensions": [0.16, 0.16, 0.16]}))

    plan = plan_evaluation(run, cube)
    assert plan.mismatches == {}
    assert "trajectories.yaml:shape.kind" in plan.differences

    _edit_yaml(cube.rendering_config, lambda value: value["camera"].update(focal_length_mm=80.0))
    plan = plan_evaluation(run, cube)
    assert set(plan.mismatches) == {"camera.focal_length_mm"}
    with pytest.raises(ValueError, match="--force"):
        evaluate_observer(context, run, cube, checkpoint=tmp_path / "x.ckpt", dry_run=True)


def test_evaluation_uses_training_data_settings_on_the_new_dataset(tmp_path):
    context = _context(tmp_path)
    run = _trained(context)
    cube = create_dataset(context, "cube", from_dataset="default")
    _edit_yaml(cube.data_config, lambda value: value["image"].update(width=999))

    output = evaluate_observer(context, run, cube, checkpoint=tmp_path / "x.ckpt", dry_run=True)
    data = yaml.safe_load((output / "effective/data.yaml").read_text())
    training = yaml.safe_load((output / "effective/training.yaml").read_text())
    trained_width = yaml.safe_load((run.dataset_snapshot / "data.yaml").read_text())["image"]["width"]
    assert data["image"]["width"] == trained_width != 999
    assert Path(data["root"]) == cube.rendered
    assert Path(data["manifest_path"]) == cube.manifest
    assert Path(training["data_config"]) == output / "effective/data.yaml"


def test_comparison_report_has_one_row_per_evaluation(tmp_path):
    context = _context(tmp_path)
    run = context.run()
    for dataset, median in (("default", 1.5), ("cube", 9.0)):
        folder = run.evaluations / dataset
        folder.mkdir(parents=True)
        metrics = {
            "windows": 10, "split": "all",
            "position": {"norm_rmse": 0.01, "r2": [0.9, None, 0.7]},
            "orientation_error_deg": {"median": median},
            "rollout_vs_horizon": {"rollout_from_current_state": {
                "position_error_m": {"median": [0.001, 0.002]}}},
        }
        (folder / "metrics.json").write_text(json.dumps(metrics))
        (folder / "evaluation.json").write_text(json.dumps(
            {"dataset": dataset, "split": "all", "training_dataset": "default", "mismatches": {}}))

    path, rows = write_comparison(context)
    table = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows == 2
    assert [row["dataset"] for row in table] == ["cube", "default"]
    assert float(table[0]["orientation_median_deg"]) == 9.0
    assert float(table[0]["position_r2_mean"]) == pytest.approx(0.8)
    assert float(table[0]["rollout_final_position_median_m"]) == 0.002


def _shape_config(**physics) -> dict:
    return {
        "shape": {"kind": "cube", "dimensions": [0.16, 0.16, 0.16]},
        "simulation": {"backend": "mujoco", "internal_dt": 0.001, "observation_dt": 0.01},
        "physics": physics,
    }


def test_magnus_and_rotational_drag_are_allowed_for_other_shapes():
    ExperimentConfig.model_validate(_shape_config(
        magnus_coefficient={"low": 0.0, "high": 0.05},
        rotational_drag={"low": 0.0, "high": 0.005},
    ))
    with pytest.raises(ValueError, match="drag_coefficient"):
        ExperimentConfig.model_validate(_shape_config(drag_coefficient={"low": 0.1, "high": 0.1}))

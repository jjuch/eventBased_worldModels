from __future__ import annotations

import json
from pathlib import Path

import pytest

from ball_project.creator import create_project
from ball_project.derive import derive_project_hard_copy
from ball_project.discovery import discover_project


def _prepared_source(tmp_path: Path) -> Path:
    root = create_project(
        "source_trial",
        experiment_type="ball",
        subtype="free-flight",
        mode="translation",
        parent=tmp_path,
    )
    dataset = discover_project(root).dataset()
    dataset.trajectories.write_bytes(b"independent trajectory copy")

    rendered = dataset.rendered / "trajectory_00000000"
    (rendered / "rgb").mkdir(parents=True, exist_ok=True)
    (rendered / "rgb/000001.png").write_bytes(b"rendered frame")
    (rendered / "_SUCCESS").write_text("{}\n", encoding="utf-8")
    dataset.manifest.write_bytes(b"manifest with absolute paths into the source")
    dataset.camera_report.parent.mkdir(parents=True, exist_ok=True)
    dataset.camera_report.write_text('{"score": 1.0}\n', encoding="utf-8")
    (dataset.root / "logs").mkdir()
    (dataset.root / "logs" / "old.log").write_text("old\n", encoding="utf-8")

    checkpoints = root / "runs/default/checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "old.ckpt").write_bytes(b"must not be copied")
    return root


def test_derive_makes_independent_hard_copy(tmp_path):
    source = _prepared_source(tmp_path)
    result = derive_project_hard_copy("derived_trial", source=source, parent=tmp_path)
    derived = result.root
    source_dataset = source / "datasets/default"
    derived_dataset = derived / "datasets/default"

    assert derived != source
    assert result.datasets == 1 and result.rendered_trajectories == 1
    assert (derived_dataset / "trajectories.h5").read_bytes() == (
        source_dataset / "trajectories.h5"
    ).read_bytes()
    assert (derived_dataset / "rendered/trajectory_00000000/_SUCCESS").is_file()
    assert (derived_dataset / "reports/camera.json").is_file()
    assert (derived_dataset / "dataset.yaml").is_file()
    assert not (derived_dataset / "manifest.parquet").exists()  # rebuilt in the new project
    assert not (derived_dataset / "logs").exists()
    assert not list((derived / "runs").rglob("*.ckpt"))
    assert (derived / "runs/default/training.yaml").is_file()

    # No link or source dependency: modifying the derived copy leaves the source unchanged.
    (derived_dataset / "trajectories.h5").write_bytes(b"changed only in derived project")
    assert (source_dataset / "trajectories.h5").read_bytes() == b"independent trajectory copy"

    context = discover_project(derived)
    assert context.manifest.project.name == "derived_trial"
    lineage = json.loads((derived / ".ball_project/lineage.json").read_text())
    assert lineage["kind"] == "hard-copy"
    assert lineage["independent_after_creation"] is True


def test_derive_refuses_existing_destination(tmp_path):
    source = _prepared_source(tmp_path)
    (tmp_path / "derived_trial").mkdir()
    with pytest.raises(FileExistsError):
        derive_project_hard_copy("derived_trial", source=source, parent=tmp_path)


def test_derive_requires_rendered_data(tmp_path):
    source = create_project(
        "incomplete",
        experiment_type="ball",
        subtype="free-flight",
        mode="translation",
        parent=tmp_path,
    )
    with pytest.raises(FileNotFoundError):
        derive_project_hard_copy("derived", source=source, parent=tmp_path)


def test_failed_derive_removes_destination(tmp_path, monkeypatch):
    source = _prepared_source(tmp_path)

    def failing_copy(source_path, destination_path):
        raise PermissionError("simulated copy failure")

    monkeypatch.setattr("ball_project.derive._copy_if_present", failing_copy)
    with pytest.raises(PermissionError):
        derive_project_hard_copy("derived_trial", source=source, parent=tmp_path)
    assert not (tmp_path / "derived_trial").exists()

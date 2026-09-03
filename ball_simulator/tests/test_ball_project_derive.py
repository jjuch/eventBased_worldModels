from __future__ import annotations

import json
from pathlib import Path

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
    trajectory = root / "data/trajectories/trajectories.h5"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_bytes(b"independent trajectory copy")

    rendered = root / "data/rendered/trajectory_00000000"
    (rendered / "rgb").mkdir(parents=True, exist_ok=True)
    (rendered / "rgb/000001.png").write_bytes(b"rendered frame")
    (rendered / "_SUCCESS").write_text("{}\n", encoding="utf-8")

    manifest = root / "data/manifests/manifest.parquet"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(b"manifest")

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "camera.json").write_text('{"score": 1.0}\n', encoding="utf-8")

    training = root / "outputs/training/checkpoints"
    training.mkdir(parents=True, exist_ok=True)
    (training / "old.ckpt").write_bytes(b"must not be copied")
    records = root / ".ball_project/records"
    records.mkdir(parents=True, exist_ok=True)
    (records / "old.json").write_text("{}\n", encoding="utf-8")
    return root


def test_derive_makes_independent_hard_copy(tmp_path):
    source = _prepared_source(tmp_path)
    result = derive_project_hard_copy(
        "derived_trial",
        source=source,
        parent=tmp_path,
    )
    derived = result.root

    assert derived != source
    assert (derived / "data/trajectories/trajectories.h5").read_bytes() == (
        source / "data/trajectories/trajectories.h5"
    ).read_bytes()
    assert (derived / "data/rendered/trajectory_00000000/_SUCCESS").is_file()
    assert (derived / "data/manifests/manifest.parquet").is_file()
    assert (derived / "reports/camera.json").is_file()
    assert not list((derived / "outputs/training").rglob("*.ckpt"))
    assert not list((derived / ".ball_project/records").glob("*.json"))

    # Prove there is no link or source dependency: modifying the derived copy leaves
    # the source bytes unchanged.
    derived_trajectory = derived / "data/trajectories/trajectories.h5"
    derived_trajectory.write_bytes(b"changed only in derived project")
    assert (source / "data/trajectories/trajectories.h5").read_bytes() == (
        b"independent trajectory copy"
    )

    context = discover_project(derived)
    assert context.manifest.project.name == "derived_trial"
    lineage = json.loads((derived / ".ball_project/lineage.json").read_text())
    assert lineage["kind"] == "hard-copy"
    assert lineage["independent_after_creation"] is True


def test_derive_refuses_existing_destination(tmp_path):
    source = _prepared_source(tmp_path)
    destination = tmp_path / "derived_trial"
    destination.mkdir()

    try:
        derive_project_hard_copy("derived_trial", source=source, parent=tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected existing destination to be rejected")


def test_derive_requires_completed_render_and_manifest(tmp_path):
    source = create_project(
        "incomplete",
        experiment_type="ball",
        subtype="free-flight",
        mode="translation",
        parent=tmp_path,
    )
    try:
        derive_project_hard_copy("derived", source=source, parent=tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected incomplete source to be rejected")


def test_failed_derive_removes_destination(tmp_path, monkeypatch):
    source = _prepared_source(tmp_path)

    def failing_copy(source_path, destination_path):
        raise PermissionError("simulated copy failure")

    monkeypatch.setattr("ball_project.derive._copy_if_present", failing_copy)
    destination = tmp_path / "derived_trial"
    try:
        derive_project_hard_copy(
            "derived_trial",
            source=source,
            parent=tmp_path,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError(
            "Expected simulated copy failure"
        )
    assert not destination.exists()

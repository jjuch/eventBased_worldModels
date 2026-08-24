from pathlib import Path

import yaml

from ball_project.creator import create_project
from ball_project.discovery import discover_project
from ball_project.effective_configs import effective_render_config
from ball_project.render_config import (
    apply_proposed_camera,
    archive_rendering_config,
)


def test_camera_proposal_updates_only_authoritative_camera(tmp_path):
    root = create_project(
        "trial",
        experiment_type="ball",
        subtype="free-flight",
        mode="translation",
        parent=tmp_path,
    )
    context = discover_project(root)
    rendering = root / "configs/rendering.yaml"
    before = yaml.safe_load(rendering.read_text(encoding="utf-8"))

    proposal = root / ".ball_project/effective/proposal.yaml"
    proposed = yaml.safe_load(rendering.read_text(encoding="utf-8"))
    proposed["camera"]["location"] = [1.0, 2.0, 3.0]
    proposed["camera"]["target"] = [0.1, 0.2, 0.3]
    proposed["camera"]["focal_length_mm"] = 55.0
    proposed["samples"] = 999
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(yaml.safe_dump(proposed, sort_keys=False), encoding="utf-8")

    backup = archive_rendering_config(context)
    result = apply_proposed_camera(context, proposal)
    after = yaml.safe_load(rendering.read_text(encoding="utf-8"))

    assert result == rendering
    assert backup.is_file()
    assert after["camera"]["location"] == [1.0, 2.0, 3.0]
    assert after["camera"]["target"] == [0.1, 0.2, 0.3]
    assert after["camera"]["focal_length_mm"] == 55.0
    # A camera proposal may not overwrite unrelated manual settings.
    assert after["samples"] == before["samples"]


def test_rendering_always_uses_authoritative_config(tmp_path):
    root = create_project(
        "trial",
        experiment_type="ball",
        subtype="free-flight",
        mode="rotation",
        parent=tmp_path,
    )
    context = discover_project(root)
    legacy_generated = root / "configs/generated/rendering_optimized.yaml"
    legacy_generated.parent.mkdir(parents=True, exist_ok=True)
    legacy_generated.write_text("camera: {}\n", encoding="utf-8")

    assert effective_render_config(context) == root / "configs/rendering.yaml"

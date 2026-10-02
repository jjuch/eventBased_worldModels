from pathlib import Path
import sys

from ball_project.commands import run_command
from ball_project.creator import create_project
from ball_project.discovery import discover_project


def test_project_creation_is_lazy(tmp_path: Path):
    root = create_project(
        "trial",
        experiment_type="ball",
        subtype="free-flight",
        mode="combined",
        parent=tmp_path,
    )
    assert (root / "configs").is_dir()
    assert (root / ".ball_project").is_dir()
    assert not (root / "data").exists()
    assert not (root / "outputs").exists()
    assert not (root / "logs").exists()


def test_run_command_creates_stage_log(tmp_path: Path):
    root = create_project(
        "trial",
        experiment_type="ball",
        subtype="free-flight",
        mode="combined",
        parent=tmp_path,
    )
    context = discover_project(root)
    run_command(context, [sys.executable, "-c", "print('logged output')"], stage="smoke")
    logs = list((root / "logs" / "smoke").glob("*.log"))
    records = list((root / ".ball_project" / "records").glob("*_smoke.json"))
    assert len(logs) == 1
    assert "logged output" in logs[0].read_text(encoding="utf-8")
    assert len(records) == 1

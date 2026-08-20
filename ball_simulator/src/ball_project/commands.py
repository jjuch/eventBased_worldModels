from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .discovery import ProjectContext


def executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(
            f"Required command {name!r} is not installed in the active world model environment."
        )
    return resolved


def run_command(
    context: ProjectContext,
    command: list[str],
    *,
    stage: str,
    dry_run: bool = False,
) -> None:
    printable = subprocess.list2cmdline(command)
    if dry_run:
        print(printable)
        return
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=context.root, check=False)
    record = {
        "stage": stage,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "return_code": completed.returncode,
    }
    record_directory = context.root / ".ball_project/records"
    record_directory.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%S_%fZ")
    (record_directory / f"{stamp}_{stage}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
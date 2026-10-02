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

def _safe_stage_name(stage: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in stage)


def run_command(
    context: ProjectContext,
    command: list[str],
    *,
    stage: str,
    dry_run: bool = False,
) -> None:
    """Run a project stage while streaming and persistently logging output."""
    printable = subprocess.list2cmdline(command)
    if dry_run:
        print(printable)
        return
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%S_%fZ")
    stage_name = _safe_stage_name(stage)
    log_directory = context.root / "logs" / stage_name
    record_directory = context.root / ".ball_project" / "records"
    log_directory.mkdir(parents=True, exist_ok=True)
    record_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"{stamp}.log"

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"$ {printable}\n\n")
        process = subprocess.Popen(
            command,
            cwd=context.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()

    record = {
        "stage": stage,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "return_code": return_code,
        "log": str(log_path.relative_to(context.root)),
    }
    (record_directory / f"{stamp}_{stage_name}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
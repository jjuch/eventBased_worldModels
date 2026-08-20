from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def choose_editor(explicit: str | None) -> list[str]:
    candidate = explicit or os.environ.get("BALL_PROJECT_EDITOR") or os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if candidate:
        return shlex.split(candidate, posix=sys.platform != "win32")
    
    fallbacks = ["notepad"] if sys.platform == "win32" else ["nano", "vim", "vi"]
    for name in fallbacks:
        if shutil.which(name):
            return [name]
    raise FileNotFoundError("No editor found. Set BALL_PROJECT_EDITOR, VISUAL, or EDITOR.")

def edit_file(path: Path, editor: str | None = None) -> None:
    subprocess.run([*choose_editor(editor), str(path)], check=True)

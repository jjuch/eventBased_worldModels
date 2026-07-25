from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class BlenderRunner:
    executable: Path
    script: Path

    @classmethod
    def discover(cls, executable: str | Path | None = None) -> "BlenderRunner":
        candidate = str(executable or os.environ.get("BLENDER_EXECUTABLE", "blender"))
        resolved = shutil.which(candidate)
        if resolved is None and Path(candidate).exists():
            resolved = str(Path(candidate).resolve())
        if resolved is None:
            raise FileNotFoundError(
                "Blender was not found. Put it on PATH, set BLENDER_EXECUTABLE, "
                "or pass --blender-executable."
            )
        script = Path(__file__).parent / "blender" / "render_trajectory.py"
        return cls(Path(resolved), script)

    def version(self) -> str:
        result = subprocess.run(
            [str(self.executable), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()[0]

    def render(self, job: str | Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = [
            str(self.executable),
            "--background",
            "--factory-startup",
            "--python",
            str(self.script),
            "--",
            "--job",
            str(Path(job).resolve()),
        ]
        return subprocess.run(command, check=check, text=True)
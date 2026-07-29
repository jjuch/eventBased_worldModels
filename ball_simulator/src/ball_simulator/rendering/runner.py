from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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

    def render(
        self, 
        job: str | Path, 
        *, 
        check: bool = True,
        threads: int = 0,
        log_file: str | Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            str(self.executable),
            "--background",
            "--factory-startup",
        ]
        if threads >= 0:
            command.extend(
                [
                    "--threads",
                    str(threads),
                ]
            )
        command.extend(
            [
                "--python",
                str(self.script),
                "--",
                "--job",
                str(Path(job).resolve()),
            ]
        )

        if log_file is None:
            return subprocess.run(command, check=check, text=True)

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("w", encoding="utf-8") as handle:
            return subprocess.run(
                command,
                check=check,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

    def render_many(
        self,
        jobs: Sequence[Path],
        *,
        workers: int,
        threads_per_blender: int = 0,
        log_directory: str | Path | None = None,
        on_complete: Callable[
            [Path],
            None,
        ] | None = None,
    ) -> list[RenderResult]:
        if workers < 1:
            raise ValueError("workers must be at least 1.")

        log_root = (
            Path(log_directory) if log_directory is not None else None
        )

        results: list[RenderResult] = []

        def launch(job: Path) -> RenderResult:
            log_file = None
            if log_root is not None:
                log_file = log_root / f"{job.stem}.log"

            completed = self.render(
                job,
                check=True,
                threads=threads_per_blender,
                log_file=log_file,
            )
            return RenderResult(job=job, return_code=completed.returncode)

        # Threads only orchestrate external Blender subprocesses. Blender performs rendering.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job: dict[
                Future[RenderResult],
                Path,
            ] = {
                executor.submit(
                    launch,
                    Path(job),
                ): Path(job)
                for job in jobs
            }

            try:
                for future in as_completed(future_to_job):
                    job = future_to_job[future]

                    try:
                        result = future.result()
                    except Exception as error:
                        raise RuntimeError(
                            "Blender rendering failed for "
                            f"{job}."
                        ) from error

                    results.append(result)

                    if on_complete is not None:
                        on_complete(job)
            except BaseException:
                for future in future_to_job:
                    future.cancel()
                raise
            
        return results


@dataclass(frozen=True, slots=True)
class RenderResult:
    job: Path
    return_code: int
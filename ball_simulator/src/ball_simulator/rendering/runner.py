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
import json
import math

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

        resolved_jobs = [
            Path(job).resolve()
            for job in jobs
        ]

        if not resolved_jobs:
            return []
        

        log_root = (
            Path(log_directory) if log_directory is not None else None
        )
        if log_root is None:
            batch_root = resolved_jobs[0].parent / "_batches"
        else:
            batch_root = log_root.parent / "_batches"
        batch_root.mkdir(parents=True, exist_ok=True)

        batches = partition_jobs(resolved_jobs, workers)
        batch_specs: list[
            tuple[int, list[Path], Path]
        ] = []

        for batch_index, batch_jobs in enumerate(batches):
            batch_file = batch_root / f"render_batch_{batch_index:03d}.json"

            batch_file.write_text(
                json.dumps(
                    {
                        "schema_version": "1.1",
                        "jobs": [
                            str(job)
                            for job in batch_jobs
                        ],
                    },
                    ident=2,
                ),
                encoding="utf-8",
            )
            batch_specs.append(
                (
                    batch_index,\
                    batch_jobs,
                    batch_file,
                )
            )


        def launch_batch(
            batch_index: int,
            batch_jobs: list[Path],
            batch_file: Path,
        ) -> list[RenderResult]:
            log_file = None
            if log_root is not None:
                log_file = log_root / f"render_batch_{batch_index:03d}.log"

            completed = self.render_batch(
                batch_file,
                check=True,
                threads=threads_per_blender,
                log_file=log_file,
            )
            return [RenderResult(job=job, return_code=completed.returncode)]

        results: list[RenderResult] = []

        # Threads only orchestrate external Blender subprocesses. Blender performs rendering.
        with ThreadPoolExecutor(max_workers=len(batch_specs)) as executor:
            future_to_batch = {
                executor.submit(
                    launch_batch,
                    batch_index,
                    batch_jobs,
                    batch_file,
                ): (
                    batch_index,
                    batch_jobs,
                )
                for batch_index, batch_job, batch_file in batch_specs
            }

            try:
                for future in as_completed(future_to_batch):
                    batch_index, batch_jobs = future_to_batch[future]

                    try:
                        batch_results = future.result()
                    except Exception as error:
                        trajectory_names = ", ".join(job.stem for job in batch_jobs)
                        raise RuntimeError(
                            f"Blender batch rendering failed for batch {batch_index}: "
                            f"{trajectory_names}."
                        ) from error

                    results.append(batch_results)

                    if on_complete is not None:
                        for job in batch_jobs:
                            on_complete(job)
            except BaseException:
                for future in future_to_batch:
                    future.cancel()
                raise
            
        return results


    def render_batch(
        self,
        batch_file: str | Path,
        *,
        check: bool = True,
        threads: int = 0,
        log_file: str | Path | None = None,
    ) -> subprocess.CompletedProcess:
        command = [
            str(self.executable),
            "--background",
            "--factory-startup",
        ]

        if threads >= 0:
            command.extend(
                [
                    "--threads",
                    str(threads)
                ]
            )

        command.extend(
            [
                "--python",
                str(self.script),
                "--",
                "--batch",
                str(Path(batch_file).resolve()),
            ]
        )

        if log_file is None:
            return subprocess.run(
                command,
                check=check,
                text=True,
            )

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


def partition_jobs(jobs: Sequence[Path], workers: int) -> list[list[Path]]:
    """
    Round-robin partitioning balances batches better than contiguous
    chunks when trajectory render times differ. (worker 0 gets trajectories 0, 2, 4, and worker 1 gets 1, 3, 5.)
    """
    batch_count = min(workers, len(jobs))

    batches: list[list[Path]] = [
        []
        for _ in range(len(batch_count))
    ]

    for index, job in enumerate(jobs):
        batches[index % batch_count].append(Path(job))

    return [
        batch
        for batch in batches if batch
    ]



@dataclass(frozen=True, slots=True)
class RenderResult:
    job: Path
    return_code: int
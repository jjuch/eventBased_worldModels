from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import ExperimentConfig
from .generator import DatasetGenerator

app = typer.Typer(help="Generate physically rich sphere-wall trajectories.")


@app.command()
def generate(config: Path, output: Path | None = None) -> None:
    cfg = ExperimentConfig.from_yaml(config)
    path = DatasetGenerator(cfg).generate(output)
    print(f"[green]Wrote dataset:[/green] {path.resolve()}")


@app.command("validate-config")
def validate_config(config: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config)
    print(cfg.model_dump_json(indent=2))

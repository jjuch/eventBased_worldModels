from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich import print

from ball_world_model.config import DataConfig
from ball_world_model.data.dataset import RenderedTrajectoryDataset, build_dataloader
from ball_world_model.data.discovery import validate_trajectory
from ball_world_model.data.inspection import describe_batch, save_batch_inspection
from ball_world_model.data.manifest import build_manifest, load_manifest, save_manifest

app = typer.Typer(
    no_args_is_help=True,
    help="Temporal JEPA tooling for rendered ball trajectories.",
)


@app.command("build-manifest")
def build_manifest_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
        ),
    ] = None,
    skip_invalid: Annotated[
        bool,
        typer.Option(
            "--skip-invalid",
            help="Record and skip malformed trajectories.",
        ),
    ] = False,
) -> None:
    data_config = DataConfig.from_yaml(config)
    manifest = build_manifest(data_config, skip_invalid=skip_invalid)

    destination = output or data_config.manifest_path or (data_config.root / "manifest.parquet")
    save_manifest(manifest, destination)
    print(f"[green]Saved manifest:[/green] {Path(destination).resolve()}")
    print(f"Trajectories: {len(manifest):,}")

    for split, count in manifest["split"].value_counts().sort_index().items():
        print(f"    {split}: {count:,}")

    errors = manifest.attrs.get("validation_errors", [])
    if errors:
        print(f"[yellow]Skipped invalid trajectories: {len(errors)}[/yellow]")


@app.command("validate-data")
def validate_data_command(
    manifest_path: Annotated[
        Path,
        typer.Argument(
            exists=True, 
            readable=True,
        ),
    ],
    require_success_marker: Annotated[
        bool,
        typer.Option(
            "--require_success_marker/--ignore-success-marker"
        ),
    ] = True,
) -> None:
    manifest = load_manifest(manifest_path)

    with typer.progressbar(
        manifest.to_dict(orient="records"),
        label="Validating trajectories",
    ) as progress:
        for row in progress:
            validate_trajectory(
                row["trajectory_directory"],
                require_success_marker=require_success_marker,
            )
    print(f"[green]Validated {len(manifest):,} trajectories.[/green]")


@app.command("inspect-data")
def inspect_data_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            readable=True,
        ),
    ],
    manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest", 
            "-m",
            exists=True,
            readable=True,
        ),
    ] = None,
    split: Annotated[
        str,
        typer.Option(
            "--split",
            help="train, validation, or test.",
        ),
    ] = "train",
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
        ),
    ] = Path("inspection/data_batch.png"),
) -> None:
    data_config = DataConfig.from_yaml(config)
    manifest = manifest_path or data_config.manifest_path or (data_config.root / "manifest.parquet")
    dataset = RenderedTrajectoryDataset(manifest, data_config, split=split)
    loader = build_dataloader(dataset, data_config, shuffle=False)
    batch = next(iter(loader))
    print(describe_batch(batch))
    saved = save_batch_inspection(batch, data_config, output)
    print(f"[green]Saved batch inspection:[/green] {saved.resolve()}")

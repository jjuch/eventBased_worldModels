from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import ExperimentConfig
from .generator import DatasetGenerator
from .visualization import (
    list_trajectory_ids,
    load_initial_conditions,
    load_trajectory,
    plot_initial_condition_spread,
    plot_trajectory_3d,
    plot_trajectory_diagnostics,
)
from .regime_inspection import render_stratified_inspection


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


@app.command("list-trajectories")
def list_trajectories(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="HDF5 trajectory dataset.",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum number of IDs to print.",
        ),
    ] = 20,
) -> None:
    """List trajectory IDs stored in a dataset."""
    trajectory_ids = list_trajectory_ids(dataset)

    print(
        f"[bold]{len(trajectory_ids):,} trajectories[/bold] "
        f"in {dataset}"
    )

    for trajectory_id in trajectory_ids[:limit]:
        print(f"  {trajectory_id}")

    if len(trajectory_ids) > limit:
        print(
            f"[dim]... and "
            f"{len(trajectory_ids) - limit:,} more[/dim]"
        )


@app.command("plot-initial-conditions")
def plot_initial_conditions_command(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="HDF5 trajectory dataset."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional output image, e.g. initial_conditions.png.",
        ),
    ] = None,
    no_show: Annotated[
        bool,
        typer.Option(
            "--no-show",
            help="Save without opening an interactive window.",
        ),
    ] = False,
    max_points: Annotated[
        int,
        typer.Option(
            "--mqx-points",
            min=100,
            help="Maximum number of points displayed per scatter plot.",
        ),
    ] = 10_000,
) -> None:
    """Visualize coverage of initial states across the dateset."""
    data = load_initial_conditions(dataset)

    plot_initial_condition_spread(
        data,
        output=output,
        show=not no_show,
        max_scatter_points=max_points,
    )

    print(
        f"[green]Visualized "
        f"{len(data.trajectory_ids):,} initial conditions.[/green]"
    )

    if output is not None:
        print(f"[green]Saved:[/green] {output.resolve()}")


@app.command("plot-trajectory")
def plot_trajectory_command(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="HDF5 trajectory dataset.",
        ),
    ],
    trajectory_id: Annotated[
        str,
        typer.Argument(
            help=(
                "Trajectory ID. Both '12' and '00000012' "
                "are accepted."
            ),
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional output image for the 3D plot.",
        ),
    ] = None,
    diagnostics_output: Annotated[
        Path | None,
        typer.Option(
            "--diagnostics-output",
            help="Optional output image for physics diagnostics.",
        ),
    ] = None,
    wall_x: Annotated[
        float,
        typer.Option(
            "--wall-x",
            help="Wall-plane x coordinate.",
        ),
    ] = 0.0,
    high_rate: Annotated[
        bool,
        typer.Option(
            "--high-rate",
            help=(
                "Use high-rate samples when present; otherwise "
                "fall back to fixed-rate observations."
            ),
        ),
    ] = False,
    diagnostics: Annotated[
        bool,
        typer.Option(
            "--diagnostics/--no-diagnostics",
            help="Show the physics-diagnostic figure.",
        ),
    ] = True,
    no_show: Annotated[
        bool,
        typer.Option(
            "--no-show",
            help="Save without opening interactive windows.",
        ),
    ] = False,
) -> None:
    """Plot one 3D trajectory and optional physics diagnostics."""
    trajectory = load_trajectory(
        dataset,
        trajectory_id,
        use_high_rate=high_rate,
    )

    plot_trajectory_3d(
        trajectory,
        wall_x=wall_x,
        output=output,
        show=not no_show,
    )

    if diagnostics:
        plot_trajectory_diagnostics(
            trajectory,
            output=diagnostics_output,
            show=not no_show,
        )

    contact_samples = int(
        trajectory.contact_active.sum()
    )
    maximum_penetration = float(
        trajectory.penetration.max(initial=0.0)
    )
    peak_normal_force = float(
        trajectory.normal_force_magnitude.max(initial=0.0)
    )


    print(
        f"[bold]Trajectory {trajectory.trajectory_id}[/bold]\n"
        f"  Samples: {len(trajectory.time):,}\n"
        f"  Contact samples: {contact_samples:,}\n"
        f"  Maximum penetration: "
        f"{1_000.0 * maximum_penetration:.4f} mm\n"
        f"  Peak normal force: {peak_normal_force:.3f} N"
    )
    
    if output is not None:
        print(f"[green]Saved 3D plot:[/green] {output.resolve()}")

    if diagnostics_output is not None:
        print(
            f"[green]Saved diagnostics:[/green] "
            f"{diagnostics_output.resolve()}"
        )


@app.command("inspect-regimes")
def inspect_regimes_command(
    dataset: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    output_directory: Annotated[
        Path,
        typer.Option("--output-directory", "-o", help="Directory for plots and manifest."),
    ] = Path("regime_inspection"),
    diagnostics: Annotated[
        bool,
        typer.Option("--diagnostics/--no-diagnostics"),
    ] = True,
    high_rate: Annotated[
        bool,
        typer.Option("--high-rate", help="Use high-rate samples when available."),
    ] = False,
    allow_reuse: Annotated[
        bool,
        typer.Option(
            "--allow-reuse",
            help="Permit one trajectory to represent multiple regimes.",
        ),
    ] = False,
    wall_x: Annotated[
        float,
        typer.Option("--wall-x", help="Wall-plane x coordinate."),
    ] = 0.0,
) -> None:
    """Select and render representative physical regimes."""
    selected = render_stratified_inspection(
        dataset,
        output_directory, 
        include_diagnostics=diagnostics,
        use_high_rate=high_rate,
        allow_reuse=allow_reuse,
        wall_x=wall_x
    )
    print(f"[green]Rendered {len(selected)} physical regimes:[/green]")
    for item in selected:
        print(f"  [bold]{item.regime:22s}[/bold] {item.trajectory_id} — {item.reason}")
    print(f"[green]Output:[/green] {output_directory.resolve()}")
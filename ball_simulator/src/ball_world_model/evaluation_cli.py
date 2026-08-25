from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from ball_world_model.evaluation import EvaluationSettings, evaluate_kinematic_observer


def register_evaluation_command(app: typer.Typer) -> None:
    @app.command("evaluate-kinematic")
    def evaluate_kinematic_command(
        checkpoint: Annotated[
            Path,
            typer.Option("--checkpoint", exists=True, readable=True),
        ],
        config: Annotated[
            Path,
            typer.Option("--config", "-c", exists=True, readable=True),
        ],
        output: Annotated[
            Path,
            typer.Option("-output", "-o"),
        ] = Path("outputs/evaluation"),
        maximum_test_windows: Annotated[
            int,
            typer.Option("--maximum-test-windows", min=1),
        ] = 2_000,
         maximum_probe_train_windows: Annotated[
            int,
            typer.Option("--maximum-probe-train-windows", min=10),
        ] = 5_000,
        trajectory_plots: Annotated[
            int,
            typer.Option("--trajectory-plots", min=0),
        ] = 12,
        seed: Annotated[int, typer.Option("--seed")] = 20260824,
    ) -> None:
        """Evaluate physical predictions, interventions, and layerwise probes."""
        saved = evaluate_kinematic_observer(
            checkpoint_path=checkpoint,
            training_config_path=config,
            output_directory=output,
            settings=EvaluationSettings(
                maximum_test_windows=maximum_test_windows,
                maximum_probe_train_windows=maximum_probe_train_windows,
                trajectory_plots=trajectory_plots,
                seed=seed,
            ),
        )
        print(f"[green]Saved observer evaluation:[/green] {saved}")

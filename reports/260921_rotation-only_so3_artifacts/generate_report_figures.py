"""Generate comparison figures for the structured SO(3) artifact report.

Expected report layout
----------------------
report_root/
├── generate_report_figures.py
├── source_data/
│   ├── current_aggregate_metrics.csv
│   ├── previous_aggregate_metrics.csv
│   ├── current_interventions.csv
│   ├── previous_interventions.csv
│   ├── current_group_validity_and_consistency.csv
│   ├── previous_group_validity_and_consistency.csv
│   ├── current_rollout_by_horizon.csv
│   ├── previous_rollout_by_horizon.csv
│   ├── current_sector_omega_probes.csv
│   ├── previous_sector_omega_probes.csv
│   ├── current_sector_statistics.csv
│   └── previous_sector_statistics.csv
└── figures/

Outputs
-------
figures/state_comparison.pdf
figures/intervention_comparison.pdf
figures/geometry_comparison.pdf
figures/leakage_comparison.pdf
figures/rank_comparison.pdf

Usage
-----
python generate_report_figures.py
python generate_report_figures.py --root /path/to/report/package
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREVIOUS_LABEL = "Previous structured model"
CURRENT_LABEL = "Residual + adversarial artifacts"
PREVIOUS_COLOUR = "#4C78A8"
CURRENT_COLOUR = "#C51B7D"
GRID_ALPHA = 0.25
BAR_WIDTH = 0.36


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the five derived PDF figures used by the report."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Report root containing source_data/ and figures/.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required source-data file is missing: {path}\n"
            "Check that the evaluation CSV files were copied into source_data/."
        )
    return path


def read_csv(source_data: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(require_file(source_data / filename))


def metric_value(
    dataframe: pd.DataFrame,
    *,
    quantity: str,
    component: str,
    metric: str,
) -> float:
    selected = dataframe[
        (dataframe["quantity"] == quantity)
        & (dataframe["component"] == component)
        & (dataframe["metric"] == metric)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one aggregate metric row for "
            f"quantity={quantity!r}, component={component!r}, metric={metric!r}; "
            f"found {len(selected)}."
        )
    return float(selected.iloc[0]["value"])


def save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {destination}")


def generate_state_comparison(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))

    orientation_labels = ["Mean", "Median", "P95", "Maximum"]
    orientation_metrics = ["mean_deg", "median_deg", "p95_deg", "maximum_deg"]
    x_orientation = np.arange(len(orientation_labels))

    previous_orientation = [
        metric_value(previous, quantity="orientation", component="all", metric=name)
        for name in orientation_metrics
    ]
    current_orientation = [
        metric_value(current, quantity="orientation", component="all", metric=name)
        for name in orientation_metrics
    ]

    axes[0].bar(
        x_orientation - BAR_WIDTH / 2,
        previous_orientation,
        BAR_WIDTH,
        label=PREVIOUS_LABEL,
        color=PREVIOUS_COLOUR,
    )
    axes[0].bar(
        x_orientation + BAR_WIDTH / 2,
        current_orientation,
        BAR_WIDTH,
        label=CURRENT_LABEL,
        color=CURRENT_COLOUR,
    )
    axes[0].set_xticks(x_orientation, orientation_labels)
    axes[0].set_ylabel("Geodesic orientation error [deg]")
    axes[0].set_title("Orientation accuracy")
    axes[0].grid(axis="y", alpha=GRID_ALPHA)
    axes[0].legend(fontsize=8)

    axes_names = ("x", "y", "z")
    x_velocity = np.arange(len(axes_names))
    previous_velocity = [
        metric_value(
            previous,
            quantity="angular_velocity",
            component=axis,
            metric="rmse",
        )
        for axis in axes_names
    ]
    current_velocity = [
        metric_value(
            current,
            quantity="angular_velocity",
            component=axis,
            metric="rmse",
        )
        for axis in axes_names
    ]

    axes[1].bar(
        x_velocity - BAR_WIDTH / 2,
        previous_velocity,
        BAR_WIDTH,
        label=PREVIOUS_LABEL,
        color=PREVIOUS_COLOUR,
    )
    axes[1].bar(
        x_velocity + BAR_WIDTH / 2,
        current_velocity,
        BAR_WIDTH,
        label=CURRENT_LABEL,
        color=CURRENT_COLOUR,
    )
    axes[1].set_xticks(x_velocity, axes_names)
    axes[1].set_ylabel("Angular-velocity RMSE [rad/s]")
    axes[1].set_title("Angular-velocity accuracy")
    axes[1].grid(axis="y", alpha=GRID_ALPHA)
    axes[1].legend(fontsize=8)

    save_figure(figure, destination)


def intervention_value(
    dataframe: pd.DataFrame,
    intervention: str,
    metric: str,
) -> float:
    selected = dataframe[dataframe["intervention"] == intervention]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for intervention={intervention!r}; found {len(selected)}."
        )
    return float(selected.iloc[0][metric])


def generate_intervention_comparison(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))

    reversal = [
        intervention_value(previous, "reversed", "omega_reversal_error"),
        intervention_value(current, "reversed", "omega_reversal_error"),
    ]
    repeated = [
        intervention_value(previous, "repeated_last", "mean_angular_speed"),
        intervention_value(current, "repeated_last", "mean_angular_speed"),
    ]
    labels = ["Previous", "Residual +\nadversarial"]
    colours = [PREVIOUS_COLOUR, CURRENT_COLOUR]

    axes[0].bar(labels, reversal, color=colours)
    axes[0].set_ylabel("Residual [rad/s]")
    axes[0].set_title("Time-reversal consistency")
    axes[0].grid(axis="y", alpha=GRID_ALPHA)

    axes[1].bar(labels, repeated, color=colours)
    axes[1].set_ylabel("Predicted angular speed [rad/s]")
    axes[1].set_title("Repeated-frame intervention")
    axes[1].grid(axis="y", alpha=GRID_ALPHA)

    save_figure(figure, destination)


def group_metric_mean_deg(dataframe: pd.DataFrame, metric: str) -> float:
    selected = dataframe[dataframe["metric"] == metric]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one group metric row for metric={metric!r}; found {len(selected)}."
        )
    return float(np.rad2deg(float(selected.iloc[0]["mean"])))


def generate_geometry_comparison(
    previous_group: pd.DataFrame,
    current_group: pd.DataFrame,
    previous_rollout: pd.DataFrame,
    current_rollout: pd.DataFrame,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))

    consistency_labels = ["One-step", "Inverse"]
    consistency_metrics = ["one_step", "inverse"]
    x = np.arange(len(consistency_labels))
    previous_values = [
        group_metric_mean_deg(previous_group, metric)
        for metric in consistency_metrics
    ]
    current_values = [
        group_metric_mean_deg(current_group, metric)
        for metric in consistency_metrics
    ]

    axes[0].bar(
        x - BAR_WIDTH / 2,
        previous_values,
        BAR_WIDTH,
        label=PREVIOUS_LABEL,
        color=PREVIOUS_COLOUR,
    )
    axes[0].bar(
        x + BAR_WIDTH / 2,
        current_values,
        BAR_WIDTH,
        label=CURRENT_LABEL,
        color=CURRENT_COLOUR,
    )
    axes[0].set_xticks(x, consistency_labels)
    axes[0].set_ylabel("Mean geodesic residual [deg]")
    axes[0].set_title("Intrinsic group consistency")
    axes[0].grid(axis="y", alpha=GRID_ALPHA)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        previous_rollout["horizon_intervals"],
        previous_rollout["mean_deg"],
        marker="o",
        color=PREVIOUS_COLOUR,
        label="Previous mean",
    )
    axes[1].plot(
        current_rollout["horizon_intervals"],
        current_rollout["mean_deg"],
        marker="o",
        color=CURRENT_COLOUR,
        label="Residual mean",
    )
    axes[1].plot(
        previous_rollout["horizon_intervals"],
        previous_rollout["p95_deg"],
        linestyle="--",
        color=PREVIOUS_COLOUR,
        label="Previous P95",
    )
    axes[1].plot(
        current_rollout["horizon_intervals"],
        current_rollout["p95_deg"],
        linestyle="--",
        color=CURRENT_COLOUR,
        label="Residual P95",
    )
    axes[1].set_xlabel("Composed intervals")
    axes[1].set_ylabel("Geodesic rollout error [deg]")
    axes[1].set_title("Intrinsic rollout by horizon")
    axes[1].grid(alpha=GRID_ALPHA)
    axes[1].legend(fontsize=8)

    save_figure(figure, destination)


def sector_mean_r2(dataframe: pd.DataFrame, sector: str) -> float:
    selected = dataframe[dataframe["sector"] == sector]
    if len(selected) != 3:
        raise ValueError(
            f"Expected three axis rows for sector={sector!r}; found {len(selected)}."
        )
    return float(selected["r2"].mean())


def generate_leakage_comparison(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    destination: Path,
) -> None:
    sectors = ["omega", "tangent", "motion_invariants", "motion_artifacts"]
    labels = ["Canonical\n$\\omega$", "Tangent\ncarrier", "Motion\nscalars", "Motion\nartifacts"]
    x = np.arange(len(sectors))
    previous_values = [sector_mean_r2(previous, sector) for sector in sectors]
    current_values = [sector_mean_r2(current, sector) for sector in sectors]

    figure, axis = plt.subplots(figsize=(9.5, 4.9))
    axis.bar(
        x - BAR_WIDTH / 2,
        previous_values,
        BAR_WIDTH,
        label=PREVIOUS_LABEL,
        color=PREVIOUS_COLOUR,
    )
    axis.bar(
        x + BAR_WIDTH / 2,
        current_values,
        BAR_WIDTH,
        label=CURRENT_LABEL,
        color=CURRENT_COLOUR,
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean component $R^2$ for angular velocity")
    axis.set_title("Where angular velocity remains linearly decodable")
    axis.set_ylim(min(-0.05, min(previous_values + current_values) - 0.05), 1.05)
    axis.grid(axis="y", alpha=GRID_ALPHA)
    axis.legend(fontsize=8)

    save_figure(figure, destination)


def sector_rank(dataframe: pd.DataFrame, sector: str) -> float:
    selected = dataframe[dataframe["sector"] == sector]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one statistics row for sector={sector!r}; found {len(selected)}."
        )
    return float(selected.iloc[0]["effective_rank"])


def generate_rank_comparison(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    destination: Path,
) -> None:
    sectors = ["invariants", "artifacts", "motion_invariants", "motion_artifacts"]
    labels = ["Context\nscalars", "Context\nartifacts", "Motion\nscalars", "Motion\nartifacts"]
    x = np.arange(len(sectors))
    previous_values = [sector_rank(previous, sector) for sector in sectors]
    current_values = [sector_rank(current, sector) for sector in sectors]

    figure, axis = plt.subplots(figsize=(9.5, 4.9))
    axis.bar(
        x - BAR_WIDTH / 2,
        previous_values,
        BAR_WIDTH,
        label=PREVIOUS_LABEL,
        color=PREVIOUS_COLOUR,
    )
    axis.bar(
        x + BAR_WIDTH / 2,
        current_values,
        BAR_WIDTH,
        label=CURRENT_LABEL,
        color=CURRENT_COLOUR,
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Effective rank")
    axis.set_title("Auxiliary-sector effective rank")
    axis.grid(axis="y", alpha=GRID_ALPHA)
    axis.legend(fontsize=8)

    save_figure(figure, destination)


def main() -> None:
    arguments = parse_arguments()
    root = arguments.root.expanduser().resolve()
    source_data = root / "source_data"
    figures = root / "figures"

    previous_aggregate = read_csv(source_data, "previous_aggregate_metrics.csv")
    current_aggregate = read_csv(source_data, "current_aggregate_metrics.csv")
    previous_interventions = read_csv(source_data, "previous_interventions.csv")
    current_interventions = read_csv(source_data, "current_interventions.csv")
    previous_group = read_csv(
        source_data, "previous_group_validity_and_consistency.csv"
    )
    current_group = read_csv(
        source_data, "current_group_validity_and_consistency.csv"
    )
    previous_rollout = read_csv(source_data, "previous_rollout_by_horizon.csv")
    current_rollout = read_csv(source_data, "current_rollout_by_horizon.csv")
    previous_probes = read_csv(source_data, "previous_sector_omega_probes.csv")
    current_probes = read_csv(source_data, "current_sector_omega_probes.csv")
    previous_statistics = read_csv(source_data, "previous_sector_statistics.csv")
    current_statistics = read_csv(source_data, "current_sector_statistics.csv")

    generate_state_comparison(
        previous_aggregate,
        current_aggregate,
        figures / "state_comparison.pdf",
    )
    generate_intervention_comparison(
        previous_interventions,
        current_interventions,
        figures / "intervention_comparison.pdf",
    )
    generate_geometry_comparison(
        previous_group,
        current_group,
        previous_rollout,
        current_rollout,
        figures / "geometry_comparison.pdf",
    )
    generate_leakage_comparison(
        previous_probes,
        current_probes,
        figures / "leakage_comparison.pdf",
    )
    generate_rank_comparison(
        previous_statistics,
        current_statistics,
        figures / "rank_comparison.pdf",
    )

    print("Generated all five report figures successfully.")


if __name__ == "__main__":
    main()

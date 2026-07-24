from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .visualization import (
    load_trajectory,
    natural_trajectory_sort_key,
    plot_trajectory_3d,
    plot_trajectory_diagnostics,
    determine_initial_target_surface
)
from .physics import ContactMode
from .dataset_geometry import EnvironmentGeometry, _decode_attribute, load_environment_geometry, load_root_metadata

FloatArray = NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class TrajectorySummary:
    trajectory_id: str
    initial_target_surface: str
    initial_normal_speed: float
    initial_tangential_speed: float
    initial_linear_speed: float
    initial_angular_speed: float
    incidence_cosine: float
    friction: float
    normal_stiffness: float
    normal_damping: float
    mass: float
    radius: float
    contact_episodes: int
    total_surface_episodes: int
    maximum_simultaneous_contacts: int
    simultaneous_contact_samples: int
    contacted_surfaces: tuple[int, ...]
    surface_epsiode_counts: tuple[int, ...]
    has_sticking: bool
    has_sliding: bool
    maximum_penetration: float
    peak_normal_force: float
    final_linear_speed: float
    final_angular_speed: float


@dataclass(frozen=True, slots=True)
class SelectedTrajectory:
    regime: str
    trajectory_id: str
    score: float
    reason: str


def _contact_episode_count(active: NDArray[np.bool_]) -> int:
    if active.size == 0:
        return 0
    padded = np.pad(active.astype(np.int8), (1, 0))
    return int(np.count_nonzero(np.diff(padded) == 1))


def _attr(attrs: h5py.AttributeManager, name: str, default: float = np.nan) -> float:
    return float(attrs[name]) if name in attrs else float(default)


def initial_incidence_features(
        position: FloatArray,
        velocity: FloatArray,
        radius: float,
        geometry: EnvironmentGeometry,
) -> tuple[str, float, float, float]:
    surface_id, _, normal_speed, tangential_speed = determine_initial_target_surface(
        position, 
        velocity, 
        radius, 
        geometry,
        )

    total_speed = float(np.linalg.norm(velocity))
    incidence_cosine = normal_speed / total_speed if total_speed > 0.0 else 0.0

    return (
        surface_id,
        normal_speed,
        tangential_speed,
        incidence_cosine
    )


def summarize_dataset(
    dataset: str | Path,
    geometry: EnvironmentGeometry) -> list[TrajectorySummary]:
    """Read compact per-trajectory features without loading positions/quaternions."""
    summaries: list[TrajectorySummary] = []
    with h5py.File(dataset, "r") as handle:
        root = handle["trajectories"]
        for trajectory_id in sorted(root.keys(), key=natural_trajectory_sort_key):
            group = root[trajectory_id]
            obs = group["observations"]
            attrs = group["parameters"].attrs

            p0 = np.asarray(obs["position"][0], dtype=float)
            v0 = np.asarray(obs["linear_velocity"][0], dtype=float)
            w0 = np.asarray(obs["angular_velocity"][0], dtype=float)
            vf = np.asarray(obs["linear_velocity"][-1], dtype=float)
            wf = np.asarray(obs["angular_velocity"][-1], dtype=float)
            speed = float(np.linalg.norm(v0))
            # normal_speed = max(0.0, -float(v0[0]))
            # tangential_speed = float(np.linalg.norm(v0[1:]))
            # incidence_cosine = normal_speed / max(speed, np.finfo(float).eps)

            active = np.asarray(obs["contact_active"][:], dtype=bool)
            any_active = np.any(active, axis=1)
            active_contact_count = np.sum(active, axis=1)

            modes = np.asarray(obs["contact_mode"][:], dtype=np.int8)
            
            has_sticking = bool(
                np.any(modes == ContactMode.STICKING)
            )
            has_sliding = bool(
                np.any(modes == ContactMode.SLIDING)
            )

            penetration = np.asarray(obs["penetration"][:], dtype=float)
            normal_force = np.asarray(obs["normal_force"][:], dtype=float)

            surface_ids = tuple(
                json.loads(_decode_attribute(group.attrs["surface_ids_json"]))
            )

            if surface_ids != geometry.surface_ids:
                raise ValueError(
                    f"Surface order mismatch for "
                    f"trajectory {trajectory_id}."
                )
            radius = _attr(attrs, "radius")
            target_surface, normal_speed, tangential_speed, incidence_cosine = initial_incidence_features(
                p0,
                v0,
                radius,
                geometry,
            )

            surface_episode_counts = tuple(
                _contact_episode_count(active[:, surface_index]) 
                for surface_index in range(len(surface_ids))
            )
            contact_episodes = _contact_episode_count(any_active)

            contacted_surfaces = tuple(
                surface_id for surface_index, surface_id in enumerate(surface_ids)
                if np.any(active[:, surface_index])
            )

            total_force = np.sum(normal_force, axis=1)

            summaries.append(
                TrajectorySummary(
                    trajectory_id=trajectory_id,
                    initial_target_surface=target_surface,
                    initial_normal_speed=normal_speed,
                    initial_tangential_speed=tangential_speed,
                    initial_linear_speed=speed,
                    initial_angular_speed=float(np.linalg.norm(w0)),
                    incidence_cosine=incidence_cosine,
                    friction=_attr(attrs, "friction"),
                    normal_stiffness=_attr(attrs, "normal_stiffness"),
                    normal_damping=_attr(attrs, "normal_damping"),
                    mass=_attr(attrs, "mass"),
                    radius=radius,
                    contact_episodes=contact_episodes,
                    total_surface_episodes=int(sum(surface_episode_counts)),
                    maximum_simultaneous_contacts=int(np.max(active_contact_count, initial=0)),
                    simultaneous_contact_samples=int(np.count_nonzero(active_contact_count >= 2)),
                    contacted_surfaces=contacted_surfaces,
                    surface_epsiode_counts=surface_episode_counts,
                    has_sticking=has_sticking,
                    has_sliding=has_sliding,
                    maximum_penetration=float(np.max(penetration, initial=0.0)),
                    peak_normal_force=float(
                        np.max(np.linalg.norm(total_force, axis=1), initial=0.0)
                    ),
                    final_linear_speed=float(np.linalg.norm(vf)),
                    final_angular_speed=float(np.linalg.norm(wf)),
                )
            )
    if not summaries:
        raise ValueError("The dataset contains no trajectories.")
    return summaries


def _values(items: Sequence[TrajectorySummary], name: str) -> FloatArray:
    return np.asarray([getattr(item, name) for item in items], dtype=float)


def _quantile(items: Sequence[TrajectorySummary], name: str, q: float) -> float:
    values = _values(items, name)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def _normalized_distance(value: float, target: float, scale: float) -> float:
    if not np.isfinite(value) or not np.isfinite(target):
        return float("inf")
    return abs(value - target) / max(scale, np.finfo(float).eps)


def _pick(
    items: Sequence[TrajectorySummary],
    regime: str,
    predicate: Callable[[TrajectorySummary], bool],
    score: Callable[[TrajectorySummary], float],
    reason: str,
    used: set[str],
    allow_reuse: bool,
) -> SelectedTrajectory | None:
    candidates = [
        item for item in items
        if predicate(item) and (allow_reuse or item.trajectory_id not in used)
    ]
    if not candidates:
        return None
    winner = min(candidates, key=score)
    used.add(winner.trajectory_id)
    return SelectedTrajectory(regime, winner.trajectory_id, float(score(winner)), reason)


def add_environment_specific_regimes(
    summaries: Sequence[TrajectorySummary],
    geometry: EnvironmentGeometry,
    selected: list[SelectedTrajectory],
    used: set[str],
    allow_reuse: bool,
) -> None:
    def add(result: SelectedTrajectory | None) -> None:
        if result is not None:
            selected.append(result)

    if geometry.kind == "u-box":
        for surface_id in geometry.surface_ids:
            add(
                _pick(
                    summaries,
                    regime=f"most-{surface_id}-contacts",
                    predicate=lambda item, sid = surface_id: (
                        sid in item.contacted_surfaces
                    ),
                    score=lambda item, sid = surface_id: (
                        -item.surface_episode_counts[
                            geometry.surface_index(sid)
                        ]
                    ),
                    reason=(
                        f"Largest number of separated "
                        f"{surface_id} contact episodes."
                    ),
                    used=used,
                    allow_reuse=allow_reuse,
                )
            )

        add(
            _pick(
                summaries,
                regime="simultaneous-contact",
                predicate=lambda item: (
                    item.maximum_simultaneous_contacts >= 2
                ),
                score=lambda item: -item.simultaneous_contact_samples,
                reason=(
                    "Contains simultaneous contact with "
                    "at least two surfaces."
                ),
                used=used,
                allow_reuse=allow_reuse,
            )
        )
        add(
            _pick(
                summaries,
                regime="contacts-all-surfaces",
                predicate=lambda item: (
                    set(item.contacted_surfaces) == set(geometry.surface_ids)
                ),
                score=lambda item: (
                -item.total_surface_episodes
                ),
                reason=(
                "Contacts every U-box surface during "
                "one trajectory."
                ),
                used=used,
                allow_reuse=allow_reuse,
            )
        )
    else:
        return


def select_stratified_regimes(
    summaries: Sequence[TrajectorySummary],
    geometry: EnvironmentGeometry,
    *,
    allow_reuse: bool = False,
) -> list[SelectedTrajectory]:
    """Select representative and extreme trajectories using dataset-relative strata.

    Quantile-based thresholds keep this valid when physical ranges change. The selector
    contains both input regimes (speed, spin, incidence, material) and outcome regimes
    (sticking/sliding, penetration, force, repeated contact).
    """
    used: set[str] = set()
    selected: list[SelectedTrajectory] = []

    fields = (
        "initial_normal_speed", "initial_angular_speed", "incidence_cosine",
        "friction", "normal_stiffness", "normal_damping",
        "maximum_penetration", "peak_normal_force",
    )
    q10 = {field: _quantile(summaries, field, 0.10) for field in fields}
    q50 = {field: _quantile(summaries, field, 0.50) for field in fields}
    q90 = {field: _quantile(summaries, field, 0.90) for field in fields}
    spread = {
        field: max(q90[field] - q10[field], np.finfo(float).eps)
        for field in fields
    }

    def add(result: SelectedTrajectory | None) -> None:
        if result is not None:
            selected.append(result)

    add(_pick(
        summaries, "representative",
        lambda x: x.contact_episodes >= 1,
        lambda x: sum(
            _normalized_distance(getattr(x, field), q50[field], spread[field])
            for field in fields[:6]
        ),
        "Closest contacted trajectory to the multivariate median.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "slow-impact",
        lambda x: x.contact_episodes >= 1 and x.initial_normal_speed <= q10["initial_normal_speed"],
        lambda x: x.initial_normal_speed,
        "Bottom decile of incoming wall-normal speed.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "fast-impact",
        lambda x: x.contact_episodes >= 1 and x.initial_normal_speed >= q90["initial_normal_speed"],
        lambda x: -x.initial_normal_speed,
        "Top decile of incoming wall-normal speed.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "near-normal-impact",
        lambda x: x.contact_episodes >= 1 and x.incidence_cosine >= q90["incidence_cosine"],
        lambda x: -x.incidence_cosine,
        "Top decile of normal-to-total incoming speed ratio.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "grazing-impact",
        lambda x: x.contact_episodes >= 1 and x.incidence_cosine <= q10["incidence_cosine"],
        lambda x: x.incidence_cosine,
        "Bottom decile of normal-to-total incoming speed ratio.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "low-spin",
        lambda x: x.contact_episodes >= 1 and x.initial_angular_speed <= q10["initial_angular_speed"],
        lambda x: x.initial_angular_speed,
        "Bottom decile of initial angular-speed magnitude.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "high-spin",
        lambda x: x.contact_episodes >= 1 and x.initial_angular_speed >= q90["initial_angular_speed"],
        lambda x: -x.initial_angular_speed,
        "Top decile of initial angular-speed magnitude.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "low-friction",
        lambda x: x.contact_episodes >= 1 and x.friction <= q10["friction"],
        lambda x: x.friction,
        "Bottom decile of friction coefficient.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "high-friction",
        lambda x: x.contact_episodes >= 1 and x.friction >= q90["friction"],
        lambda x: -x.friction,
        "Top decile of friction coefficient.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "soft-contact",
        lambda x: x.contact_episodes >= 1 and x.normal_stiffness <= q10["normal_stiffness"],
        lambda x: x.normal_stiffness,
        "Bottom decile of normal contact stiffness.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "stiff-contact",
        lambda x: x.contact_episodes >= 1 and x.normal_stiffness >= q90["normal_stiffness"],
        lambda x: -x.normal_stiffness,
        "Top decile of normal contact stiffness.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "sticking-contact",
        lambda x: x.has_sticking,
        lambda x: abs(x.initial_tangential_speed),
        "Contains at least one recorded sticking-contact observation.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "sliding-contact",
        lambda x: x.has_sliding,
        lambda x: -x.initial_tangential_speed,
        "Contains at least one recorded sliding-contact observation.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "repeated-contact",
        lambda x: x.contact_episodes >= 2,
        lambda x: -float(x.contact_episodes),
        "Contains two or more separated contact episodes.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "maximum-penetration",
        lambda x: x.contact_episodes >= 1,
        lambda x: -x.maximum_penetration,
        "Largest recorded compliant penetration.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "peak-force",
        lambda x: x.contact_episodes >= 1,
        lambda x: -x.peak_normal_force,
        "Largest recorded normal contact force.", used, allow_reuse,
    ))
    add(_pick(
        summaries, "missed-wall",
        lambda x: x.contact_episodes == 0,
        lambda x: x.initial_normal_speed,
        "No contact episode was recorded; useful for detecting invalid sampling horizons.",
        used, allow_reuse,
    ))
    add_environment_specific_regimes(
        summaries=summaries,
        geometry=geometry,
        selected=selected,
        used=used,
        allow_reuse=allow_reuse
    )
    return selected


def write_manifest(
    path: str | Path,
    selected: Sequence[SelectedTrajectory],
    summaries: Sequence[TrajectorySummary],
) -> None:
    lookup = {item.trajectory_id: item for item in summaries}
    rows = []
    for item in selected:
        row = asdict(item)
        row.update(asdict(lookup[item.trajectory_id]))
        rows.append(row)
    if not rows:
        raise ValueError("No trajectories were selected.")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_stratified_inspection(
    dataset: str | Path,
    output_directory: str | Path,
    *,
    include_diagnostics: bool = True,
    use_high_rate: bool = False,
    allow_reuse: bool = False,
) -> list[SelectedTrajectory]:
    geometry = load_environment_geometry(dataset)
    metadata = load_root_metadata(dataset)
    gravity_vector = np.asarray(
        metadata["simulation"]["gravity"], dtype=float
    )
    summaries = summarize_dataset(dataset, geometry)
    selected = select_stratified_regimes(summaries, geometry, allow_reuse=allow_reuse)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output / f"manifest_{geometry.kind}.csv", selected, summaries)

    for item in selected:
        trajectory = load_trajectory(dataset, item.trajectory_id, use_high_rate=use_high_rate)
        plot_trajectory_3d(
            trajectory=trajectory,
            geometry=geometry,
            output=output / f"{item.regime}__{item.trajectory_id}__3d.png",
            show=False,
        )
        plt.close("all")
        if include_diagnostics:
            plot_trajectory_diagnostics(
                trajectory,
                gravity_vector=gravity_vector,
                output=output / f"{item.regime}__{item.trajectory_id}__diagnostics.png",
                show=False,
            )
            plt.close("all")
    return selected

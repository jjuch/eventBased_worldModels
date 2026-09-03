from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml

from .config import RenderConfig

@dataclass(frozen=True, slots=True)
class CameraCandidate:
    location: np.ndarray
    target: np.ndarray
    focal_length_mm: float
    score: float
    minimum_diameter_px: float
    median_diameter_px: float
    maximum_diameter_px: float
    minimum_margin_px: float
    conditioning: float


def load_position_cloud(dataset: str | Path, max_samples: int = 50_000):
    positions, radii = [], []
    with h5py.File(dataset, "r") as handle:
        trajectories = handle["trajectories"]
        for group in trajectories.values():
            samples = np.asarray(group["observations/position"], dtype=float)
            radius = float(group["parameters"].attrs["radius"])
            positions.append(samples)
            radii.append(np.full(len(samples), radius, dtype=float))

    position = np.concatenate(positions)
    radius = np.concatenate(radii)

    if len(position) > max_samples:
        indices = np.linspace(0, len(position) - 1, max_samples, dtype=int)
        position, radius = position[indices], radius[indices]
    return position, radius


def _basis(location: np.ndarray, target: np.ndarray):
    forward = target - location
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def project_spheres(
    position: np.ndarray,
    radius: np.ndarray,
    *,
    location: np.ndarray,
    target: np.ndarray,
    focal_length_mm: float,
    sensor_width_mm: float,
    width: int,
    height: int,
):
    right, up, forward = _basis(location, target)
    relative = position - location
    depth = relative @ forward
    focal_pixels_x = width * focal_length_mm / sensor_width_mm
    sensor_height_mm = sensor_width_mm * height / width
    focal_pixels_y = height * focal_length_mm / sensor_height_mm
    u = width / 2.0 + focal_pixels_x * (relative @ right) / depth
    v = height / 2.0 + focal_pixels_y * (relative @ up) / depth
    diameter = 2.0 * focal_pixels_x * radius/depth
    return u, v, diameter, depth


def _conditioning(
    center: np.ndarray,
    *,
    location: np.ndarray,
    target: np.ndarray,
    focal_length_mm: float,
    sensor_width_mm: float,
    width: int,
    height: int,
) -> float:
    radius = np.asarray([0.05])
    epsilon = 1.0e-3

    def observation(point):
        u, v, diameter, depth = project_spheres(
            point[None, :], radius,
            location=location, 
            target=target,
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            width=width,
            height=height,
        )
        if depth[0] <= 0.0:
            return np.full(3, np.nan)
        # Log diameter makes depth sensitivity dimensionless and better conditioned.
        return np.asarray([u[0] / width, v[0] / height, np.log(diameter[0])])

    jacobian = np.column_stack(
        [
            (observation(center + epsilon * axis) - observation(center - epsilon * axis))
            / (2.0 * epsilon)
            for axis in np.eye(3)
        ]
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(singular_values[-1] / max(singular_values[0], 1.0e-12))


def optimise_camera(
    dataset: str | Path,
    render_config: RenderConfig,
    *,
    minimum_diameter_px: float = 35.0,
    target_diameter_px: float = 70.0,
    image_margin_fraction: float = 0.06,
) -> CameraCandidate:
    positions, radii = load_position_cloud(dataset)
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    center = 0.5 * (lower + upper)
    extent = np.maximum(upper - lower, 0.10)
    scene_radius = 0.5 * float(np.linalg.norm(extent)) + float(radii.max())
    width, height = render_config.width, render_config.height
    sensor = render_config.camera.sensor_width_mm

    candidates: list[CameraCandidate] = []
    for azimuth_degree in np.linspace(205.0, 335.0, 9):
        azimuth = np.deg2rad(azimuth_degree)
        for elevation_degree in (18.0, 28.0, 38.0, 48.0):
            elevation = np.deg2rad(elevation_degree)
            direction = np.asarray(
                [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]
            )
            for focal_length in (32.0, 42.0, 50.0, 60.0):
                base_distance = max(
                    2.0 * scene_radius,
                    width * focal_length * float(radii.max())
                    / (sensor * target_diameter_px),
                )
                for distance_scale in (1.0, 1.25, 1.55, 1.90, 2.30):
                    location = center + direction * base_distance * distance_scale
                    u, v, diameter, depth = project_spheres(
                        positions, radii,
                        location=location,
                        target=center,
                        focal_length_mm=focal_length,
                        sensor_width_mm=sensor,
                        width=width,
                        height=height,
                    )
                    if np.any(depth <= 0.0):
                        continue

                    radius_pixels = diameter / 2.0
                    margin = np.minimum.reduce(
                        (u - radius_pixels, width - u - radius_pixels,
                        v - radius_pixels, height - v - radius_pixels)
                    )
                    minimum_margin = float(margin.min())
                    required_margin = image_margin_fraction * min(width, height)
                    if minimum_margin < required_margin:
                        continue

                    minimum_diameter = float(diameter.min())
                    if minimum_diameter < minimum_diameter_px:
                        continue

                    median_diameter = float(np.median(diameter))
                    conditioning = _conditioning(
                        center,
                        location=location,
                        target=center,
                        focal_length_mm=focal_length,
                        sensor_width_mm=sensor,
                        width=width,
                        height=height,
                    )

                    size_score = np.exp(
                        -abs(np.log(max(median_diameter, 1.0) / target_diameter_px))
                    )
                    margin_score = min(1.0, minimum_margin / (0.20 * min(width, height)))
                    score = 0.50 * size_score + 0.30 * margin_score + 0.20 * conditioning

                    candidates.append(
                        CameraCandidate(
                            location=location,
                            target=center,
                            focal_length_mm=focal_length,
                            score=float(score),
                            minimum_diameter_px=minimum_diameter,
                            median_diameter_px=median_diameter,
                            maximum_diameter_px=float(diameter.max()),
                            minimum_margin_px=minimum_margin,
                            conditioning=conditioning,
                        )
                    )
    if not candidates:
        raise RuntimeError(
            "No valid camera candidate was found. Reduce minimum_diameter_px, "
            "increase render resolution, or narrow the trajectory bounds."
        )
    return max(candidates, key=lambda candidate: candidate.score)


def write_optimised_reder_config(
    input_config: str | Path,
    output_config: str | Path,
    candidate: CameraCandidate,
    report_path: str | Path | None = None,
) -> Path:
    input_config = Path(input_config)
    output_config = Path(output_config)
    raw = yaml.safe_load(input_config.read_text(encoding="utf-8"))
    raw["camera"]["location"] = candidate.location.tolist()
    raw["camera"]["target"] = candidate.target.tolist()
    raw["camera"]["focal_length_mm"] = candidate.focal_length_mm
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    if report_path is not None:
        report = {
            "score": candidate.score,
            "location": candidate.location.tolist(),
            "target": candidate.target.tolist(),
            "focal_length_mm": candidate.focal_length_mm,
            "minimum_diameter_px":candidate.minimum_diameter_px,
            "median_diameter_px": candidate.median_diameter_px,
            "maximum_diameter_px": candidate.maximum_diameter_px,
            "minimum_margin_px": candidate.minimum_margin_px,
            "conditioning": candidate.conditioning,
        }
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_config
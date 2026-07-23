# Single truth for all visualisation commands after parsing the hf5 file.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class PlaneGeometry:
    surface_id: str
    point: FloatArray
    normal: FloatArray

    def signed_distance(
        self,
        points: FloatArray,
    ) -> FloatArray:
        points = np.asarray(points, dtype=float)
        return (points - self.point) @ self.normal

    def sphere_clearance(
        self,
        points: FloatArray,
        radius: float,
    ) -> FloatArray:
        return self.signed_distance(points) - radius


@dataclass(frozen=True, slots=True)
class EnvironmentGeometry:
    kind: str
    surfaces: tuple[PlaneGeometry, ...]
    channel_width: float | None = None
    unbound_axis: str | None = None

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return tuple(
            surface.surface_id
            for surface in self.surfaces
        )

    def surface(
        self,
        surface_id: str,
    ) -> PlaneGeometry:
        for surface in self.surfaces:
            if surface.surface_id == surface_id:
                return surface

        raise KeyError(
            f"Unknown surface {surface_id!r}."
            f"Available surfaces: {self.surface_ids}."
        )

    def surface_index(
        self,
        surface_id: str,
    ) -> int:
        try:
            return self.surface_ids.index(surface_id)
        except ValueError as error:
            raise KeyError(
                f"Unknown surface {surface_id!r}."
            ) from error

def _decode_attribute(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)

def load_root_metadata(
    dataset: str | Path,
) -> dict[str, object]:
    with h5py.File(dataset, "r") as handle:
        if "metadata_json" not in handle.attrs:
            raise ValueError(
                "Dataset has no root 'metadata_json' attribute."
                "Datasets should store their complete"
                "experiment metadata."
            )

        raw_metadata = _decode_attribute(
            handle.attrs["metadata_json"]
        )

    metadata = json.loads(raw_metadata)

    if not isinstance(metadata, dict):
        raise ValueError(
            "metadata_json must decode to a dictionary."
        )
    
    return metadata

def load_environment_geometry(
        dataset: str | Path,
) -> EnvironmentGeometry:
    metadata = load_root_metadata(dataset)

    if "environment_geometry" not in metadata:
        raise ValueError(
            "metadata_json does not contain "
            "'environment_geometry'."
        )

    raw_geometry = metadata["environment_geometry"]

    if not isinstance(raw_geometry, dict):
        raise ValueError(
            "environment_geometry must be a dictionary."
        )

    raw_surfaces = raw_geometry.get("surfaces")

    if not isinstance(raw_surfaces, dict):
        raise ValueError(
            "environment_geometry must contain a "
            "'surfaces' dictionary."
        )
    surfaces: list[PlaneGeometry] = []

    for surface_id, raw_surface in raw_surfaces.items():
        if not isinstance(raw_surface, dict):
            raise ValueError(
                f"Surface {surface_id!r} is malformed."
            )

        point = np.asarray(raw_surface["point"], dtype=float)
        normal = np.asarray(raw_surface["normal"], dtype=float)

        if point.shape != (3, ):
            raise ValueError(
                f"Surface {surface_id!r} point must have "
                f"shape (3, ), received {point.shape}."
            )
        if normal.shape != (3, ):
            raise ValueError(
                f"Surface {surface_id!r} normal must have "
                f"shape (3, ), received {normal.shape}."
            )

        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 0.0:
            raise ValueError(
                f"Surface{surface_id!r} has a zero normal."
            )

        surfaces.append(
            PlaneGeometry(
                surface_id=surface_id,
                point=point,
                normal=normal / normal_norm,
            )
        )

    return EnvironmentGeometry(
        kind=str(raw_geometry["kind"]),
        surfaces=tuple(surfaces),
        channel_width=(
            float(raw_geometry["channel_width"])
            if "channel_width" in raw_geometry else None
        ),
        unbound_axis=(
            str(raw_geometry["axis"])
            if "axis" in raw_geometry else None
        ),
    )
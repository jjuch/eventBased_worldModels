"""Rigid-body shapes shared by the MuJoCo backend, the renderer and the evaluator.

Every shape lives in its body frame, whose origin is the centre of mass of a uniform-density solid. The simulator position ``p`` is therefore the centre of mass, and ``R`` maps these body coordinates to the world. All lengths are in m.

Polyhedra are convex. Faces are polygons of vertex indices, counter-clockwise seen from outside.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .config import ShapeConfig

MarkerPlacement = Literal["none", "tetrahedral", "vertices"]

APPEARANCE_FILE = "appearance.json"

# The first four colours are those of the original ball markers, in the same order, so a
# tetrahedral layout shows the training colours on any shape.
MARKER_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.90, 0.05, 0.25, 1.0),  # red
    (0.05, 0.25, 0.95, 1.0),  # blue
    (0.05, 0.80, 0.20, 1.0),  # green
    (0.95, 0.95, 0.05, 1.0),  # yellow
    (1.00, 0.45, 0.00, 1.0),  # orange
    (0.00, 0.85, 0.90, 1.0),  # cyan
    (0.85, 0.10, 0.85, 1.0),  # magenta
    (0.05, 0.05, 0.05, 1.0),  # black
)

# Marker directions of the original ball. On a cube they point exactly at alternate corners.
TETRAHEDRAL_DIRECTIONS = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
) / math.sqrt(3.0)


@dataclass(frozen=True, slots=True)
class RigidShape:
    """Geometry and mass distribution of one rigid body, in its body frame.

    vertices          [V, 3] in m, empty for a sphere
    faces             outward polygons as vertex indices, empty for a sphere
    sphere_radius     in m for a sphere, otherwise None
    inertia_per_mass  [3, 3] in m², inertia about the centre of mass divided by the mass
    """

    kind: str
    vertices: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    sphere_radius: float | None
    inertia_per_mass: np.ndarray

    @property
    def is_sphere(self) -> bool:
        return self.sphere_radius is not None

    @property
    def bounding_radius(self) -> float:
        """Largest distance from the centre of mass to the surface, in m."""
        if self.is_sphere:
            return float(self.sphere_radius)
        return float(np.linalg.norm(self.vertices, axis=1).max())

    def inertia(self, mass: float) -> np.ndarray:
        """Inertia tensor [3, 3] in kg m² about the centre of mass, body frame. ``mass`` in kg."""
        return mass * self.inertia_per_mass

    def triangles(self) -> np.ndarray:
        """Faces split into triangles [F, 3] with the same outward orientation."""
        return np.array(
            [
                (face[0], face[k], face[k + 1])
                for face in self.faces
                for k in range(1, len(face) - 1)
            ],
            dtype=int,
        )

    def surface_point(self, direction: np.ndarray) -> np.ndarray:
        """Point [3] in m where the ray from the centre of mass along ``direction`` exits."""
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        if self.is_sphere:
            return self.sphere_radius * direction
        distances = [
            offset / approach
            for normal, offset in _face_planes(self.vertices, self.faces)
            if (approach := float(normal @ direction)) > 1.0e-12
        ]
        return min(distances) * direction


def build_shape(config: ShapeConfig, sphere_radius: float) -> RigidShape:
    """Shape described by the trajectory config. ``sphere_radius`` (m) is only used for spheres."""
    if config.kind == "sphere":
        return RigidShape(
            kind="sphere",
            vertices=np.zeros((0, 3)),
            faces=(),
            sphere_radius=float(sphere_radius),
            inertia_per_mass=0.4 * sphere_radius**2 * np.eye(3),
        )
    x_extent, y_extent, z_extent = (float(value) for value in config.dimensions)
    if config.kind in ("cube", "box"):
        vertices, faces = _box(x_extent, y_extent, z_extent)
    elif config.kind == "pyramid":
        vertices, faces = _pyramid(x_extent, y_extent, z_extent)
    elif config.kind == "tetrahedron":
        vertices, faces = _tetrahedron(x_extent)
    else:
        raise ValueError(f"Unknown shape kind: {config.kind!r}.")
    return _centred_polyhedron(config.kind, vertices, faces)


def _box(x_extent: float, y_extent: float, z_extent: float):
    half = 0.5 * np.array([x_extent, y_extent, z_extent])
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    faces = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    return signs * half, faces


def _pyramid(x_extent: float, y_extent: float, height: float):
    """Rectangular base in the plane z = 0 with the apex above its centre."""
    half_x, half_y = 0.5 * x_extent, 0.5 * y_extent
    vertices = np.array(
        [
            [-half_x, -half_y, 0.0],
            [half_x, -half_y, 0.0],
            [half_x, half_y, 0.0],
            [-half_x, half_y, 0.0],
            [0.0, 0.0, height],
        ],
    )
    faces = ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))
    return vertices, faces


def _tetrahedron(cube_extent: float):
    """Regular tetrahedron on alternate corners of a cube with edge ``cube_extent``."""
    vertices = (
        0.5
        * cube_extent
        * np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    )
    faces = ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))
    return vertices, faces


def _centred_polyhedron(kind: str, vertices: np.ndarray, faces) -> RigidShape:
    faces = _orient_outward(vertices, faces)
    volume, centre, second_moment = polyhedron_mass_properties(vertices, faces)
    inertia_per_volume = np.trace(second_moment) * np.eye(3) - second_moment
    return RigidShape(
        kind=kind,
        vertices=vertices - centre,
        faces=faces,
        sphere_radius=None,
        inertia_per_mass=inertia_per_volume / volume,
    )


def _orient_outward(vertices: np.ndarray, faces) -> tuple[tuple[int, ...], ...]:
    """Reverse any face whose normal points into the (convex) body."""
    interior = vertices.mean(axis=0)
    oriented = []
    for face in faces:
        normal = _newell_normal(vertices[list(face)])
        outward = float(normal @ (vertices[face[0]] - interior)) > 0.0
        oriented.append(tuple(face) if outward else tuple(reversed(face)))
    return tuple(oriented)


def _newell_normal(polygon: np.ndarray) -> np.ndarray:
    """Unit normal of a planar polygon [N, 3], right-handed with its vertex order."""
    following = np.roll(polygon, -1, axis=0)
    normal = np.cross(polygon, following).sum(axis=0)
    return normal / np.linalg.norm(normal)


def _face_planes(vertices: np.ndarray, faces):
    """(unit outward normal, distance from the centre of mass) for every face."""
    for face in faces:
        normal = _newell_normal(vertices[list(face)])
        yield normal, float(normal @ vertices[face[0]])


def polyhedron_mass_properties(vertices: np.ndarray, faces) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume (m³), centre of mass [3] (m) and second moment ∫(x-c)(x-c)ᵀ dV [3, 3] (m⁵).

    Sums signed tetrahedra spanned by the origin and every face triangle. For a tetrahedron with
    edge matrix A = [a b c], ∫ x xᵀ dV = det(A) A C Aᵀ with C the canonical covariance below.
    """
    canonical = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 120.0
    volume, first_moment, second_moment = 0.0, np.zeros(3), np.zeros((3, 3))
    for face in faces:
        for k in range(1, len(face) - 1):
            edges = np.column_stack((vertices[face[0]], vertices[face[k]], vertices[face[k + 1]]))
            determinant = float(np.linalg.det(edges))
            volume += determinant / 6.0
            first_moment += determinant / 24.0 * edges.sum(axis=1)
            second_moment += determinant * edges @ canonical @ edges.T
    if volume <= 0.0:
        raise ValueError("Polyhedron faces must be closed and oriented outward.")
    centre = first_moment / volume
    return volume, centre, second_moment - volume * np.outer(centre, centre)


@dataclass(frozen=True, slots=True)
class MarkerLayout:
    """Coloured surface markers in the body frame.

    positions      [M, 3] in m, on the body surface
    color_indices  [M] indices into MARKER_COLORS; markers with the same index look identical
    """

    positions: np.ndarray
    color_indices: np.ndarray

    @property
    def colors(self) -> list[tuple[float, float, float, float]]:
        return [MARKER_COLORS[index] for index in self.color_indices]


def marker_layout(
    shape: RigidShape, placement: MarkerPlacement, count: int | None = None
) -> MarkerLayout:
    """Marker positions for a placement rule, each marker with its own colour.

    tetrahedral  where the original ball's four marker directions leave the surface
                 (four alternate corners of a cube)
    vertices     every corner of a polyhedron
    count        keep only the first ``count`` markers
    """
    if placement == "none":
        points = np.zeros((0, 3))
    elif placement == "tetrahedral":
        points = np.stack([shape.surface_point(direction) for direction in TETRAHEDRAL_DIRECTIONS])
    elif placement == "vertices":
        if shape.is_sphere:
            raise ValueError("A sphere has no vertices; use tetrahedral marker placement.")
        points = shape.vertices.copy()
    else:
        raise ValueError(f"Unknown marker placement: {placement!r}.")
    if count is not None:
        points = points[:count]
    if len(points) > len(MARKER_COLORS):
        raise ValueError(f"At most {len(MARKER_COLORS)} distinctly coloured markers are available.")
    return MarkerLayout(positions=points, color_indices=np.arange(len(points)))


def rotation_symmetries(
    points: np.ndarray, labels: np.ndarray, relative_tolerance: float = 1.0e-6
) -> np.ndarray | None:
    """All proper rotations [G, 3, 3] mapping a labelled point set onto itself, identity first.

    A rotation is a symmetry when every point lands on a point with the same label. Returns None
    when the set does not fix the orientation up to a finite group (all points on one line
    through the origin, or no points at all): the symmetry is then continuous.
    """
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    norms = np.linalg.norm(points, axis=1)
    if len(points) == 0 or norms.max() == 0.0:
        return None
    scale = float(norms.max())
    tolerance = relative_tolerance * scale

    first = int(np.argmax(norms > tolerance))
    crosses = np.linalg.norm(np.cross(points[first], points), axis=1)
    if crosses.max() <= tolerance * scale:
        return None
    second = int(np.argmax(crosses > tolerance * scale))
    reference = _frame(points[first], points[second])
    reference_dot = float(points[first] @ points[second])

    symmetries: list[np.ndarray] = []
    for image_first in range(len(points)):
        if (
            labels[image_first] != labels[first]
            or abs(norms[image_first] - norms[first]) > tolerance
        ):
            continue
        for image_second in range(len(points)):
            if (
                image_second == image_first
                or labels[image_second] != labels[second]
                or abs(norms[image_second] - norms[second]) > tolerance
                or abs(points[image_first] @ points[image_second] - reference_dot)
                > tolerance * scale
            ):
                continue
            rotation = _frame(points[image_first], points[image_second]) @ reference.T
            if _maps_onto_itself(rotation, points, labels, tolerance) and not any(
                np.allclose(rotation, known, atol=relative_tolerance) for known in symmetries
            ):
                symmetries.append(rotation)
    # Smallest rotation angle first, so the identity leads.
    symmetries.sort(key=lambda rotation: -np.trace(rotation))
    return np.stack(symmetries)


def _frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Right-handed orthonormal frame [3, 3] (columns) spanned by two non-parallel vectors."""
    axis_1 = first / np.linalg.norm(first)
    axis_2 = second - (second @ axis_1) * axis_1
    axis_2 = axis_2 / np.linalg.norm(axis_2)
    return np.column_stack((axis_1, axis_2, np.cross(axis_1, axis_2)))


def _maps_onto_itself(rotation, points, labels, tolerance) -> bool:
    mapped = points @ rotation.T
    distances = np.linalg.norm(mapped[:, None, :] - points[None, :, :], axis=-1)
    matches = (distances < tolerance) & (labels[:, None] == labels[None, :])
    return bool(matches.any(axis=1).all())


def appearance_symmetries(shape: RigidShape, markers: MarkerLayout) -> np.ndarray | None:
    """Body-frame rotations S [G, 3, 3] under which the rendered body looks identical.

    Orientations R and R S cannot be told apart from images. None means the appearance does not
    fix the orientation at all (a plain sphere, or markers on a single axis).
    """
    if shape.is_sphere:
        points, labels = markers.positions, markers.color_indices
    else:
        points = np.vstack((shape.vertices, markers.positions))
        labels = np.concatenate((np.full(len(shape.vertices), -1), markers.color_indices))
    return rotation_symmetries(points, labels)


def render_body_description(
    shape: RigidShape,
    markers: MarkerLayout,
    marker_angular_radius_degree: float,
    marker_surface_offset: float,
) -> dict[str, object]:
    """JSON-ready body for a Blender render job, everything in the body frame (m).

    Marker spheres have radius ``bounding_radius * sin(angular radius)`` and are sunk half a
    marker radius into the surface (minus ``marker_surface_offset``), as on the original ball.
    """
    marker_radius = shape.bounding_radius * math.sin(math.radians(marker_angular_radius_degree))
    markers_json = []
    for position, color in zip(markers.positions, markers.colors, strict=True):
        outward = position / np.linalg.norm(position)
        centre = position + outward * (marker_surface_offset - 0.5 * marker_radius)
        markers_json.append({"centre": centre.tolist(), "color": list(color)})
    return {
        "kind": shape.kind,
        "sphere_radius": shape.sphere_radius,
        "bounding_radius": shape.bounding_radius,
        "vertices": shape.vertices.tolist(),
        "faces": [list(face) for face in shape.faces],
        "marker_radius": marker_radius,
        "markers": markers_json,
    }


def write_appearance(directory: str | Path, shape: RigidShape, markers: MarkerLayout) -> Path:
    """Write the appearance symmetries next to the renders, for symmetry-aware evaluation."""
    symmetries = appearance_symmetries(shape, markers)
    payload = {
        "kind": shape.kind,
        "marker_count": int(len(markers.positions)),
        "symmetry_rotations": None if symmetries is None else symmetries.tolist(),
    }
    path = Path(directory) / APPEARANCE_FILE
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def shape_config_from_dataset(dataset: str | Path) -> ShapeConfig:
    """The shape config a trajectory dataset was generated with (stored in its metadata)."""
    from .config import ShapeConfig
    from .dataset_geometry import load_root_metadata

    return ShapeConfig.model_validate(load_root_metadata(dataset).get("shape", {}))

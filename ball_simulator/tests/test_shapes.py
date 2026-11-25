from __future__ import annotations

import math

import numpy as np
import pytest

from ball_simulator.trajectories.config import ShapeConfig
from ball_simulator.trajectories.shapes import (
    TETRAHEDRAL_DIRECTIONS,
    appearance_symmetries,
    build_shape,
    marker_layout,
    render_body_description,
)


def _shape(kind: str, dimensions=(0.16, 0.16, 0.16), radius: float = 0.1):
    return build_shape(ShapeConfig(kind=kind, dimensions=dimensions), radius)


def test_box_inertia_matches_closed_form():
    x, y, z = 0.10, 0.20, 0.30
    shape = _shape("box", (x, y, z))
    expected = np.diag([y**2 + z**2, x**2 + z**2, x**2 + y**2]) / 12.0
    np.testing.assert_allclose(shape.inertia_per_mass, expected, atol=1e-12)
    np.testing.assert_allclose(shape.vertices.mean(axis=0), 0.0, atol=1e-12)


def test_pyramid_centre_of_mass_and_inertia():
    base, height = 0.16, 0.20
    shape = _shape("pyramid", (base, base, height))
    # The centre of mass sits a quarter of the height above the base.
    np.testing.assert_allclose(shape.vertices[:4, 2], -height / 4.0, atol=1e-12)
    np.testing.assert_allclose(shape.vertices[4], [0.0, 0.0, 0.75 * height], atol=1e-12)
    lateral = base**2 / 20.0 + 3.0 * height**2 / 80.0
    expected = np.diag([lateral, lateral, base**2 / 10.0])
    np.testing.assert_allclose(shape.inertia_per_mass, expected, atol=1e-12)


def test_regular_tetrahedron_inertia_is_isotropic():
    cube_edge = 0.16
    shape = _shape("tetrahedron", (cube_edge,) * 3)
    edge = cube_edge * math.sqrt(2.0)
    np.testing.assert_allclose(shape.inertia_per_mass, edge**2 / 20.0 * np.eye(3), atol=1e-12)


def test_cube_tetrahedral_markers_sit_on_alternate_corners():
    shape = _shape("cube")
    markers = marker_layout(shape, "tetrahedral")
    for position in markers.positions:
        assert np.min(np.linalg.norm(shape.vertices - position, axis=1)) < 1e-12
    np.testing.assert_allclose(
        markers.positions[:, 0] * markers.positions[:, 1] * markers.positions[:, 2] > 0,
        [True, True, True, True],
    )


@pytest.mark.parametrize(
    ("kind", "dimensions", "group_size"),
    [
        ("cube", (0.16, 0.16, 0.16), 24),
        ("box", (0.10, 0.20, 0.30), 4),
        ("pyramid", (0.16, 0.16, 0.20), 4),
        ("pyramid", (0.12, 0.16, 0.20), 2),
        ("tetrahedron", (0.16, 0.16, 0.16), 12),
    ],
)
def test_plain_shape_symmetry_groups(kind, dimensions, group_size):
    shape = _shape(kind, dimensions)
    symmetries = appearance_symmetries(shape, marker_layout(shape, "none"))
    assert symmetries is not None and len(symmetries) == group_size
    np.testing.assert_allclose(symmetries[0], np.eye(3), atol=1e-12)
    for rotation in symmetries:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_coloured_markers_remove_symmetry():
    for shape in (_shape("cube"), _shape("sphere")):
        symmetries = appearance_symmetries(shape, marker_layout(shape, "tetrahedral"))
        assert symmetries is not None and len(symmetries) == 1


def test_plain_sphere_orientation_is_unobservable():
    sphere = _shape("sphere")
    assert appearance_symmetries(sphere, marker_layout(sphere, "none")) is None


def test_sphere_marker_centres_match_original_renderer():
    radius, angular_radius, offset = 0.1, 16.0, 0.003
    sphere = _shape("sphere", radius=radius)
    body = render_body_description(
        sphere, marker_layout(sphere, "tetrahedral"), angular_radius, offset
    )
    marker_radius = radius * math.sin(math.radians(angular_radius))
    expected = TETRAHEDRAL_DIRECTIONS * (radius + offset - 0.5 * marker_radius)
    np.testing.assert_allclose(
        [marker["centre"] for marker in body["markers"]], expected, atol=1e-12
    )
    assert body["marker_radius"] == pytest.approx(marker_radius)


def test_unequal_cube_dimensions_are_rejected():
    with pytest.raises(ValueError):
        ShapeConfig(kind="cube", dimensions=(0.1, 0.1, 0.2))

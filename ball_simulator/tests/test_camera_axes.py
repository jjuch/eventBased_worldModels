"""Camera axes conversion, look-at and pinhole projection."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ball_simulator.rendering.camera_axes import (
    CameraIntrinsics,
    flip_camera_axes,
    look_at_blender,
    look_at_opencv,
    world_to_camera,
)

INTRINSICS = CameraIntrinsics(width=384, height=384, focal_length_mm=42.0, sensor_width_mm=36.0)


def test_axis_flip_round_trips_and_keeps_a_rotation():
    rotations = Rotation.random(20, random_state=1).as_matrix()
    flipped = flip_camera_axes(rotations)
    np.testing.assert_allclose(flip_camera_axes(flipped), rotations, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(flipped), 1.0, atol=1e-12)
    # Only the camera's own y and z axes change sign; the viewing axis flips direction.
    np.testing.assert_allclose(flipped[:, :, 0], rotations[:, :, 0])
    np.testing.assert_allclose(flipped[:, :, 2], -rotations[:, :, 2])


def test_look_at_points_the_optical_axis_at_the_target_with_image_down_towards_floor():
    location, target = np.array([4.0, -5.0, 3.0]), np.array([0.0, 0.0, 0.65])
    rotation = look_at_opencv(location, target)
    forward = (target - location) / np.linalg.norm(target - location)
    np.testing.assert_allclose(rotation[:, 2], forward, atol=1e-12)
    assert rotation[2, 1] < 0.0  # camera y (image down) has a negative world z component
    assert abs(rotation[2, 0]) < 1e-12  # no roll: camera x stays horizontal
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


def test_target_projects_to_image_centre_and_offsets_to_expected_pixels():
    location, target = np.array([4.0, -5.0, 3.0]), np.array([0.0, 0.0, 0.65])
    rotation = look_at_opencv(location, target)
    centre = INTRINSICS.project(world_to_camera(target, location, rotation))
    np.testing.assert_allclose(centre, [192.0, 192.0], atol=1e-9)

    # A point 0.1 m along camera x and camera y at the target's depth.
    depth = np.linalg.norm(target - location)
    point = target + 0.1 * rotation[:, 0] + 0.1 * rotation[:, 1]
    pixel = INTRINSICS.project(world_to_camera(point, location, rotation))
    offset = INTRINSICS.focal_pixels * 0.1 / depth
    np.testing.assert_allclose(pixel, [192.0 + offset, 192.0 + offset], atol=1e-9)
    assert INTRINSICS.focal_pixels == pytest.approx(384 * 42.0 / 36.0)


def test_world_up_points_up_in_the_image():
    location, target = np.array([4.0, -5.0, 3.0]), np.array([0.0, 0.0, 0.65])
    rotation = look_at_opencv(location, target)
    above = INTRINSICS.project(world_to_camera(target + [0.0, 0.0, 0.2], location, rotation))
    assert above[1] < 192.0  # smaller row = higher in the image


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender not on PATH")
def test_look_at_matches_blender_track_quaternion():
    location, target = [4.0, -5.0, 3.0], [0.3, 0.2, 0.65]
    script = (
        "import json; from mathutils import Vector; "
        f"d = Vector({target}) - Vector({location}); "
        "m = d.to_track_quat('-Z', 'Y').to_matrix(); "
        "print('MATRIX', json.dumps([list(row) for row in m]))"
    )
    output = subprocess.run(
        ["blender", "-b", "--factory-startup", "--python-expr", script],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    matrix = json.loads(output.split("MATRIX", 1)[1].strip().splitlines()[0])
    np.testing.assert_allclose(look_at_blender(location, target), matrix, atol=1e-6)

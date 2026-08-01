from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

TRAJECTORY_PATTERN = re.compile(r"^trajectory_(\d{8})$")
REQUIRED_STATE_FIELDS = (
    "time",
    "position",
    "quaternion_xyzw",
    "linear_velocity",
    "angular_velocity",
)


@dataclass(frozen=True, slots=True)
class TrajectoryValidation:
    trajectory_id: str
    trajectory_directory: Path
    rgb_directory: Path
    states_path: Path
    metadata_path: Path
    frame_count: int
    width: int
    height: int
    fps: float
    duration: float
    environment_kind: str


def discover_trajectory_directories(root: Path | str) -> list[Path]:
    root = Path(root)
    print("root: ", root)
    if not root.is_dir():
        raise FileNotFoundError(f"Rendered dataset root does not exist: {root}")
    directories = [
        path for path in root.iterdir()
        if path.is_dir() and TRAJECTORY_PATTERN.fullmatch(path.name)
    ]
    return sorted(directories, key=lambda path: int(path.name.rsplit('_', 1)[1]))


def _numbered_rgb_paths(rgb_directory: Path) -> list[Path]:
    paths = sorted(rgb_directory.glob("*.png"))
    expected = [f"{index:06d}.png" for index in range(len(paths))]
    actual = [path.name for path in paths]

    if actual != expected:
        raise ValueError(
            f"RGB frames in {rgb_directory} are missing, duplicated, or not numbered contiguously from 000000.png."
        )
    return paths


def validate_trajectory(
    trajectory_directory: str | Path,
    *,
    require_success_marker: bool = True,
) -> TrajectoryValidation:
    trajectory_directory = Path(trajectory_directory)
    match = TRAJECTORY_PATTERN.fullmatch(trajectory_directory.name)
    if match is None:
        raise ValueError(f"Invalid trajectory directory name: {trajectory_directory.name}")
    trajectory_id = match.group(1)
    rgb_directory = trajectory_directory / "rgb"
    states_path = trajectory_directory / "states.npz"
    metadata_path = trajectory_directory / "metadata.json"
    success_path = trajectory_directory / " _SUCCESS"


    if not rgb_directory.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_directory}")
    for required in (states_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing required trajectory file: {required}")
    if require_success_marker and not success_path.is_file():
        raise FileNotFoundError(f"Missing render completion marker: {success_path}")

    rgb_paths = _numbered_rgb_paths(rgb_directory)
    if not rgb_paths:
        raise ValueError(f"Trajectory has no RGB frames: {trajectory_directory}")

    with np.load(states_path) as states:
        missing = [field for field in REQUIRED_STATE_FIELDS if field not in states]
        if missing:
            raise ValueError(f"{states_path} is missing fields: {missing}")
        time = np.asarray(states["time"], dtype=np.float64)
        position = np.asarray(states["position"])
        quaternion = np.asarray(states["quaternion_xyzw"])
        linear_velocity = np.asarray(states["linear_velocity"])
        angular_velocity = np.asarray(states["angular_velocity"])

    frame_count = len(rgb_paths)
    expected_shapes = {
        "position": (frame_count, 3),
        "quaternion_xyzw": (frame_count, 4),
        "linear_velocity": (frame_count, 3),
        "angular_velocity": (frame_count, 3),
    }
    arrays = {
        "position": position,
        "quaternion_xyzw": quaternion,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
    }
    if time.shape != (frame_count,):
        raise ValueError(f"time shape {time.shape} does not match {frame_count} frames")
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{name} shape {arrays[name].shape}; expected {expected}")
    if frame_count > 1 and not np.all(np.diff(time) > 0.0):
        raise ValueError(f"Timestamps are not strictly increasing in {states_path}")
    norms = np.linalg.norm(quaternion, axis=1)
    if not np.allclose(norms, 1.0, atol=1.0e-5):
        raise ValueError(f"Non-unit quaternions found in {states_path}")

    with Image.open(rgb_paths[0]) as image:
        width, height = image.size
    # Cheap shape consistency check on the final frame as well.
    with Image.open(rgb_paths[-1]) as image:
        if image.size != (width, height):
            raise ValueError(f"Inconsistent RGB dimensions in {rgb_directory}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    render = metadata.get("render", {})
    fps = float(render.get("fps", 1.0 / np.median(np.diff(time)))) if frame_count > 1 else 0.0
    environment = metadata.get("environment", {})
    environment_kind = str(environment.get("kind", "unknown"))
    duration = float(time[-1] - time[0]) if frame_count > 1 else 0.0

    return TrajectoryValidation(
        trajectory_id=trajectory_id,
        trajectory_directory=trajectory_directory.resolve(),
        rgb_directory=rgb_directory.resolve(),
        states_path=states_path.resolve(),
        metadata_path=metadata_path.resolve(),
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        environment_kind=environment_kind,
    )

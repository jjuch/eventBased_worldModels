from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ball_world_model.config import DataConfig
from .manifest import load_manifest
from .transforms import FrameTransform
from .discovery import animation_frame_paths

@dataclass(frozen=True, slots=True)
class WindowIndex:
    row_index: int
    window_index: int


class RenderedTrajectoryDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        config: DataConfig,
        *,
        split: str,
    ) -> None:
        frame = (
            load_manifest(manifest) 
            if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        )
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Manifest contains no trajectories for split {split!r}.")

        self.config = config
        self.split = split
        self.frame_transform = FrameTransform(config.image)
        self.windows = self._build_window_index()


    def _build_window_index(self) -> list[WindowIndex]:
        required = self.config.temporal.required_source_frames
        windows: list[WindowIndex] = []

        for row_index, row in self.frame.iterrows():
            available = int(row["frame_count"]) - required + 1
            if available <= 0:
                raise ValueError(
                    f"Trajectory {row['trajectory_id']} has {row['frame_count']} frames, "
                    f"but {required} are required."
                )

            count = min(self.config.sampling.windows_per_trajectory, available)
            for window_index in range(count):
                windows.append(
                    WindowIndex(
                        row_index=row_index,
                        window_index=window_index,
                    )
                )
        return windows


    def __len__(self) -> int:
        return len(self.windows)


    def _start_frame(self, item: WindowIndex, frame_count: int) -> int:
        required = self.config.temporal.required_source_frames
        maximum_start = frame_count - required

        if not self.config.sampling.random_start:
            if self.config.sampling.windows_per_trajectory == 1:
                return 0
            return min(item.window_index, maximum_start)

        # Deterministic per item and epoch-independent for reproducible baselines.
        seed = np.random.SeedSequence([
            self.config.sampling.seed,
            item.row_index,
            item.window_index,
        ])
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, maximum_start + 1))


    def _indices(self, start_frame: int) -> tuple[np.ndarray, np.ndarray]:
        temporal = self.config.temporal
        all_indices = start_frame + np.arange(
            temporal.context_frames + temporal.prediction_frames
        ) * temporal.frame_stride
        return (
            all_indices[: temporal.context_frames],
            all_indices[temporal.context_frames :],
        )


    def _load_images(self, rgb_directory: Path, indices: np.ndarray) -> torch.Tensor:
        frame_paths = animation_frame_paths(rgb_directory)

        images = []
        for index in indices:
            # Dataset indices are zero-based.
            # Blender animation frame numbers are one-based.
            zero_based_index = int(index)

            if zero_based_index >= len(frame_paths):
                raise IndexError(
                    f"Requested frame{zero_based_index}, "
                    f"but {rgb_directory} contains only "
                    f"{len(frame_paths)} frames."
                )

            path = frame_paths[zero_based_index]
            with Image.open(path) as image:
                images.append(self.frame_transform(image))
        return torch.stack(images, dim=0)


    @staticmethod
    def _tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), dtype=torch.float32)


    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        row = self.frame.iloc[window.row_index]
        frame_count = int(row["frame_count"])
        start_frame = self._start_frame(window, frame_count)
        context_indices, future_indices = self._indices(start_frame)
        rgb_directory = Path(row["rgb_directory"])

        context_rgb = self._load_images(rgb_directory, context_indices)
        future_rgb = self._load_images(rgb_directory, future_indices)

        with np.load(Path(row["states_path"])) as states:
            data = {name: np.asarray(states[name]) for name in (
                "time",
                "position",
                "quaternion_xyzw",
                "linear_velocity",
                "angular_velocity",
            )}

        sample: dict[str, object] = {
            "trajectory_id": str(row["trajectory_id"]).zfill(8),
            "start_frame": int(start_frame),
            "context_indices": torch.as_tensor(context_indices, dtype=torch.int64),
            "future_indices": torch.as_tensor(future_indices, dtype=torch.int64),
            "context_rgb": context_rgb,
            "future_rgb": future_rgb,
        }

        for name, array in data.items():
            sample[f"context_{name}"] = self._tensor(array[context_indices])
            sample[f"future_{name}"] = self._tensor(array[future_indices])

        return sample


def build_dataloader(
    dataset: RenderedTrajectoryDataset,
    config: DataConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    loader = config.loader
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": loader.batch_size,
        "shuffle": shuffle,
        "num_workers": loader.workers,
        "pin_memory": loader.pin_memory and torch.cuda.is_available(),
        "drop_last": False,
    }
    if loader.workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
        kwargs["persistent_workers"] = loader.persistent_workers

    return DataLoader(**kwargs)

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ImageConfig(BaseModel):
    width: int = Field(default=128, ge=32)
    height: int = Field(default=128, ge=32)
    channels: Literal["imagenet", "unit"] = "imagenet"


class TemporalConfig(BaseModel):
    rendered_fps: float = Field(default=100.0, gt=0.0)
    frame_stride: int = Field(default=1, ge=1)
    context_frames: int = Field(default=10, ge=2)
    prediction_frames: int = Field(default=10, ge=1)
    target_history_frames: int = Field(default=4, ge=1)

    @property
    def required_source_frames(self) -> int:
        return 1 + (
            self.context_frames + self.prediction_frames - 1
        ) * self.frame_stride

    @model_validator(mode="after")
    def validate_target_history(self) -> "TemporalConfig":
        if self.target_history_frames > self.context_frames:
            raise ValueError(
                "target_history_frames cannot exceed context_frames."
            )
        return self


class SamplingConfig(BaseModel):
    windows_per_trajectory: int = Field(default=1, ge=1)
    random_start: bool = False
    seed: int = 20260801


class SplitConfig(BaseModel):
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    seed: int = 20260801

    @model_validator(mode="after")
    def validate_fractions(self) -> "SplitConfig":
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1.0e-9:
            raise ValueError(f"Split fractions must sum to 1.0, received {total}.")
        return self


class LoaderConfig(BaseModel):
    batch_size: int = Field(default=8, ge=1)
    workers: int = Field(default=4, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    pin_memory: bool = True
    persistent_workers: bool = True


class DataConfig(BaseModel):
    root: Path
    manifest_path: Path | None = None
    image: ImageConfig = ImageConfig()
    temporal: TemporalConfig = TemporalConfig()
    sampling: SamplingConfig = SamplingConfig()
    splits: SplitConfig = SplitConfig()
    loader: LoaderConfig = LoaderConfig()
    require_success_marker: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        config = cls.model_validate(raw)
        # Paths in YAML are interpreted relative to the process working directory.
        return config
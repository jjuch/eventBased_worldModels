from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

class CameraConfig(BaseModel):
    location: tuple[float, float, float] = (3.8, -4.5, 2.8)
    target: tuple[float, float, float] = (1.0, 0.0, 0.7)
    focal_length_mm: float = Field(default=42.0, gt=1.0)
    sensor_width_mm: float = Field(default=36.0, gt=1.0)


class LightingConfig(BaseModel):
    world_strength: float = Field(default=0.18, ge=0.0)
    key_energy: float = Field(default=850.0, ge=0.0)
    key_size: float = Field(default=4.0, gt=0.0)
    fill_energy: float = Field(default=300.0, ge=0.0)
    fill_size: float = Field(default=3.0, gt=0.0)


class OutputConfig(BaseModel):
    rgb: bool = True
    depth: bool = True
    segmentation: bool = True
    preview_mp4: bool = True
    keep_exr: bool = False


class RenderConfig(BaseModel):
    width: int = Field(default=128, ge=32)
    height: int = Field(default=128, ge=32)
    fps: int = Field(default=100, ge=1)
    engine: Literal["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"] = "BLENDER_EEVEE_NEXT"
    samples: int = Field(default=16, ge=1)
    transparent_background: bool = False
    source: Literal["observations", "high_rate"] = "high_rate"
    frame_dt: float | None = Field(default=0.01, gt=0.0)
    fixed_visual_y_extent: float = Field(default=4.0, gt=0.0)
    wall_thickness: float = Field(default=0.08, gt=0.0)
    texture_scale: float = Field(default=5.0, gt=0.0)
    camera: CameraConfig = CameraConfig()
    lighting: LightingConfig = LightingConfig()
    outputs: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def validate_outputs(self) -> "RenderConfig":
        if not self.outputs.rgb and not self.outputs.depth and not self.outputs.segmentation:
            raise ValueError("At least one render output must be enabled.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RenderConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))
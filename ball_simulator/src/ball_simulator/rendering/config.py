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


class EnvironmentRenderConfig(BaseModel):
    representation: Literal[
        "opaque",
        "transparent_near_wall",
        "cutaway",
        "boundaries_only",
    ] = "cutaway"

    near_wall_alpha: float = Field(default=0.12, ge=0.0, le=1.0)
    show_near_wall_frame: bool = True
    frame_thickness: float = Field(default=0.025, gt=0.0)
    show_near_wall_grid: bool = True
    grid_spacing: float = Field(default=0.50, gt=0.0)
    grid_thickness: float = Field(default=0.008, gt=0.0)
    checkerboard_enabled: bool = True
    checker_size: float = Field(default=0.25, gt=0.0)
    checker_contrast: float = Field(default=0.20, ge=0.0, le=1.0)
    floor_margin: float = Field(defualt=0.08, ge=0.0, description=(
        "Static visual extension of the floor beyong "
        "each vertical channel wall, in meters."
    ),)

class BallRenderConfig(BaseModel):
    base_color: tuple[float, float, float, float] = (0.78, 0.80, 0.84, 1.0)
    roughness: float = Field(default=0.42, ge=0.0, le=1.0)
    markers_enabled: bool = True
    marker_angular_radius_degree: float = Field(default=13.0, gt=1.0, lt=45.0)
    marker_surface_offset: float = Field(default=0.003, ge=0.0)


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
    camera: CameraConfig = CameraConfig()
    environment: EnvironmentRenderConfig = EnvironmentRenderConfig()
    ball: BallRenderConfig = BallRenderConfig()
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



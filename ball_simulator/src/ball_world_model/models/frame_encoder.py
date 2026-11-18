"""Coordinate-aware image encoder retained from the successful observer idea."""

from __future__ import annotations

import torch
from torch import Tensor, nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class CoordinateAwareFrameEncoder(nn.Module):
    def __init__(self, feature_channels: int = 128, descriptor_dim: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(5, 32), 
            ConvBlock(32, 64),
            ConvBlock(64, feature_channels), 
            ConvBlock(feature_channels, feature_channels),
        )
        self.descriptor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(feature_channels, descriptor_dim), 
            nn.LayerNorm(descriptor_dim), 
            nn.SiLU(),
        )
        self.feature_channels = feature_channels

    @staticmethod
    def _coordinates(batch: int, height: int, width: int, reference: Tensor) -> Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        flat = torch.cat((flat, self._coordinates(len(flat), height, width, flat)), dim=1)
        maps = self.backbone(flat)
        descriptors = self.descriptor(maps)
        return maps.reshape(batch, time, *maps.shape[1:]), descriptors.reshape(batch, time, -1)

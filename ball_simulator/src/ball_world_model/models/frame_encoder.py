"""Coordinate-aware image encoder retained from the successful observer idea."""

from __future__ import annotations

from dataclasses import dataclass
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
    def __init__(self, feature_channels: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(5, 32), 
            ConvBlock(32, 64),
            ConvBlock(64, feature_channels), 
            ConvBlock(feature_channels, feature_channels),
        )

        self.feature_channels = feature_channels

    @staticmethod
    def _coordinates(batch: int, height: int, width: int, reference: Tensor) -> Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, frames: Tensor) -> Tensor:
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        flat = torch.cat((flat, self._coordinates(len(flat), height, width, flat)), dim=1)
        maps = self.backbone(flat)
        return maps.reshape(batch, time, *maps.shape[1:])


@dataclass(frozen=True)
class SpatialFeatures:
    coordinates: Tensor       # [B, T, K, 2]
    descriptors: Tensor       # [B, T, K, D]
    confidence: Tensor        # [B, T, K]
    attention_maps: Tensor    # [B, T, K, H, W]
    frame_descriptor: Tensor  # [B, T, E]

class SpatialKeypointEncoder(nn.Module):
    """Soft keypoints retain position and marker geometry without legacy sectors."""
    def __init__(self, feature_channels: int, keypoints: int = 8, keypoint_dim: int = 32, descriptor_dim: int = 256) -> None:
        super().__init__()
        self.keypoints = keypoints
        self.attention = nn.Conv2d(feature_channels, keypoints, 1)
        self.descriptor_map = nn.Conv2d(feature_channels, keypoint_dim, 1)
        self.project = nn.Sequential(
            nn.Linear(keypoints * (2 + keypoint_dim + 1), descriptor_dim),
            nn.LayerNorm(descriptor_dim), 
            nn.SiLU(),
            nn.Linear(descriptor_dim, descriptor_dim), 
            nn.LayerNorm(descriptor_dim), 
            nn.SiLU(),
        )

    def forward(self, maps: Tensor) -> SpatialFeatures:
        batch, time, channels, height, width = maps.shape
        flat = maps.flatten(0, 1)
        logits = self.attention(flat).flatten(-2)
        weights = torch.softmax(logits, dim=-1)

        y = torch.linspace(-1, 1, height, device=maps.device, dtype=maps.dtype)
        x = torch.linspace(-1, 1, width, device=maps.device, dtype=maps.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

        coordinates = torch.einsum("bkn,nc->bkc", weights, grid)
        descriptor_map = self.descriptor_map(flat).flatten(-2)
        descriptors = torch.einsum("bkn,bdn->bkd", weights, descriptor_map)
        confidence = weights.amax(dim=-1)

        packed = torch.cat((coordinates, descriptors, confidence.unsqueeze(-1)), dim=-1).flatten(1)
        frame_descriptor = self.project(packed)
        
        return SpatialFeatures(
            coordinates.reshape(batch, time, self.keypoints, 2),
            descriptors.reshape(batch, time, self.keypoints, -1),
            confidence.reshape(batch, time, self.keypoints),
            weights.reshape(batch, time, self.keypoints, height, width),
            frame_descriptor.reshape(batch, time, -1),
        )

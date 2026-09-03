from __future__ import annotations

import torch
from torch import nn


def coordinate_channels(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalised image coordinates with shape (B, 2, H, W)."""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels:int, *, stride: int = 1) -> None:
        """
        Classic ResNet idea: 
            y = sigma(F(x) + x)
        where F(x) learns a correction, skip path preserves information, gradients can flow directly through the identity branch. 
        """
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SoftkeypointReadout(nn.Module):
    """
    Convert a feature map into differentiable keypoint-like measurements.

    No masks or bounding boxes are used. Each learned heatmap produces an expected
    image coordinate, spread, confidence, and locally pooled appearance vector.
    This preserves position and marker information that global average pooling loses.
    """
    def __init__(self, channels: int, keypoints: int = 8, appearance_dim: int = 16) -> None:
        super().__init__()
        self.keypoints = keypoints
        self.heatmaps = nn.Conv2d(channels, keypoints, 1)
        self.appearance = nn.Conv2d(channels, keypoints * appearance_dim, 1)
        self.appearance_dim = appearance_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        logits = self.heatmaps(features).flatten(2)
        probabilities = torch.softmax(logits, dim=-1)

        coords = coordinate_channels(
            1, height, width, device=features.device, dtype=features.dtype
        )[0].flatten(1).transpose(0, 1) # (H*W, 2)
        expected = probabilities @ coords
        second_moment = probabilities @ (coords * coords)
        variance = (second_moment - expected * expected).clamp_min(0.0)
        confidence = probabilities.amax(dim=-1, keepdim=True)

        appearance = self.appearance(features).reshape(
            batch, self.keypoints, self.appearance_dim, height * width
        )
        pooled_appearance = torch.einsum("bkn,bkdn->bkd", probabilities, appearance)
        return torch.cat((expected, variance, confidence, pooled_appearance), dim=-1).flatten(1)


class CoordinateAwareFrameEncoder(nn.Module):
    """Shared per-frame encoder mapping each RGB frame to one compact embedding that preserves object and marker coordinates."""

    def __init__(
        self, 
        embedding_dim: int = 256,
        base_channels: int = 32,
        keypoints: int = 8,
        keypoint_appearance_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        # Go from a single RGB image to a latent vector of length 'embedding_dim'
        self.feature_channels = base_channels * 4
        self.backbone = nn.Sequential(
            ConvBlock(5, base_channels, stride=2),
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ConvBlock(base_channels * 4, base_channels * 4, stride=2),
        )
        self.readout = SoftkeypointReadout(
            self.feature_channels,
            keypoints=keypoints,
            appearance_dim=keypoint_appearance_dim,
        )
        readout_dim = keypoints * (2 + 2 + 1 + keypoint_appearance_dim)
        self.projector = nn.Sequential(
            nn.Linear(readout_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )


    def encode_feature_maps(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(f"Expected (B, T, 3, H, W), recieved {tuple(frames.shape)}.")

        batch, time, _, height, width = frames.shape
        flat = frames.flatten(0, 1)
        coords = coordinate_channels(
            batch * time, height, width, device=frames.device, dtype=frames.dtype
        )
        maps = self.backbone(torch.cat((flat, coords), dim=1))
        return maps.reshape(batch, time, *maps.shape[1:])


    def embeddings_from_maps(self, maps: torch.Tensor) -> torch.Tensor:
        if maps.ndim != 5:
            raise ValueError("Expected feature maps with shape (B, T, C, H, W).")
        batch, time = maps.shape[:2]
        embeddings = self.projector(self.readout(maps.flatten(0, 1)))
        return embeddings.reshape(batch, time, self.embedding_dim)


    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Process a video sequence, by applying the same encoder to each frame. The frames tensor is batch B, time T, color (RGB), height H, width W

        Parameters
        ----------
        frames:
            Tensor with shape ``(B, T, 3, H, W)``.
        Returns
        -------
        Tensor with shape ``(B, T, D)``.
        """
        return self.embeddings_from_maps(self.encode_feature_maps(frames))
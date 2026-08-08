from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, stride: int = 1) -> None:
        """
        Classic ResNet idea: 
            y = sigma(F(x) + x)
        where F(x) learns a correction, skip path preserves information, gradients can flow directly through the identity branch. 
        """
        super().__init__()
        output_channels = channels * 2 if stride == 2 else channels
        self.residual = nn.Sequential(
            nn.Conv2d(channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, output_channels), output_channels)
        )
        self.skip = (
            nn.Conv2d(channels, output_channels, 1, stride=stride, bias=False)
            if stride != 1
            else nn.Identity()
        )
        self.activation = nn.SiLU()


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.residual(inputs) + self.skip(inputs))



class SmallFrameEncoder(nn.Module):
    """Shared encoder mapping each RGB frame to one compact embedding."""

    def __init__(self, embedding_dim: int = 384, base_channels: int = 32) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        # Go from a single RGB image to a latent vector of length 'embedding_dim'
        self.network = nn.Sequential(
            nn.Conv2d(3, base_channels, 5, stride=2, padding=2, bias=False), # 5x5 convolution
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels, stride=2), # downsample
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2, stride=2), # downsample
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, stride=2), # downsample
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )


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
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(f"Expected (B,T,3,H,W), received {tuple(frames.shape)}.")
        batch, time = frames.shape[:2]
        embeddings = self.network(frames.flatten(0, 1))
        return embeddings.reshape(batch, time, self.embedding_dim)
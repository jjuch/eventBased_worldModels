from __future__ import annotations

import torch
from torch import nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.network(inputs))
    

class TemporalConvEncoder(nn.Module):
    """Fixed-depth, easy-to-debug temporal encoder returning one feature per frame."""
    def __init__(
        self,
        embedding_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    embedding_dim,
                    dilation=2 ** min(index, 3),
                    dropout=dropout,
                ) for index in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)


    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError("Expected frame embeddings with shape (B, T, D).")
        
        encoded = self.blocks(frame_embeddings.transpose(1, 2)).transpose(1,2)
        return self.output_norm(encoded)
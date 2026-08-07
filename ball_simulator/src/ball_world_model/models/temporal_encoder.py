from __future__ import annotations

import torch
from torch import nn


class TemporalContextEncoder(nn.Module):
    """Fixed-depth causal-free context encoder with a learned belief token."""
    def __init__(
        self,
        embedding_dim: int = 384,
        depth: int = 4,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        maximum_frames: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads.")
        
        self.maximum_frames = maximum_frames
        self.belief_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, maximum_frames + 1, embedding_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(embedding_dim)
        nn.init.trunc_normal_(self.belief_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)


    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError("Expected frame embeddings with shape (B, T, D).")
        
        batch, time, _ = frame_embeddings.shape
        if time > self.maximum_frames:
            raise ValueError(f"Recieved {time} frames, maximum supported is {self.maximum_frames}.")

        belief = self.belief_token.expand(batch, -1, -1)
        tokens = torch.cat((belief, frame_embeddings), dim=1)
        tokens = tokens + self.position_embedding[:, : time + 1]
        encoded = self.transformer(tokens)
        return self.output_norm(encoded[:, 0])
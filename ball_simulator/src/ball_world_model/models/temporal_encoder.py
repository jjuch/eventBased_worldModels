"""Motion evidence between consecutive frames and its temporal context.

Pipeline per window:
    feature maps F[0..T-1]
    -> standardised feature rate (F[k+1] - F[k]) / dt          (RunningStandardiser)
    -> one motion vector per interval                         (IntervalMotionEncoder)
    -> neighbouring intervals mixed along time                 (IntervalTemporalMixer)

The dense two-frame encoder is the design that reached R^2 > 0.97 on the physics_latents branch; the temporal mixer replaces the order-agnostic attention pool so that every interval keeps its own estimate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

def group_count(channels: int, maximum: int = 8) -> int:
    """Largest number of GroupNorm groups <= maximum that divides channels."""
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -=1
    return groups

class RunningStandardiser(nn.Module):
    """Per-channel standardisation of feature rates with running statistics.

    Raw rates are O(1/dt) larger than the feature maps and their scale differs per channel. Standardising keeps relative magnitudes (slow vs fast motion) while bringing every channel to unit scale. Statistics are frozen in eval mode.
    Input shape: [B, I, C, H, W].
    """
    def __init__(self, channels: int, momentum: float = 0.01, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _updates(self, rate: Tensor) -> None:
        reduce_dims = (0, 1, 3, 4)
        mean = rate.mean(dim=reduce_dims)
        variance = rate.var(dim=reduce_dims, unbiased=False)
        if self.updates.item() == 0:
            self.running_mean.copy_(mean)
            self.running_var.copy_(variance)
        else:
            self.running_mean.lerp_(mean, self.momentum)
            self.running_var.lerp_(variance, self.momentum)
        self.updates.add_(1)

    def forward(self, rate: Tensor) -> Tensor:
        if rate.ndim != 5:
            raise ValueError(f"Expected rates shaped [B, I, C, H, W], received {tuple(rate.shape)}.")
        if self.training:
            self._updates(rate.detach())

        mean = self.running_mean.view(1, 1, -1, 1, 1)
        std = torch.sqrt(self.running_var + self.epsilon).view(1, 1, -1, 1,1)
        return (rate - mean) / std


class IntervalMotionEncoder(nn.Module):
    """Encode frame pairs (F_a, F_b, standardised rate (F_b - F_a) / gap) into motion vectors.

    ``forward`` pairs consecutive frames (one vector per interval); ``encode_pairs`` takes any
    pairs, e.g. every context frame with the anchor frame.
    """

    def __init__(self, feature_channels: int, motion_dim: int, momentum: float = 0.01) -> None:
        super().__init__()
        groups = group_count(feature_channels)
        self.rate_standardiser = RunningStandardiser(feature_channels, momentum)
        self.network = nn.Sequential(
            nn.Conv2d(3 * feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels, motion_dim),
            nn.LayerNorm(motion_dim),
            nn.SiLU(),
        )
        
    def encode_pairs(self, first: Tensor, second: Tensor, gap: Tensor) -> Tensor:
        """first, second [B, N, C, H, W], gap [B, N] in s -> motion features [B, N, D]."""
        # Differences of nearly equal values need float32.
        first, second = first.float(), second.float()
        rate = self.rate_standardiser((second - first) / gap[..., None, None, None])
        pairs = torch.cat((first, second, rate), dim=2)
        batch, count = pairs.shape[:2]
        return self.network(pairs.flatten(0, 1)).reshape(batch, count, -1)

    def forward(self, maps: Tensor, dt: Tensor) -> Tensor:
        """maps [B, T, C, H, W], dt [B, I] -> motion features [B, I, D] per interval."""
        return self.encode_pairs(maps[:, :-1], maps[:, 1:], dt)


class IntervalTemporalMixer(nn.Module):
    """Residual 1-D convolutions along the interval axis.

    Each layer widens the temporal receptive field by two intervals. Comparing neighbouring intervals is what makes acceleration observable, and padding makes the network aware of where the window ends (the "current" interval).
    Input and output shape: [B, I, D].
    """
    def __init__(self, dim: int, layers: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(dim, dim, 3, padding=1, padding_mode="replicate"),
                nn.GroupNorm(group_count(dim), dim),
                nn.SiLU(),
            )
            for _ in range(layers)
        )

        
    def forward(self, motion: Tensor) -> Tensor:
        hidden = motion.transpose(1, 2)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return hidden.transpose(1, 2)
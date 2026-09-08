"""Lightweight structural stem for explicit semantic masks."""
from __future__ import annotations
import torch
import torch.nn as nn


class StructuralStem(nn.Module):
    """Three-level conv stem for one-hot semantic channels.

    Input: [B, 3, 512, 512] (outline, unshaded, target_region)
    Outputs:
        feat_512: [B, 32, 512, 512]  (stride 1)
        feat_128: [B, 96, 128, 128]  (stride 4)
    """

    def __init__(self, in_channels: int = 3, channels: list[int] | None = None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 96]
        c0, c1, c2 = channels

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, c0, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(32, c0), c0),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(c0, c1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, c1), c1),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, c2), c2),
            nn.GELU(),
        )
        self.out_channels_512 = c0
        self.out_channels_128 = c2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.layer0(x)   # [B, 32, 512, 512]
        f1 = self.layer1(f0)  # [B, 64, 256, 256]
        f2 = self.layer2(f1)  # [B, 96, 128, 128]
        return f0, f2

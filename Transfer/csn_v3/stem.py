"""High-resolution structural stem from line_mask."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + residual)


class HighResStructuralStem(nn.Module):
    """Trainable stem preserving 1px structures.

    Input: line_mask [B,1,512,512]
    Outputs:
        H0: [B,32,512,512] stride 1
        H1: [B,64,256,256] stride 2
        H2: [B,96,128,128] stride 4
    """

    def __init__(self, channels: list[int] | None = None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 96]
        c0, c1, c2 = channels

        self.stem_in = nn.Sequential(
            nn.Conv2d(1, c0, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(32, c0), c0),
            nn.GELU(),
        )
        self.block0 = ResidualBlock(c0)
        self.down1 = nn.Sequential(
            nn.Conv2d(c0, c1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, c1), c1),
            nn.GELU(),
        )
        self.block1 = ResidualBlock(c1)
        self.down2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, c2), c2),
            nn.GELU(),
        )
        self.block2 = ResidualBlock(c2)
        self.out_channels = (c0, c1, c2)

    def forward(self, line_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = self.block0(self.stem_in(line_mask))
        h1 = self.block1(self.down1(h0))
        h2 = self.block2(self.down2(h1))
        return h0, h1, h2

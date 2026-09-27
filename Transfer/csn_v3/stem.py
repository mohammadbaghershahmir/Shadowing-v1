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

    Input:
      - bw mode: line_mask [B,1,H,W]
      - indexed_guided: structural channels [B,3,H,W] for (X==0, X==1, X==2)
    Outputs:
        H0: [B,32,H,W] stride 1
        H1: [B,64,H/2,W/2] stride 2
        H2: [B,96,H/4,W/4] stride 4
    """

    def __init__(self, channels: list[int] | None = None, in_channels: int = 1):
        super().__init__()
        if channels is None:
            channels = [32, 64, 96]
        c0, c1, c2 = channels
        self.in_channels = in_channels

        self.stem_in = nn.Sequential(
            nn.Conv2d(in_channels, c0, 3, stride=1, padding=1, bias=False),
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

    def forward(self, structural: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if structural.shape[1] != self.in_channels:
            raise ValueError(
                f"StructuralStem expected {self.in_channels} channels, got {structural.shape[1]}"
            )
        h0 = self.block0(self.stem_in(structural))
        h1 = self.block1(self.down1(h0))
        h2 = self.block2(self.down2(h1))
        return h0, h1, h2

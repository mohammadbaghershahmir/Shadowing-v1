"""Trainable mask and transition conditioners."""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        return self.net(prob)


class TransitionEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        return self.net(prob)

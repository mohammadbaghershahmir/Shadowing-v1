"""Trainable adapters and coordinate embedding."""
from __future__ import annotations

import torch
import torch.nn as nn


class BranchAdapter(nn.Module):
    """Project branch features to fusion_dim."""

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        if in_dim == out_dim:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.GroupNorm(min(32, out_dim), out_dim),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenAdapter(nn.Module):
    """Project DINO token grid to fusion_dim."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.GroupNorm(min(32, out_dim), out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CropCoordEmbedding(nn.Module):
    """6 -> 128 -> 256 coordinate embedding for global cross-attention."""

    def __init__(self, in_dim: int = 6, hidden: int = 128, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class ZeroInitConv2d(nn.Module):
    """Conv2d with zero-initialized weights for residual paths."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualFusionBlock(nn.Module):
    """C + R(x) with zero-init final projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, dim), dim),
            nn.GELU(),
            ZeroInitConv2d(dim, dim, 3),
        )

    def forward(self, x: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        return lateral + self.refine(x)

"""Prediction heads for classification, transition, and auxiliary."""
from __future__ import annotations
import torch
import torch.nn as nn


class ClassHead(nn.Module):
    """1x1 conv producing class logits at stride 1."""

    def __init__(self, in_channels: int, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TransitionHead(nn.Module):
    """1x1 conv producing single-channel transition logits at stride 1."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AuxHead(nn.Module):
    """1x1 conv producing class logits at stride 4 (128x128)."""

    def __init__(self, in_channels: int, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

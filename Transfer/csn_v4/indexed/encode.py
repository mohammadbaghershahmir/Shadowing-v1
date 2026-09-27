"""Structural encodings from X only (never from Y)."""
from __future__ import annotations

import numpy as np
import torch

from csn_v4.indexed.schema import (
    INDEX_ALLOWED,
    INDEX_BACKGROUND,
    INDEX_OUTLINE,
    STRUCTURAL_RGB_FLOAT,
)

# Keep ImageNet stats local so this module does not import the full DINO stack.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def structural_channels(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Return [3,H,W] float masks for (X==0), (X==1), (X==2)."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.asarray(x, dtype=np.int64))
    else:
        t = x.long()
    ch0 = (t == INDEX_BACKGROUND).float()
    ch1 = (t == INDEX_OUTLINE).float()
    ch2 = (t == INDEX_ALLOWED).float()
    return torch.stack([ch0, ch1, ch2], dim=0)


def structural_channels_batch(x: torch.Tensor) -> torch.Tensor:
    """x: [B,H,W] or [B,1,H,W] int → [B,3,H,W] float."""
    if x.ndim == 4:
        x = x[:, 0]
    x = x.long()
    return torch.stack(
        [
            (x == INDEX_BACKGROUND).float(),
            (x == INDEX_OUTLINE).float(),
            (x == INDEX_ALLOWED).float(),
        ],
        dim=1,
    )


def indices_to_structural_rgb(x: np.ndarray | torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Fixed 3-channel RGB view of X for ImageNet-pretrained backbones.

    Built only from input indices {0,1,2}. Does not encode shade class from Y.
    Returns [3,H,W] float (normalized if requested).
    """
    if isinstance(x, torch.Tensor):
        idx = x.detach().cpu().numpy()
        if idx.ndim == 3 and idx.shape[0] == 1:
            idx = idx[0]
    else:
        idx = np.asarray(x)
    h, w = idx.shape[-2], idx.shape[-1]
    rgb = np.zeros((3, h, w), dtype=np.float32)
    for key, (r, g, b) in STRUCTURAL_RGB_FLOAT.items():
        m = idx == key
        rgb[0][m] = r
        rgb[1][m] = g
        rgb[2][m] = b
    t = torch.from_numpy(rgb)
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=t.dtype).view(3, 1, 1)
        t = (t - mean) / std
    return t


def indices_to_structural_rgb_batch(x: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """x [B,H,W] or [B,1,H,W] → [B,3,H,W]."""
    if x.ndim == 4:
        x = x[:, 0]
    parts = [indices_to_structural_rgb(x[i], normalize=normalize) for i in range(x.shape[0])]
    out = torch.stack(parts, dim=0).to(device=x.device)
    return out

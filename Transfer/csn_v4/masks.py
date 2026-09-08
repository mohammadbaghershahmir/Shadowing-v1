"""Mask utilities for CSN-V4."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def as_bool_mask(mask: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert bool / {0,1} / {0,255} masks to boolean.

    Materialized NPZ stores bool masks as uint8 0/1. Scene BMPs use 0/255.
    Comparing only with ``> 127`` incorrectly zeros out 0/1 masks.
    """
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            return mask
        if mask.is_floating_point():
            return mask > 0.5
        return mask > 0
    arr = np.asarray(mask)
    if arr.dtype == np.bool_ or arr.dtype == bool:
        return arr.astype(bool, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        return arr > 0.5
    return arr > 0


def build_categorical_boundary_mask(
    target_class: torch.Tensor,
    transition_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    black_lock: torch.Tensor,
    *,
    dilate: int = 2,
) -> torch.Tensor:
    """Union of transition mask and class boundaries, excluding invalid/black."""
    valid = as_bool_mask(valid_mask)
    black = as_bool_mask(black_lock)
    supervised = valid & ~black & (target_class >= 0)

    tc = target_class.clone()
    tc[~supervised] = -1
    oh = F.one_hot(tc.clamp(min=0), num_classes=4).permute(0, 3, 1, 2).float()
    edge_h = (oh[:, :, :, 1:] != oh[:, :, :, :-1]).any(dim=1)
    edge_v = (oh[:, :, 1:, :] != oh[:, :, :-1, :]).any(dim=1)
    boundary = torch.zeros_like(supervised)
    boundary[:, :, :-1] |= edge_h
    boundary[:, :-1, :] |= edge_v
    trans = as_bool_mask(transition_mask)
    boundary = boundary | trans
    boundary = boundary & supervised
    if dilate <= 0:
        return boundary
    x = boundary.float().unsqueeze(1)
    k = 2 * dilate + 1
    d = F.max_pool2d(x, k, stride=1, padding=dilate)
    return (d.squeeze(1) > 0.5) & supervised

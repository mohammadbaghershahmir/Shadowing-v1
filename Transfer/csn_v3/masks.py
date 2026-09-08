"""Target builders and mask utilities for CSN-V3."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from csn_v3.constants import (
    CLASS_DARK,
    CLASS_LIGHT,
    CLASS_MEDIUM,
    CLASS_WHITE,
    GRAY_BLACK,
    GRAY_DARK,
    GRAY_LIGHT,
    GRAY_MEDIUM,
    GRAY_WHITE,
    IGNORE_INDEX,
    SHADE_LEVEL_DARK,
    SHADE_LEVEL_IGNORE,
    SHADE_LEVEL_LIGHT,
    SHADE_LEVEL_MEDIUM,
    SHADE_LEVEL_TO_CLASS,
)


def semantic_gray_to_class(gray: torch.Tensor) -> torch.Tensor:
    """Map target_semantic gray values to 4-class labels with ignore."""
    out = torch.full_like(gray, IGNORE_INDEX, dtype=torch.long)
    out[gray == GRAY_WHITE] = CLASS_WHITE
    out[gray == GRAY_LIGHT] = CLASS_LIGHT
    out[gray == GRAY_MEDIUM] = CLASS_MEDIUM
    out[gray == GRAY_DARK] = CLASS_DARK
    out[gray == GRAY_BLACK] = IGNORE_INDEX
    return out


def build_valid_supervision_mask(
    valid_mask: torch.Tensor,
    black_lock: torch.Tensor,
) -> torch.Tensor:
    """Pixels valid for semantic supervision: valid and not immutable black."""
    return (valid_mask > 127) & (black_lock < 127)


def build_where_target(shade_mask: torch.Tensor) -> torch.Tensor:
    return (shade_mask > 127).float()


def build_where_loss_mask(valid_mask: torch.Tensor, black_lock: torch.Tensor) -> torch.Tensor:
    return (valid_mask > 127) & (black_lock < 127)


def build_level_target_from_semantic(target_class: torch.Tensor) -> torch.Tensor:
    """Map 4-class to 3-level: light=0, medium=1, dark=2, else ignore."""
    out = torch.full_like(target_class, IGNORE_INDEX)
    out[target_class == CLASS_LIGHT] = 0
    out[target_class == CLASS_MEDIUM] = 1
    out[target_class == CLASS_DARK] = 2
    return out


def build_ordinal_targets(level_target: torch.Tensor) -> torch.Tensor:
    """Two cumulative thresholds: medium-or-darker, dark."""
    B = level_target.shape[0] if level_target.ndim == 3 else 1
    if level_target.ndim == 2:
        level_target = level_target.unsqueeze(0)
    H, W = level_target.shape[-2:]
    out = level_target.new_zeros((level_target.shape[0], 2, H, W))
    valid = level_target >= 0
    out[:, 0][valid & (level_target >= 1)] = 1.0
    out[:, 1][valid & (level_target >= 2)] = 1.0
    return out


def build_center_supervision_mask(
    height: int,
    width: int,
    halo: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """True on central (H-2halo) x (W-2halo) region — matches tiled inference core."""
    mask = torch.zeros(height, width, dtype=torch.bool, device=device)
    if halo * 2 >= min(height, width):
        mask[:] = True
        return mask
    mask[halo : height - halo, halo : width - halo] = True
    return mask


def build_boundary_boost_mask(
    transition_mask: torch.Tensor,
    dilate: int = 2,
) -> torch.Tensor:
    if dilate <= 0:
        return transition_mask > 127
    x = transition_mask.float().unsqueeze(1)
    k = 2 * dilate + 1
    d = F.max_pool2d(x, k, stride=1, padding=dilate)
    return (d.squeeze(1) > 0.5)


def build_affinity_targets_from_level(
    shade_level_id: torch.Tensor,
    shade_mask: torch.Tensor,
    offsets: list[int],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build pairwise same-level targets for each offset direction."""
    pairs = []
    H, W = shade_level_id.shape[-2:]
    valid_shade = (shade_mask > 127) & (shade_level_id != SHADE_LEVEL_IGNORE)
    for delta in offsets:
        for dy, dx in [(0, delta), (delta, 0)]:
            py = slice(0, H - abs(dy)) if dy >= 0 else slice(abs(dy), H)
            px = slice(0, W - abs(dx)) if dx >= 0 else slice(abs(dx), W)
            qy = slice(abs(dy), H) if dy >= 0 else slice(0, H - abs(dy))
            qx = slice(abs(dx), W) if dx >= 0 else slice(0, W - abs(dx))
            lp = shade_level_id[..., py, px]
            lq = shade_level_id[..., qy, qx]
            vp = valid_shade[..., py, px]
            vq = valid_shade[..., qy, qx]
            both = vp & vq
            same = (lp == lq) & both
            target = same.float()
            pairs.append((dy, dx, target))
    return pairs

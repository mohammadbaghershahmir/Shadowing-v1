"""Target builders for CSN-V4 tile batches."""
from __future__ import annotations

import numpy as np
import torch

from csn_v4.constants import SHADE_SUBTYPE_IGNORE, SHADE_SUBTYPE_LIGHT, SHADE_SUBTYPE_MEDIUM, SHADE_SUBTYPE_DARK, AFFINITY_OFFSETS
from csn_v3.masks import build_level_target_from_semantic, semantic_gray_to_class


def load_gray_bmp(path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def build_affinity_channel_targets(level: np.ndarray, shade_mask: np.ndarray, offsets: list[int] | None = None) -> list[torch.Tensor]:
    if offsets is None:
        offsets = list(AFFINITY_OFFSETS)
    H, W = level.shape
    if shade_mask.dtype == np.bool_:
        shade = shade_mask
    else:
        shade = shade_mask > 127
    valid = shade & (level != SHADE_SUBTYPE_IGNORE)
    targets = []
    for delta in offsets:
        for dy, dx in [(0, delta), (delta, 0)]:
            t_arr = np.full((H, W), -1.0, dtype=np.float32)
            py = slice(0, H - abs(dy)) if dy >= 0 else slice(abs(dy), H)
            px = slice(0, W - abs(dx)) if dx >= 0 else slice(abs(dx), W)
            qy = slice(abs(dy), H) if dy >= 0 else slice(0, H - abs(dy))
            qx = slice(abs(dx), W) if dx >= 0 else slice(0, W - abs(dx))
            lp, lq = level[py, px], level[qy, qx]
            vp, vq = valid[py, px], valid[qy, qx]
            both = vp & vq
            same = (lp == lq) & both
            t_arr[qy, qx] = np.where(both, same.astype(np.float32), -1.0)
            targets.append(torch.from_numpy(t_arr))
    return targets


def build_affinity_targets_tensor(
    level: np.ndarray,
    shade_mask: np.ndarray,
    offsets: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = build_affinity_channel_targets(level, shade_mask, offsets)
    stacked = torch.stack(channels, dim=0)
    valid = (stacked > -0.5).float()
    return stacked, valid


def _level_array_for_affinity(
    local_level: np.ndarray | None,
    level_target: torch.Tensor,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Affinity needs 0/1/2 on shade pixels; fall back to semantic-derived levels."""
    if local_level is not None:
        return local_level.astype(np.int64)
    lt = level_target.numpy()
    out = np.full(shape, SHADE_SUBTYPE_IGNORE, dtype=np.int64)
    for subtype in (SHADE_SUBTYPE_LIGHT, SHADE_SUBTYPE_MEDIUM, SHADE_SUBTYPE_DARK):
        out[lt == subtype] = subtype
    return out


def build_tile_targets(
    local_semantic: np.ndarray,
    local_transition: np.ndarray | None,
    local_black: np.ndarray | None,
    local_valid: np.ndarray | None,
    local_shade: np.ndarray | None,
    local_level: np.ndarray | None,
    supervision_weights: np.ndarray,
) -> dict[str, torch.Tensor]:
    sem_t = torch.from_numpy(local_semantic.astype(np.int64))
    target_class = semantic_gray_to_class(sem_t)
    valid = torch.from_numpy((local_valid if local_valid is not None else np.ones_like(local_semantic) * 255) > 127)
    black = torch.from_numpy((local_black if local_black is not None else np.zeros_like(local_semantic)) > 127)
    if local_shade is not None:
        shade_np = local_shade > 127 if local_shade.dtype != np.bool_ else local_shade
    else:
        shade_np = (local_semantic > 0) & (local_semantic < 255)
    shade = torch.from_numpy(shade_np)
    level_np = local_level if local_level is not None else np.full_like(local_semantic, SHADE_SUBTYPE_IGNORE)
    level_target = build_level_target_from_semantic(target_class)

    level_for_aff = _level_array_for_affinity(local_level, level_target, local_semantic.shape)
    aff_targets, aff_valid = build_affinity_targets_tensor(level_for_aff, shade.numpy())

    return {
        "target_class": target_class,
        "shade_mask": shade,
        "transition_mask": torch.from_numpy((local_transition if local_transition is not None else np.zeros_like(local_semantic)).astype(np.uint8)),
        "black_lock": black,
        "valid_mask": valid,
        "where_target": shade.float(),
        "level_target": level_target,
        "affinity_targets": aff_targets,
        "affinity_valid": aff_valid,
        "supervision_weights": torch.from_numpy(supervision_weights.astype(np.float32)),
    }

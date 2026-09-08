"""Target tensor builders for CSN-V3 batches."""
from __future__ import annotations

import numpy as np
import torch

from csn_v3.constants import IGNORE_INDEX, SHADE_LEVEL_IGNORE
from csn_v3.masks import (
    build_level_target_from_semantic,
    build_ordinal_targets,
    build_valid_supervision_mask,
    build_where_target,
    semantic_gray_to_class,
)


def load_gray_bmp(path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def build_sample_targets(
    target_semantic: np.ndarray,
    shade_mask: np.ndarray,
    shade_level_id: np.ndarray,
    transition_mask: np.ndarray,
    black_lock: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    sem_t = torch.from_numpy(target_semantic.astype(np.int64))
    target_class = semantic_gray_to_class(sem_t)
    valid = torch.from_numpy(valid_mask > 127)
    black = torch.from_numpy(black_lock > 127)
    shade = torch.from_numpy(shade_mask > 127)
    level_target = build_level_target_from_semantic(target_class)
    ordinal = build_ordinal_targets(level_target.unsqueeze(0)).squeeze(0)

    # Affinity channel targets (8 dirs)
    aff_targets = _build_affinity_channel_targets(shade_level_id, shade_mask)

    return {
        "target_semantic": sem_t,
        "target_class": target_class,
        "shade_mask": shade,
        "shade_level_id": torch.from_numpy(shade_level_id.astype(np.int64)),
        "transition_mask": torch.from_numpy(transition_mask.astype(np.uint8)),
        "black_lock": black,
        "valid_mask": valid,
        "where_target": build_where_target(torch.from_numpy(shade_mask.astype(np.uint8))),
        "level_target": level_target,
        "ordinal_target": ordinal,
        "affinity_channel_targets": aff_targets,
    }


def _build_affinity_channel_targets(
    shade_level_id: np.ndarray,
    shade_mask: np.ndarray,
    offsets: list[int] | None = None,
) -> list[torch.Tensor]:
    if offsets is None:
        offsets = [1, 2, 4, 8]
    H, W = shade_level_id.shape
    level = shade_level_id.astype(np.int64)
    shade = shade_mask > 127
    valid = shade & (level != SHADE_LEVEL_IGNORE)
    targets = []
    for delta in offsets:
        for dy, dx in [(0, delta), (delta, 0)]:
            tgt = np.full((H, W), -1.0, dtype=np.float32)
            py = slice(0, H - abs(dy)) if dy >= 0 else slice(abs(dy), H)
            px = slice(0, W - abs(dx)) if dx >= 0 else slice(abs(dx), W)
            qy = slice(abs(dy), H) if dy >= 0 else slice(0, H - abs(dy))
            qx = slice(abs(dx), W) if dx >= 0 else slice(0, W - abs(dx))
            lp = level[py, px]
            lq = level[qy, qx]
            vp = valid[py, px]
            vq = valid[qy, qx]
            both = vp & vq
            same = (lp == lq) & both
            t = np.zeros((H, W), dtype=np.float32)
            t_slice = np.zeros((qy.stop - qy.start if isinstance(qy, slice) else 0,
                                qx.stop - qx.start if isinstance(qx, slice) else 0), dtype=np.float32)
            # assign to p positions
            sy = py
            sx = px
            t_arr = np.full((H, W), -1.0, dtype=np.float32)
            p_valid = np.zeros((H, W), dtype=bool)
            p_valid[sy, sx] = both
            t_arr[sy, sx] = same.astype(np.float32)
            targets.append(torch.from_numpy(t_arr))
    return targets

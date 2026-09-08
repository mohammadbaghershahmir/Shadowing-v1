"""Global semantic letterbox helper."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SEMANTIC_COLORS = {
    "outline": (0, 0, 0),
    "unshaded": (255, 255, 255),
    "shade_candidate": (255, 0, 255),
}


def letterbox_semantic_guide(
    outline_mask: np.ndarray,
    unshaded_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    target_size: int = 512,
) -> dict[str, Any]:
    """Resize semantic masks with nearest-neighbor and rebuild exact RGB guide."""
    masks = {
        "outline": outline_mask.astype(bool),
        "unshaded": unshaded_mask.astype(bool),
        "candidate": candidate_mask.astype(bool),
    }
    h, w = outline_mask.shape
    scale = min(target_size / h, target_size / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    resized: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        t = torch.from_numpy(mask.astype(np.float32))[None, None]
        rt = F.interpolate(t, size=(nh, nw), mode="nearest").squeeze().numpy().astype(bool)
        resized[name] = rt

    canvas = np.full((target_size, target_size, 3), SEMANTIC_COLORS["unshaded"], dtype=np.uint8)
    valid = np.zeros((target_size, target_size), dtype=bool)
    pad_top = (target_size - nh) // 2
    pad_left = (target_size - nw) // 2
    valid[pad_top : pad_top + nh, pad_left : pad_left + nw] = True

    sub = canvas[pad_top : pad_top + nh, pad_left : pad_left + nw]
    sub[:] = SEMANTIC_COLORS["unshaded"]
    sub[resized["outline"]] = SEMANTIC_COLORS["outline"]
    sub[resized["unshaded"]] = SEMANTIC_COLORS["unshaded"]
    sub[resized["candidate"]] = SEMANTIC_COLORS["shade_candidate"]

    return {
        "guide_rgb": canvas,
        "valid_mask": valid,
        "content_size": (nh, nw),
        "pad_box": (pad_top, pad_left, target_size - nh - pad_top, target_size - nw - pad_left),
        "scale": scale,
    }

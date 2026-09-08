"""Input encoding for BW vs magenta-target modes."""
from __future__ import annotations

import numpy as np
import torch

from shadow_dataset.constants import (
    ONEHOT_CHANNEL_OUTLINE,
    ONEHOT_CHANNEL_TARGET_REGION,
    ONEHOT_CHANNEL_UNSHADED,
    OUTLINE,
    TARGET_REGION,
    UNSHADED,
)
from shadow_dataset.dataset import build_input_onehot


def rgb_uint8_to_onehot(rgb: np.ndarray) -> np.ndarray:
    """Return [3,H,W] float one-hot from RGB uint8 semantic guide."""
    return build_input_onehot(rgb)


def onehot_to_line_mask(onehot: torch.Tensor) -> torch.Tensor:
    """Outline channel as line mask [B,1,H,W]."""
    if onehot.ndim == 3:
        onehot = onehot.unsqueeze(0)
    return onehot[:, ONEHOT_CHANNEL_OUTLINE : ONEHOT_CHANNEL_OUTLINE + 1].float()


def build_where_target_magenta(onehot: torch.Tensor) -> torch.Tensor:
    if onehot.ndim == 3:
        ch = onehot[ONEHOT_CHANNEL_TARGET_REGION]
    else:
        ch = onehot[:, ONEHOT_CHANNEL_TARGET_REGION]
    return ch.float()


def load_rgb_guide(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def match_magenta_guide(sample_id: str, guide_dir: str) -> str | None:
    from pathlib import Path
    root = Path(guide_dir)
    for pat in (f"{sample_id}.bmp", f"{sample_id}_*.bmp", "*.bmp"):
        hits = list(root.glob(pat))
        if hits:
            return str(hits[0])
    return None

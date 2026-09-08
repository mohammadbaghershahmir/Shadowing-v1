"""Deterministic black lock and BMP rendering."""
from __future__ import annotations

import numpy as np
import torch

from csn_v3.constants import ALLOWED_OUTPUT_GRAY, CLASS_TO_GRAY
from dataset_v2.bmp_io import save_gray_bmp


def class_to_gray_tensor(pred_class: torch.Tensor) -> torch.Tensor:
    """Map class indices to grayscale BMP values."""
    out = torch.zeros_like(pred_class, dtype=torch.uint8)
    for cls, gray in CLASS_TO_GRAY.items():
        out[pred_class == cls] = gray
    return out


def apply_black_lock(
    gray: torch.Tensor | np.ndarray,
    source_bw: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Force original black pixels to 0."""
    if isinstance(gray, np.ndarray):
        out = gray.copy()
        if source_bw.ndim == 3:
            src = source_bw[..., 0]
        else:
            src = source_bw
        out[src == 0] = 0
        return out
    out = gray.clone()
    src = source_bw if source_bw.ndim == 2 else source_bw[:, 0]
    out[src < 0.5] = 0
    return out


def render_bmp_from_logits(
    refined_logits: torch.Tensor,
    source_bw: torch.Tensor,
    hard_black_lock: bool = True,
) -> np.ndarray:
    """Argmax refined logits, map to gray, apply black lock."""
    pred = refined_logits.argmax(dim=1).squeeze(0).cpu()
    gray = class_to_gray_tensor(pred).numpy()
    if source_bw.ndim == 3:
        src = (source_bw.squeeze(0)[0].cpu().numpy() * 255).astype(np.uint8)
    else:
        src = (source_bw.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    if hard_black_lock:
        gray = apply_black_lock(gray, src)
    return gray.astype(np.uint8)


def validate_palette(gray: np.ndarray) -> tuple[bool, set[int]]:
    uniq = set(int(v) for v in np.unique(gray))
    valid = uniq.issubset(ALLOWED_OUTPUT_GRAY)
    return valid, uniq


def save_output_bmp(path: str, gray: np.ndarray) -> None:
    ok, uniq = validate_palette(gray)
    if not ok:
        raise ValueError(f"Invalid palette values: {uniq - ALLOWED_OUTPUT_GRAY}")
    save_gray_bmp(path, gray)

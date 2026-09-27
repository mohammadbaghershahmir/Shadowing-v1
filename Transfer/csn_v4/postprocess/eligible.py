"""Eligible-region detection from source input + user masks (never from GT)."""
from __future__ import annotations

import numpy as np

from csn_v4.constants import GRAY_BLACK, GRAY_WHITE
from csn_v4.indexed.contract import allowed_mask
from csn_v4.indexed.schema import INDEX_OUTLINE


def locked_outline_mask(source_bw: np.ndarray) -> np.ndarray:
    """Original black pixels that must never be edited."""
    src = np.asarray(source_bw)
    if src.ndim == 3:
        src = src[..., 0]
    return src == GRAY_BLACK


def find_eligible_mask(
    source_bw: np.ndarray,
    *,
    editable_mask: np.ndarray | None = None,
    locked_mask: np.ndarray | None = None,
    source_index: np.ndarray | None = None,
    require_explicit_editable: bool = False,
) -> tuple[np.ndarray, dict]:
    """Identify pixels where shading may be changed.

    Priority:
      1. ``source_index`` → eligible = (X == INDEX_ALLOWED), lock outline/background
      2. ``editable_mask`` (user-supplied)
      3. Else: non-black pixels only when ``require_explicit_editable`` is False
         (used by conservative cleanup). Style profiles that need a true motif
         mask set ``require_explicit_editable=True`` and get an empty eligible
         region with a clear reason when no mask is provided.

    Never consults ground-truth targets.
    """
    src = np.asarray(source_bw)
    if src.ndim == 3:
        src = src[..., 0]
    h, w = src.shape[:2]
    outline = locked_outline_mask(src)
    info: dict = {
        "source": "none",
        "require_explicit_editable": bool(require_explicit_editable),
        "reason": "",
    }

    if source_index is not None:
        idx = np.asarray(source_index)
        if idx.shape[:2] != (h, w):
            raise ValueError(f"source_index shape {idx.shape} != source {(h, w)}")
        eligible = allowed_mask(idx)
        eligible &= ~outline
        eligible &= idx != INDEX_OUTLINE
        info["source"] = "source_index"
    elif editable_mask is not None:
        em = np.asarray(editable_mask)
        if em.shape[:2] != (h, w):
            raise ValueError(f"editable_mask shape {em.shape} != source {(h, w)}")
        eligible = em.astype(bool) & ~outline
        info["source"] = "editable_mask"
    elif require_explicit_editable:
        eligible = np.zeros((h, w), dtype=bool)
        info["source"] = "missing_required_mask"
        info["reason"] = (
            "Style-specific edits require an editable mask or source index; "
            "BW input alone does not define an unambiguous editable region."
        )
    else:
        # Conservative cleanup only: all non-outline pixels are candidates.
        eligible = ~outline
        info["source"] = "non_outline_fallback"
        info["reason"] = (
            "No editable mask supplied; using non-outline pixels for "
            "conservative cleanup only (not for decorative style repair)."
        )

    if locked_mask is not None:
        lm = np.asarray(locked_mask).astype(bool)
        if lm.shape[:2] != (h, w):
            raise ValueError(f"locked_mask shape {lm.shape} != source {(h, w)}")
        eligible &= ~lm
        info["locked_pixels"] = int(lm.sum())

    # Always exclude original blacks.
    eligible &= ~outline
    info["eligible_pixels"] = int(eligible.sum())
    info["outline_pixels"] = int(outline.sum())
    return eligible, info

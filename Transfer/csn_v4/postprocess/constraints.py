"""Final constraint application and palette validation."""
from __future__ import annotations

import numpy as np

from csn_v4.constants import ALLOWED_OUTPUT_GRAY, GRAY_BLACK, SEMANTIC_GRAY_TO_CLASS, CLASS_TO_GRAY
from csn_v4.postprocess.eligible import locked_outline_mask


def gray_to_class(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.uint8)
    out = np.zeros(g.shape, dtype=np.int64)
    for gray_v, cls in SEMANTIC_GRAY_TO_CLASS.items():
        out[g == gray_v] = cls
    # Any unexpected non-black values → white class (safe)
    unknown = ~np.isin(g, list(ALLOWED_OUTPUT_GRAY))
    out[unknown] = 0
    out[g == GRAY_BLACK] = 0
    return out


def class_to_gray(cls: np.ndarray) -> np.ndarray:
    c = np.asarray(cls, dtype=np.int64)
    out = np.full(c.shape, 255, dtype=np.uint8)
    for ci, gv in CLASS_TO_GRAY.items():
        out[c == ci] = gv
    return out


def apply_final_constraints(
    repaired_gray: np.ndarray,
    *,
    source_bw: np.ndarray,
    raw_gray: np.ndarray,
    editable_mask: np.ndarray | None = None,
    locked_mask: np.ndarray | None = None,
    eligible_used: np.ndarray | None = None,
) -> np.ndarray:
    """Restore blacks/locked/outside-eligible; enforce palette and shape."""
    out = np.asarray(repaired_gray, dtype=np.uint8).copy()
    raw = np.asarray(raw_gray, dtype=np.uint8)
    src = np.asarray(source_bw)
    if src.ndim == 3:
        src = src[..., 0]
    if out.shape != raw.shape or out.shape != src.shape[:2]:
        raise ValueError(
            f"Shape mismatch repaired={out.shape} raw={raw.shape} source={src.shape}"
        )

    outline = locked_outline_mask(src)
    out[outline] = GRAY_BLACK

    if locked_mask is not None:
        lm = np.asarray(locked_mask).astype(bool)
        out[lm] = raw[lm]

    # Outside editable / eligible → restore raw prediction bit-exact
    if eligible_used is not None:
        outside = ~np.asarray(eligible_used, dtype=bool)
        out[outside] = raw[outside]
    elif editable_mask is not None:
        outside = ~np.asarray(editable_mask, dtype=bool)
        out[outside] = raw[outside]

    # Force outline again after restores
    out[outline] = GRAY_BLACK

    # Snap any illegal values back to raw (never invent palette values)
    illegal = ~np.isin(out, list(ALLOWED_OUTPUT_GRAY))
    if illegal.any():
        out[illegal] = raw[illegal]
        out[outline] = GRAY_BLACK

    return out


def validate_result_palette(gray: np.ndarray, source_bw: np.ndarray) -> tuple[bool, set[int]]:
    g = np.asarray(gray, dtype=np.uint8)
    src = np.asarray(source_bw)
    if src.ndim == 3:
        src = src[..., 0]
    if g.shape != src.shape[:2]:
        return False, set(int(x) for x in np.unique(g))
    uniq = set(int(x) for x in np.unique(g))
    ok = uniq.issubset(ALLOWED_OUTPUT_GRAY) and bool(np.all(g[src == GRAY_BLACK] == GRAY_BLACK))
    return ok, uniq

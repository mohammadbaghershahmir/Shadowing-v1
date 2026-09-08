"""Exact symmetric 4-connected region boundaries."""

from __future__ import annotations

import numpy as np


def compute_region_boundary_mask(contiguous_used_index_map: np.ndarray) -> np.ndarray:
    cont = contiguous_used_index_map.astype(np.int32)
    h, w = cont.shape
    boundary = np.zeros((h, w), dtype=np.uint8)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = np.roll(np.roll(cont, dy, 0), dx, 1)
        diff = (cont != shifted) & (cont >= 0) & (shifted >= 0)
        boundary = np.logical_or(boundary, diff)
    return (boundary.astype(np.uint8) * 255)


def compute_boundary_pair_map(contiguous_used_index_map: np.ndarray, boundary_mask: np.ndarray) -> dict[str, np.ndarray]:
    cont = contiguous_used_index_map.astype(np.int32)
    h, w = cont.shape
    class_a = np.full((h, w), -1, dtype=np.int16)
    class_b = np.full((h, w), -1, dtype=np.int16)
    b = boundary_mask > 0
    for dy, dx in ((0, 1), (1, 0)):
        shifted = np.roll(np.roll(cont, dy, 0), dx, 1)
        edge = b & (shifted != cont)
        pair_a = np.minimum(cont, shifted)
        pair_b = np.maximum(cont, shifted)
        class_a = np.where(edge, pair_a, class_a)
        class_b = np.where(edge, pair_b, class_b)
    return {"class_a": class_a, "class_b": class_b}

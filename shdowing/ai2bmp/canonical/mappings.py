"""Reversible contiguous index and unique-RGB mappings."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_contiguous_used_index_mapping(
    used_original_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    sorted_used = np.sort(used_original_indices.astype(np.int64))
    original_to_contiguous = {int(o): int(i) for i, o in enumerate(sorted_used.tolist())}
    contiguous_to_original = {int(v): int(k) for k, v in original_to_contiguous.items()}
    return sorted_used, {
        "original_to_contiguous": {str(k): v for k, v in original_to_contiguous.items()},
        "contiguous_to_original": {str(k): v for k, v in contiguous_to_original.items()},
        "k_index": int(len(sorted_used)),
    }


def apply_contiguous_used_index_map(
    index_map: np.ndarray, mapping: dict[str, Any]
) -> np.ndarray:
    lut = np.full(int(index_map.max()) + 1, -1, dtype=np.int32)
    for orig, cont in mapping["original_to_contiguous"].items():
        lut[int(orig)] = int(cont)
    out = lut[index_map.astype(np.int64)]
    if np.any(out < 0):
        raise ValueError("Index map contains value outside used-index set")
    return out.astype(np.int32)


def build_unique_rgb_mapping(
    used_original_indices: np.ndarray, palette: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    rgb_to_id: dict[tuple[int, int, int], int] = {}
    original_to_unique: dict[int, int] = {}
    unique_id_to_rgb: dict[int, list[int]] = {}
    unique_id_to_original_indices: dict[int, list[int]] = {}

    next_id = 0
    for idx in np.sort(used_original_indices).tolist():
        rgb = tuple(int(x) for x in palette[int(idx)])
        if rgb not in rgb_to_id:
            rgb_to_id[rgb] = next_id
            unique_id_to_rgb[next_id] = list(rgb)
            unique_id_to_original_indices[next_id] = [int(idx)]
            next_id += 1
        else:
            uid = rgb_to_id[rgb]
            unique_id_to_original_indices[uid].append(int(idx))
        original_to_unique[int(idx)] = rgb_to_id[rgb]

    mapping = {
        "original_to_unique_rgb_id": {str(k): v for k, v in original_to_unique.items()},
        "unique_rgb_id_to_rgb": {str(k): v for k, v in unique_id_to_rgb.items()},
        "unique_rgb_id_to_original_indices": {
            str(k): v for k, v in unique_id_to_original_indices.items()
        },
        "k_rgb": int(len(unique_id_to_rgb)),
    }
    return mapping, np.array(list(original_to_unique.keys()), dtype=np.int64)


def apply_unique_rgb_map(index_map: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    lut = np.full(int(index_map.max()) + 1, -1, dtype=np.int32)
    for orig, uid in mapping["original_to_unique_rgb_id"].items():
        lut[int(orig)] = int(uid)
    out = lut[index_map.astype(np.int64)]
    if np.any(out < 0):
        raise ValueError("Index map contains value outside used-index set")
    return out.astype(np.int32)

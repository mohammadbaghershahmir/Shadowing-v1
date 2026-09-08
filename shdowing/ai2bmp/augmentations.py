"""Exact rotation and flip augmentations."""

from __future__ import annotations

from typing import Any

import numpy as np


AUGMENTATIONS = [
    {"name": "rot0", "op": "identity"},
    {"name": "rot90", "op": "rot90", "k": 1},
    {"name": "rot180", "op": "rot90", "k": 2},
    {"name": "rot270", "op": "rot90", "k": 3},
    {"name": "flip_h", "op": "flip", "axis": 1},
    {"name": "flip_v", "op": "flip", "axis": 0},
]


def apply_augmentation(arr: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    op = spec["op"]
    if op == "identity":
        return arr.copy()
    if op == "rot90":
        return np.rot90(arr, k=int(spec["k"]))
    if op == "flip":
        return np.flip(arr, axis=int(spec["axis"]))
    raise ValueError(f"Unknown augmentation {spec}")


def inverse_augmentation(arr: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    op = spec["op"]
    if op == "identity":
        return arr.copy()
    if op == "rot90":
        return np.rot90(arr, k=(4 - int(spec["k"])) % 4)
    if op == "flip":
        return np.flip(arr, axis=int(spec["axis"]))
    raise ValueError(f"Unknown augmentation {spec}")


def verify_inverse(arr: np.ndarray, spec: dict[str, Any]) -> bool:
    back = inverse_augmentation(apply_augmentation(arr, spec), spec)
    return bool(np.array_equal(arr, back))

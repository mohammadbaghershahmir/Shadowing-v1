"""Exact D4 transforms — standard rot(k%4) + optional horizontal flip (8 unique)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class D4Transform:
    k: int

    def __post_init__(self) -> None:
        if not 0 <= self.k <= 7:
            raise ValueError(f"D4 k must be 0..7, got {self.k}")

    @property
    def rotation(self) -> int:
        return self.k % 4

    @property
    def reflected(self) -> bool:
        return self.k >= 4

    @property
    def inverse(self) -> D4Transform:
        if self.reflected:
            return D4Transform(self.k)
        inv_r = (4 - self.rotation) % 4
        return D4Transform(inv_r)

    def apply(self, arr: np.ndarray) -> np.ndarray:
        out = np.rot90(arr, k=self.rotation)
        if self.reflected:
            out = np.flip(out, axis=1)
        return out.copy()

    def apply_pair(self, bw: np.ndarray, semantic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.apply(bw), self.apply(semantic)

    def transformed_wh(self, fw: int, fh: int) -> tuple[int, int]:
        tbw = self.apply(np.zeros((fh, fw), dtype=np.uint8))
        h, w = tbw.shape[:2]
        return w, h


def apply_d4_pair(bw: np.ndarray, semantic: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return D4Transform(k).apply_pair(bw, semantic)


def inverse_apply(arr: np.ndarray, k: int) -> np.ndarray:
    return D4Transform(k).inverse.apply(arr)

"""Versioned adapters: legacy gray BMPs ↔ indexed maps. Never mix gray↔index silently."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from csn_v4.indexed.schema import (
    INDEX_ALLOWED,
    INDEX_BACKGROUND,
    INDEX_OUTLINE,
    INDEX_SHADOW_1,
    INDEX_SHADOW_2,
)


class GrayAdapter(Protocol):
    version: str

    def gray_to_index(self, gray: np.ndarray) -> np.ndarray: ...

    def index_to_gray(self, indices: np.ndarray) -> np.ndarray: ...


class GrayLegacyAdapterV1:
    """Map historical 5-tone gray carpets to indexed schema.

    Explicit mapping (documented, not inferred from brightness):
      gray 0   → outline (1)
      gray 255 → background (0) when used as full-scene canvas white
      gray 200 → allowed / base (2)
      gray 150 → shadow_1 (3)
      gray 100 → shadow_2 (4)

    Note: older pipelines sometimes treated white as "unshaded fill" inside the
    design. Prefer storing native index maps; use this adapter only for legacy
    gray BMPs. Round-trip of non-mapped gray values raises.
    """

    version = "gray_legacy.v1"

    GRAY_TO_INDEX = {
        0: INDEX_OUTLINE,
        255: INDEX_BACKGROUND,
        200: INDEX_ALLOWED,
        150: INDEX_SHADOW_1,
        100: INDEX_SHADOW_2,
    }
    INDEX_TO_GRAY = {v: k for k, v in GRAY_TO_INDEX.items()}

    def gray_to_index(self, gray: np.ndarray) -> np.ndarray:
        g = np.asarray(gray)
        if g.ndim == 3:
            # Take first channel if RGB accidentally passed
            g = g[..., 0]
        out = np.empty(g.shape, dtype=np.int16)
        unknown = np.ones(g.shape, dtype=bool)
        for gray_v, idx in self.GRAY_TO_INDEX.items():
            m = g == gray_v
            out[m] = idx
            unknown &= ~m
        if unknown.any():
            vals = np.unique(g[unknown])
            raise ValueError(
                f"{self.version}: unsupported gray value(s) {vals.tolist()}; "
                f"allowed grays={sorted(self.GRAY_TO_INDEX)}"
            )
        return out.astype(np.int64)

    def index_to_gray(self, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        out = np.empty(idx.shape, dtype=np.uint8)
        unknown = np.ones(idx.shape, dtype=bool)
        for i, gray_v in self.INDEX_TO_GRAY.items():
            m = idx == i
            out[m] = gray_v
            unknown &= ~m
        if unknown.any():
            vals = np.unique(idx[unknown])
            raise ValueError(
                f"{self.version}: unsupported index value(s) {vals.tolist()}; "
                f"allowed indices={sorted(self.INDEX_TO_GRAY)}"
            )
        return out


_ADAPTERS: dict[str, GrayAdapter] = {
    GrayLegacyAdapterV1.version: GrayLegacyAdapterV1(),
}


def get_gray_adapter(version: str = GrayLegacyAdapterV1.version) -> GrayAdapter:
    if version not in _ADAPTERS:
        raise KeyError(f"Unknown gray adapter {version!r}; known={sorted(_ADAPTERS)}")
    return _ADAPTERS[version]

"""Canonical index schema. Indices are integers — not gray levels or RGB."""
from __future__ import annotations

from typing import Final

INDEXED_SCHEMA_VERSION: Final[str] = "indexed_guided.v1"

# Semantic indices (storage / model contract)
INDEX_BACKGROUND: Final[int] = 0
INDEX_OUTLINE: Final[int] = 1
INDEX_ALLOWED: Final[int] = 2  # base fill inside shadeable region
INDEX_SHADOW_1: Final[int] = 3
INDEX_SHADOW_2: Final[int] = 4

INDEXED_INPUT_INDICES: Final[frozenset[int]] = frozenset(
    {INDEX_BACKGROUND, INDEX_OUTLINE, INDEX_ALLOWED}
)
INDEXED_OUTPUT_INDICES: Final[frozenset[int]] = frozenset(
    {
        INDEX_BACKGROUND,
        INDEX_OUTLINE,
        INDEX_ALLOWED,
        INDEX_SHADOW_1,
        INDEX_SHADOW_2,
    }
)

# Model predicts 3 classes inside M=(X==2), mapped back to 2/3/4
INTERNAL_BASE: Final[int] = 0
INTERNAL_SHADOW_1: Final[int] = 1
INTERNAL_SHADOW_2: Final[int] = 2
INTERNAL_NUM_CLASSES: Final[int] = 3

INTERNAL_TO_INDEX: Final[dict[int, int]] = {
    INTERNAL_BASE: INDEX_ALLOWED,
    INTERNAL_SHADOW_1: INDEX_SHADOW_1,
    INTERNAL_SHADOW_2: INDEX_SHADOW_2,
}
INDEX_TO_INTERNAL: Final[dict[int, int]] = {
    INDEX_ALLOWED: INTERNAL_BASE,
    INDEX_SHADOW_1: INTERNAL_SHADOW_1,
    INDEX_SHADOW_2: INTERNAL_SHADOW_2,
}

# Display-only palettes (never confuse with index IDs)
DISPLAY_GRAY_V1: Final[dict[int, int]] = {
    INDEX_BACKGROUND: 255,
    INDEX_OUTLINE: 0,
    INDEX_ALLOWED: 200,
    INDEX_SHADOW_1: 150,
    INDEX_SHADOW_2: 100,
}
DISPLAY_RGB_V1: Final[dict[int, tuple[int, int, int]]] = {
    INDEX_BACKGROUND: (245, 245, 245),
    INDEX_OUTLINE: (0, 0, 0),
    INDEX_ALLOWED: (220, 180, 120),
    INDEX_SHADOW_1: (140, 100, 60),
    INDEX_SHADOW_2: (70, 45, 25),
}

# Fixed structural RGB for ImageNet-pretrained backbones (from X only)
STRUCTURAL_RGB_FLOAT: Final[dict[int, tuple[float, float, float]]] = {
    INDEX_BACKGROUND: (0.55, 0.55, 0.55),
    INDEX_OUTLINE: (0.0, 0.0, 0.0),
    INDEX_ALLOWED: (1.0, 1.0, 1.0),
}

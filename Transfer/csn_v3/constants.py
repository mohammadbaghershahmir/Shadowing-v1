"""CSN-V3 semantic constants and palette mapping."""
from __future__ import annotations

from typing import Final

MODEL_NAME: Final[str] = "CarpetShadeNetV3"
SHORT_NAME: Final[str] = "CSN-V3"

# Neural class indices (black is never predicted)
CLASS_WHITE: Final[int] = 0
CLASS_LIGHT: Final[int] = 1
CLASS_MEDIUM: Final[int] = 2
CLASS_DARK: Final[int] = 3
NUM_CLASSES: Final[int] = 4
NUM_LEVEL_CLASSES: Final[int] = 3
NUM_ORDINAL_THRESHOLDS: Final[int] = 2

IGNORE_INDEX: Final[int] = -100

# BMP output grayscale values
GRAY_BLACK: Final[int] = 0
GRAY_WHITE: Final[int] = 255
GRAY_LIGHT: Final[int] = 200
GRAY_MEDIUM: Final[int] = 150
GRAY_DARK: Final[int] = 100

ALLOWED_OUTPUT_GRAY: Final[frozenset[int]] = frozenset(
    {GRAY_BLACK, GRAY_WHITE, GRAY_LIGHT, GRAY_MEDIUM, GRAY_DARK}
)

CLASS_TO_GRAY: Final[dict[int, int]] = {
    CLASS_WHITE: GRAY_WHITE,
    CLASS_LIGHT: GRAY_LIGHT,
    CLASS_MEDIUM: GRAY_MEDIUM,
    CLASS_DARK: GRAY_DARK,
}

SEMANTIC_GRAY_TO_CLASS: Final[dict[int, int]] = {
    GRAY_WHITE: CLASS_WHITE,
    GRAY_LIGHT: CLASS_LIGHT,
    GRAY_MEDIUM: CLASS_MEDIUM,
    GRAY_DARK: CLASS_DARK,
}

# shade_level_id.bmp values
SHADE_LEVEL_LIGHT: Final[int] = 0
SHADE_LEVEL_MEDIUM: Final[int] = 1
SHADE_LEVEL_DARK: Final[int] = 2
SHADE_LEVEL_IGNORE: Final[int] = 255

SHADE_LEVEL_TO_CLASS: Final[dict[int, int]] = {
    SHADE_LEVEL_LIGHT: CLASS_LIGHT,
    SHADE_LEVEL_MEDIUM: CLASS_MEDIUM,
    SHADE_LEVEL_DARK: CLASS_DARK,
}

# Affinity offsets: horizontal/vertical at distances 1,2,4,8
AFFINITY_OFFSETS: Final[tuple[int, ...]] = (1, 2, 4, 8)
AFFINITY_NUM_CHANNELS: Final[int] = len(AFFINITY_OFFSETS) * 2  # 8

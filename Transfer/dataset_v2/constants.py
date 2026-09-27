"""Dataset V2 pixel contract and version identifiers."""
from __future__ import annotations

from typing import Final

DATASET_NAME: Final[str] = "Dataset V2"
DATASET_VERSION: Final[str] = "V2"

BLACK: Final[tuple[int, int, int]] = (0, 0, 0)
WHITE: Final[tuple[int, int, int]] = (255, 255, 255)
SHADE_LIGHT: Final[tuple[int, int, int]] = (200, 200, 200)
SHADE_MEDIUM: Final[tuple[int, int, int]] = (150, 150, 150)
SHADE_DARK: Final[tuple[int, int, int]] = (100, 100, 100)

EXPECTED_RGB: Final[frozenset[tuple[int, int, int]]] = frozenset(
    {BLACK, WHITE, SHADE_LIGHT, SHADE_MEDIUM, SHADE_DARK}
)

ORACLE_MAGENTA: Final[tuple[int, int, int]] = (255, 0, 255)

SHADE_LEVEL_LIGHT: Final[int] = 0
SHADE_LEVEL_MEDIUM: Final[int] = 1
SHADE_LEVEL_DARK: Final[int] = 2
SHADE_LEVEL_IGNORE: Final[int] = 255

RGB_TO_SEMANTIC_GRAY: Final[dict[tuple[int, int, int], int]] = {
    BLACK: 0,
    WHITE: 255,
    SHADE_LIGHT: 200,
    SHADE_MEDIUM: 150,
    SHADE_DARK: 100,
}

RGB_TO_SHADE_LEVEL: Final[dict[tuple[int, int, int], int]] = {
    SHADE_LIGHT: SHADE_LEVEL_LIGHT,
    SHADE_MEDIUM: SHADE_LEVEL_MEDIUM,
    SHADE_DARK: SHADE_LEVEL_DARK,
}

SHADE_LEVEL_TO_GRAY: Final[dict[int, int]] = {
    SHADE_LEVEL_LIGHT: 200,
    SHADE_LEVEL_MEDIUM: 150,
    SHADE_LEVEL_DARK: 100,
    SHADE_LEVEL_IGNORE: 255,
}

CROP_CATEGORIES: Final[tuple[str, ...]] = (
    "transition_rich",
    "shade_rich",
    "dark_rare",
    "negative",
    "mixed_random",
)

FULL_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "target_semantic.bmp",
    "source_bw.bmp",
    "source_oracle.bmp",
    "shade_mask.bmp",
    "shade_level_id.bmp",
    "shade_level_preview.bmp",
    "transition_mask.bmp",
    "black_lock.bmp",
    "valid_mask.bmp",
    "reconstructed_target.bmp",
    "roundtrip_diff.bmp",
    "contact_sheet.bmp",
)

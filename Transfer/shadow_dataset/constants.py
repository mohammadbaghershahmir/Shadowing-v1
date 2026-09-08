"""Pixel contracts, categorical label encodings, and pipeline defaults.

Target RGB values INDEX_200 / INDEX_150 / INDEX_100 are fixed categorical
colour identifiers. Their numeric magnitudes have no ordinal, lighting,
severity, geometric, or distance meaning.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[int] = 1

# Supported lossless raster extensions (lowercase, with leading dot).
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".png", ".bmp", ".tif", ".tiff"}
)

# ---------------------------------------------------------------------------
# Input guide colours (RGB uint8)
# ---------------------------------------------------------------------------
OUTLINE: Final[tuple[int, int, int]] = (0, 0, 0)
UNSHADED: Final[tuple[int, int, int]] = (255, 255, 255)
TARGET_REGION: Final[tuple[int, int, int]] = (255, 0, 255)

INPUT_ALLOWED_COLORS: Final[frozenset[tuple[int, int, int]]] = frozenset(
    {OUTLINE, UNSHADED, TARGET_REGION}
)

# One-hot channel order for local_input_onehot:
#   channel 0 = outline
#   channel 1 = unshaded
#   channel 2 = target region (magenta)
ONEHOT_CHANNEL_OUTLINE: Final[int] = 0
ONEHOT_CHANNEL_UNSHADED: Final[int] = 1
ONEHOT_CHANNEL_TARGET_REGION: Final[int] = 2
ONEHOT_NUM_CHANNELS: Final[int] = 3

# ---------------------------------------------------------------------------
# Ground-truth categorical colour indices (RGB uint8)
# These are immutable class identifiers, not intensities or ordinal ranks.
# ---------------------------------------------------------------------------
INDEX_200: Final[tuple[int, int, int]] = (200, 200, 200)
INDEX_150: Final[tuple[int, int, int]] = (150, 150, 150)
INDEX_100: Final[tuple[int, int, int]] = (100, 100, 100)

TARGET_ALLOWED_COLORS: Final[frozenset[tuple[int, int, int]]] = frozenset(
    {OUTLINE, UNSHADED, INDEX_200, INDEX_150, INDEX_100}
)

# Canonical class order is by class id (0, 1, 2), never by RGB magnitude.
CLASS_IDS: Final[tuple[int, int, int]] = (0, 1, 2)

# class id -> RGB identifier (immutable mapping stored in dataset_config.json)
CLASS_TO_RGB: Final[dict[int, tuple[int, int, int]]] = {
    0: INDEX_200,
    1: INDEX_150,
    2: INDEX_100,
}

# RGB grey value -> class id (inside TARGET_REGION mask only).
RGB_VALUE_TO_CLASS: Final[dict[int, int]] = {
    200: 0,  # INDEX_200
    150: 1,  # INDEX_150
    100: 2,  # INDEX_100
}

# String keys used in manifests (canonical class-id order: 0, 1, 2).
CLASS_COUNT_KEYS: Final[tuple[str, str, str]] = ("200", "150", "100")

IGNORE_INDEX: Final[int] = -100

# ---------------------------------------------------------------------------
# Crop / split defaults
# ---------------------------------------------------------------------------
DEFAULT_CROP_SIZE: Final[int] = 512
DEFAULT_VALID_MARGIN: Final[int] = 64
DEFAULT_INFERENCE_STRIDE_HINT: Final[int] = 384

DEFAULT_TRAIN_RATIO: Final[float] = 0.8
DEFAULT_VAL_RATIO: Final[float] = 0.1
DEFAULT_TEST_RATIO: Final[float] = 0.1
DEFAULT_SEED: Final[int] = 42

DEFAULT_MIN_CROPS_PER_IMAGE: Final[int] = 8
DEFAULT_MAX_CROPS_PER_IMAGE: Final[int] = 64
DEFAULT_GRID_STRIDE: Final[int] = 256
DEFAULT_TARGET_TRAIN_CROPS: Final[int] = 10000
DEFAULT_EDGE_BAND: Final[int] = 64

# Sampling strategy mix (must sum to 1.0).
STRATEGY_SHADE_BOUNDARY: Final[str] = "shade_boundary"
STRATEGY_CANDIDATE_COMPONENT: Final[str] = "candidate_component"
STRATEGY_RARE_CLASS: Final[str] = "rare_class"
STRATEGY_UNIFORM_CANDIDATE: Final[str] = "uniform_candidate"
STRATEGY_SPATIAL_GRID: Final[str] = "spatial_grid"
STRATEGY_EDGE_CANDIDATE: Final[str] = "edge_candidate"

STRATEGY_WEIGHTS: Final[dict[str, float]] = {
    STRATEGY_SHADE_BOUNDARY: 0.45,
    STRATEGY_CANDIDATE_COMPONENT: 0.30,
    STRATEGY_RARE_CLASS: 0.15,
    STRATEGY_UNIFORM_CANDIDATE: 0.10,
}

# V2 mix: spatial coverage first (corners/edges), then class-aware.
STRATEGY_WEIGHTS_V2: Final[dict[str, float]] = {
    STRATEGY_SPATIAL_GRID: 0.40,
    STRATEGY_EDGE_CANDIDATE: 0.20,
    STRATEGY_SHADE_BOUNDARY: 0.20,
    STRATEGY_CANDIDATE_COMPONENT: 0.10,
    STRATEGY_RARE_CLASS: 0.05,
    STRATEGY_UNIFORM_CANDIDATE: 0.05,
}

# Near-duplicate IoU threshold for crop boxes (high overlap rejection).
CROP_IOU_DEDUP_THRESHOLD: Final[float] = 0.90
CROP_IOU_DEDUP_THRESHOLD_V2: Final[float] = 0.70

DEFAULT_PREVIEW_COUNT: Final[int] = 100
DEFAULT_LETTERBOX_SIZE: Final[int] = 512

STATUS_VALID: Final[str] = "valid"
STATUS_WARNING: Final[str] = "warning"
STATUS_INVALID: Final[str] = "invalid"


def class_to_rgb_config() -> dict[str, list[int]]:
    """JSON-serialisable class-id -> RGB mapping for dataset_config.json."""
    return {str(cid): list(rgb) for cid, rgb in CLASS_TO_RGB.items()}

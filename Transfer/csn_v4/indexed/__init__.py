"""Indexed-guided I/O contract for CSN-V4 (indices ≠ display colors)."""
from __future__ import annotations

from csn_v4.indexed.adapters import GrayLegacyAdapterV1, get_gray_adapter
from csn_v4.indexed.contract import (
    allowed_mask,
    build_input_from_target,
    compose_output,
    internal_to_index,
    target_to_internal,
    validate_index_map,
    validate_pair,
)
from csn_v4.indexed.encode import (
    indices_to_structural_rgb,
    structural_channels,
)
from csn_v4.indexed.schema import (
    INDEX_ALLOWED,
    INDEX_BACKGROUND,
    INDEX_OUTLINE,
    INDEX_SHADOW_1,
    INDEX_SHADOW_2,
    INDEXED_INPUT_INDICES,
    INDEXED_OUTPUT_INDICES,
    INDEXED_SCHEMA_VERSION,
    INTERNAL_NUM_CLASSES,
    DISPLAY_GRAY_V1,
    DISPLAY_RGB_V1,
)

__all__ = [
    "INDEX_BACKGROUND",
    "INDEX_OUTLINE",
    "INDEX_ALLOWED",
    "INDEX_SHADOW_1",
    "INDEX_SHADOW_2",
    "INDEXED_INPUT_INDICES",
    "INDEXED_OUTPUT_INDICES",
    "INDEXED_SCHEMA_VERSION",
    "INTERNAL_NUM_CLASSES",
    "DISPLAY_GRAY_V1",
    "DISPLAY_RGB_V1",
    "GrayLegacyAdapterV1",
    "get_gray_adapter",
    "allowed_mask",
    "build_input_from_target",
    "compose_output",
    "internal_to_index",
    "target_to_internal",
    "validate_index_map",
    "validate_pair",
    "structural_channels",
    "indices_to_structural_rgb",
]

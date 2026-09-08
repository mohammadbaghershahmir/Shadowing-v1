"""Constants and schema defaults for AI2BMP-SYN-V1."""

from __future__ import annotations

DATASET_VERSION = "AI2BMP-SYN-V1"
SCHEMA_VERSION = 1

DEFAULT_TASK_AVAILABILITY: dict[str, bool] = {
    "palette_target_available": True,
    "index_target_available": True,
    "generic_boundary_target_available": True,
    "thin_structure_target_available": True,
    "semantic_background_target_available": False,
    "semantic_outline_target_available": False,
    "shade_family_target_available": False,
    "shade_level_target_available": False,
}

ONEHOT_CHANNEL_ORDER = ("outline", "unshaded", "shade_candidate")

SUPPORTED_BMP_SUFFIXES = (".bmp",)

# AI-like resolution profile (observed ~2016x2640 -> ~896x1175)
AI_SCALE_PROFILE_445 = 0.445

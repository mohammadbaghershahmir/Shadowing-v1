"""Generate Dataset V2 full-image and crop artifact maps from resolved RGB."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset_v2.constants import (
    BLACK,
    ORACLE_MAGENTA,
    RGB_TO_SEMANTIC_GRAY,
    RGB_TO_SHADE_LEVEL,
    SHADE_DARK,
    SHADE_LEVEL_IGNORE,
    SHADE_LEVEL_TO_GRAY,
    SHADE_LIGHT,
    SHADE_MEDIUM,
    WHITE,
)


@dataclass(frozen=True)
class SampleArtifacts:
    target_semantic: np.ndarray
    source_bw: np.ndarray
    source_oracle: np.ndarray
    shade_mask: np.ndarray
    shade_level_id: np.ndarray
    shade_level_preview: np.ndarray
    transition_mask: np.ndarray
    black_lock: np.ndarray
    valid_mask: np.ndarray
    reconstructed_target: np.ndarray
    roundtrip_diff: np.ndarray


def _rgb_equals(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    r, g, b = color
    return (rgb[:, :, 0] == r) & (rgb[:, :, 1] == g) & (rgb[:, :, 2] == b)


def build_valid_mask(rgb: np.ndarray) -> np.ndarray:
    valid = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for color in RGB_TO_SEMANTIC_GRAY:
        valid[_rgb_equals(rgb, color)] = 255
    return valid


def build_target_semantic(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for color, gray in RGB_TO_SEMANTIC_GRAY.items():
        out[_rgb_equals(rgb, color)] = gray
    return out


def build_source_bw(rgb: np.ndarray) -> np.ndarray:
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    out[_rgb_equals(rgb, BLACK)] = 0
    return out


def build_source_oracle(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[_rgb_equals(rgb, BLACK)] = np.array(BLACK, dtype=np.uint8)
    out[_rgb_equals(rgb, WHITE)] = np.array(WHITE, dtype=np.uint8)
    shade = (
        _rgb_equals(rgb, SHADE_LIGHT)
        | _rgb_equals(rgb, SHADE_MEDIUM)
        | _rgb_equals(rgb, SHADE_DARK)
    )
    out[shade] = np.array(ORACLE_MAGENTA, dtype=np.uint8)
    return out


def build_shade_mask(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    shade = (
        _rgb_equals(rgb, SHADE_LIGHT)
        | _rgb_equals(rgb, SHADE_MEDIUM)
        | _rgb_equals(rgb, SHADE_DARK)
    )
    out[shade] = 255
    return out


def build_shade_level_id(rgb: np.ndarray) -> np.ndarray:
    out = np.full(rgb.shape[:2], SHADE_LEVEL_IGNORE, dtype=np.uint8)
    for color, level in RGB_TO_SHADE_LEVEL.items():
        out[_rgb_equals(rgb, color)] = level
    return out


def build_shade_level_preview(shade_level_id: np.ndarray) -> np.ndarray:
    out = np.full(shade_level_id.shape, 255, dtype=np.uint8)
    for level, gray in SHADE_LEVEL_TO_GRAY.items():
        out[shade_level_id == level] = gray
    return out


def build_transition_mask(shade_level_id: np.ndarray) -> np.ndarray:
    """4-connected boundary between different shade level IDs (symmetric, no dilation)."""
    h, w = shade_level_id.shape
    out = np.zeros((h, w), dtype=np.uint8)
    ids = shade_level_id
    shade = (ids >= 0) & (ids <= 2)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        sy = slice(max(0, -dy), h + min(0, -dy))
        sx = slice(max(0, -dx), w + min(0, -dx))
        ty = slice(max(0, dy), h + min(0, dy))
        tx = slice(max(0, dx), w + min(0, dx))
        a = ids[sy, sx]
        b = ids[ty, tx]
        both = shade[sy, sx] & shade[ty, tx]
        diff = both & (a != b)
        out[sy, sx] = np.maximum(out[sy, sx], diff.astype(np.uint8) * 255)
        out[ty, tx] = np.maximum(out[ty, tx], diff.astype(np.uint8) * 255)
    return out


def build_black_lock(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    out[_rgb_equals(rgb, BLACK)] = 255
    return out


def reconstruct_target(
    black_lock: np.ndarray,
    shade_mask: np.ndarray,
    shade_level_id: np.ndarray,
) -> np.ndarray:
    out = np.full(black_lock.shape, 255, dtype=np.uint8)
    out[shade_level_id == 0] = 200
    out[shade_level_id == 1] = 150
    out[shade_level_id == 2] = 100
    out[shade_mask == 0] = 255
    out[black_lock == 255] = 0
    return out


def build_roundtrip_diff(target_semantic: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    diff = target_semantic != reconstructed
    return (diff.astype(np.uint8)) * 255


def build_all_artifacts(rgb: np.ndarray, valid_mask: np.ndarray) -> SampleArtifacts:
    target_semantic = build_target_semantic(rgb)
    source_bw = build_source_bw(rgb)
    source_oracle = build_source_oracle(rgb)
    shade_mask = build_shade_mask(rgb)
    shade_level_id = build_shade_level_id(rgb)
    shade_level_preview = build_shade_level_preview(shade_level_id)
    transition_mask = build_transition_mask(shade_level_id)
    black_lock = build_black_lock(rgb)
    reconstructed = reconstruct_target(black_lock, shade_mask, shade_level_id)
    roundtrip_diff = build_roundtrip_diff(target_semantic, reconstructed)
    return SampleArtifacts(
        target_semantic=target_semantic,
        source_bw=source_bw,
        source_oracle=source_oracle,
        shade_mask=shade_mask,
        shade_level_id=shade_level_id,
        shade_level_preview=shade_level_preview,
        transition_mask=transition_mask,
        black_lock=black_lock,
        valid_mask=valid_mask,
        reconstructed_target=reconstructed,
        roundtrip_diff=roundtrip_diff,
    )


def crop_array(arr: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr[y : y + size, x : x + size].copy()
    return arr[y : y + size, x : x + size, :].copy()

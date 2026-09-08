"""Extract local/context/global inputs from a TileSpec on a transformed scene."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from csn_v4.geometry.tile_spec import HALO, INPUT_SIZE, TileSpec
from shadow_dataset.dataset import extract_padded_crop

from csn_v3.data.multiscale import (
    derive_context_crop,
    letterbox_full_bw,
)


def extract_local_bw(full_bw: np.ndarray, spec: TileSpec, pad_value: int = 255) -> np.ndarray:
    tile, _ = extract_padded_crop(
        full_bw,
        x=spec.input_x,
        y=spec.input_y,
        crop_size=spec.input_size,
        pad_value=pad_value,
    )
    return tile


def extract_context_bw(
    full_bw: np.ndarray,
    spec: TileSpec,
    context_size: int = 1024,
    pad_value: int = 255,
) -> np.ndarray:
    return derive_context_crop(
        full_bw,
        spec.input_x,
        spec.input_y,
        local_size=spec.input_size,
        context_size=context_size,
        pad_value=pad_value,
    )


def build_supervision_weights(
    spec: TileSpec,
    artifact_valid: np.ndarray,
    *,
    halo_loss_weight: float = 1.0,
) -> np.ndarray:
    """Spatial loss weights: core=1.0, valid halo=halo_loss_weight, pad=0."""
    s = spec.input_size
    w = np.zeros((s, s), dtype=np.float32)
    ys, xs = spec.model_core_slices()
    core = artifact_valid[ys, xs] if artifact_valid.shape == (s, s) else spec.valid_input_mask[ys, xs]
    w[ys, xs] = np.where(core, 1.0, 0.0)
    halo_ring = spec.valid_input_mask & (w == 0)
    w[halo_ring] = halo_loss_weight
    w[~spec.valid_input_mask] = 0.0
    return w


def build_artifact_valid_mask(local_valid: np.ndarray, spec: TileSpec) -> np.ndarray:
    out = np.zeros((spec.input_size, spec.input_size), dtype=bool)
    out[spec.valid_input_mask] = local_valid[spec.valid_input_mask] > 127
    return out


def extract_tile_bundle(
    source_bw: np.ndarray,
    target_semantic: np.ndarray,
    spec: TileSpec,
    *,
    context_size: int = 1024,
    global_long_side: int = 1024,
    halo_loss_weight: float = 1.0,
    shade_level_id: np.ndarray | None = None,
    transition_mask: np.ndarray | None = None,
    black_lock: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    shade_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build all tensors for one tile from transformed full scene arrays."""
    local_bw = extract_local_bw(source_bw, spec)
    context_bw = extract_context_bw(source_bw, spec, context_size=context_size)
    full_thumb, letterbox_meta = letterbox_full_bw(source_bw, global_long_side)

    fw, fh = spec.image_wh
    ix, iy = spec.input_x, spec.input_y
    s = spec.input_size
    cx, cy = spec.core_x, spec.core_y
    cw, ch = spec.core_w, spec.core_h

    local_sem = np.full((s, s), 255, dtype=np.uint8)
    for sy in range(s):
        for sx in range(s):
            gy, gx = iy + sy, ix + sx
            if 0 <= gy < fh and 0 <= gx < fw:
                local_sem[sy, sx] = target_semantic[gy, gx]

    def _crop_full(arr: np.ndarray | None, fill: int = 0) -> np.ndarray | None:
        if arr is None:
            return None
        out = np.full((s, s), fill, dtype=arr.dtype)
        for sy in range(s):
            for sx in range(s):
                gy, gx = iy + sy, ix + sx
                if 0 <= gy < fh and 0 <= gx < fw:
                    out[sy, sx] = arr[gy, gx]
        return out

    local_trans = _crop_full(transition_mask, fill=0)
    local_black = _crop_full(black_lock, fill=0)
    local_valid = _crop_full(valid_mask, fill=0)
    local_shade = _crop_full(shade_mask, fill=0)
    local_level = _crop_full(shade_level_id, fill=255)

    artifact_valid = build_artifact_valid_mask(
        local_valid if local_valid is not None else np.ones((s, s), dtype=np.uint8) * 255,
        spec,
    )
    supervision_weights = build_supervision_weights(spec, artifact_valid, halo_loss_weight=halo_loss_weight)

    return {
        "local_bw": local_bw,
        "context_bw": context_bw,
        "full_bw": full_thumb,
        "letterbox_meta": letterbox_meta,
        "local_semantic": local_sem,
        "local_transition": local_trans,
        "local_black_lock": local_black,
        "local_valid": local_valid,
        "local_shade_mask": local_shade,
        "local_shade_level_id": local_level,
        "supervision_weights": supervision_weights,
        "crop_coords_norm": spec.input_bbox_norm.copy(),
        "input_bbox_norm": spec.input_bbox_norm.copy(),
        "core_bbox_norm": spec.core_bbox_norm.copy(),
        "tile_spec": spec,
        "core_semantic": target_semantic[cy : cy + ch, cx : cx + cw],
    }


def global_cache_key(scene_id: str, transform_id: int, preprocess_ver: str = "v4") -> str:
    return f"{scene_id}__d4_{transform_id}__{preprocess_ver}"

"""Overlapping tiled inference with seamless feature blending."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from dinov3_carpet_probe.src.feature_adapters import FeatureBundle, extract_features
from dinov3_carpet_probe.src.model_loader import peak_vram_mb
from dinov3_carpet_probe.src.preprocessing import TileSpec, make_tile_grid


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    ww = np.outer(wy, wx).astype(np.float32)
    # Avoid zero weights at corners for tiny tiles
    return np.clip(ww, 1e-3, None)


def run_tiled_inference(
    loaded_model,
    full_tensor: torch.Tensor,
    *,
    tile_size: int = 512,
    overlap: int = 128,
    tile_batch_size: int = 1,
    device: str = "cuda",
    autocast_dtype: str = "bfloat16",
    extract_fn: Callable | None = None,
) -> tuple[FeatureBundle, FeatureBundle, list[TileSpec], dict]:
    """
    full_tensor: normalized [3,H,W].
    Blends overlapping tile features with a 2D Hann window.
    On CUDA OOM, halves tile_batch_size (min 1). Never silently reduces resolution.
    """
    if extract_fn is None:
        extract_fn = extract_features

    _, h, w = full_tensor.shape
    tiles = make_tile_grid(h, w, tile_size=tile_size, overlap=overlap)
    batch_size = max(1, int(tile_batch_size))

    # Probe stride/channels with first tile
    first = full_tensor[:, tiles[0].y0 : tiles[0].y1, tiles[0].x0 : tiles[0].x1].unsqueeze(0)
    while True:
        try:
            final0, fused0 = extract_fn(
                loaded_model, first, device=device, autocast_dtype=autocast_dtype
            )
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                continue
            raise RuntimeError(
                "CUDA OOM at tile_batch_size=1. Reduce --tile-size explicitly; "
                "refusing to silently shrink image resolution."
            ) from None

    stride = final0.stride
    # Assembled feature grid size
    feat_h = max(1, h // stride)
    feat_w = max(1, w // stride)

    def _init_accum(channels: int):
        return (
            np.zeros((feat_h, feat_w, channels), dtype=np.float64),
            np.zeros((feat_h, feat_w), dtype=np.float64),
        )

    sum_final, wt_final = _init_accum(final0.channels)
    sum_fused, wt_fused = _init_accum(fused0.channels)

    meta = {
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_batch_size_effective": batch_size,
        "n_tiles": len(tiles),
        "stride": stride,
        "assembled_hw": (feat_h, feat_w),
        "peak_vram_mb": None,
        "oom_retries": 0,
    }

    i = 0
    while i < len(tiles):
        chunk = tiles[i : i + batch_size]
        batch_list = []
        for t in chunk:
            tile = full_tensor[:, t.y0 : t.y1, t.x0 : t.x1]
            # Pad if edge tile smaller (should be rare with make_tile_grid)
            th, tw = tile.shape[1], tile.shape[2]
            if th != tile_size or tw != tile_size:
                pad = torch.zeros(3, tile_size, tile_size, dtype=tile.dtype)
                pad[:, :th, :tw] = tile
                tile = pad
            batch_list.append(tile)
        batch = torch.stack(batch_list, dim=0)
        try:
            final_b, fused_b = extract_fn(
                loaded_model, batch, device=device, autocast_dtype=autocast_dtype
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                meta["tile_batch_size_effective"] = batch_size
                meta["oom_retries"] += 1
                torch.cuda.empty_cache()
                continue
            raise RuntimeError(
                "CUDA OOM at tile_batch_size=1 during tiled inference. "
                "Reduce --tile-size; resolution will not be auto-reduced."
            ) from None

        for j, t in enumerate(chunk):
            # Map image coords to feature coords
            fy0 = t.y0 // stride
            fx0 = t.x0 // stride
            fy1 = min(feat_h, (t.y1 + stride - 1) // stride)
            fx1 = min(feat_w, (t.x1 + stride - 1) // stride)
            # Actual feature crop from this tile output
            f_final = final_b.features[j].numpy()
            f_fused = fused_b.features[j].numpy()
            # Tile was possibly padded to tile_size; feature grid for full tile:
            th_feat = f_final.shape[0]
            tw_feat = f_final.shape[1]
            # Content size in feature space for this placement
            out_h = fy1 - fy0
            out_w = fx1 - fx0
            use_h = min(out_h, th_feat)
            use_w = min(out_w, tw_feat)
            window = _hann2d(use_h, use_w)
            sum_final[fy0 : fy0 + use_h, fx0 : fx0 + use_w] += (
                f_final[:use_h, :use_w].astype(np.float64) * window[..., None]
            )
            wt_final[fy0 : fy0 + use_h, fx0 : fx0 + use_w] += window
            # Fused may have different spatial size than final (ConvNeXt)
            if f_fused.shape[:2] != f_final.shape[:2]:
                # Scale placement for fused stride
                f_stride = fused_b.stride
                ff_h = max(1, h // f_stride)
                ff_w = max(1, w // f_stride)
                if sum_fused.shape[0] != ff_h or sum_fused.shape[1] != ff_w:
                    # Reallocate once if needed
                    sum_fused = np.zeros((ff_h, ff_w, fused_b.channels), dtype=np.float64)
                    wt_fused = np.zeros((ff_h, ff_w), dtype=np.float64)
                    meta["fused_assembled_hw"] = (ff_h, ff_w)
                    meta["fused_stride"] = f_stride
                fy0f = t.y0 // f_stride
                fx0f = t.x0 // f_stride
                fy1f = min(ff_h, (t.y1 + f_stride - 1) // f_stride)
                fx1f = min(ff_w, (t.x1 + f_stride - 1) // f_stride)
                use_hf = min(fy1f - fy0f, f_fused.shape[0])
                use_wf = min(fx1f - fx0f, f_fused.shape[1])
                window_f = _hann2d(use_hf, use_wf)
                sum_fused[fy0f : fy0f + use_hf, fx0f : fx0f + use_wf] += (
                    f_fused[:use_hf, :use_wf].astype(np.float64) * window_f[..., None]
                )
                wt_fused[fy0f : fy0f + use_hf, fx0f : fx0f + use_wf] += window_f
            else:
                sum_fused[fy0 : fy0 + use_h, fx0 : fx0 + use_w] += (
                    f_fused[:use_h, :use_w].astype(np.float64) * window[..., None]
                )
                wt_fused[fy0 : fy0 + use_h, fx0 : fx0 + use_w] += window
        i += len(chunk)

    def _finalize(sum_arr, wt, kind, stride_v, channels, layer_ids):
        wt_safe = np.clip(wt, 1e-6, None)
        feats = (sum_arr / wt_safe[..., None]).astype(np.float32)
        if not np.isfinite(feats).all():
            raise ValueError(f"tiled {kind} features contain NaN/Inf")
        return FeatureBundle(
            features=torch.from_numpy(feats).unsqueeze(0),
            kind=kind,
            stride=int(stride_v),
            grid_hw=(feats.shape[0], feats.shape[1]),
            channels=channels,
            layer_or_stage_ids=layer_ids,
            extras={"tiled": True, "blend": "hann2d"},
        )

    fused_stride = meta.get("fused_stride", fused0.stride)
    final_out = _finalize(
        sum_final, wt_final, "final", stride, final0.channels, final0.layer_or_stage_ids
    )
    fused_out = _finalize(
        sum_fused, wt_fused, "fused", fused_stride, fused0.channels, fused0.layer_or_stage_ids
    )
    meta["peak_vram_mb"] = peak_vram_mb()
    return final_out, fused_out, tiles, meta

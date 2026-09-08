"""Tiled logit inference for CSN-V3 (512×512 crops + overlap stitch)."""
from __future__ import annotations

import numpy as np
import torch

from csn_v3.data.multiscale import (
    build_global_token_padding_mask,
    derive_context_crop,
    letterbox_full_bw,
    normalize_crop_coords,
)
from csn_v3.data.targets import load_gray_bmp
from csn_v3.renderer import apply_black_lock, class_to_gray_tensor
from shadow_dataset.dataset import extract_padded_crop


def _axis_origins(length: int, tile_size: int, stride: int) -> list[int]:
    """Top-left origins along one axis, always including the final clamped origin."""
    if length <= tile_size:
        return [0]
    last = length - tile_size
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def tile_positions(
    image_height: int,
    image_width: int,
    tile_size: int = 512,
    stride: int = 384,
) -> list[tuple[int, int]]:
    """Generate (y, x) top-left positions for overlapping 512×512 tiles."""
    positions: list[tuple[int, int]] = []
    for y in _axis_origins(image_height, tile_size, stride):
        for x in _axis_origins(image_width, tile_size, stride):
            positions.append((y, x))
    return positions


def _smooth_window(tile_size: int, halo: int) -> np.ndarray:
    """Raised-cosine blend: fade overlap regions, full weight in tile core."""
    w = np.ones((tile_size, tile_size), dtype=np.float32)
    if halo <= 0 or halo * 2 >= tile_size:
        return w
    ramp = np.linspace(0.0, 1.0, halo, dtype=np.float32)
    w[:halo, :] *= ramp[:, None]
    w[-halo:, :] *= ramp[::-1][:, None]
    w[:, :halo] *= ramp[None, :]
    w[:, -halo:] *= ramp[::-1][None, :]
    return w


def _tile_blend_window(
    base: np.ndarray,
    *,
    y: int,
    x: int,
    image_height: int,
    image_width: int,
    tile_size: int,
    halo: int,
) -> np.ndarray:
    """Keep weight=1 on image borders so edge tiles are not zeroed out."""
    w = base.copy()
    if halo <= 0:
        return w
    if y <= 0:
        w[:halo, :] = 1.0
    if x <= 0:
        w[:, :halo] = 1.0
    if y + tile_size >= image_height:
        w[-halo:, :] = 1.0
    if x + tile_size >= image_width:
        w[:, -halo:] = 1.0
    return w


def tiled_predict_logits(
    model,
    source_bw: np.ndarray,
    *,
    tile_size: int = 512,
    halo: int = 64,
    stride: int = 384,
    context_size: int = 1024,
    global_long_side: int = 1024,
    device: torch.device,
) -> torch.Tensor:
    """Run 512×512 tiled forward passes and stitch logits with smooth blending."""
    H, W = source_bw.shape[:2]
    full_thumb, letterbox_meta = letterbox_full_bw(source_bw, global_long_side)
    gf = torch.from_numpy(full_thumb.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        full_rgb = model._prep_rgb(gf)
        global_tokens = model.dino.forward_global_tokens(full_rgb)
        global_pad = build_global_token_padding_mask(
            letterbox_meta, global_tokens.shape[1], model.cfg.model.dino.patch_size, device,
        )

    accum = torch.zeros(1, 4, H, W, device=device)
    weight = torch.zeros(1, 1, H, W, device=device)
    blend = _smooth_window(tile_size, halo)
    positions = tile_positions(H, W, tile_size, stride)

    model.eval()
    with torch.no_grad():
        for y, x in positions:
            tile, valid = extract_padded_crop(
                source_bw,
                x=x,
                y=y,
                crop_size=tile_size,
                pad_value=255,
            )
            context = derive_context_crop(source_bw, x, y, tile_size, context_size)
            lb = torch.from_numpy(tile.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            cb = torch.from_numpy(context.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            cc = torch.from_numpy(normalize_crop_coords(x, y, tile_size, tile_size, W, H)).unsqueeze(0).to(device)

            out = model(
                lb, cb, gf, cc,
                global_tokens=global_tokens,
                letterbox_meta=letterbox_meta,
                teacher_forcing=0.0,
            )
            logits = out["refined_logits"].squeeze(0).float()

            w_np = _tile_blend_window(
                blend,
                y=y,
                x=x,
                image_height=H,
                image_width=W,
                tile_size=tile_size,
                halo=halo,
            )
            w_np[~valid] = 0.0
            w = torch.from_numpy(w_np).to(device)

            src_y0 = max(0, y)
            src_x0 = max(0, x)
            src_y1 = min(H, y + tile_size)
            src_x1 = min(W, x + tile_size)
            cy0 = src_y0 - y
            cx0 = src_x0 - x
            cy1 = cy0 + (src_y1 - src_y0)
            cx1 = cx0 + (src_x1 - src_x0)

            tile_w = w[cy0:cy1, cx0:cx1]
            accum[:, :, src_y0:src_y1, src_x0:src_x1] += (
                logits[:, cy0:cy1, cx0:cx1].unsqueeze(0) * tile_w
            )
            weight[:, :, src_y0:src_y1, src_x0:src_x1] += tile_w.unsqueeze(0).unsqueeze(0)

    return accum / weight.clamp(min=1e-6)


def predict_full_image_gray(
    model,
    source_bw: np.ndarray,
    device: torch.device,
    **kwargs,
) -> tuple[np.ndarray, int]:
    """Tiled inference → stitched grayscale array. Returns (gray, num_tiles)."""
    H, W = source_bw.shape[:2]
    tile_size = kwargs.get("tile_size", 512)
    stride = kwargs.get("stride", 384)
    n_tiles = len(tile_positions(H, W, tile_size, stride))

    logits = tiled_predict_logits(model, source_bw, device=device, **kwargs)
    pred = logits.argmax(dim=1).squeeze(0).cpu()
    gray = class_to_gray_tensor(pred).numpy()
    gray = apply_black_lock(gray, source_bw)
    return gray, n_tiles


def predict_full_image_bmp(model, source_bw_path: str, device: torch.device, **kwargs) -> np.ndarray:
    src = load_gray_bmp(source_bw_path)
    gray, _ = predict_full_image_gray(model, src, device, **kwargs)
    return gray

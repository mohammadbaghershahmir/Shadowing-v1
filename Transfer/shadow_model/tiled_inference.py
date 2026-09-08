"""Tiled inference for full-image prediction with dual-DINO support."""
from __future__ import annotations

import numpy as np
import torch

from shadow_dataset.constants import UNSHADED
from shadow_dataset.dataset import build_input_onehot, extract_padded_crop
from shadow_model.backbone_local import IMAGENET_MEAN, IMAGENET_STD


def _smooth_window(tile_size: int, halo: int) -> np.ndarray:
    """Raised-cosine blend window: 0 at edges, 1 in core."""
    w = np.ones((tile_size, tile_size), dtype=np.float32)
    if halo <= 0 or halo * 2 >= tile_size:
        return w
    ramp = np.linspace(0.0, 1.0, halo, dtype=np.float32)
    w[:halo, :] *= ramp[:, None]
    w[-halo:, :] *= ramp[::-1][:, None]
    w[:, :halo] *= ramp[None, :]
    w[:, -halo:] *= ramp[::-1][None, :]
    return w


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
    """Generate (y, x) top-left positions for overlapping tiles.

    Always includes a final tile flush with the bottom/right so the full image
    is covered when H/W are not aligned to stride.
    """
    positions: list[tuple[int, int]] = []
    for y in _axis_origins(image_height, tile_size, stride):
        for x in _axis_origins(image_width, tile_size, stride):
            positions.append((y, x))
    return positions


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
    """Disable edge fade on sides that touch the image border (weight would be 0)."""
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


def tiled_predict(
    model: torch.nn.Module,
    input_rgb: np.ndarray,
    *,
    tile_size: int = 512,
    halo: int = 64,
    stride: int = 384,
    device: str = "cuda",
    global_tokens: torch.Tensor | None = None,
    global_valid_mask: torch.Tensor | None = None,
    letterbox_meta: dict | None = None,
    imagenet_mean: tuple[float, ...] = IMAGENET_MEAN,
    imagenet_std: tuple[float, ...] = IMAGENET_STD,
) -> np.ndarray:
    """Run core-based tiled inference with smooth window blending.

    Global DINO tokens are computed once; local DINO runs per tile.

    Returns:
        logits: [C, H, W] float32 merged logits
    """
    H, W = input_rgb.shape[:2]
    num_classes = 3
    logit_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    positions = tile_positions(H, W, tile_size, stride)
    blend = _smooth_window(tile_size, halo)

    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)

    model.eval()
    with torch.no_grad():
        for y, x in positions:
            crop, valid = extract_padded_crop(
                input_rgb, x=x, y=y, crop_size=tile_size, pad_value=UNSHADED,
            )

            rgb_t = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)
            rgb_norm = ((rgb_t - mean) / std).unsqueeze(0).to(device)
            onehot = torch.from_numpy(build_input_onehot(crop)).unsqueeze(0).to(device)

            crop_box = torch.tensor(
                [[x / max(W, 1), y / max(H, 1), tile_size / max(W, 1), tile_size / max(H, 1)]],
                dtype=torch.float32,
                device=device,
            )

            kwargs: dict = {}
            if global_tokens is not None:
                kwargs["global_tokens"] = global_tokens.to(device)
            if global_valid_mask is not None:
                kwargs["global_valid_mask"] = global_valid_mask.to(device)
            if letterbox_meta is not None:
                kwargs["letterbox_meta"] = letterbox_meta
            kwargs["crop_box"] = crop_box

            outputs = model(rgb_norm, onehot, **kwargs)
            logits = outputs["class_logits"].squeeze(0).float().cpu().numpy()

            w = _tile_blend_window(
                blend, y=y, x=x, image_height=H, image_width=W, tile_size=tile_size, halo=halo,
            )
            w[~valid] = 0.0

            src_y0 = max(0, y)
            src_x0 = max(0, x)
            src_y1 = min(H, y + tile_size)
            src_x1 = min(W, x + tile_size)

            cy0 = src_y0 - y
            cx0 = src_x0 - x
            cy1 = cy0 + (src_y1 - src_y0)
            cx1 = cx0 + (src_x1 - src_x0)

            tile_w = w[cy0:cy1, cx0:cx1]
            logit_sum[:, src_y0:src_y1, src_x0:src_x1] += logits[:, cy0:cy1, cx0:cx1] * tile_w
            weight_sum[src_y0:src_y1, src_x0:src_x1] += tile_w

    from shadow_dataset.geometry import candidate_mask_from_input

    candidate = candidate_mask_from_input(input_rgb)
    if not candidate.any():
        weight_safe = np.maximum(weight_sum, 1e-8)
        return (logit_sum / weight_safe[np.newaxis, :, :]).astype(np.float32)
    if (weight_sum[candidate] <= 0).any():
        bad = int((weight_sum[candidate] <= 0).sum())
        raise RuntimeError(f"Tiled merge has weight_sum<=0 on {bad} candidate pixels")

    weight_safe = np.maximum(weight_sum, 1e-8)
    merged = logit_sum / weight_safe[np.newaxis, :, :]
    return merged.astype(np.float32)

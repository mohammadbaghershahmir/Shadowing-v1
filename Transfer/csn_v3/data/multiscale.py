"""Multiscale crop derivation for CSN-V3."""
from __future__ import annotations

import numpy as np

from shadow_dataset.dataset import extract_padded_crop


def derive_context_crop(
    full_bw: np.ndarray,
    x: int,
    y: int,
    local_size: int = 512,
    context_size: int = 1024,
    pad_value: int = 255,
) -> np.ndarray:
    """Centered context crop from full source_bw with white padding."""
    half = (context_size - local_size) // 2
    x0, y0 = x - half, y - half
    crop, _ = extract_padded_crop(
        full_bw,
        x=x0,
        y=y0,
        crop_size=context_size,
        pad_value=pad_value,
    )
    return crop


def normalize_crop_coords(x: int, y: int, w: int, h: int, W: int, H: int) -> np.ndarray:
    """6-vector normalized coordinate embedding."""
    return np.array(
        [x / W, y / H, w / W, h / H, (x + w / 2) / W, (y + h / 2) / H],
        dtype=np.float32,
    )


def letterbox_full_bw(
    full_bw: np.ndarray,
    long_side: int = 1024,
    pad_value: int = 255,
) -> tuple[np.ndarray, dict]:
    """Aspect-preserving resize, then pad to fixed square canvas (multiple of 16)."""
    h, w = full_bw.shape[:2]
    scale = long_side / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    from PIL import Image
    img = Image.fromarray(full_bw)
    resized = np.asarray(img.resize((nw, nh), Image.NEAREST), dtype=np.uint8)

    canvas_size = max(16, ((long_side + 15) // 16) * 16)
    canvas = np.full((canvas_size, canvas_size), pad_value, dtype=np.uint8)
    canvas[:nh, :nw] = resized

    meta = {
        "scale": scale,
        "content_width": nw,
        "content_height": nh,
        "canvas_width": canvas_size,
        "canvas_height": canvas_size,
        "image_width": w,
        "image_height": h,
        "offset_x": 0.0,
        "offset_y": 0.0,
    }
    return canvas, meta


def build_global_token_padding_mask(
    letterbox_meta: dict,
    num_tokens: int,
    patch_size: int = 16,
    device: torch.device | None = None,
) -> "torch.Tensor":
    """True = padded (ignore) token positions on letterboxed global canvas."""
    import torch

    cw = letterbox_meta["canvas_width"]
    ch = letterbox_meta["canvas_height"]
    nw = letterbox_meta["content_width"]
    nh = letterbox_meta["content_height"]
    gh = ch // patch_size
    gw = cw // patch_size
    if gh * gw != num_tokens:
        gh = int(num_tokens**0.5)
        gw = num_tokens // max(gh, 1)

    mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    content_gw = max(1, (nw + patch_size - 1) // patch_size)
    content_gh = max(1, (nh + patch_size - 1) // patch_size)
    idx = 0
    for gy in range(gh):
        for gx in range(gw):
            if idx >= num_tokens:
                break
            if gy < content_gh and gx < content_gw:
                mask[idx] = False
            idx += 1
    return mask.unsqueeze(0)


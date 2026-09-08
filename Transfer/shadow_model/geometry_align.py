"""Absolute coordinate mapping between crop queries and letterboxed global canvas."""
from __future__ import annotations

import torch


def crop_uv_grid(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized u,v in [0,1] for each spatial position, shape [H*W]."""
    gy = torch.arange(h, device=device, dtype=torch.float32)
    gx = torch.arange(w, device=device, dtype=torch.float32)
    if h > 1:
        gy = gy / (h - 1)
    if w > 1:
        gx = gx / (w - 1)
    vv, uu = torch.meshgrid(gy, gx, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def map_crop_to_letterbox(
    crop_box: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    letterbox_meta: dict[str, torch.Tensor | float | int],
    global_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local query (u,v) in crop to normalized letterbox coords.

    crop_box: [B,4] normalized (x,y,w,h) in full image coords.
    letterbox_meta keys: scale, offset_x, offset_y, content_width, content_height, image_width, image_height
    """
    B = crop_box.shape[0]
    x, y, w, h = crop_box[:, 0], crop_box[:, 1], crop_box[:, 2], crop_box[:, 3]
    img_w = float(letterbox_meta["image_width"])
    img_h = float(letterbox_meta["image_height"])
    scale = float(letterbox_meta["scale"])
    off_x = float(letterbox_meta["offset_x"])
    off_y = float(letterbox_meta["offset_y"])

    # pixel coords in full image
    x_g = (x.unsqueeze(1) + u.unsqueeze(0) * w.unsqueeze(1)) * img_w
    y_g = (y.unsqueeze(1) + v.unsqueeze(0) * h.unsqueeze(1)) * img_h

    x_lb = (off_x + x_g * scale) / global_size
    y_lb = (off_y + y_g * scale) / global_size
    return x_lb, y_lb


def sincos_posembed_from_coords(x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    """2D sincos PE from normalized coords [B,N] -> [B,N,dim]."""
    assert dim % 4 == 0
    half = dim // 4
    device = x.device
    omega = 1.0 / (10000.0 ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    pe_y = torch.cat([torch.sin(y.unsqueeze(-1) * omega), torch.cos(y.unsqueeze(-1) * omega)], dim=-1)
    pe_x = torch.cat([torch.sin(x.unsqueeze(-1) * omega), torch.cos(x.unsqueeze(-1) * omega)], dim=-1)
    return torch.cat([pe_y, pe_x], dim=-1)

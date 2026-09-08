"""Crop geometry helpers."""

from __future__ import annotations

import numpy as np


def normalized_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> list[float]:
    return [
        round(x / img_w, 6),
        round(y / img_h, 6),
        round(w / img_w, 6),
        round(h / img_h, 6),
    ]


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], threshold: float = 0.9) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = aw * ah
    area_b = bw * bh
    iou = inter / max(1, area_a + area_b - inter)
    return iou >= threshold


def clamp_crop(x: int, y: int, size: int, img_w: int, img_h: int) -> tuple[int, int, int, int, np.ndarray]:
    """Return x,y,w,h and image_valid_mask for crop with constant padding."""
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + size)
    y1 = min(img_h, y + size)
    valid = np.zeros((size, size), dtype=bool)
    dx = x0 - x
    dy = y0 - y
    valid[dy : dy + (y1 - y0), dx : dx + (x1 - x0)] = True
    return x, y, size, size, valid


def central_valid_mask(size: int, margin: int) -> np.ndarray:
    inner = size - 2 * margin
    mask = np.zeros((size, size), dtype=bool)
    mask[margin : margin + inner, margin : margin + inner] = True
    return mask

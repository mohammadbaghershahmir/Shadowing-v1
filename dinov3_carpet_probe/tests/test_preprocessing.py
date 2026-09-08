"""Unit tests for preprocessing (no model weights required)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from dinov3_carpet_probe.src.preprocessing import (
    LVD_MEAN,
    LVD_STD,
    make_tile_grid,
    pad_to_multiple,
    preprocess_global,
    resize_long_side_preserve_aspect,
)


def test_resize_preserves_aspect():
    img = Image.new("RGB", (3000, 2000), color=(10, 20, 30))
    resized, scale = resize_long_side_preserve_aspect(img, 1024)
    w, h = resized.size
    assert max(w, h) == 1024
    assert abs((w / h) - (3000 / 2000)) < 0.02


def test_pad_to_multiple():
    t = torch.zeros(3, 100, 77)
    padded, pad_box = pad_to_multiple(t, 16)
    assert padded.shape[1] % 16 == 0
    assert padded.shape[2] % 16 == 0
    assert pad_box[0] == 0 and pad_box[1] == 0


def test_preprocess_global_pad_multiple():
    img = Image.new("RGB", (3000, 2000), color=(128, 64, 32))
    tensor, meta, resized = preprocess_global(
        img, long_side=1024, pad_multiple=16, mean=LVD_MEAN, std=LVD_STD
    )
    assert tensor.shape[0] == 3
    assert meta.processed_hw[0] % 16 == 0
    assert meta.processed_hw[1] % 16 == 0
    assert abs((resized.size[0] / resized.size[1]) - (3000 / 2000)) < 0.02


def test_tile_grid_covers():
    tiles = make_tile_grid(1000, 800, tile_size=512, overlap=128)
    assert len(tiles) >= 4
    # Coverage: every pixel belongs to at least one tile
    cover = np.zeros((1000, 800), dtype=np.int32)
    for t in tiles:
        cover[t.y0 : t.y1, t.x0 : t.x1] += 1
    assert cover.min() >= 1
    assert all((t.y1 - t.y0) == 512 or t.y0 == 0 for t in tiles)

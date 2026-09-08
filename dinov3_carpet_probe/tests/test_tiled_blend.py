"""Tiled blend weight tests."""

from __future__ import annotations

import numpy as np

from dinov3_carpet_probe.src.preprocessing import make_tile_grid
from dinov3_carpet_probe.src.tiled_inference import _hann2d


def test_hann2d_positive():
    w = _hann2d(64, 64)
    assert w.shape == (64, 64)
    assert w.min() > 0
    assert w.max() <= 1.0 + 1e-6


def test_overlap_regions_have_multiple_tiles():
    h, w = 768, 768
    tiles = make_tile_grid(h, w, tile_size=512, overlap=128)
    cover = np.zeros((h, w), dtype=np.int32)
    for t in tiles:
        cover[t.y0 : t.y1, t.x0 : t.x1] += 1
    # Center of image should be covered by overlapping tiles
    assert cover[384, 384] >= 2

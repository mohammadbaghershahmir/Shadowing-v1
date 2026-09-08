"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shdowing.ai2bmp.bmp.fixtures import (
    make_index_map_checkerboard,
    make_palette_8_indices,
    write_test_bmp_4,
    write_test_bmp_8,
)


@pytest.fixture
def tmp_bmp_dir(tmp_path: Path) -> Path:
    d = tmp_path / "bmps"
    d.mkdir()
    return d


@pytest.fixture
def palette8() -> list[tuple[int, int, int]]:
    return make_palette_8_indices()


@pytest.fixture
def bmp_8_8indices(tmp_bmp_dir: Path, palette8) -> Path:
    idx = make_index_map_checkerboard(256, [0, 1, 2, 3, 4, 5, 6, 7])
    return write_test_bmp_8(tmp_bmp_dir / "test8.bmp", idx, palette8, clr_used=16)


@pytest.fixture
def bmp_4_noncontiguous(tmp_bmp_dir: Path) -> Path:
    palette = [(i * 10, i * 5, i * 3) for i in range(16)]
    idx = np.zeros((32, 32), dtype=np.uint8)
    idx[:16, :16] = 0
    idx[16:, :16] = 4
    idx[:16, 16:] = 7
    idx[16:, 16:] = 12
    return write_test_bmp_4(tmp_bmp_dir / "test4.bmp", idx, palette, clr_used=16)

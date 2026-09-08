"""BMP parsing and validation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shdowing.ai2bmp.bmp.fixtures import make_index_map_checkerboard, write_test_bmp_4, write_test_bmp_8
from shdowing.ai2bmp.bmp.validate import load_validated_index_and_palette, validate_bmp_file
from shdowing.ai2bmp.canonical.artifacts import generate_canonical_artifacts
from shdowing.ai2bmp.canonical.mappings import (
    apply_contiguous_used_index_map,
    apply_unique_rgb_map,
    build_contiguous_used_index_mapping,
    build_unique_rgb_mapping,
)


def test_8bpp_parsing(bmp_8_8indices: Path) -> None:
    rec = validate_bmp_file(bmp_8_8indices)
    assert rec.status == "valid"
    assert rec.bits_per_pixel == 8
    assert rec.used_index_count == 8
    assert rec.declared_palette_entry_count == 16


def test_4bpp_parsing(bmp_4_noncontiguous: Path) -> None:
    rec = validate_bmp_file(bmp_4_noncontiguous)
    assert rec.status == "valid"
    assert rec.bits_per_pixel == 4
    assert rec.used_index_count == 4


def test_clr_used_zero_default_256(tmp_bmp_dir: Path, palette8) -> None:
    idx = make_index_map_checkerboard(32, [0, 1, 2])
    path = write_test_bmp_8(tmp_bmp_dir / "clr0.bmp", idx, palette8[:4], clr_used=0)
    rec = validate_bmp_file(path)
    assert rec.declared_palette_entry_count == 256


def test_declared_7_used_6(tmp_bmp_dir: Path) -> None:
    palette = [(i, i, i) for i in range(7)]
    idx = np.zeros((32, 32), dtype=np.uint8)
    for i in range(6):
        idx[i * 4 : i * 4 + 4, :] = i
    path = write_test_bmp_8(tmp_bmp_dir / "d7u6.bmp", idx, palette, clr_used=7)
    rec = validate_bmp_file(path)
    assert rec.used_index_count == 6


def test_declared_16_used_8(bmp_8_8indices: Path) -> None:
    rec = validate_bmp_file(bmp_8_8indices)
    assert rec.declared_palette_entry_count == 16
    assert rec.used_index_count == 8


def test_duplicate_rgb_indices(tmp_bmp_dir: Path) -> None:
    palette = [(10, 10, 10), (20, 20, 20), (10, 10, 10), (30, 30, 30)]
    idx = np.array([[0, 2], [1, 3]], dtype=np.uint8)
    path = write_test_bmp_8(tmp_bmp_dir / "dup.bmp", idx, palette, clr_used=4)
    rec = validate_bmp_file(path)
    assert rec.unique_used_rgb_count == 3
    assert rec.used_index_count == 4
    assert len(rec.duplicate_rgb_indices) >= 1


def test_used_vs_unique_count(bmp_8_8indices: Path) -> None:
    rec = validate_bmp_file(bmp_8_8indices)
    assert rec.used_index_count >= rec.unique_used_rgb_count


def test_reversible_mappings(bmp_8_8indices: Path) -> None:
    _, index_map, palette = load_validated_index_and_palette(bmp_8_8indices)
    used = np.unique(index_map)
    _, cm = build_contiguous_used_index_mapping(used)
    out = apply_contiguous_used_index_map(index_map, cm)
    assert out.min() >= 0
    um, _ = build_unique_rgb_mapping(used, palette)
    uout = apply_unique_rgb_map(index_map, um)
    assert uout.min() >= 0


def test_canonical_roundtrip_zero(bmp_8_8indices: Path, tmp_path: Path) -> None:
    canonical = generate_canonical_artifacts(bmp_8_8indices, tmp_path / "out")
    assert canonical.roundtrip_mismatch_count == 0


def test_malformed_rejection(tmp_bmp_dir: Path) -> None:
    bad = tmp_bmp_dir / "bad.bmp"
    bad.write_bytes(b"NOTABMP")
    rec = validate_bmp_file(bad)
    assert rec.status == "invalid"

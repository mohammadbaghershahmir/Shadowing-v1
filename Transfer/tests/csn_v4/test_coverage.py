"""Tests for tile coverage overwrite detection."""
from __future__ import annotations

import pytest

from csn_v4.geometry.tile_spec import CORE_SIZE, HALO, assert_full_coverage, build_coverage_map, build_tile_spec


def test_assert_full_coverage_rejects_overwrite():
    fw, fh = 384, 384
    specs = [
        build_tile_spec(fw, fh, 0, 0),
        build_tile_spec(fw, fh, 0, 0),
    ]
    with pytest.raises(AssertionError, match="overwrite"):
        assert_full_coverage(fw, fh, specs)


def test_build_coverage_map_detects_double_cover():
    fw, fh = 384 + 2 * HALO, 384 + 2 * HALO
    spec = build_tile_spec(fw, fh, HALO, HALO)
    cov = build_coverage_map(fw, fh, [spec, spec])
    assert int((cov > 1).sum()) == CORE_SIZE * CORE_SIZE

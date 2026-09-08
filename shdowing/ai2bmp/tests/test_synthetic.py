"""Synthetic corruption tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shdowing.ai2bmp.canonical.artifacts import generate_canonical_artifacts
from shdowing.ai2bmp.labels.pipeline import generate_exact_labels
from shdowing.ai2bmp.synthetic.engine import apply_recipe
from shdowing.ai2bmp.synthetic.io import generate_variant
from shdowing.ai2bmp.synthetic.profiles import pilot_variants
from shdowing.ai2bmp.synthetic.recipe import derive_seed, rng_for


def test_deterministic_variant_seed(bmp_8_8indices: Path) -> None:
    s1 = derive_seed(42, "design", "v1")
    s2 = derive_seed(42, "design", "v1")
    s3 = derive_seed(42, "design", "v2")
    assert s1 == s2
    assert s1 != s3


def test_canonical_unchanged_after_corruption(bmp_8_8indices: Path, tmp_path: Path) -> None:
    canonical = generate_canonical_artifacts(bmp_8_8indices, tmp_path / "out")
    before = canonical.rgb.copy()
    labels = generate_exact_labels(canonical)
    recipe = pilot_variants(42, canonical.sample_id)[1]
    result = generate_variant(canonical, recipe, labels["boundary_mask"], labels["label_metadata"])
    assert result["status"] == "generated"
    after = np.load(canonical.canonical_dir / "original_index_map.npy", mmap_mode="r")
    assert np.array_equal(canonical.rgb, before)


def test_native_aligned_metadata(bmp_8_8indices: Path, tmp_path: Path) -> None:
    canonical = generate_canonical_artifacts(bmp_8_8indices, tmp_path / "out")
    labels = generate_exact_labels(canonical)
    recipe = next(r for r in pilot_variants(42, canonical.sample_id) if r.variant_id == "resolution_445")
    result = generate_variant(canonical, recipe, labels["boundary_mask"], labels["label_metadata"])
    meta = result["metadata"]
    assert meta["aligned_width"] == canonical.audit.width
    assert meta["aligned_height"] == canonical.audit.absolute_height


def test_different_seeds_differ(bmp_8_8indices: Path, tmp_path: Path) -> None:
    from shdowing.ai2bmp.synthetic.recipe import derive_seed

    canonical = generate_canonical_artifacts(bmp_8_8indices, tmp_path / "out")
    s1 = derive_seed(1, canonical.sample_id, "palette_shift_mild")
    s2 = derive_seed(99, canonical.sample_id, "palette_shift_mild")
    assert s1 != s2

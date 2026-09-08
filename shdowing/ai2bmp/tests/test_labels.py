"""Structural label tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shdowing.ai2bmp.bmp.validate import load_validated_index_and_palette
from shdowing.ai2bmp.canonical.artifacts import generate_canonical_artifacts
from shdowing.ai2bmp.labels.boundaries import compute_boundary_pair_map, compute_region_boundary_mask
from shdowing.ai2bmp.labels.pipeline import generate_exact_labels
from shdowing.ai2bmp.labels.regions import label_regions


def test_region_and_boundary(bmp_8_8indices: Path, tmp_path: Path) -> None:
    canonical = generate_canonical_artifacts(bmp_8_8indices, tmp_path / "out")
    labels = generate_exact_labels(canonical)
    assert (canonical.design_dir / "exact_labels" / "region_id_map.npy").exists()
    boundary = labels["boundary_mask"]
    assert boundary.dtype == np.uint8
    pair = compute_boundary_pair_map(canonical.contiguous_used_index_map, boundary)
    assert "class_a" in pair


def test_thin_structure_masks(bmp_8_8indices: Path, tmp_path: Path) -> None:
    canonical = generate_canonical_artifacts(bmp_8_8indices, tmp_path / "out2")
    generate_exact_labels(canonical)
    assert (canonical.design_dir / "exact_labels" / "thin_structure_mask.bmp").exists()
    assert (canonical.design_dir / "exact_labels" / "endpoint_mask.bmp").exists()

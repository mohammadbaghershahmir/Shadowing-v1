"""Generate exact structural labels for a canonical design."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shdowing.ai2bmp.canonical.artifacts import CanonicalArtifacts
from shdowing.ai2bmp.io_utils import ensure_dir, save_index_bmp, write_json
from shdowing.ai2bmp.labels.boundaries import compute_boundary_pair_map, compute_region_boundary_mask
from shdowing.ai2bmp.labels.provisional import generate_provisional_labels
from shdowing.ai2bmp.labels.regions import label_regions
from shdowing.ai2bmp.labels.thin_structure import generate_thin_structure_labels


def generate_exact_labels(canonical: CanonicalArtifacts) -> dict[str, Any]:
    exact_dir = ensure_dir(canonical.design_dir / "exact_labels")
    provisional_dir = ensure_dir(canonical.design_dir / "provisional_labels")

    region_id_map, region_records = label_regions(
        canonical.index_map,
        canonical.contiguous_used_index_map,
        canonical.palette,
        canonical.contiguous_used_mapping,
    )
    np.save(exact_dir / "region_id_map.npy", region_id_map)

    boundary_mask = compute_region_boundary_mask(canonical.contiguous_used_index_map)
    save_index_bmp(exact_dir / "region_boundary_mask.bmp", boundary_mask)

    pair_map = compute_boundary_pair_map(canonical.contiguous_used_index_map, boundary_mask)
    np.savez_compressed(exact_dir / "boundary_pair_map.npz", **pair_map)

    thin_meta = generate_thin_structure_labels(
        canonical.contiguous_used_index_map, boundary_mask, exact_dir
    )

    label_metadata = {
        "region_count": len(region_records),
        "boundary_pixel_count": int((boundary_mask > 0).sum()),
        "thin_structure": thin_meta,
        "regions": region_records,
    }
    write_json(exact_dir / "label_metadata.json", label_metadata)

    provisional_meta = generate_provisional_labels(
        canonical.index_map,
        canonical.palette,
        region_records,
        provisional_dir,
    )

    return {
        "exact_labels_dir": str(exact_dir),
        "provisional_labels_dir": str(provisional_dir),
        "label_metadata": label_metadata,
        "provisional_meta": provisional_meta,
        "region_id_map": region_id_map,
        "boundary_mask": boundary_mask,
        "region_records": region_records,
    }

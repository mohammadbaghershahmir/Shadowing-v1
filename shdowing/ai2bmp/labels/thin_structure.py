"""Thin-structure geometric labels."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from shdowing.ai2bmp.io_utils import save_index_bmp


THIN_RULES = {
    "distance_threshold_px": 2.0,
    "min_component_area": 4,
    "endpoint_degree_max": 1,
    "junction_degree_min": 3,
}


def generate_thin_structure_labels(
    contiguous_used_index_map: np.ndarray,
    boundary_mask: np.ndarray,
    exact_labels_dir,
) -> dict[str, Any]:
    cont = contiguous_used_index_map.astype(np.int32)
    h, w = cont.shape
    combined = np.zeros((h, w), dtype=bool)
    for cls in np.unique(cont):
        mask = cont == cls
        if mask.sum() < THIN_RULES["min_component_area"]:
            combined |= mask
            continue
        dist = ndimage.distance_transform_edt(mask)
        thin = dist <= THIN_RULES["distance_threshold_px"]
        combined |= thin

    thin_u8 = combined.astype(np.uint8)
    neighbor_count = ndimage.convolve(thin_u8, np.ones((3, 3), dtype=np.uint8), mode="constant")
    endpoint = combined & (neighbor_count <= 2)
    junction = combined & (neighbor_count >= 4)

    dist_to_boundary = ndimage.distance_transform_edt(boundary_mask == 0).astype(np.float32)

    save_index_bmp(exact_labels_dir / "thin_structure_mask.bmp", (combined.astype(np.uint8) * 255))
    save_index_bmp(exact_labels_dir / "endpoint_mask.bmp", (endpoint.astype(np.uint8) * 255))
    save_index_bmp(exact_labels_dir / "junction_mask.bmp", (junction.astype(np.uint8) * 255))
    np.save(exact_labels_dir / "distance_to_boundary.npy", dist_to_boundary)

    return {
        "rules": THIN_RULES,
        "thin_pixel_count": int(combined.sum()),
        "endpoint_count": int(endpoint.sum()),
        "junction_count": int(junction.sum()),
    }

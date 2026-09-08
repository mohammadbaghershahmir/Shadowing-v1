"""Deterministic shade-aware crop selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from shdowing.ai2bmp.crops.geometry import boxes_overlap, central_valid_mask, clamp_crop, normalized_box
from shdowing.ai2bmp.synthetic.recipe import rng_for


CATEGORY_QUOTAS = {
    256: 12,
    512: 16,
    1024: 8,
}


def _integral_image(mask: np.ndarray) -> np.ndarray:
    return ndimage.sum(mask.astype(np.float64), labels=np.ones_like(mask), index=[1]) if False else np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def window_sum(integral: np.ndarray, y: int, x: int, h: int, w: int) -> int:
    y2, x2 = y + h, x + w
    return int(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def select_crops_for_design(
    *,
    sample_id: str,
    source_id: str,
    split: str,
    input_path: str,
    target_path: str,
    index_map: np.ndarray,
    contiguous_map: np.ndarray,
    boundary_mask: np.ndarray,
    thin_mask: np.ndarray,
    image_width: int,
    image_height: int,
    crop_sizes: list[int],
    global_seed: int,
    valid_margin: int = 64,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = rng_for(global_seed + hash(sample_id) % 100000)
    crops: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    rare_indices = [int(i) for i in np.unique(index_map) if (index_map == i).sum() < index_map.size * 0.02]

    for size in crop_sizes:
        if image_width < size or image_height < size:
            rejected.append({"crop_size": size, "reason": "image_smaller_than_crop"})
            continue
        quota = CATEGORY_QUOTAS.get(size, 8)
        candidates: list[tuple[int, int, str, float]] = []

        bpts = np.argwhere(boundary_mask > 0)
        if len(bpts):
            for _ in range(max(2, quota // 3)):
                cy, cx = bpts[rng.integers(0, len(bpts))]
                candidates.append((int(cx - size // 2), int(cy - size // 2), "boundary_rich", 1.0))

        tpts = np.argwhere(thin_mask > 0)
        if len(tpts):
            for _ in range(max(1, quota // 4)):
                cy, cx = tpts[rng.integers(0, len(tpts))]
                candidates.append((int(cx - size // 2), int(cy - size // 2), "thin_structure_rich", 0.9))

        if rare_indices:
            ri = rare_indices[rng.integers(0, len(rare_indices))]
            pts = np.argwhere(index_map == ri)
            cy, cx = pts[rng.integers(0, len(pts))]
            candidates.append((int(cx - size // 2), int(cy - size // 2), "rare_palette_index", 0.85))

        for _ in range(max(1, quota // 5)):
            cx = int(rng.integers(0, max(1, image_width - size)))
            cy = int(rng.integers(0, max(1, image_height - size)))
            candidates.append((cx, cy, "mixed_random", 0.5))

        accepted_boxes: list[tuple[int, int, int, int]] = []
        center_mask = central_valid_mask(size, valid_margin)

        for x, y, category, score in candidates:
            if x < 0 or y < 0 or x + size > image_width or y + size > image_height:
                rejected.append({"x": x, "y": y, "crop_size": size, "reason": "out_of_bounds"})
                continue
            if any(boxes_overlap((x, y, size, size), b) for b in accepted_boxes):
                rejected.append({"x": x, "y": y, "crop_size": size, "reason": "near_duplicate"})
                continue

            window = index_map[y : y + size, x : x + size]
            if np.unique(window).size < 2:
                rejected.append({"x": x, "y": y, "crop_size": size, "reason": "insufficient_diversity"})
                continue

            bwin = boundary_mask[y : y + size, x : x + size]
            twin = thin_mask[y : y + size, x : x + size]
            accepted_boxes.append((x, y, size, size))

            shade_counts: dict[str, int] = {}
            for idx in np.unique(window).tolist():
                shade_counts[str(int(idx))] = int((window == idx).sum())

            crop_id = f"{source_id}__s{size}__x{x}_y{y}"
            crops.append(
                {
                    "schema_version": 1,
                    "sample_id": crop_id,
                    "source_id": source_id,
                    "input_path": input_path,
                    "target_path": target_path,
                    "split": split,
                    "image_width": image_width,
                    "image_height": image_height,
                    "x": x,
                    "y": y,
                    "crop_size": size,
                    "valid_margin": valid_margin,
                    "sampling_strategy": category,
                    "candidate_pixels": int(window.size),
                    "candidate_pixels_in_valid_center": int(center_mask.sum()),
                    "shade_counts": shade_counts,
                    "boundary_pixels": int((bwin > 0).sum()),
                    "component_count": int(len(np.unique(window))),
                    "normalized_crop_box": normalized_box(x, y, size, size, image_width, image_height),
                    "selection_score": score,
                    "thin_structure_density": float((twin > 0).mean()),
                }
            )
            if len([c for c in crops if c["crop_size"] == size]) >= quota:
                break

    return crops, rejected

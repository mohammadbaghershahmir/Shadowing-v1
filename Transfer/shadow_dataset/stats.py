"""Aggregate and serialize dataset statistics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from shadow_dataset.constants import CLASS_COUNT_KEYS
from shadow_dataset.types import (
    CropRecord,
    CropSamplingStats,
    ImageRecord,
    ValidationResult,
)


def compute_dataset_stats(
    *,
    n_input_files: int,
    n_target_files: int,
    validation_results: Sequence[ValidationResult],
    image_records: Sequence[ImageRecord],
    train_crops: Sequence[CropRecord],
    val_crops: Sequence[CropRecord],
    crop_sampling_stats: CropSamplingStats,
    component_sizes: Sequence[int],
    boundary_pixel_total: int,
) -> dict[str, Any]:
    status_counts = Counter(r.status for r in validation_results)
    split_counts = Counter(r.split for r in image_records)

    widths = [r.width for r in image_records]
    heights = [r.height for r in image_records]

    total_candidates = sum(r.candidate_pixel_count for r in image_records)
    class_freq = {k: 0 for k in CLASS_COUNT_KEYS}
    for r in image_records:
        for k in class_freq:
            class_freq[k] += int(r.class_counts.get(k, 0))

    crop_strategy = Counter(c.sampling_strategy for c in train_crops)
    crop_class_hist = {k: 0 for k in CLASS_COUNT_KEYS}
    for c in list(train_crops) + list(val_crops):
        for k in crop_class_hist:
            crop_class_hist[k] += int(c.class_counts.get(k, 0))

    tiny_crops = sum(1 for c in train_crops if c.is_tiny_component)

    size_arr = (
        np.asarray(list(component_sizes), dtype=np.int64)
        if component_sizes
        else np.asarray([])
    )
    if size_arr.size:
        comp_dist = {
            "count": int(size_arr.size),
            "min": int(size_arr.min()),
            "max": int(size_arr.max()),
            "mean": float(size_arr.mean()),
            "median": float(np.median(size_arr)),
            "p90": float(np.percentile(size_arr, 90)),
            "tiny_le_64": int((size_arr <= 64).sum()),
        }
    else:
        comp_dist = {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "tiny_le_64": 0,
        }

    dim_dist = {
        "count": len(widths),
        "width_min": int(min(widths)) if widths else 0,
        "width_max": int(max(widths)) if widths else 0,
        "width_mean": float(np.mean(widths)) if widths else 0.0,
        "height_min": int(min(heights)) if heights else 0,
        "height_max": int(max(heights)) if heights else 0,
        "height_mean": float(np.mean(heights)) if heights else 0.0,
    }

    return {
        "discovered_input_files": int(n_input_files),
        "discovered_target_files": int(n_target_files),
        "validation_status_counts": dict(status_counts),
        "valid_pairs": int(status_counts.get("valid", 0)),
        "invalid_pairs": int(status_counts.get("invalid", 0)),
        "warning_pairs": int(status_counts.get("warning", 0)),
        "images_per_split": dict(split_counts),
        "dimension_distribution": dim_dist,
        "total_candidate_pixels": int(total_candidates),
        "class_frequencies": class_freq,
        "connected_component_size_distribution": comp_dist,
        "internal_class_boundary_pixels": int(boundary_pixel_total),
        "crop_counts": {
            "train": len(train_crops),
            "val": len(val_crops),
        },
        "crop_count_per_strategy": dict(crop_strategy),
        "class_histogram_across_crops": crop_class_hist,
        "tiny_component_crops": int(tiny_crops),
        "crop_sampling": crop_sampling_stats.to_dict(),
    }


def format_summary(stats: dict[str, Any]) -> str:
    lines = [
        "=== Dataset preparation summary ===",
        f"Input files discovered : {stats['discovered_input_files']}",
        f"Target files discovered: {stats['discovered_target_files']}",
        f"Valid pairs            : {stats['valid_pairs']}",
        f"Invalid pairs          : {stats['invalid_pairs']}",
        f"Warning pairs          : {stats['warning_pairs']}",
        f"Images per split       : {stats['images_per_split']}",
        f"Dimension distribution : {stats['dimension_distribution']}",
        f"Total candidate pixels : {stats['total_candidate_pixels']}",
        f"Class frequencies      : {stats['class_frequencies']}",
        f"Component sizes        : {stats['connected_component_size_distribution']}",
        f"Class boundary pixels  : {stats['internal_class_boundary_pixels']}",
        f"Crop counts            : {stats['crop_counts']}",
        f"Crops per strategy     : {stats['crop_count_per_strategy']}",
        f"Class histogram (crops): {stats['class_histogram_across_crops']}",
        f"Tiny-component crops   : {stats['tiny_component_crops']}",
        f"Rejected/dedup crops   : {stats['crop_sampling']}",
    ]
    return "\n".join(lines)

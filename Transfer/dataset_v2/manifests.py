"""CSV manifest writers for Dataset V2."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


CANONICAL_SAMPLE_FIELDS = [
    "dataset_version",
    "sample_id",
    "group_id",
    "original_filename",
    "relative_path",
    "width",
    "height",
    "sha256",
    "validation_status",
    "black_pixels",
    "white_pixels",
    "shade_light_pixels",
    "shade_medium_pixels",
    "shade_dark_pixels",
    "roundtrip_mismatch_count",
    "full_artifact_dir",
]

CROP_MANIFEST_FIELDS = [
    "dataset_version",
    "sample_id",
    "group_id",
    "parent_sample_id",
    "original_filename",
    "crop_id",
    "base_crop_id",
    "crop_category",
    "x",
    "y",
    "width",
    "height",
    "class_0_count",
    "class_1_count",
    "class_2_count",
    "class_0_pct",
    "class_1_pct",
    "class_2_pct",
    "shade_coverage_pct",
    "transition_density",
    "content_hash",
    "validation_status",
    "artifact_dir",
]

AUGMENTATION_FIELDS = [
    "dataset_version",
    "group_id",
    "parent_sample_id",
    "base_crop_id",
    "augmentation_id",
    "operation",
    "crop_size",
    "inverse_transform_exact",
    "roundtrip_exact",
    "approved_for_training",
    "artifact_dir",
]

PILOT_SUMMARY_FIELDS = [
    "sample_id",
    "status",
    "original_filename",
    "width",
    "height",
    "canonical_crops",
    "augmentation_candidates",
    "roundtrip_mismatch_count",
    "unexpected_colors",
]

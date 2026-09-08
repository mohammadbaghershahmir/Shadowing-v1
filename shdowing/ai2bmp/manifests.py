"""Manifest CSV/JSONL writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shdowing.ai2bmp.constants import DATASET_VERSION
from shdowing.ai2bmp.io_utils import rel_path, write_csv, write_json, write_jsonl


def write_all_manifests(
    output_root: Path,
    *,
    canonical_rows: list[dict[str, Any]],
    palette_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    crop_rows: list[dict[str, Any]],
    augmentation_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> None:
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    def _rel_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            nr = dict(row)
            for k in keys:
                if k in nr and nr[k]:
                    nr[k] = rel_path(nr[k], output_root) if not str(nr[k]).startswith("http") else nr[k]
            nr["dataset_version"] = DATASET_VERSION
            out.append(nr)
        return out

    path_keys = [
        "source_bmp_path",
        "input_path",
        "target_path",
        "recipe_path",
        "clean_input_path",
        "native_input_path",
        "aligned_input_path",
    ]

    write_csv(manifest_dir / "canonical_designs.csv", _rel_rows(canonical_rows, path_keys), list(canonical_rows[0].keys()) if canonical_rows else ["sample_id"])
    write_csv(manifest_dir / "palette_entries.csv", palette_rows, list(palette_rows[0].keys()) if palette_rows else ["sample_id"])
    write_csv(manifest_dir / "variants.csv", _rel_rows(variant_rows, path_keys), list(variant_rows[0].keys()) if variant_rows else ["variant_id"])
    write_csv(manifest_dir / "crops.csv", _rel_rows(crop_rows, path_keys), list(crop_rows[0].keys()) if crop_rows else ["sample_id"])
    write_csv(manifest_dir / "augmentations.csv", augmentation_rows, list(augmentation_rows[0].keys()) if augmentation_rows else ["augmentation_id"])
    write_csv(manifest_dir / "duplicate_candidates.csv", duplicate_rows, list(duplicate_rows[0].keys()) if duplicate_rows else ["group_type"])
    write_csv(manifest_dir / "rejected_files.csv", rejected_rows, list(rejected_rows[0].keys()) if rejected_rows else ["filename"])
    write_jsonl(manifest_dir / "all_training_samples.jsonl", _rel_rows(training_rows, path_keys))

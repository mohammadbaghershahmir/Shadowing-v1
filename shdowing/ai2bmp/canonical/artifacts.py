"""Canonical artifact generation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shdowing.ai2bmp.bmp.raster import index_to_rgb
from shdowing.ai2bmp.bmp.validate import BmpAuditRecord, load_validated_index_and_palette
from shdowing.ai2bmp.canonical.ids import make_sample_id
from shdowing.ai2bmp.canonical.mappings import (
    apply_contiguous_used_index_map,
    apply_unique_rgb_map,
    build_contiguous_used_index_mapping,
    build_unique_rgb_mapping,
)
from shdowing.ai2bmp.constants import DATASET_VERSION, SCHEMA_VERSION
from shdowing.ai2bmp.io_utils import ensure_dir, save_index_bmp, save_rgb_png, sha256_file, write_json


@dataclass
class CanonicalArtifacts:
    sample_id: str
    design_dir: Path
    canonical_dir: Path
    audit: BmpAuditRecord
    index_map: np.ndarray
    palette: np.ndarray
    rgb: np.ndarray
    contiguous_used_index_map: np.ndarray
    contiguous_used_mapping: dict[str, Any]
    unique_rgb_map: np.ndarray
    unique_rgb_mapping: dict[str, Any]
    roundtrip_mismatch_count: int


def _palette_json_tables(
    audit: BmpAuditRecord, palette: np.ndarray, used: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    total_pixels = sum(audit.used_index_histogram.values())
    full_entries = []
    for i in range(len(palette)):
        rgb = palette[i].tolist()
        count = audit.used_index_histogram.get(str(i), 0)
        full_entries.append(
            {
                "index": i,
                "rgb": rgb,
                "hex": f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
                "pixel_count": count,
                "percentage": round(100.0 * count / total_pixels, 6) if total_pixels else 0.0,
                "used": i in set(int(x) for x in used.tolist()),
            }
        )

    used_entries = []
    for idx in used.tolist():
        idx = int(idx)
        rgb = palette[idx].tolist()
        count = audit.used_index_histogram.get(str(idx), 0)
        used_entries.append(
            {
                "original_index": idx,
                "rgb": rgb,
                "hex": f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
                "pixel_count": count,
                "percentage": round(100.0 * count / total_pixels, 6) if total_pixels else 0.0,
            }
        )

    unique_entries = []
    seen: set[tuple[int, int, int]] = set()
    for idx in used.tolist():
        rgb_t = tuple(int(x) for x in palette[int(idx)])
        if rgb_t in seen:
            continue
        seen.add(rgb_t)
        count = sum(
            audit.used_index_histogram.get(str(i), 0)
            for i in used.tolist()
            if tuple(int(x) for x in palette[int(i)]) == rgb_t
        )
        unique_entries.append(
            {
                "rgb": list(rgb_t),
                "hex": f"#{rgb_t[0]:02X}{rgb_t[1]:02X}{rgb_t[2]:02X}",
                "pixel_count": count,
                "percentage": round(100.0 * count / total_pixels, 6) if total_pixels else 0.0,
                "original_indices": [
                    int(i)
                    for i in used.tolist()
                    if tuple(int(x) for x in palette[int(i)]) == rgb_t
                ],
            }
        )

    return (
        {"entries": full_entries, "declared_count": audit.declared_palette_entry_count},
        {"entries": used_entries, "used_index_count": audit.used_index_count},
        {"entries": unique_entries, "unique_rgb_count": audit.unique_used_rgb_count},
    )


def generate_canonical_artifacts(
    source_bmp: Path,
    output_root: Path,
    *,
    input_root: Path | None = None,
) -> CanonicalArtifacts:
    source_bmp = Path(source_bmp)
    output_root = Path(output_root)
    audit, index_map, palette = load_validated_index_and_palette(
        source_bmp, relative_to=input_root
    )
    sample_id = make_sample_id(source_bmp.stem, audit.sha256)
    design_dir = ensure_dir(output_root / "designs" / sample_id)
    canonical_dir = ensure_dir(design_dir / "canonical")

    target_bmp = canonical_dir / "target_original.bmp"
    shutil.copy2(source_bmp, target_bmp)
    copied_hash = sha256_file(target_bmp)
    if copied_hash != audit.sha256:
        raise RuntimeError("Byte copy SHA-256 mismatch")

    rgb = index_to_rgb(index_map, palette)
    save_rgb_png(canonical_dir / "target_rgb.png", rgb)
    np.save(canonical_dir / "original_index_map.npy", index_map)

    used = np.unique(index_map)
    _, contiguous_mapping = build_contiguous_used_index_mapping(used)
    contiguous_map = apply_contiguous_used_index_map(index_map, contiguous_mapping)
    np.save(canonical_dir / "contiguous_used_index_map.npy", contiguous_map)
    write_json(canonical_dir / "contiguous_used_index_mapping.json", contiguous_mapping)

    unique_mapping, _ = build_unique_rgb_mapping(used, palette)
    unique_map = apply_unique_rgb_map(index_map, unique_mapping)
    np.save(canonical_dir / "contiguous_unique_rgb_map.npy", unique_map)
    write_json(canonical_dir / "contiguous_unique_rgb_mapping.json", unique_mapping)

    full_pal, used_pal, uniq_pal = _palette_json_tables(audit, palette, used)
    write_json(canonical_dir / "palette_full.json", full_pal)
    write_json(canonical_dir / "palette_used_indices.json", used_pal)
    write_json(canonical_dir / "palette_unique_rgb.json", uniq_pal)

    reconstructed = index_to_rgb(index_map, palette)
    save_rgb_png(canonical_dir / "reconstructed_rgb.png", reconstructed)
    diff = np.any(reconstructed != rgb, axis=-1).astype(np.uint8) * 255
    save_index_bmp(canonical_dir / "roundtrip_diff.bmp", diff)
    roundtrip_mismatch_count = int(np.sum(diff > 0))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "sample_id": sample_id,
        "original_filename": audit.filename,
        "original_relative_path": audit.relative_path,
        "source_bmp_path": audit.source_path,
        "source_sha256": audit.sha256,
        **audit.to_dict(),
        "roundtrip_mismatch_count": roundtrip_mismatch_count,
    }
    write_json(canonical_dir / "metadata.json", metadata)

    return CanonicalArtifacts(
        sample_id=sample_id,
        design_dir=design_dir,
        canonical_dir=canonical_dir,
        audit=audit,
        index_map=index_map,
        palette=palette,
        rgb=rgb,
        contiguous_used_index_map=contiguous_map,
        contiguous_used_mapping=contiguous_mapping,
        unique_rgb_map=unique_map,
        unique_rgb_mapping=unique_mapping,
        roundtrip_mismatch_count=roundtrip_mismatch_count,
    )

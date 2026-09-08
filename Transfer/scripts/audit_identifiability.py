#!/usr/bin/env python3
"""Audit training pairs for identifiability issues."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shadow_dataset.geometry import find_candidate_components
from shadow_dataset.image_io import load_rgb_exact
from shadow_dataset.io_utils import read_jsonl

CHUNK_SIZE = 8192
FAMILY_9_RE = re.compile(r"^9 \d")


def _sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _find_duplicates(records: list[dict]) -> list[list[str]]:
    hash_to_stems: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rec in records:
        inp = Path(rec["input_path"])
        tgt = Path(rec["target_path"])
        if not inp.exists() or not tgt.exists():
            continue
        key = (_sha256(inp), _sha256(tgt))
        hash_to_stems[key].append(inp.stem)
    return [stems for stems in hash_to_stems.values() if len(stems) > 1]


def _find_family_9(records: list[dict]) -> list[dict]:
    results = []
    for rec in records:
        stem = Path(rec["input_path"]).stem
        if FAMILY_9_RE.match(stem):
            results.append({"stem": stem, "split": rec.get("split", "train")})
    return results


def _cross_split_hashes(metadata_dir: Path) -> dict[str, list[dict]]:
    """Detect identical input hashes appearing in multiple splits."""
    split_files = ["train_images.jsonl", "val_images.jsonl", "test_images.jsonl"]
    hash_to_entries: dict[str, list[dict]] = defaultdict(list)

    for split_file in split_files:
        path = metadata_dir / split_file
        if not path.exists():
            continue
        split_name = split_file.replace("_images.jsonl", "")
        for rec in read_jsonl(path):
            inp = Path(rec["input_path"])
            if not inp.exists():
                continue
            h = _sha256(inp)
            hash_to_entries[h].append({"stem": inp.stem, "split": split_name, "source_id": rec.get("source_id")})

    conflicts = []
    for h, entries in hash_to_entries.items():
        splits = {e["split"] for e in entries}
        if len(splits) > 1:
            conflicts.append({"input_hash": h, "entries": entries, "splits": sorted(splits)})
    return {"cross_split_conflicts": conflicts, "count": len(conflicts)}


def _component_shape_conflicts(records: list[dict], *, max_check: int = 50) -> list[dict]:
    """Flag pairs with unusual component count / size mismatches (basic)."""
    conflicts: list[dict] = []
    for rec in records[:max_check]:
        inp_path = Path(rec["input_path"])
        if not inp_path.exists():
            continue
        try:
            input_rgb, _ = load_rgb_exact(inp_path)
            from shadow_dataset.geometry import candidate_mask_from_input

            cand = candidate_mask_from_input(input_rgb)
            comps = find_candidate_components(cand, connectivity=8)
            sizes = sorted([c.pixel_count for c in comps], reverse=True)
            if len(sizes) >= 2 and sizes[0] > 10 * max(sizes[1], 1):
                conflicts.append({
                    "stem": inp_path.stem,
                    "source_id": rec.get("source_id"),
                    "component_sizes": sizes,
                    "reason": "dominant_component_ratio",
                })
        except OSError:
            continue
    return conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit training data for identifiability issues.")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.metadata_dir / "train_images.jsonl")

    duplicate_groups = _find_duplicates(records)
    family_9 = _find_family_9(records)
    cross_split = _cross_split_hashes(args.metadata_dir)
    comp_conflicts = _component_shape_conflicts(records)

    summary = {
        "total_pairs": len(records),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": sum(len(g) for g in duplicate_groups),
        "family_9_count": len(family_9),
        "cross_split_hash_conflicts": cross_split["count"],
        "component_shape_conflicts": len(comp_conflicts),
    }

    with open(args.out / "duplicate_hashes.json", "w", encoding="utf-8") as f:
        json.dump(duplicate_groups, f, indent=2)

    with open(args.out / "family_9_report.json", "w", encoding="utf-8") as f:
        json.dump(family_9, f, indent=2)

    with open(args.out / "cross_split_hashes.json", "w", encoding="utf-8") as f:
        json.dump(cross_split, f, indent=2)

    with open(args.out / "component_shape_conflicts.json", "w", encoding="utf-8") as f:
        json.dump(comp_conflicts, f, indent=2)

    with open(args.out / "identifiability_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Audit complete. {summary}")


if __name__ == "__main__":
    main()

"""Duplicate detection and design grouping."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from shdowing.ai2bmp.bmp.validate import validate_bmp_file


def normalize_filename(stem: str) -> str:
    s = re.sub(r"\(\d+\)$", "", stem.strip())
    s = re.sub(r"\s+", " ", s)
    return s


def content_hash_index_and_palette(index_map: np.ndarray, palette: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(index_map.tobytes())
    h.update(palette.tobytes())
    return h.hexdigest()


def build_duplicate_report(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sha_groups: dict[str, list[str]] = {}
    content_groups: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []

    for rec in records:
        sha = rec.get("sha256", "")
        sha_groups.setdefault(sha, []).append(rec["filename"])
        ch = rec.get("content_hash", "")
        if ch:
            content_groups.setdefault(ch, []).append(rec["filename"])

    group_id = 0
    for sha, files in sha_groups.items():
        if len(files) < 2:
            continue
        rows.append(
            {
                "group_type": "exact_duplicate_group",
                "approved_group_id": f"grp_{group_id:04d}",
                "confidence": 1.0,
                "evidence": {"sha256": sha, "files": files},
            }
        )
        group_id += 1

    for ch, files in content_groups.items():
        if len(files) < 2:
            continue
        if len(set(files)) == len(files):
            rows.append(
                {
                    "group_type": "near_duplicate_candidate_group",
                    "approved_group_id": f"grp_{group_id:04d}",
                    "confidence": 0.8,
                    "evidence": {"content_hash": ch, "files": files},
                }
            )
            group_id += 1

    return rows


def assign_design_group_id(filename: str, sha256: str, duplicate_rows: list[dict[str, Any]]) -> str:
    for row in duplicate_rows:
        files = row.get("evidence", {}).get("files", [])
        if filename in files:
            return row["approved_group_id"]
    return f"grp_single_{sha256[:8]}"

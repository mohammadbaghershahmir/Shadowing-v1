"""Collision-safe sample and crop identifiers."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sanitize_sample_id(stem: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    if not base:
        base = "sample"
    candidate = base[:120]
    if candidate not in used:
        used.add(candidate)
        return candidate
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base[:100]}_{digest}"
    while candidate in used:
        digest = hashlib.sha256((candidate + digest).encode()).hexdigest()[:4]
        candidate = f"{base[:100]}_{digest}"
    used.add(candidate)
    return candidate


def sample_id_from_path(path: Path, input_root: Path, used: set[str]) -> str:
    rel = path.relative_to(input_root)
    stem = "__".join(rel.with_suffix("").parts)
    return sanitize_sample_id(stem, used)

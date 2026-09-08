"""Collision-safe sample IDs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sanitize_stem(stem: str, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "design"
    return cleaned[:max_len]


def make_sample_id(stem: str, sha256: str) -> str:
    short = sha256[:8]
    return f"{sanitize_stem(stem)}_{short}"

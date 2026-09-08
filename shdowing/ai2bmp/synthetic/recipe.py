"""Deterministic corruption recipes."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CorruptionRecipe:
    variant_id: str
    family: str
    name: str
    seed: int
    operations: list[dict[str, Any]] = field(default_factory=list)
    difficulty: str = "medium"
    default_dense_supervision: bool = True
    use_for_confidence_or_robustness: bool = False
    status: str = "applicable"  # applicable | not_applicable | skipped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_seed(global_seed: int, sample_id: str, variant_id: str) -> int:
    payload = f"{global_seed}|{sample_id}|{variant_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

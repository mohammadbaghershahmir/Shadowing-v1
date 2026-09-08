"""Deterministic group-aware dataset splitting."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Sequence

import numpy as np

from shadow_dataset.constants import SCHEMA_VERSION
from shadow_dataset.types import ImageRecord, ValidationResult, path_to_str


def extract_group_id(stem: str, group_regex: str | None) -> str:
    """Extract a group/family id from a filename stem.

    If ``group_regex`` is None, the stem itself is the group.
    The regex must contain a named group ``group`` or use capture group 1.
    """
    if not group_regex:
        return stem
    match = re.search(group_regex, stem)
    if match is None:
        return stem
    if "group" in match.groupdict():
        value = match.group("group")
        return value if value else stem
    if match.lastindex and match.lastindex >= 1:
        value = match.group(1)
        return value if value else stem
    return stem


def _stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def assign_splits(
    results: Sequence[ValidationResult],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    group_regex: str | None = None,
) -> dict[str, str]:
    """Assign each usable image stem to a split.

    Returns a mapping stem -> split name. Entire groups stay in one split.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio}+{val_ratio}+{test_ratio}={total}"
        )
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative")

    # Group stems.
    groups: dict[str, list[str]] = defaultdict(list)
    for result in results:
        group_id = extract_group_id(result.stem, group_regex)
        groups[group_id].append(result.stem)

    # Deterministic group order by hashed group id under seed.
    rng = np.random.default_rng(seed)
    group_ids = sorted(groups.keys(), key=lambda g: (_stable_hash_int(f"{seed}:{g}"), g))
    # Shuffle with seeded RNG for reproducibility.
    order = np.arange(len(group_ids))
    rng.shuffle(order)
    ordered_groups = [group_ids[i] for i in order.tolist()]

    n = len(ordered_groups)
    if n == 0:
        return {}

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Ensure all groups are assigned; fix rounding on test.
    if n_train + n_val > n:
        overflow = n_train + n_val - n
        if n_val >= overflow:
            n_val -= overflow
        else:
            n_train -= overflow - n_val
            n_val = 0
    n_test = n - n_train - n_val

    # Prefer at least one sample in each non-zero-ratio split when possible.
    if n >= 3 and train_ratio > 0 and val_ratio > 0 and test_ratio > 0:
        if n_train == 0:
            n_train = 1
            if n_test > n_val:
                n_test -= 1
            else:
                n_val -= 1
        if n_val == 0:
            n_val = 1
            if n_train > n_test:
                n_train -= 1
            else:
                n_test -= 1
        if n_test == 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
        # Reconcile.
        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        n_test = n - n_train - n_val

    split_of_group: dict[str, str] = {}
    idx = 0
    for _ in range(n_train):
        split_of_group[ordered_groups[idx]] = "train"
        idx += 1
    for _ in range(n_val):
        split_of_group[ordered_groups[idx]] = "val"
        idx += 1
    for _ in range(n_test):
        split_of_group[ordered_groups[idx]] = "test"
        idx += 1

    stem_to_split: dict[str, str] = {}
    for group_id, stems in groups.items():
        split = split_of_group[group_id]
        for stem in stems:
            stem_to_split[stem] = split
    return stem_to_split


def build_image_records(
    results: Sequence[ValidationResult],
    stem_to_split: dict[str, str],
    *,
    group_regex: str | None = None,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for result in results:
        split = stem_to_split[result.stem]
        group_id = extract_group_id(result.stem, group_regex)
        records.append(
            ImageRecord(
                schema_version=SCHEMA_VERSION,
                source_id=result.stem,
                stem=result.stem,
                input_path=result.input_path,
                target_path=result.target_path,
                width=result.width,
                height=result.height,
                split=split,
                class_counts={
                    "200": result.class_200_count,
                    "150": result.class_150_count,
                    "100": result.class_100_count,
                },
                candidate_pixel_count=result.candidate_pixel_count,
                group_id=group_id,
            )
        )
    # Stable order: by split then stem.
    split_order = {"train": 0, "val": 1, "test": 2}
    records.sort(key=lambda r: (split_order.get(r.split, 9), r.stem))
    return records


def splits_summary(records: Sequence[ImageRecord]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for record in records:
        summary.setdefault(record.split, []).append(record.stem)
    for key in summary:
        summary[key] = sorted(summary[key])
    return summary

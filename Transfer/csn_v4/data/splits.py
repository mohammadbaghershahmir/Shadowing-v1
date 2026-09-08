"""Group-aware scene splitting for CSN-V4."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from csn_v4.data.paths import sha256_file


@dataclass
class SplitResult:
    train_scenes: list[dict]
    val_scenes: list[dict]
    policy: str
    group_key_field: str
    n_groups: int
    n_train_groups: int
    n_val_groups: int
    warnings: list[str]


def load_scene_metadata(full_dir: Path) -> dict:
    meta_path = full_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json under {full_dir}")
    with meta_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def discover_scenes_from_source(source_root: Path) -> list[dict]:
    samples = source_root / "samples"
    if not samples.is_dir():
        raise FileNotFoundError(f"samples/ not found under {source_root}")

    scenes: list[dict] = []
    for art in sorted(samples.iterdir()):
        if not art.is_dir():
            continue
        full_dir = art / "full"
        if not full_dir.is_dir():
            continue
        src = full_dir / "source_bw.bmp"
        sem = full_dir / "target_semantic.bmp"
        if not src.exists() or not sem.exists():
            continue
        meta = load_scene_metadata(full_dir)
        group_id = meta.get("group_id")
        if not group_id:
            raise ValueError(f"metadata.json missing group_id for {art.name}")
        file_hash = sha256_file(src)
        meta_hash = meta.get("sha256")
        if meta_hash and meta_hash != file_hash:
            # metadata.json may be stale; file hash is authoritative
            pass
        scenes.append({
            "scene_id": str(meta.get("sample_id") or art.name),
            "group_id": str(group_id),
            "family_id": str(meta.get("family_id") or group_id),
            "design_id": str(meta.get("design_id") or group_id),
            "source_sha256": file_hash,
            "source_bw_path": str(src.resolve()),
            "target_semantic_path": str(sem.resolve()),
            "artifact_dir": str(art.resolve()),
            "full_artifact_dir": str(full_dir.resolve()),
            "image_width": int(meta.get("width", 0)),
            "image_height": int(meta.get("height", 0)),
        })
    if not scenes:
        raise FileNotFoundError(f"No valid scenes under {samples}")
    return scenes


def _build_split_units(scenes: list[dict]) -> tuple[list[dict], list[str]]:
    """Merge scenes sharing group_id OR source_sha256 into atomic split units."""
    warnings: list[str] = []
    by_group: dict[str, list[dict]] = {}
    for s in scenes:
        by_group.setdefault(s["group_id"], []).append(s)

    # Union groups linked by identical source_sha256
    hash_to_groups: dict[str, set[str]] = {}
    for gid, members in by_group.items():
        for s in members:
            hash_to_groups.setdefault(s["source_sha256"], set()).add(gid)

    parent = {gid: gid for gid in by_group}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def unite(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for gids in hash_to_groups.values():
        gids = sorted(gids)
        if len(gids) > 1:
            warnings.append(
                f"source_sha256 links groups {gids}; forcing same split"
            )
            for g in gids[1:]:
                unite(gids[0], g)

    clusters: dict[str, list[dict]] = {}
    for gid, members in by_group.items():
        root = find(gid)
        clusters.setdefault(root, []).extend(members)

    units: list[dict] = []
    for root, members in clusters.items():
        gids = sorted({m["group_id"] for m in members})
        fids = sorted({m["family_id"] for m in members})
        units.append({
            "unit_id": root,
            "group_ids": gids,
            "family_ids": fids,
            "source_sha256": members[0]["source_sha256"],
            "scene_ids": [m["scene_id"] for m in members],
        })
    return units, warnings


def assign_scene_splits(
    scenes: list[dict],
    *,
    seed: int = 42,
    val_fraction: float = 0.20,
) -> SplitResult:
    units, warnings = _build_split_units(scenes)
    n = len(units)
    if n == 0:
        raise ValueError("No scenes to split")

    if n == 1:
        return SplitResult(
            train_scenes=[dict(s, split="train") for s in scenes],
            val_scenes=[],
            policy="intra_scene_spatial_only",
            group_key_field="group_id",
            n_groups=1,
            n_train_groups=1,
            n_val_groups=0,
            warnings=warnings + ["Only one independent group; true held-out validation unavailable."],
        )

    rng = np.random.default_rng(seed)
    order = list(range(n))
    rng.shuffle(order)

    if 2 <= n <= 4:
        policy = "leave_one_group_out"
        val_unit_ids = {units[order[0]]["unit_id"]}
    else:
        policy = "group_holdout_fraction"
        n_val = max(1, int(round(n * val_fraction)))
        val_unit_ids = {units[i]["unit_id"] for i in order[:n_val]}

    val_unit_ids_frozen = frozenset(val_unit_ids)
    train_scenes, val_scenes = [], []
    for s in scenes:
        row = dict(s)
        # Scene belongs to val if ANY of its group's units are val
        unit_for_scene = next(
            u for u in units
            if s["scene_id"] in u["scene_ids"]
        )
        if unit_for_scene["unit_id"] in val_unit_ids_frozen:
            row["split"] = "val"
            val_scenes.append(row)
        else:
            row["split"] = "train"
            train_scenes.append(row)

    if not train_scenes:
        raise ValueError(
            f"Split produced 0 train scenes (n_units={n}, policy={policy}). "
            "Adjust val_fraction or add more scenes."
        )

    return SplitResult(
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        policy=policy,
        group_key_field="group_id",
        n_groups=n,
        n_train_groups=n - len(val_unit_ids),
        n_val_groups=len(val_unit_ids),
        warnings=warnings,
    )


def assert_no_split_leakage(train_scenes: list[dict], val_scenes: list[dict]) -> None:
    train_groups = {s["group_id"] for s in train_scenes}
    val_groups = {s["group_id"] for s in val_scenes}
    group_overlap = train_groups & val_groups
    if group_overlap:
        raise AssertionError(f"Split leakage: shared group_id {group_overlap}")

    train_families = {s["family_id"] for s in train_scenes}
    val_families = {s["family_id"] for s in val_scenes}
    family_overlap = train_families & val_families
    if family_overlap:
        raise AssertionError(f"Split leakage: shared family_id {family_overlap}")

    train_hashes = {s["source_sha256"] for s in train_scenes}
    val_hashes = {s["source_sha256"] for s in val_scenes}
    hash_overlap = train_hashes & val_hashes
    if hash_overlap:
        raise AssertionError(f"Split leakage: shared source_sha256 {hash_overlap}")

    train_ids = {s["scene_id"] for s in train_scenes}
    val_ids = {s["scene_id"] for s in val_scenes}
    if train_ids & val_ids:
        raise AssertionError(f"Split leakage: shared scene_id {train_ids & val_ids}")

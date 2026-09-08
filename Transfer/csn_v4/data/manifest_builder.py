"""Deterministic manifest generation for CSN-V4."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from csn_v4.data.paths import rel_path_if_under, sha256_file, sha256_manifest
from csn_v4.data.splits import SplitResult, assert_no_split_leakage
from csn_v4.data.targets import load_gray_bmp
from csn_v4.geometry.d4 import D4Transform
from csn_v4.geometry.tile_spec import (
    CORE_SIZE,
    HALO,
    INPUT_SIZE,
    build_tile_spec,
    core_starts,
    iter_core_tiles,
    normalize_core_coords,
    tile_uid,
)


@dataclass
class ManifestBuildConfig:
    seed: int = 42
    offgrid_tiles_per_scene: int = 4
    spatial_offgrid_per_scene: int = 8
    subscene_sizes: tuple[int, ...] = (512, 768, 1024)
    subscenes_per_size: int = 2
    input_mode: str = "bw"


@dataclass
class ManifestBuildResult:
    scenes_path: Path
    train_capacity_path: Path
    val_spatial_path: Path
    val_subscenes_path: Path
    val_holdout_path: Path
    eval_full_path: Path
    summary: dict[str, Any] = field(default_factory=dict)


def _canonical_input_origins(fw: int, fh: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for cy in core_starts(fh):
        for cx in core_starts(fw):
            out.add((cx - HALO, cy - HALO))
    return out


def _sample_offgrid_origins(
    fw: int,
    fh: int,
    rng: np.random.Generator,
    n: int,
    *,
    region: str,
) -> list[tuple[int, int, str]]:
    canonical = _canonical_input_origins(fw, fh)
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, str]] = []
    max_attempts = max(n * 80, 200)
    for attempt in range(max_attempts):
        if region == "interior":
            max_ox = max(1, fw - INPUT_SIZE)
            max_oy = max(1, fh - INPUT_SIZE)
            if max_ox <= 0 or max_oy <= 0:
                ox, oy = 0, 0
            else:
                ox = int(rng.integers(0, max_ox + 1))
                oy = int(rng.integers(0, max_oy + 1))
        elif region == "edge":
            ox = int(rng.choice([0, max(0, fw - INPUT_SIZE)]))
            oy = int(rng.integers(0, max(1, fh - INPUT_SIZE + 1)))
        else:
            ox = int(rng.choice([0, max(0, fw - INPUT_SIZE)]))
            oy = int(rng.choice([0, max(0, fh - INPUT_SIZE)]))
        ox = max(0, min(ox, max(0, fw - INPUT_SIZE)))
        oy = max(0, min(oy, max(0, fh - INPUT_SIZE)))
        if (ox, oy) in seen:
            continue
        if (ox, oy) in canonical:
            continue
        if ox % 384 == 0 and oy % 384 == 0:
            continue
        seen.add((ox, oy))
        candidates.append((ox, oy, region))
        if len(candidates) >= n:
            break
    return candidates


def _tile_row(
    scene: dict,
    *,
    transform_id: int,
    core_x: int,
    core_y: int,
    fw: int,
    fh: int,
    split: str,
    manifest_kind: str,
    sample_id: str,
    sampling_region: str = "grid",
    diagnostic: bool = False,
    dataset_root: Path | None = None,
) -> dict:
    spec = build_tile_spec(fw, fh, core_x, core_y, transform_id=transform_id, scene_id=scene["scene_id"])
    canonical = _canonical_input_origins(fw, fh)
    overlap = (spec.input_x, spec.input_y) in canonical
    row = {
        "sample_id": sample_id,
        "scene_id": scene["scene_id"],
        "group_id": scene.get("group_id"),
        "family_id": scene.get("family_id"),
        "design_id": scene.get("design_id"),
        "split": split,
        "manifest_kind": manifest_kind,
        "diagnostic": diagnostic,
        "transform_id": transform_id,
        "core_origin_x": spec.core_x,
        "core_origin_y": spec.core_y,
        "core_w": spec.core_w,
        "core_h": spec.core_h,
        "source_window_x": spec.input_x,
        "source_window_y": spec.input_y,
        "source_window_size": INPUT_SIZE,
        "model_input_size": INPUT_SIZE,
        "scale_factor": 1.0,
        "sampling_region": sampling_region,
        "image_width": fw,
        "image_height": fh,
        "input_bbox_norm": spec.input_bbox_norm.tolist(),
        "core_bbox_norm": spec.core_bbox_norm.tolist(),
        "random_seed": None,
        "canonical_overlap": overlap,
        "record_type": "tile",
        "tile_uid": tile_uid(scene["scene_id"], transform_id, spec.core_x, spec.core_y),
        "input_mode": "bw",
    }
    if dataset_root is not None:
        row["source_bw_path"] = rel_path_if_under(Path(scene["source_bw_path"]), dataset_root)
        row["target_semantic_path"] = rel_path_if_under(Path(scene["target_semantic_path"]), dataset_root)
    return row


def build_capacity_tiles(scenes: list[dict], cfg: ManifestBuildConfig) -> list[dict]:
    rows: list[dict] = []
    seen_uids: set[str] = set()
    rng = np.random.default_rng(cfg.seed)
    regions = ["interior", "interior", "edge", "corner"]
    for scene in scenes:
        bw = load_gray_bmp(scene["source_bw_path"])
        fh0, fw0 = bw.shape
        for k in range(8):
            tfw, tfh = D4Transform(k).transformed_wh(fw0, fh0)
            for spec in iter_core_tiles(tfw, tfh, transform_id=k, scene_id=scene["scene_id"]):
                uid = tile_uid(scene["scene_id"], k, spec.core_x, spec.core_y)
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
                rows.append(_tile_row(
                    scene, transform_id=k, core_x=spec.core_x, core_y=spec.core_y,
                    fw=tfw, fh=tfh,
                    split=scene["split"], manifest_kind="train_capacity",
                    sample_id=f"cap__{scene['scene_id']}__d4_{k}__{spec.core_x}_{spec.core_y}",
                    sampling_region="grid",
                ))
            n_off = max(1, cfg.offgrid_tiles_per_scene // 4)
            for region in regions:
                origins = _sample_offgrid_origins(tfw, tfh, rng, n_off, region=region)
                for idx, (ox, oy, reg) in enumerate(origins):
                    cx, cy = ox + HALO, oy + HALO
                    if cx + CORE_SIZE > tfw or cy + CORE_SIZE > tfh:
                        continue
                    uid = tile_uid(scene["scene_id"], k, cx, cy)
                    if uid in seen_uids:
                        continue
                    seen_uids.add(uid)
                    rows.append(_tile_row(
                        scene, transform_id=k, core_x=cx, core_y=cy,
                        fw=tfw, fh=tfh,
                        split=scene["split"], manifest_kind="train_capacity",
                        sample_id=f"cap_off__{scene['scene_id']}__d4_{k}__{reg}__{idx}__{ox}_{oy}",
                        sampling_region=reg,
                    ))
    return rows


def build_spatial_diagnostic_tiles(scenes: list[dict], cfg: ManifestBuildConfig) -> list[dict]:
    rows: list[dict] = []
    seen_uids: set[str] = set()
    rng = np.random.default_rng(cfg.seed + 1)
    for scene in scenes:
        bw = load_gray_bmp(scene["source_bw_path"])
        fh0, fw0 = bw.shape
        for k in range(8):
            tfw, tfh = D4Transform(k).transformed_wh(fw0, fh0)
            origins = _sample_offgrid_origins(tfw, tfh, rng, cfg.spatial_offgrid_per_scene, region="interior")
            for idx, (ox, oy, reg) in enumerate(origins):
                cx, cy = ox + HALO, oy + HALO
                if cx + CORE_SIZE > tfw or cy + CORE_SIZE > tfh:
                    continue
                uid = tile_uid(scene["scene_id"], k, cx, cy)
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
                rows.append(_tile_row(
                    scene, transform_id=k, core_x=cx, core_y=cy,
                    fw=tfw, fh=tfh,
                    split="val", manifest_kind="val_spatial_fixed",
                    sample_id=f"diag_off__{scene['scene_id']}__d4_{k}__{idx}__{ox}_{oy}",
                    sampling_region=reg, diagnostic=True,
                ))
    return rows


def build_spatial_diagnostic_subscenes(scenes: list[dict], cfg: ManifestBuildConfig) -> list[dict]:
    rows: list[dict] = []
    rng = np.random.default_rng(cfg.seed + 1)
    for scene in scenes:
        bw = load_gray_bmp(scene["source_bw_path"])
        fh0, fw0 = bw.shape
        for k in range(8):
            tfw, tfh = D4Transform(k).transformed_wh(fw0, fh0)
            for size in cfg.subscene_sizes:
                sw = min(size, tfw)
                sh = min(size, tfh)
                for i in range(cfg.subscenes_per_size):
                    ox = int(rng.integers(0, max(1, tfw - sw + 1)))
                    oy = int(rng.integers(0, max(1, tfh - sh + 1)))
                    rows.append({
                        "sample_id": f"diag_sub__{scene['scene_id']}__d4_{k}__{size}__{i}__{ox}_{oy}",
                        "scene_id": scene["scene_id"],
                        "group_id": scene.get("group_id"),
                        "family_id": scene.get("family_id"),
                        "design_id": scene.get("design_id"),
                        "split": "val",
                        "manifest_kind": "val_spatial_subscenes",
                        "record_type": "subscene",
                        "diagnostic": True,
                        "validation_suite": "val_intra_scene_spatial_holdout",
                        "transform_id": k,
                        "source_window_x": ox,
                        "source_window_y": oy,
                        "source_window_size": sw,
                        "native_width": sw,
                        "native_height": sh,
                        "model_input_size": sw,
                        "scale_factor": 1.0,
                        "resize_mode": "none",
                        "sampling_region": "subscene",
                        "random_seed": cfg.seed + 1,
                        "input_mode": "bw",
                    })
    return rows


def build_scene_holdout_records(scenes: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for scene in scenes:
        for k in range(8):
            rows.append({
                "sample_id": f"holdout__{scene['scene_id']}__d4_{k}",
                "scene_id": scene["scene_id"],
                "group_id": scene.get("group_id"),
                "family_id": scene.get("family_id"),
                "design_id": scene.get("design_id"),
                "split": "val",
                "manifest_kind": "val_scene_holdout",
                "diagnostic": False,
                "validation_suite": "val_unseen_scene_native",
                "transform_id": k,
                "source_window_x": 0,
                "source_window_y": 0,
                "source_window_size": scene.get("image_width", 0),
                "model_input_size": None,
                "scale_factor": 1.0,
                "sampling_region": "full",
                "random_seed": 42,
                "source_bw_path": scene["source_bw_path"],
                "target_semantic_path": scene["target_semantic_path"],
                "full_artifact_dir": scene["full_artifact_dir"],
            })
    return rows


def build_eval_full(scenes: list[dict], *, include_train: bool = False) -> list[dict]:
    out: list[dict] = []
    for s in scenes:
        if not include_train and s.get("split") != "val":
            continue
        out.append({
            "sample_id": s["scene_id"],
            "scene_id": s["scene_id"],
            "split": s.get("split", "val"),
            "source_bw_path": s["source_bw_path"],
            "target_semantic_path": s["target_semantic_path"],
            "full_artifact_dir": s["full_artifact_dir"],
        })
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _dedupe_sample_ids(rows: list[dict]) -> None:
    seen: set[str] = set()
    for row in rows:
        sid = row["sample_id"]
        if sid in seen:
            raise ValueError(f"Duplicate sample_id: {sid}")
        seen.add(sid)


def build_all_manifests(
    split: SplitResult,
    dataset_root: Path,
    cfg: ManifestBuildConfig,
) -> ManifestBuildResult:
    assert_no_split_leakage(split.train_scenes, split.val_scenes)
    all_scenes = split.train_scenes + split.val_scenes
    for s in all_scenes:
        if not s.get("image_width") or not s.get("image_height"):
            bw = load_gray_bmp(s["source_bw_path"])
            s["image_height"], s["image_width"] = bw.shape

    manifest_dir = dataset_root / "manifests"
    scenes_path = manifest_dir / "scenes.jsonl"
    train_cap_path = manifest_dir / "train_capacity_tiles_bw.jsonl"
    val_spatial_path = manifest_dir / "val_spatial_fixed_bw.jsonl"
    val_subscenes_path = manifest_dir / "val_spatial_subscenes_bw.jsonl"
    val_holdout_path = manifest_dir / "val_scene_holdout_bw.jsonl"
    eval_full_path = manifest_dir / "eval_full_images.jsonl"

    _write_jsonl(scenes_path, all_scenes)

    train_tiles = build_capacity_tiles(split.train_scenes, cfg)
    _dedupe_sample_ids(train_tiles)
    _write_jsonl(train_cap_path, train_tiles)

    spatial_tiles = build_spatial_diagnostic_tiles(split.train_scenes, cfg)
    _dedupe_sample_ids(spatial_tiles)
    _write_jsonl(val_spatial_path, spatial_tiles)

    subscenes = build_spatial_diagnostic_subscenes(split.train_scenes, cfg)
    if subscenes:
        _dedupe_sample_ids(subscenes)
    _write_jsonl(val_subscenes_path, subscenes)

    holdout = build_scene_holdout_records(split.val_scenes) if split.val_scenes else []
    if holdout:
        _dedupe_sample_ids(holdout)
    _write_jsonl(val_holdout_path, holdout)

    # Full-image eval includes train scenes when no val holdout exists (capacity mode).
    eval_scenes = split.val_scenes if split.val_scenes else split.train_scenes
    eval_full = build_eval_full(eval_scenes, include_train=not split.val_scenes)
    _write_jsonl(eval_full_path, eval_full)

    summary = {
        "policy": split.policy,
        "group_key_field": split.group_key_field,
        "n_groups": split.n_groups,
        "n_train_groups": split.n_train_groups,
        "n_val_groups": split.n_val_groups,
        "warnings": split.warnings,
        "n_scenes_train": len(split.train_scenes),
        "n_scenes_val": len(split.val_scenes),
        "n_train_capacity_tiles": len(train_tiles),
        "n_val_spatial_tiles": len(spatial_tiles),
        "n_val_subscenes": len(subscenes),
        "n_val_holdout": len(holdout),
        "n_eval_full": len(eval_full),
        "true_unseen_scene_validation": len(split.val_scenes) > 0 and split.policy != "intra_scene_spatial_only",
        "validation_scope": "unseen_scene_native" if split.val_scenes else "intra_scene_spatial_only",
        "fingerprints": {
            "scenes.jsonl": sha256_manifest(scenes_path),
            "train_capacity_tiles_bw.jsonl": sha256_manifest(train_cap_path),
            "val_spatial_fixed_bw.jsonl": sha256_manifest(val_spatial_path),
            "val_spatial_subscenes_bw.jsonl": sha256_manifest(val_subscenes_path),
            "val_scene_holdout_bw.jsonl": sha256_manifest(val_holdout_path),
        },
    }
    return ManifestBuildResult(
        scenes_path=scenes_path,
        train_capacity_path=train_cap_path,
        val_spatial_path=val_spatial_path,
        val_subscenes_path=val_subscenes_path,
        val_holdout_path=val_holdout_path,
        eval_full_path=eval_full_path,
        summary=summary,
    )

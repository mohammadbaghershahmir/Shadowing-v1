"""Build one CSN-V4 training sample from a tile manifest record."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from csn_v4.data.paths import resolve_dataset_path
from csn_v4.data.targets import build_tile_targets, load_gray_bmp
from csn_v4.geometry.d4 import D4Transform
from csn_v4.geometry.extract import extract_tile_bundle, global_cache_key
from csn_v4.geometry.tile_spec import build_tile_spec, tile_uid


def _resolve_scene_path(dataset_root: Path, scene_rec: dict, key: str, fallback_name: str, full_dir: Path) -> Path:
    raw = scene_rec.get(key)
    if raw:
        p = resolve_dataset_path(raw, dataset_root)
        if p.is_file():
            return p
    p = full_dir / fallback_name
    if not p.is_file():
        raise FileNotFoundError(f"Missing {fallback_name} for scene {scene_rec.get('scene_id')} at {p}")
    return p


def load_raw_scene(dataset_root: Path, scene_rec: dict) -> dict[str, np.ndarray | None]:
    full_dir = resolve_dataset_path(scene_rec.get("full_artifact_dir", ""), dataset_root)
    if not full_dir.is_dir():
        art = resolve_dataset_path(scene_rec.get("artifact_dir", ""), dataset_root)
        full_dir = art / "full" if (art / "full").is_dir() else art
    src_path = _resolve_scene_path(dataset_root, scene_rec, "source_bw_path", "source_bw.bmp", full_dir)
    sem_path = _resolve_scene_path(dataset_root, scene_rec, "target_semantic_path", "target_semantic.bmp", full_dir)
    src = load_gray_bmp(src_path)
    sem = load_gray_bmp(sem_path)
    trans_p = full_dir / "transition_mask.bmp"
    black_p = full_dir / "black_lock.bmp"
    valid_p = full_dir / "valid_mask.bmp"
    shade_p = full_dir / "shade_mask.bmp"
    level_p = full_dir / "shade_level_id.bmp"
    return {
        "source_bw": src,
        "target_semantic": sem,
        "transition_mask": load_gray_bmp(trans_p) if trans_p.exists() else None,
        "black_lock": load_gray_bmp(black_p) if black_p.exists() else None,
        "valid_mask": load_gray_bmp(valid_p) if valid_p.exists() else np.ones_like(src) * 255,
        "shade_mask": load_gray_bmp(shade_p) if shade_p.exists() else ((sem > 0) & (sem < 255)).astype(np.uint8) * 255,
        "shade_level_id": load_gray_bmp(level_p) if level_p.exists() else None,
    }


def transform_scene(raw: dict[str, np.ndarray | None], transform_id: int) -> dict[str, np.ndarray | None]:
    t = D4Transform(transform_id)
    return {
        "source_bw": t.apply(raw["source_bw"]),
        "target_semantic": t.apply(raw["target_semantic"]),
        "transition_mask": t.apply(raw["transition_mask"]) if raw["transition_mask"] is not None else None,
        "black_lock": t.apply(raw["black_lock"]) if raw["black_lock"] is not None else None,
        "valid_mask": t.apply(raw["valid_mask"]),
        "shade_mask": t.apply(raw["shade_mask"]),
        "shade_level_id": t.apply(raw["shade_level_id"]) if raw["shade_level_id"] is not None else None,
    }


def build_tile_sample_arrays(
    rec: dict,
    scene: dict,
    transformed: dict[str, np.ndarray | None],
    *,
    context_size: int = 1024,
    global_long_side: int = 1024,
    halo_loss_weight: float = 1.0,
) -> dict[str, Any]:
    scene_id = rec["scene_id"]
    k = int(rec["transform_id"])
    fh, fw = transformed["source_bw"].shape[:2]
    spec = build_tile_spec(
        fw, fh, int(rec["core_origin_x"]), int(rec["core_origin_y"]),
        transform_id=k, scene_id=scene_id,
    )
    bundle = extract_tile_bundle(
        transformed["source_bw"], transformed["target_semantic"], spec,
        context_size=context_size,
        global_long_side=global_long_side,
        halo_loss_weight=halo_loss_weight,
        transition_mask=transformed["transition_mask"],
        black_lock=transformed["black_lock"],
        valid_mask=transformed["valid_mask"],
        shade_mask=transformed["shade_mask"],
        shade_level_id=transformed["shade_level_id"],
    )
    targets = build_tile_targets(
        bundle["local_semantic"],
        bundle["local_transition"],
        bundle["local_black_lock"],
        bundle["local_valid"],
        bundle["local_shade_mask"],
        bundle["local_shade_level_id"],
        bundle["supervision_weights"],
    )
    meta = {
        "sample_id": rec["sample_id"],
        "tile_uid": rec.get("tile_uid") or tile_uid(scene_id, k, spec.core_x, spec.core_y),
        "global_cache_key": global_cache_key(scene_id, k),
        "transform_id": k,
        "split": rec.get("split", "train"),
        "scene_id": scene_id,
        "letterbox_meta": bundle["letterbox_meta"],
    }
    arrays = {
        "local_bw": bundle["local_bw"],
        "context_bw": bundle["context_bw"],
        "full_bw": bundle["full_bw"],
        "crop_coords_norm": bundle["crop_coords_norm"],
    }
    return {"arrays": arrays, "targets": targets, "meta": meta}


def arrays_to_training_sample(arrays: dict[str, np.ndarray], targets: dict[str, torch.Tensor], meta: dict) -> dict:
    sample = {
        "local_bw": torch.from_numpy(arrays["local_bw"].astype(np.float32) / 255.0).unsqueeze(0),
        "context_bw": torch.from_numpy(arrays["context_bw"].astype(np.float32) / 255.0).unsqueeze(0),
        "full_bw": torch.from_numpy(arrays["full_bw"].astype(np.float32) / 255.0).unsqueeze(0),
        "crop_coords_norm": torch.from_numpy(arrays["crop_coords_norm"].astype(np.float32)),
        "letterbox_meta": meta["letterbox_meta"],
        "tile_uid": meta["tile_uid"],
        "global_cache_key": meta["global_cache_key"],
        "transform_id": meta["transform_id"],
        "split": meta["split"],
        **targets,
    }
    return sample

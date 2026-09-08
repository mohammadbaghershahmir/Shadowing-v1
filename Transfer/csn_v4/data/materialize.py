"""Materialize CSN-V4 tile samples to compressed NPZ on disk."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from csn_v4.data.paths import sha256_file
from csn_v4.data.tile_sample import (
    build_tile_sample_arrays,
    load_raw_scene,
    transform_scene,
)

MATERIALIZE_VERSION = "v1"


def _safe_filename(sample_id: str) -> str:
    return sample_id.replace("/", "_").replace("\\", "_").replace(":", "_")


def save_materialized_npz(path: Path, arrays: dict[str, np.ndarray], targets: dict[str, torch.Tensor], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "local_bw": arrays["local_bw"].astype(np.uint8),
        "context_bw": arrays["context_bw"].astype(np.uint8),
        "full_bw": arrays["full_bw"].astype(np.uint8),
        "crop_coords_norm": arrays["crop_coords_norm"].astype(np.float32),
        "target_class": targets["target_class"].numpy().astype(np.int8),
        "shade_mask": targets["shade_mask"].numpy().astype(np.uint8),
        "transition_mask": targets["transition_mask"].numpy().astype(np.uint8),
        "black_lock": targets["black_lock"].numpy().astype(np.uint8),
        "valid_mask": targets["valid_mask"].numpy().astype(np.uint8),
        "where_target": targets["where_target"].numpy().astype(np.float16),
        "level_target": targets["level_target"].numpy().astype(np.int8),
        "affinity_targets": targets["affinity_targets"].numpy().astype(np.float16),
        "affinity_valid": targets["affinity_valid"].numpy().astype(np.uint8),
        "supervision_weights": targets["supervision_weights"].numpy().astype(np.float16),
        "meta_json": np.array(json.dumps({
            "sample_id": meta["sample_id"],
            "tile_uid": meta["tile_uid"],
            "global_cache_key": meta["global_cache_key"],
            "transform_id": meta["transform_id"],
            "split": meta["split"],
            "scene_id": meta["scene_id"],
            "letterbox_meta": meta["letterbox_meta"],
            "materialize_version": MATERIALIZE_VERSION,
        }), dtype=object),
    }
    np.savez_compressed(path, **payload)


def load_materialized_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta_json"]))
        targets = {
            "target_class": torch.from_numpy(z["target_class"].astype(np.int64)),
            "shade_mask": torch.from_numpy(z["shade_mask"] > 0),
            "transition_mask": torch.from_numpy(z["transition_mask"].astype(np.uint8)),
            "black_lock": torch.from_numpy(z["black_lock"] > 0),
            "valid_mask": torch.from_numpy(z["valid_mask"] > 0),
            "where_target": torch.from_numpy(z["where_target"].astype(np.float32)),
            "level_target": torch.from_numpy(z["level_target"].astype(np.int64)),
            "affinity_targets": torch.from_numpy(z["affinity_targets"].astype(np.float32)),
            "affinity_valid": torch.from_numpy(z["affinity_valid"].astype(np.float32)),
            "supervision_weights": torch.from_numpy(z["supervision_weights"].astype(np.float32)),
        }
        sample = {
            "local_bw": torch.from_numpy(z["local_bw"].astype(np.float32) / 255.0).unsqueeze(0),
            "context_bw": torch.from_numpy(z["context_bw"].astype(np.float32) / 255.0).unsqueeze(0),
            "full_bw": torch.from_numpy(z["full_bw"].astype(np.float32) / 255.0).unsqueeze(0),
            "crop_coords_norm": torch.from_numpy(z["crop_coords_norm"].astype(np.float32)),
            "letterbox_meta": meta["letterbox_meta"],
            "tile_uid": meta["tile_uid"],
            "global_cache_key": meta["global_cache_key"],
            "transform_id": int(meta["transform_id"]),
            "split": meta["split"],
            **targets,
        }
    return sample


def materialize_manifest(
    tiles: list[dict],
    scenes: dict[str, dict],
    out_dir: Path,
    *,
    dataset_root: Path,
    context_size: int = 1024,
    global_long_side: int = 1024,
    halo_loss_weight: float = 1.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    dataset_root = dataset_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in tiles:
        grouped[(rec["scene_id"], int(rec["transform_id"]))].append(rec)

    written = 0
    skipped = 0
    index_rows: list[dict] = []

    for (scene_id, transform_id), recs in tqdm(
        grouped.items(), desc="materialize scenes", unit="scene×d4",
    ):
        scene = scenes[scene_id]
        raw = load_raw_scene(dataset_root, scene)
        transformed = transform_scene(raw, transform_id)
        for rec in recs:
            npz_path = out_dir / f"{_safe_filename(rec['sample_id'])}.npz"
            if npz_path.is_file() and not overwrite:
                skipped += 1
                index_rows.append({
                    "sample_id": rec["sample_id"],
                    "path": str(npz_path.relative_to(dataset_root)),
                    "tile_uid": rec.get("tile_uid"),
                    "split": rec.get("split"),
                })
                continue
            built = build_tile_sample_arrays(
                rec, scene, transformed,
                context_size=context_size,
                global_long_side=global_long_side,
                halo_loss_weight=halo_loss_weight,
            )
            save_materialized_npz(npz_path, built["arrays"], built["targets"], built["meta"])
            written += 1
            index_rows.append({
                "sample_id": rec["sample_id"],
                "path": str(npz_path.relative_to(dataset_root)),
                "tile_uid": built["meta"]["tile_uid"],
                "split": built["meta"]["split"],
            })

    with index_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in index_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "materialize_version": MATERIALIZE_VERSION,
        "out_dir": str(out_dir.resolve()),
        "n_tiles": len(tiles),
        "n_written": written,
        "n_skipped": skipped,
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
    }
    with (out_dir / "materialize_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary

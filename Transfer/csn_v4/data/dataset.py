"""CSN-V4 training dataset: scene-first D4 + TileSpec extraction."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from csn_v4.data.paths import resolve_dataset_path
from csn_v4.data.targets import build_tile_targets, load_gray_bmp
from csn_v4.geometry.d4 import D4Transform
from csn_v4.geometry.extract import extract_tile_bundle, global_cache_key
from csn_v4.geometry.tile_spec import build_tile_spec, tile_uid


class _BoundedCache(OrderedDict):
    def __init__(self, max_items: int = 32):
        super().__init__()
        self.max_items = max_items

    def set(self, key, value) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.max_items:
            self.popitem(last=False)


class CSNV4TrainDataset(Dataset):
    def __init__(
        self,
        tiles_manifest: str | Path,
        scenes_manifest: str | Path,
        dataset_root: str | Path,
        *,
        input_mode: str = "bw",
        context_size: int = 1024,
        global_long_side: int = 1024,
        halo_loss_weight: float = 1.0,
        eval_mode: bool = False,
        cache_size: int = 32,
        split_filter: str | None = "train",
    ):
        if input_mode != "bw":
            raise ValueError(f"CSN-V4 train dataset supports input_mode='bw' only, got {input_mode!r}")
        self.dataset_root = Path(dataset_root)
        self.input_mode = input_mode
        self.context_size = context_size
        self.global_long_side = global_long_side
        self.halo_loss_weight = halo_loss_weight
        self.eval_mode = eval_mode
        self.split_filter = None if eval_mode else split_filter

        scenes_path = resolve_dataset_path(scenes_manifest, self.dataset_root)
        tiles_path = resolve_dataset_path(tiles_manifest, self.dataset_root)
        if not scenes_path.is_file():
            raise FileNotFoundError(f"Scenes manifest not found: {scenes_path}")
        if not tiles_path.is_file():
            raise FileNotFoundError(f"Tiles manifest not found: {tiles_path}")

        scenes: dict[str, dict] = {}
        with scenes_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if "scene_id" not in rec:
                        raise KeyError(f"Scene record missing scene_id in {scenes_path}")
                    scenes[rec["scene_id"]] = rec

        self.tiles: list[dict] = []
        with tiles_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("record_type") == "subscene" or rec.get("sampling_region") == "subscene":
                    continue
                if "core_origin_x" not in rec or "core_origin_y" not in rec:
                    raise KeyError(
                        f"Tile record missing core_origin_x/y in {tiles_path}: "
                        f"sample_id={rec.get('sample_id')!r}"
                    )
                sid = rec.get("scene_id")
                if sid not in scenes:
                    raise KeyError(f"Tile references unknown scene_id={sid!r} in {tiles_path}")
                if self.split_filter is not None and rec.get("split") != self.split_filter:
                    continue
                rec["_scene"] = scenes[sid]
                self.tiles.append(rec)

        if not self.tiles:
            raise ValueError(f"No tiles loaded from {tiles_path} with split_filter={self.split_filter!r}")

        self._raw_scene_cache: _BoundedCache = _BoundedCache(cache_size)
        self._transform_cache: _BoundedCache = _BoundedCache(cache_size * 8)

    def __len__(self) -> int:
        return len(self.tiles)

    def _resolve_scene_path(self, scene_rec: dict, key: str, fallback_name: str, full_dir: Path) -> Path:
        raw = scene_rec.get(key)
        if raw:
            p = resolve_dataset_path(raw, self.dataset_root)
            if p.is_file():
                return p
        p = full_dir / fallback_name
        if not p.is_file():
            raise FileNotFoundError(f"Missing {fallback_name} for scene {scene_rec.get('scene_id')} at {p}")
        return p

    def _load_raw_scene(self, scene_id: str, scene_rec: dict) -> dict:
        if scene_id in self._raw_scene_cache:
            return self._raw_scene_cache[scene_id]
        full_dir = resolve_dataset_path(scene_rec.get("full_artifact_dir", ""), self.dataset_root)
        if not full_dir.is_dir():
            art = resolve_dataset_path(scene_rec.get("artifact_dir", ""), self.dataset_root)
            full_dir = art / "full" if (art / "full").is_dir() else art
        src_path = self._resolve_scene_path(scene_rec, "source_bw_path", "source_bw.bmp", full_dir)
        sem_path = self._resolve_scene_path(scene_rec, "target_semantic_path", "target_semantic.bmp", full_dir)
        src = load_gray_bmp(src_path)
        sem = load_gray_bmp(sem_path)
        trans_p = full_dir / "transition_mask.bmp"
        black_p = full_dir / "black_lock.bmp"
        valid_p = full_dir / "valid_mask.bmp"
        shade_p = full_dir / "shade_mask.bmp"
        level_p = full_dir / "shade_level_id.bmp"
        raw = {
            "source_bw": src,
            "target_semantic": sem,
            "transition_mask": load_gray_bmp(trans_p) if trans_p.exists() else None,
            "black_lock": load_gray_bmp(black_p) if black_p.exists() else None,
            "valid_mask": load_gray_bmp(valid_p) if valid_p.exists() else np.ones_like(src) * 255,
            "shade_mask": load_gray_bmp(shade_p) if shade_p.exists() else ((sem > 0) & (sem < 255)).astype(np.uint8) * 255,
            "shade_level_id": load_gray_bmp(level_p) if level_p.exists() else None,
        }
        self._raw_scene_cache.set(scene_id, raw)
        return raw

    def _load_transformed_scene(self, scene_id: str, scene_rec: dict, k: int) -> dict:
        key = (scene_id, k)
        if key in self._transform_cache:
            return self._transform_cache[key]
        raw = self._load_raw_scene(scene_id, scene_rec)
        t = D4Transform(k)
        transformed = {
            "source_bw": t.apply(raw["source_bw"]),
            "target_semantic": t.apply(raw["target_semantic"]),
            "transition_mask": t.apply(raw["transition_mask"]) if raw["transition_mask"] is not None else None,
            "black_lock": t.apply(raw["black_lock"]) if raw["black_lock"] is not None else None,
            "valid_mask": t.apply(raw["valid_mask"]),
            "shade_mask": t.apply(raw["shade_mask"]),
            "shade_level_id": t.apply(raw["shade_level_id"]) if raw["shade_level_id"] is not None else None,
        }
        self._transform_cache.set(key, transformed)
        return transformed

    def __getitem__(self, idx: int) -> dict:
        rec = self.tiles[idx]
        scene = rec["_scene"]
        scene_id = rec["scene_id"]
        k = int(rec["transform_id"])
        raw = self._load_transformed_scene(scene_id, scene, k)

        fh, fw = raw["source_bw"].shape[:2]
        spec = build_tile_spec(
            fw, fh, int(rec["core_origin_x"]), int(rec["core_origin_y"]),
            transform_id=k, scene_id=scene_id,
        )

        bundle = extract_tile_bundle(
            raw["source_bw"], raw["target_semantic"], spec,
            context_size=self.context_size,
            global_long_side=self.global_long_side,
            halo_loss_weight=self.halo_loss_weight,
            transition_mask=raw["transition_mask"],
            black_lock=raw["black_lock"],
            valid_mask=raw["valid_mask"],
            shade_mask=raw["shade_mask"],
            shade_level_id=raw["shade_level_id"],
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

        gkey = global_cache_key(scene_id, k)
        sample = {
            "local_bw": torch.from_numpy(bundle["local_bw"].astype(np.float32) / 255.0).unsqueeze(0),
            "context_bw": torch.from_numpy(bundle["context_bw"].astype(np.float32) / 255.0).unsqueeze(0),
            "full_bw": torch.from_numpy(bundle["full_bw"].astype(np.float32) / 255.0).unsqueeze(0),
            "crop_coords_norm": torch.from_numpy(bundle["crop_coords_norm"]),
            "letterbox_meta": bundle["letterbox_meta"],
            "tile_uid": rec.get("tile_uid") or rec.get("sample_id") or tile_uid(scene_id, k, spec.core_x, spec.core_y),
            "global_cache_key": gkey,
            "transform_id": k,
            "split": rec.get("split", "train"),
            **targets,
        }
        if "local_rgb" in sample or "local_onehot" in sample:
            raise RuntimeError("BW dataset must not expose target-derived inputs")
        return sample

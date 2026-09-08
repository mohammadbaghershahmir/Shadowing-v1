"""Data loading with D4 augmentation, global cache, and grouped sampling."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler

from shadow_dataset.dataset import ShadowCropDataset
from shadow_model.augment import apply_dihedral_sample
from shadow_model.backbone_local import IMAGENET_MEAN, IMAGENET_STD
from shadow_model.global_cache import load_cached_tokens


class ShadowTrainDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        global_cache_dir: str | Path | None = None,
        dino_cfg: Any | None = None,
        require_cache: bool = False,
        augment_dihedral: bool = False,
        allowed_dihedral: list[int] | None = None,
        seed: int = 42,
    ):
        self.base = ShadowCropDataset(manifest_path)
        self.global_cache_dir = Path(global_cache_dir) if global_cache_dir else None
        self.dino_cfg = dino_cfg
        self.require_cache = require_cache
        self.augment_dihedral = augment_dihedral
        self.allowed_dihedral = allowed_dihedral or [0]
        self.seed = seed
        self.epoch = 0
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def _dihedral_k(self, index: int, baked: int = 0) -> int:
        if self.augment_dihedral:
            rng = random.Random(self.seed + self.epoch * 1000003 + index)
            return rng.choice(self.allowed_dihedral)
        return int(baked)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        baked = int(sample.get("transform_id", 0))
        k = self._dihedral_k(index, baked=baked)
        if k != 0:
            sample = apply_dihedral_sample(sample, k)

        sample["local_input_rgb_norm"] = (sample["local_input_rgb"] - self.mean) / self.std

        if self.global_cache_dir is not None and self.dino_cfg is not None:
            tokens, patch_valid, meta = load_cached_tokens(
                self.global_cache_dir,
                sample["source_id"],
                k,
                self.dino_cfg,
                require=self.require_cache,
            )
            sample["global_tokens"] = torch.from_numpy(tokens)
            sample["global_valid_mask"] = torch.from_numpy(patch_valid)
            sample["letterbox_meta"] = {
                "scale": meta["scale"],
                "offset_x": meta["offset_x"],
                "offset_y": meta["offset_y"],
                "content_width": meta["content_width"],
                "content_height": meta["content_height"],
                "image_width": meta["image_width"],
                "image_height": meta["image_height"],
            }
        elif self.require_cache:
            raise FileNotFoundError("Global cache required but cache_dir not configured")

        sample["dihedral_k"] = k
        return sample


class GroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        batch_size: int,
        max_per_source: int = 2,
        seed: int = 42,
        drop_last: bool = False,
        accum_size: int = 1,
    ):
        self.records = records
        self.batch_size = batch_size
        self.max_per_source = max_per_source
        self.seed = seed
        self.drop_last = drop_last
        self.accum_size = max(1, accum_size)
        self.epoch = 0
        self._num_batches: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._num_batches = None

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.records)))
        rng.shuffle(indices)
        deferred: list[int] = []
        batches: list[list[int]] = []
        current: list[int] = []
        source_counts: dict[str, int] = {}

        def flush() -> None:
            nonlocal current, source_counts
            if current:
                batches.append(current)
            current = []
            source_counts = {}

        pool = deferred + indices
        deferred = []
        i = 0
        while i < len(pool) or current:
            if i >= len(pool):
                if len(current) < self.batch_size and not self.drop_last and current:
                    flush()
                break
            idx = pool[i]
            i += 1
            source = self.records[idx].get("source_id", str(idx))
            if source_counts.get(source, 0) >= self.max_per_source:
                deferred.append(idx)
                continue
            current.append(idx)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(current) == self.batch_size:
                flush()
        if deferred:
            pool2 = deferred
            i = 0
            while i < len(pool2):
                idx = pool2[i]
                i += 1
                source = self.records[idx].get("source_id", str(idx))
                if source_counts.get(source, 0) >= self.max_per_source and len(current) < self.batch_size:
                    continue
                current.append(idx)
                source_counts[source] = source_counts.get(source, 0) + 1
                if len(current) == self.batch_size:
                    flush()
            if current and not self.drop_last:
                flush()
        rng.shuffle(batches)
        self._num_batches = len(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self._build_batches():
            yield batch

    def __len__(self) -> int:
        if self._num_batches is None:
            self._build_batches()
        return self._num_batches or 0

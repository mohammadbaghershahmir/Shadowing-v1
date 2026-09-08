"""Create CSN-V4 train/eval dataset (materialized or runtime)."""
from __future__ import annotations

from pathlib import Path

from csn_v4.config import CSNV4Config
from csn_v4.data.dataset import CSNV4TrainDataset
from csn_v4.data.materialized_dataset import CSNV4MaterializedDataset
from csn_v4.data.paths import resolve_dataset_path


def materialized_index_for_manifest(cfg: CSNV4Config, manifest_rel: str) -> Path | None:
    if not cfg.data.prefer_materialized:
        return None
    base = cfg.data.materialized_root
    if not base:
        return None
    stem = Path(manifest_rel).stem
    index = resolve_dataset_path(f"{base}/{stem}/index.jsonl", cfg.data.dataset_root)
    return index if index.is_file() else None


def create_tile_dataset(
    cfg: CSNV4Config,
    manifest_rel: str,
    *,
    split_filter: str | None = "train",
    eval_mode: bool = False,
):
    root = Path(cfg.data.dataset_root)
    index = materialized_index_for_manifest(cfg, manifest_rel)
    if index is not None:
        sf = None if eval_mode else split_filter
        return CSNV4MaterializedDataset(index, root, split_filter=sf)

    return CSNV4TrainDataset(
        root / manifest_rel,
        root / cfg.data.scenes,
        root,
        input_mode=cfg.data.input_mode,
        context_size=cfg.model.context_size,
        global_long_side=cfg.model.global_long_side,
        halo_loss_weight=cfg.data.halo_loss_weight,
        eval_mode=eval_mode,
        split_filter=None if eval_mode else split_filter,
    )

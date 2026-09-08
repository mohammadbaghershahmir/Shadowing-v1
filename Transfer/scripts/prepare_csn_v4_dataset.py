#!/usr/bin/env python3
"""Build CSN-V4 manifests from a source dataset tree."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v4.config import config_to_dict, load_config
from csn_v4.data.manifest_builder import ManifestBuildConfig, build_all_manifests
from csn_v4.data.splits import assign_scene_splits, discover_scenes_from_source

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare CSN-V4 dataset manifests")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v4_capacity.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Split only; no manifest files written")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    t0 = time.perf_counter()

    cfg = load_config(args.config)
    dataset_root = args.dataset_root.resolve()
    source_root = args.source_dataset.resolve()

    LOGGER.info("Discovering scenes from %s …", source_root)
    scenes = discover_scenes_from_source(source_root)
    LOGGER.info("Found %d scenes", len(scenes))

    split = assign_scene_splits(scenes, seed=args.seed)
    LOGGER.info(
        "Split policy=%s groups=%d train_groups=%d val_groups=%d → %d train / %d val scenes",
        split.policy,
        split.n_groups,
        split.n_train_groups,
        split.n_val_groups,
        len(split.train_scenes),
        len(split.val_scenes),
    )
    for w in split.warnings:
        LOGGER.warning(w)

    build_cfg = ManifestBuildConfig(seed=args.seed, input_mode=cfg.data.input_mode)

    if args.dry_run:
        summary = {
            "dry_run": True,
            "policy": split.policy,
            "group_key_field": split.group_key_field,
            "n_groups": split.n_groups,
            "n_train_groups": split.n_train_groups,
            "n_val_groups": split.n_val_groups,
            "warnings": split.warnings,
            "n_scenes_train": len(split.train_scenes),
            "n_scenes_val": len(split.val_scenes),
            "train_scene_ids": [s["scene_id"] for s in split.train_scenes],
            "val_scene_ids": [s["scene_id"] for s in split.val_scenes],
            "source_dataset": str(source_root),
            "dataset_root": str(dataset_root),
            "seed": args.seed,
            "config": config_to_dict(cfg),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
        LOGGER.info("Dry run complete in %.1fs — no manifests written", summary["elapsed_sec"])
    else:
        LOGGER.info("Building manifests under %s (this may take several minutes) …", dataset_root)
        result = build_all_manifests(split, dataset_root, build_cfg)
        summary = {
            "dry_run": False,
            **result.summary,
            "source_dataset": str(source_root),
            "dataset_root": str(dataset_root),
            "seed": args.seed,
            "manifest_paths": {
                "scenes": str(result.scenes_path),
                "train_capacity": str(result.train_capacity_path),
                "val_spatial": str(result.val_spatial_path),
                "val_subscenes": str(result.val_subscenes_path),
                "val_holdout": str(result.val_holdout_path),
                "eval_full": str(result.eval_full_path),
            },
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
        LOGGER.info(
            "Built in %.1fs: %d capacity tiles, %d spatial tiles, %d subscenes, %d holdout, %d eval_full",
            summary["elapsed_sec"],
            summary["n_train_capacity_tiles"],
            summary["n_val_spatial_tiles"],
            summary.get("n_val_subscenes", 0),
            summary.get("n_val_holdout", 0),
            summary.get("n_eval_full", 0),
        )

    report_dir = dataset_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "dataset_build_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    LOGGER.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

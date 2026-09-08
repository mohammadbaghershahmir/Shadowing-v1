#!/usr/bin/env python3
"""Pre-materialize CSN-V4 tile samples to NPZ (V2-style fast I/O for training)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v4.config import config_to_dict, load_config
from csn_v4.data.materialize import materialize_manifest
from csn_v4.data.paths import resolve_dataset_path

LOGGER = logging.getLogger(__name__)

DEFAULT_MANIFESTS = (
    "manifests/train_capacity_tiles_bw.jsonl",
    "manifests/val_spatial_fixed_bw.jsonl",
)


def _load_tiles(manifest_path: Path, split_filter: str | None) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("record_type") == "subscene" or rec.get("sampling_region") == "subscene":
                continue
            if "core_origin_x" not in rec or "core_origin_y" not in rec:
                continue
            if split_filter is not None and rec.get("split") != split_filter:
                continue
            rows.append(rec)
    return rows


def _load_scenes(scenes_path: Path) -> dict[str, dict]:
    scenes: dict[str, dict] = {}
    with scenes_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                scenes[rec["scene_id"]] = rec
    return scenes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize CSN-V4 tiles to NPZ cache")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v4_generalization.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--manifest", type=str, action="append", default=None, help="Tile manifest rel path (repeatable)")
    parser.add_argument("--split", type=str, default=None, help="Only tiles with this split (default: all in manifest)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    root = Path(args.dataset_root or cfg.data.dataset_root).resolve()
    mat_root = root / cfg.data.materialized_root

    scenes_path = resolve_dataset_path(cfg.data.scenes, root)
    scenes = _load_scenes(scenes_path)
    manifests = args.manifest or list(DEFAULT_MANIFESTS)

    for rel in manifests:
        manifest_path = root / rel
        if not manifest_path.is_file():
            LOGGER.warning("Skip missing manifest: %s", manifest_path)
            continue
        tiles = _load_tiles(manifest_path, args.split)
        if not tiles:
            LOGGER.warning("No tile rows in %s", manifest_path)
            continue
        out_dir = mat_root / Path(rel).stem
        LOGGER.info("Materializing %d tiles from %s → %s", len(tiles), rel, out_dir)
        summary = materialize_manifest(
            tiles, scenes, out_dir,
            dataset_root=root,
            context_size=cfg.model.context_size,
            global_long_side=cfg.model.global_long_side,
            halo_loss_weight=cfg.data.halo_loss_weight,
            overwrite=args.overwrite,
        )
        LOGGER.info(
            "Done %s: written=%d skipped=%d index=%s",
            rel, summary["n_written"], summary["n_skipped"], summary["index"],
        )

    report = {
        "dataset_root": str(root),
        "materialized_root": str(mat_root),
        "manifests": manifests,
        "config": config_to_dict(cfg),
    }
    report_path = mat_root / "materialize_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    LOGGER.info("Wrote %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

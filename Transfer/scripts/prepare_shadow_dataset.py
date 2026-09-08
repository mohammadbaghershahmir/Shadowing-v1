#!/usr/bin/env python3
"""CLI entry point for shadow dataset preparation.

Example (Dataset V2 / overfit coverage)
---------------------------------------
python scripts/prepare_shadow_dataset.py ^
  --input-dir "E:\\Shadowing\\converted_ready" ^
  --target-dir "E:\\Shadowing\\converted_output" ^
  --output-dir "E:\\Shadowing\\dataset_metadata_v2" ^
  --crop-size 512 --valid-margin 0 ^
  --include-spatial-grid --allow-padded-origins --bake-dihedral ^
  --merge-train-val-crops --target-train-crops 10000 ^
  --min-crops-per-image 32 --max-crops-per-image 512 ^
  --grid-stride 256 --overwrite --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shadow_dataset.constants import (
    CROP_IOU_DEDUP_THRESHOLD_V2,
    DEFAULT_CROP_SIZE,
    DEFAULT_EDGE_BAND,
    DEFAULT_GRID_STRIDE,
    DEFAULT_INFERENCE_STRIDE_HINT,
    DEFAULT_MAX_CROPS_PER_IMAGE,
    DEFAULT_MIN_CROPS_PER_IMAGE,
    DEFAULT_PREVIEW_COUNT,
    DEFAULT_SEED,
    DEFAULT_TARGET_TRAIN_CROPS,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VALID_MARGIN,
    DEFAULT_VAL_RATIO,
)
from shadow_dataset.pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_shadow_dataset",
        description=(
            "Validate paired guide/GT images, split them deterministically, "
            "sample shade-aware crops, and write manifests, reports, and QA previews."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument(
        "--valid-margin",
        type=int,
        default=DEFAULT_VALID_MARGIN,
        help="Halo outside supervised region; use 0 for full-crop supervision (V2).",
    )
    parser.add_argument("--inference-stride-hint", type=int, default=DEFAULT_INFERENCE_STRIDE_HINT)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--min-crops-per-image", type=int, default=DEFAULT_MIN_CROPS_PER_IMAGE)
    parser.add_argument("--max-crops-per-image", type=int, default=DEFAULT_MAX_CROPS_PER_IMAGE)
    parser.add_argument("--val-crops-per-image", type=int, default=None)
    parser.add_argument("--preview-count", type=int, default=DEFAULT_PREVIEW_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--group-regex", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-spatial-grid",
        action="store_true",
        help="Enable dense spatial grid + corner/edge forced crops (Dataset V2).",
    )
    parser.add_argument(
        "--allow-padded-origins",
        action="store_true",
        help="Allow crop windows to overhang image borders (pad at load time).",
    )
    parser.add_argument(
        "--bake-dihedral",
        action="store_true",
        help="Emit transform_id=0..7 rows per spatial crop (D4 baked into manifest).",
    )
    parser.add_argument(
        "--merge-train-val-crops",
        action="store_true",
        help="Put train+val image crops into train_crops.jsonl (test stays locked).",
    )
    parser.add_argument("--grid-stride", type=int, default=DEFAULT_GRID_STRIDE)
    parser.add_argument("--edge-band", type=int, default=DEFAULT_EDGE_BAND)
    parser.add_argument(
        "--target-train-crops",
        type=int,
        default=None,
        help=f"Aim for ~N train crop rows after bake (default unset; V2 often {DEFAULT_TARGET_TRAIN_CROPS}).",
    )
    parser.add_argument(
        "--iou-dedup-threshold",
        type=float,
        default=None,
        help=f"Override IoU dedup (V2 default {CROP_IOU_DEDUP_THRESHOLD_V2} when grid on).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input_dir.is_dir():
        parser.error(f"--input-dir does not exist or is not a directory: {args.input_dir}")
    if not args.target_dir.is_dir():
        parser.error(f"--target-dir does not exist or is not a directory: {args.target_dir}")

    config = PipelineConfig(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        valid_margin=args.valid_margin,
        inference_stride_hint=args.inference_stride_hint,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_crops_per_image=args.min_crops_per_image,
        max_crops_per_image=args.max_crops_per_image,
        preview_count=args.preview_count,
        seed=args.seed,
        group_regex=args.group_regex,
        overwrite=args.overwrite,
        val_crops_per_image=args.val_crops_per_image,
        include_spatial_grid=args.include_spatial_grid,
        allow_padded_origins=args.allow_padded_origins,
        bake_dihedral=args.bake_dihedral,
        merge_train_val_crops=args.merge_train_val_crops,
        grid_stride=args.grid_stride,
        edge_band=args.edge_band,
        target_train_crops=args.target_train_crops,
        iou_dedup_threshold=args.iou_dedup_threshold,
    )

    try:
        run_pipeline(config)
    except FileExistsError as exc:
        logging.error("%s", exc)
        return 2
    except Exception:
        logging.exception("Dataset preparation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

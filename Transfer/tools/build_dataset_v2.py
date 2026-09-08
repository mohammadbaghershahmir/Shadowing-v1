#!/usr/bin/env python3
"""Build Dataset V2 pilot samples from large BMP ground-truth images."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_v2.constants import DATASET_NAME, DATASET_VERSION
from dataset_v2.pipeline import PilotConfig, run_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_dataset_v2",
        description=f"Build {DATASET_NAME} pilot artifacts from BMP ground-truth images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="pilot", choices=["pilot"])
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--sample-files", type=str, nargs="*", default=None)
    parser.add_argument("--rotations", type=int, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--include-flips", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode != "pilot":
        raise SystemExit("Only --mode pilot is implemented.")
    if not args.input_dir.is_dir():
        raise SystemExit(f"--input-dir not found: {args.input_dir}")

    cfg = PilotConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        crop_sizes=args.crop_sizes,
        recursive=args.recursive,
        sample_files=args.sample_files,
        rotations=args.rotations,
        include_flips=args.include_flips,
    )

    result = run_pilot(cfg)
    summary = result.validation_report["summary"]
    print(f"{DATASET_NAME} ({DATASET_VERSION}) pilot complete.")
    print(f"  selected samples : {summary['selected_valid_samples']}")
    print(f"  rejected files   : {summary['rejected_files']}")
    print(f"  canonical crops  : {summary['canonical_crops']}")
    print(f"  aug candidates   : {summary['augmentation_candidates']}")
    print(f"  report           : {cfg.output_dir / 'reports' / 'pilot_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

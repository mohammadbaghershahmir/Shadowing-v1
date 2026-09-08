"""Validate palette dataset and persist deterministic split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dinov3_carpet_probe.palette_train.src.dataset import (  # noqa: E402
    load_or_create_split_manifest,
    validate_dataset,
)
from dinov3_carpet_probe.src.io_utils import ensure_dir, load_yaml, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "dinov3_carpet_probe" / "palette_train" / "configs" / "palette_train.yaml",
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--palette-dir", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--max-num-colors", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    image_dir = args.image_dir or Path(cfg["image_dir"])
    palette_dir = args.palette_dir or Path(cfg["palette_dir"])
    manifest_dir = args.manifest_dir or Path(cfg["manifest_dir"])

    records, report = validate_dataset(
        image_dir,
        palette_dir,
        max_num_colors_override=args.max_num_colors if args.max_num_colors is not None else cfg.get("max_num_colors"),
        show_progress=True,
    )
    ensure_dir(manifest_dir)
    report_path = manifest_dir / "validation_report.json"
    write_json(report_path, report.to_dict())

    if report.errors:
        print(f"Validation failed. Report: {report_path}", file=sys.stderr)
        for line in report.errors:
            print(f"ERROR: {line}", file=sys.stderr)
        return 1

    manifest, manifest_path = load_or_create_split_manifest(
        records,
        manifest_dir=manifest_dir,
        split_seed=int(cfg["split_seed"]),
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
    )
    write_json(manifest_dir / "validated_records.json", [record.__dict__ for record in records])
    print(f"Validated {report.paired_count} pairs.")
    print(f"trainable_pairs={report.trainable_count}")
    print(f"excluded_over_12={len(report.excluded_over_cap)}")
    print(f"max_num_colors={report.max_num_colors}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")
    print(f"dataset_id={manifest['dataset_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

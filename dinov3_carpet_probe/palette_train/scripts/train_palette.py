"""Train the palette prediction head on frozen vitl16_lvd features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dinov3_carpet_probe.palette_train.src.trainer import train  # noqa: E402
from dinov3_carpet_probe.src.io_utils import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "dinov3_carpet_probe" / "palette_train" / "configs" / "palette_train.yaml",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--limit-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of training samples to use from the train split.",
    )
    parser.add_argument(
        "--limit-val-samples",
        type=int,
        default=None,
        help="Optional cap on the number of validation samples to use from the val split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for cfg['batch_size'].",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    artifacts = train(
        cfg,
        resume=args.resume,
        limit_train_samples=args.limit_train_samples,
        limit_val_samples=args.limit_val_samples,
    )
    print(f"run_dir={artifacts.run_dir}")
    print(f"checkpoint_last={artifacts.checkpoint_last}")
    print(f"checkpoint_best={artifacts.checkpoint_best}")
    print(f"metrics={artifacts.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

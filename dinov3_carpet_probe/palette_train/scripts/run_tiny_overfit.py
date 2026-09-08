"""Run a tiny-subset overfit gate for palette prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dinov3_carpet_probe.palette_train.src.trainer import train  # noqa: E402
from dinov3_carpet_probe.src.io_utils import load_yaml, read_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "dinov3_carpet_probe" / "palette_train" / "configs" / "palette_train.yaml",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg["epochs"] = int(args.epochs)
    artifacts = train(cfg, limit_train_samples=int(args.num_samples))
    metrics = read_json(artifacts.metrics_path)
    summary = metrics[-1] if metrics else {}
    out_path = artifacts.run_dir / "tiny_overfit_summary.json"
    write_json(out_path, summary)
    print(f"run_dir={artifacts.run_dir}")
    print(f"checkpoint_best={artifacts.checkpoint_best}")
    print(f"tiny_overfit_summary={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

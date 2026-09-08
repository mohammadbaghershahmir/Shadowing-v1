"""Evaluate a trained palette head on a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dinov3_carpet_probe.palette_train.src.checkpointing import load_checkpoint  # noqa: E402
from dinov3_carpet_probe.palette_train.src.dataset import PaletteTrainingDataset, collate_palette_batch  # noqa: E402
from dinov3_carpet_probe.palette_train.src.trainer import _build_dataloader, prepare_training_state, run_eval  # noqa: E402
from dinov3_carpet_probe.src.io_utils import load_yaml, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "dinov3_carpet_probe" / "palette_train" / "configs" / "palette_train.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    model, manifest, splits, _ = prepare_training_state(cfg)
    loader = _build_dataloader(cfg, splits[args.split], training=False)
    load_checkpoint(args.checkpoint, model=model, optimizer=None, scheduler=None, scaler=None, restore_rng=False)
    metrics = run_eval(model, loader, cfg)
    out_path = args.checkpoint.parent / f"eval_{args.split}.json"
    write_json(out_path, metrics)
    print(f"metrics={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

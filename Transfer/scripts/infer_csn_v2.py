#!/usr/bin/env python3
"""Full-image CSN-V2 BMP inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from csn_v2.config import load_config
from csn_v2.factory import load_model_from_checkpoint
from csn_v2.inference.tiled import predict_full_image_bmp
from csn_v2.renderer import save_output_bmp, validate_palette


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN-V2 full-image inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v2_overfit.yaml"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_model_from_checkpoint(args.checkpoint, args.config, device=device)
    cfg = load_config(args.config)
    gray = predict_full_image_bmp(
        model,
        str(args.input),
        device,
        tile_size=cfg.inference.tile_size,
        halo=cfg.inference.halo,
        stride=cfg.inference.stride,
        context_size=cfg.data.context_size,
        global_long_side=cfg.inference.global_long_side,
    )
    ok, uniq = validate_palette(gray)
    print(f"Palette valid={ok} unique={uniq}")
    save_output_bmp(args.output, gray)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

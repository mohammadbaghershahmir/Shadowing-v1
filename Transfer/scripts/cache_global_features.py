#!/usr/bin/env python3
"""Precompute frozen DINO global tokens for all source images."""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shadow_model.config import load_config
from shadow_model.global_cache import cache_global_tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache global DINO tokens.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dihedral-variants", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    variants = list(range(min(8, max(1, args.dihedral_variants))))
    if cfg.data.allowed_dihedral:
        variants = [k for k in variants if k in cfg.data.allowed_dihedral]
    cache_global_tokens(
        metadata_dir=cfg.data.metadata_dir,
        cache_dir=cfg.dino.cache_dir,
        dino_cfg=cfg.dino,
        dihedral_variants=variants,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

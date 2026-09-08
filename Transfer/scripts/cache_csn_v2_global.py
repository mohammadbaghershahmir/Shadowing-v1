#!/usr/bin/env python3
"""Pre-cache global DINO tokens for CSN-V2 (placeholder — tokens computed on-the-fly in v1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v2.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN-V2 global token cache")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v2_overfit.yaml"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cache_dir = Path(cfg.model.dino.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"CSN-V2 uses on-the-fly global DINO in training. Cache dir ready: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Smoke validation — same path as training eval, limited tiles + tqdm.

Does NOT train. Use this to verify validation metrics / masks / progress bars.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader

from csn_v4.config import load_config
from csn_v4.factory import build_model, load_model_from_checkpoint
from csn_v4.training.metrics_logger import MetricsLogger
from scripts.train_csn_v4 import (
    VAL_SPATIAL_MANIFEST,
    _metrics_to_dict,
    collate_batch,
    evaluate_tiles,
    make_tile_loader,
    resolve_amp,
    write_validation_report,
)

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN-V4 smoke validation (training-eval path)")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v4_generalization.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional .pt; else random-init model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tiles", type=int, default=5, help="Tiles per manifest (default 5)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 = safer on Windows)")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/smoke_val_v4"))
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--skip-spatial", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    cfg.data.dataloader_workers = max(0, int(args.workers))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype, _scaler = resolve_amp(cfg, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        LOGGER.info("Loading checkpoint %s", args.checkpoint)
        model, state = load_model_from_checkpoint(args.checkpoint, args.config, device=device)
        LOGGER.info("Loaded optimizer_step=%s", state.get("optimizer_step"))
    else:
        LOGGER.info("No checkpoint — building fresh model (metrics will be low; path check only)")
        model = build_model(cfg).to(device)

    model.eval()

    loaders = {}
    if not args.skip_capacity:
        loaders["capacity"] = make_tile_loader(cfg, cfg.data.train_tiles, shuffle=False)
    if not args.skip_spatial:
        loaders["spatial"] = make_tile_loader(cfg, VAL_SPATIAL_MANIFEST, shuffle=False)

    summary: dict = {}
    report: dict = {"eval_tile_limit": args.max_tiles, "smoke": True}

    for name, loader in loaders.items():
        if loader is None:
            LOGGER.warning("Skip %s — loader unavailable", name)
            continue
        LOGGER.info("Running %s eval on up to %d tiles (tqdm) …", name, args.max_tiles)
        results = evaluate_tiles(
            model, loader, device, amp_dtype,
            max_tiles=args.max_tiles,
            desc=f"eval {name}",
            fast_metrics=True,
        )
        agg = _metrics_to_dict(results)
        report[name] = agg
        report[f"{name}_n_tiles"] = len(results)
        summary[f"score_{'capacity' if name == 'capacity' else 'spatial'}"] = agg.get("checkpoint_score_v4")
        if name == "spatial" or (args.skip_spatial and name == "capacity"):
            summary.update(agg)
        prefix = "cap_" if name == "capacity" else "spat_"
        summary.update({f"{prefix}{k}": v for k, v in agg.items()})
        LOGGER.info(
            "%s | n=%d | global_miou=%.4f core=%.4f halo=%.4f boundary=%.4f score=%.4f",
            name, len(results),
            agg.get("global_miou", float("nan")),
            agg.get("core_miou", float("nan")),
            agg.get("halo_miou", float("nan")),
            agg.get("boundary_miou", float("nan")),
            agg.get("checkpoint_score_v4", float("nan")),
        )

    write_validation_report(args.out_dir, optimizer_step=0, summary=summary, report=report)
    metrics_logger = MetricsLogger(args.out_dir, plot_every_steps=10**9)
    metrics_logger.log_eval(0, summary, optimizer_step=0, report=report)

    out_json = args.out_dir / "smoke_val_summary.json"
    out_json.write_text(json.dumps({"summary": summary, "report": report}, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s and %s", out_json, args.out_dir / "validation_latest.txt")
    LOGGER.info("CSV: %s", args.out_dir / "metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

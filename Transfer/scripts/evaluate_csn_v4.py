#!/usr/bin/env python3
"""Evaluate CSN-V4 on validation manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from csn_v4.config import load_config
from csn_v4.data.targets import load_gray_bmp
from csn_v4.evaluation.evaluator import CSNV4Evaluator
from csn_v4.factory import load_model_from_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CSN-V4")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v4_capacity.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--suite", type=str, default="val_scene_holdout", choices=[
        "val_scene_holdout", "val_spatial_fixed", "train_capacity",
    ])
    parser.add_argument("--output-dir", type=Path, default=Path("eval_v4"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.data.dataset_root = str(args.dataset_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, cfg, device)
    evaluator = CSNV4Evaluator(model, cfg, device)

    manifest_map = {
        "val_scene_holdout": "manifests/val_scene_holdout_bw.jsonl",
        "val_spatial_fixed": "manifests/val_spatial_fixed_bw.jsonl",
        "train_capacity": "manifests/train_capacity_tiles_bw.jsonl",
    }
    root = Path(cfg.data.dataset_root)
    rows = []
    with (root / manifest_map[args.suite]).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    scenes: dict[str, dict] = {}
    with (root / "manifests/scenes.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                scenes[rec["scene_id"]] = rec

    results = []
    if args.suite == "val_scene_holdout":
        seen = set()
        for row in rows:
            sid = row["scene_id"]
            if sid in seen:
                continue
            seen.add(sid)
            scene = scenes[sid]
            full_dir = Path(scene["full_artifact_dir"])
            src = load_gray_bmp(scene["source_bw_path"])
            sem = load_gray_bmp(scene["target_semantic_path"])
            trans = load_gray_bmp(full_dir / "transition_mask.bmp") if (full_dir / "transition_mask.bmp").exists() else None
            valid = load_gray_bmp(full_dir / "valid_mask.bmp") if (full_dir / "valid_mask.bmp").exists() else None
            black = load_gray_bmp(full_dir / "black_lock.bmp") if (full_dir / "black_lock.bmp").exists() else None
            res = evaluator.evaluate_all_orientations(
                src, sem, scene_id=sid,
                transition_mask=trans, valid_mask=valid, black_lock=black,
            )
            results.append(res.to_dict())
    else:
        print(f"Suite {args.suite}: tile-level eval not yet wired in CLI; use holdout for now.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"eval_{args.suite}.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump([r.to_dict() if hasattr(r, "to_dict") else r for r in results], fh, indent=2)
    print(f"Wrote {out} ({len(results)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

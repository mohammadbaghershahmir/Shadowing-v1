#!/usr/bin/env python3
"""Full-image tiled inference and evaluation on val/test split."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from shadow_dataset.geometry import candidate_mask_from_input, class_label_map, detect_class_boundaries
from shadow_dataset.image_io import load_rgb_exact
from shadow_dataset.io_utils import read_jsonl
from shadow_model.config import load_config
from shadow_model.factory import load_model_from_checkpoint
from shadow_model.global_cache import load_cached_tokens
from shadow_model.metrics import (
    ConfusionMatrix,
    compute_bf1,
    compute_checkpoint_score,
    compute_small_component_miou,
)
from shadow_model.render import render_prediction, validate_output_colors
from shadow_model.tiled_inference import tiled_predict

LOGGER = logging.getLogger(__name__)


def evaluate_split(
    model: torch.nn.Module,
    cfg,
    records: list[dict],
    device: torch.device,
) -> dict:
    cm = ConfusionMatrix(3)
    bf1_scores = {t: [] for t in [1, 2, 3]}
    small_comp_ious: list[float] = []
    per_image: list[dict] = []

    for rec in records:
        stem = rec["stem"]
        input_rgb, _ = load_rgb_exact(rec["input_path"])
        target_rgb, _ = load_rgb_exact(rec["target_path"])

        gtokens = gvalid = letterbox_meta = None
        if cfg.dino.use_global_view and cfg.dino.cache_dir:
            try:
                tokens, pvalid, meta = load_cached_tokens(
                    cfg.dino.cache_dir, rec["source_id"], 0, dino_cfg=cfg.dino, require=False,
                )
                gtokens = torch.from_numpy(tokens).unsqueeze(0)
                gvalid = torch.from_numpy(pvalid).unsqueeze(0)
                letterbox_meta = {
                    "scale": float(meta["scale"]),
                    "offset_x": float(meta["offset_x"]),
                    "offset_y": float(meta["offset_y"]),
                    "content_width": int(meta["content_width"]),
                    "content_height": int(meta["content_height"]),
                    "image_width": int(meta["image_width"]),
                    "image_height": int(meta["image_height"]),
                }
            except FileNotFoundError:
                LOGGER.warning("Missing global cache for %s", rec["source_id"])

        LOGGER.info("Processing %s (%dx%d)", stem, input_rgb.shape[1], input_rgb.shape[0])

        logits = tiled_predict(
            model,
            input_rgb,
            tile_size=cfg.eval.tile,
            halo=cfg.eval.halo,
            stride=cfg.eval.stride,
            device=str(device),
            global_tokens=gtokens,
            global_valid_mask=gvalid,
            letterbox_meta=letterbox_meta,
        )

        pred_class = np.argmax(logits, axis=0)
        candidate = candidate_mask_from_input(input_rgb)
        gt_labels = class_label_map(target_rgb, candidate)
        gt_np = gt_labels.astype(np.int64)
        valid = candidate & (gt_np >= 0)

        cm.update(pred_class, gt_np, valid)
        small_comp_ious.append(compute_small_component_miou(pred_class, gt_labels, candidate))

        output_rgb = render_prediction(input_rgb, logits)
        color_check = validate_output_colors(output_rgb)

        gt_boundary = detect_class_boundaries(gt_labels).astype(bool) & valid
        pred_labels_map = np.full_like(gt_labels, -1)
        pred_labels_map[candidate] = pred_class[candidate].astype(np.int16)
        pred_boundary = detect_class_boundaries(pred_labels_map).astype(bool) & valid

        img_results = {"stem": stem, "invalid_colors": color_check["invalid_color_count"]}
        for tol in [1, 2, 3]:
            score = compute_bf1(pred_boundary, gt_boundary, tolerance=tol)
            bf1_scores[tol].append(score)
            img_results[f"bf1_{tol}px"] = score

        correct = int(np.sum((pred_class == gt_np) & valid))
        total = int(np.sum(valid))
        img_results["pixel_acc"] = correct / max(total, 1)
        img_results["small_comp_miou"] = small_comp_ious[-1]
        per_image.append(img_results)
        LOGGER.info(
            "  acc=%.4f bf1@2=%.4f small_comp=%.4f invalid=%d",
            img_results["pixel_acc"],
            img_results["bf1_2px"],
            img_results["small_comp_miou"],
            img_results["invalid_colors"],
        )

    summary = cm.summary()
    for tol in [1, 2, 3]:
        summary[f"bf1_{tol}px"] = float(np.mean(bf1_scores[tol])) if bf1_scores[tol] else 0.0
    summary["small_comp_miou"] = float(np.mean(small_comp_ious)) if small_comp_ious else 0.0
    summary["checkpoint_score"] = compute_checkpoint_score(
        summary["macro_miou"],
        summary.get("bf1_2px", 0.0),
        summary["small_comp_miou"],
        weights=(cfg.eval.score.miou, cfg.eval.score.bf1_2px, cfg.eval.score.small_comp_miou),
    )
    summary["per_image"] = per_image
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full-image validation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    device = torch.device(args.device)

    manifest = Path(cfg.data.metadata_dir) / f"{args.split}_images.jsonl"
    records = read_jsonl(manifest)
    LOGGER.info("Evaluating %d images from %s", len(records), manifest)

    model, ckpt_state = load_model_from_checkpoint(str(args.checkpoint), device=str(device))
    model.eval()
    LOGGER.info("Loaded checkpoint profile=%s step=%d", ckpt_state.get("profile"), ckpt_state.get("global_step", 0))

    summary = evaluate_split(model, cfg, records, device)

    LOGGER.info("=== %s Results ===", args.split.upper())
    LOGGER.info(
        "mIoU=%.4f  BF1@2=%.4f  small_comp=%.4f  acc=%.4f  score=%.4f",
        summary["macro_miou"],
        summary["bf1_2px"],
        summary["small_comp_miou"],
        summary["pixel_accuracy"],
        summary["checkpoint_score"],
    )

    out_dir = args.out or Path("results") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Results saved to %s", out_dir / "eval_results.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

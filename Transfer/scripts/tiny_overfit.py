#!/usr/bin/env python3
"""Phase 1: Tiny overfit test on a few fixed crops (BEST_V1 path)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from shadow_model.config import load_config
from shadow_model.data import ShadowTrainDataset
from shadow_model.factory import build_model
from shadow_model.losses import ShadowLoss
from shadow_model.masks import build_supervision_mask
from shadow_model.numerical_safety import NumericalGuard
from shadow_model.optimizer_utils import audit_optimizer_coverage
from shadow_model.profiles import get_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiny overfit test.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", type=str, default="BEST_V1")
    parser.add_argument("--num-crops", type=int, default=6)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    profile = get_profile(args.profile)
    device = torch.device(args.device)
    guard = NumericalGuard(enabled=True)

    ds = ShadowTrainDataset(
        Path(cfg.data.metadata_dir) / cfg.data.train_crops,
        global_cache_dir=cfg.dino.cache_dir if profile.use_global_fusion else None,
        augment_dihedral=False,
        require_cache=False,
        dino_cfg=cfg.dino if profile.use_global_fusion else None,
    )

    indices = list(range(min(args.num_crops, len(ds))))
    samples = [ds[i] for i in indices]

    model = build_model(cfg, profile).to(device)
    criterion = ShadowLoss(
        ce_weight=cfg.loss.ce.weight,
        dice_weight=cfg.loss.dice.weight,
        transition_weight=cfg.loss.transition.weight if profile.use_transition_head else 0.0,
        affinity_weight=cfg.loss.affinity.weight if profile.use_affinity_loss else 0.0,
        aux_weight=cfg.loss.aux.weight if profile.use_aux_head else 0.0,
        class_weights=cfg.loss.ce.class_weights,
        ignore_index=cfg.data.ignore_index,
        use_main_logit_boundary=profile.use_main_logit_boundary_loss,
        total_steps=args.steps,
    )
    optimizer = torch.optim.AdamW(model.trainable_param_groups(lr_new=1e-3, lr_backbone=1e-4))
    audit_optimizer_coverage(model, optimizer)

    model.train()
    acc = 0.0
    for step in range(args.steps):
        for sample in samples:
            rgb = sample["local_input_rgb_norm"].unsqueeze(0).to(device)
            onehot = sample["local_input_onehot"].unsqueeze(0).to(device)
            label = sample["target_label"].unsqueeze(0).to(device)
            cand = sample["candidate_mask"].unsqueeze(0).to(device)
            valid = sample["image_valid_mask"].unsqueeze(0).to(device)
            center = sample["central_valid_mask"].unsqueeze(0).to(device)

            kwargs = {}
            if sample.get("global_tokens") is not None:
                kwargs["global_tokens"] = sample["global_tokens"].unsqueeze(0).to(device)
            if sample.get("global_valid_mask") is not None:
                kwargs["global_valid_mask"] = sample["global_valid_mask"].unsqueeze(0).to(device)
            if sample.get("crop_box_normalized") is not None:
                kwargs["crop_box"] = sample["crop_box_normalized"].unsqueeze(0).to(device)
            if sample.get("letterbox_meta"):
                kwargs["letterbox_meta"] = sample["letterbox_meta"]

            M = build_supervision_mask(cand, valid, center)
            guard.check_mask(M)

            outputs = model(rgb, onehot, **kwargs)
            losses = criterion(outputs, label, cand, valid, center, global_step=step)
            guard.check_losses(losses)

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()

        if (step + 1) % 50 == 0:
            model.eval()
            total_correct = 0
            total_pixels = 0
            with torch.no_grad():
                for sample in samples:
                    rgb = sample["local_input_rgb_norm"].unsqueeze(0).to(device)
                    onehot = sample["local_input_onehot"].unsqueeze(0).to(device)
                    label = sample["target_label"]
                    cand = sample["candidate_mask"]
                    valid = sample["image_valid_mask"]
                    center = sample["central_valid_mask"]

                    outputs = model(rgb, onehot)
                    pred = outputs["class_logits"].argmax(dim=1).squeeze(0).cpu()

                    M = build_supervision_mask(cand, valid, center)
                    total_correct += ((pred == label) & M).sum().item()
                    total_pixels += M.sum().item()

            acc = total_correct / max(total_pixels, 1) * 100
            logging.info("Step %d/%d: candidate_acc=%.2f%%", step + 1, args.steps, acc)
            model.train()

    logging.info("=== Tiny overfit gate check ===")
    logging.info("Final accuracy: %.2f%%", acc)
    if acc > 99.5:
        logging.info("PASS: accuracy > 99.5%%")
        return 0
    logging.warning("FAIL: accuracy %.2f%% < 99.5%%", acc)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

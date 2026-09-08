#!/usr/bin/env python3
"""VRAM dry-run: measure GPU memory for BEST_V1 forward/backward steps."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from shadow_model.config import load_config
from shadow_model.factory import build_model
from shadow_model.losses import ShadowLoss
from shadow_model.profiles import get_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VRAM dry run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", type=str, default="BEST_V1")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = load_config(args.config)
    profile = get_profile(args.profile)
    device = torch.device(args.device)

    model = build_model(cfg, profile).to(device)
    criterion = ShadowLoss(
        ce_weight=1.0,
        dice_weight=0.3,
        transition_weight=cfg.loss.transition.weight if profile.use_transition_head else 0.0,
        affinity_weight=cfg.loss.affinity.weight if profile.use_affinity_loss else 0.0,
        aux_weight=cfg.loss.aux.weight if profile.use_aux_head else 0.0,
        use_main_logit_boundary=profile.use_main_logit_boundary_loss,
        total_steps=args.steps,
    )
    optimizer = torch.optim.AdamW(model.trainable_param_groups())
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else torch.amp.GradScaler()

    B = cfg.train.micro_batch
    S = cfg.data.crop_size

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.steps):
        rgb = torch.randn(B, 3, S, S, device=device)
        onehot = torch.randn(B, 3, S, S, device=device)
        labels = torch.randint(0, 3, (B, S, S), device=device)
        cand = torch.ones(B, S, S, dtype=torch.bool, device=device)
        valid = torch.ones(B, S, S, dtype=torch.bool, device=device)
        center = torch.zeros(B, S, S, dtype=torch.bool, device=device)
        center[:, S // 8 : 7 * S // 8, S // 8 : 7 * S // 8] = True

        letterbox_meta = {
            "scale": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "content_width": S,
            "content_height": S,
            "image_width": S,
            "image_height": S,
        }
        kwargs = {}
        if profile.use_global_fusion:
            kwargs["global_tokens"] = torch.randn(B, 1024, cfg.dino.token_dim, device=device)
            kwargs["global_valid_mask"] = torch.ones(B, 1024, dtype=torch.bool, device=device)
            kwargs["crop_box"] = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device).expand(B, -1)
            kwargs["letterbox_meta"] = letterbox_meta

        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            outputs = model(rgb, onehot, **kwargs)
            losses = criterion(outputs, labels, cand, valid, center, global_step=step)
            loss = losses["total"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dt = time.time() - t0
        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            logging.info("Step %d: loss=%.4f time=%.2fs peak_mem=%.2fGB", step, loss.item(), dt, mem)
        else:
            logging.info("Step %d: loss=%.4f time=%.2fs (no CUDA)", step, loss.item(), dt)

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        logging.info("Peak VRAM: %.2f GB", peak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

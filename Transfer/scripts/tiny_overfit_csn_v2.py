#!/usr/bin/env python3
"""Phase 1: tiny overfit gate for CSN-V2 (1 crop, near-zero loss target)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from csn_v2.config import load_config
from csn_v2.data.dataset import CSNV2TrainDataset
from csn_v2.factory import build_model
from csn_v2.losses import CSNV2Loss
from shadow_model.optimizer_utils import audit_optimizer_coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN-V2 tiny overfit gate")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v2_overfit.yaml"))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss-target", type=float, default=0.05)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    manifest = Path(cfg.data.dataset_root) / cfg.data.train_crops
    ds = CSNV2TrainDataset(manifest, dataset_root=cfg.data.dataset_root)
    sample = ds[0]

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.trainable_param_groups(lr_new=3e-4), weight_decay=0.01)
    audit_optimizer_coverage(model, optimizer)
    criterion = CSNV2Loss(cfg.loss)

    model.train()
    lb = sample["local_bw"].unsqueeze(0).to(device)
    cb = sample["context_bw"].unsqueeze(0).to(device)
    fb = sample["full_bw"].unsqueeze(0).to(device)
    cc = sample["crop_coords_norm"].unsqueeze(0).to(device)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(lb, cb, fb, cc, teacher_forcing=1.0,
                    gt_shade_mask=sample["shade_mask"].unsqueeze(0).to(device),
                    gt_transition_mask=(sample["transition_mask"] > 127).float().unsqueeze(0).to(device))
        targets = {k: sample[k].unsqueeze(0).to(device) if isinstance(sample[k], torch.Tensor) else sample[k]
                   for k in sample if k in ("valid_mask", "black_lock", "shade_mask", "transition_mask", "shade_level_id", "target_class")}
        targets["where_target"] = sample["where_target"].unsqueeze(0).to(device)
        targets["level_target"] = sample["level_target"].unsqueeze(0).to(device)
        targets["ordinal_target"] = sample["ordinal_target"].unsqueeze(0).to(device)
        targets["affinity_channel_targets"] = sample["affinity_channel_targets"]
        losses = criterion(out, targets, step)
        losses["total"].backward()
        optimizer.step()
        if step % 50 == 0:
            logging.info("step=%d total=%.4f ce=%.4f trans=%.4f", step, losses["total"].item(),
                         losses["final_ce"].item(), losses["transition"].item())

    final = losses["total"].item()
    if final > args.loss_target:
        logging.error("Tiny overfit FAILED: total=%.4f > target=%.4f", final, args.loss_target)
        return 1
    logging.info("Tiny overfit PASSED: total=%.4f", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

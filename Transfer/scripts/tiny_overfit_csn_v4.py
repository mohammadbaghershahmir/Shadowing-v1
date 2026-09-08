#!/usr/bin/env python3
"""Single-tile overfit smoke test for CSN-V4."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import Subset, DataLoader

from csn_v4.config import load_config
from csn_v4.data.dataset import CSNV4TrainDataset
from csn_v4.factory import build_model
from csn_v4.losses import CSNV4Loss
from scripts.train_csn_v4 import collate_batch, _forward_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v4_bw_overfit_full_crop.yaml"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(cfg.data.dataset_root)
    ds = CSNV4TrainDataset(
        root / cfg.data.train_tiles, root / cfg.data.scenes, root,
        halo_loss_weight=cfg.data.halo_loss_weight,
    )
    loader = DataLoader(Subset(ds, [0]), batch_size=1, collate_fn=collate_batch)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.trainable_param_groups(lr_new=3e-4), lr=3e-4)
    crit = CSNV4Loss(cfg.loss)

    for step, batch in zip(range(args.steps), loader):
        opt.zero_grad(set_to_none=True)
        out = _forward_batch(model, batch, device, torch.float32, 0.0)
        targets = {k: batch[k].to(device) for k in batch if isinstance(batch[k], torch.Tensor) and k in (
            "target_class", "shade_mask", "transition_mask", "black_lock", "valid_mask",
            "where_target", "level_target", "affinity_targets", "affinity_valid", "supervision_weights",
        )}
        losses = crit(out, targets, step)
        losses["total"].backward()
        opt.step()
        if step % 10 == 0:
            print(f"step={step} total={losses['total'].item():.4f}")
    print("Smoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

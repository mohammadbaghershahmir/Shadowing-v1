#!/usr/bin/env python3
"""Single-crop overfit smoke test for CSN-V3 (teacher_forcing=0 by default)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from csn_v3.config import load_config
from csn_v3.factory import build_model
from csn_v3.losses import CSNV3Loss
from csn_v3.data.dataset import CSNV3TrainDataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v3_bw_overfit.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--oracle-smoke", action="store_true", help="Use teacher_forcing=1.0 wiring test only")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = Path(cfg.data.dataset_root) / cfg.data.train_crops
    ds = CSNV3TrainDataset(manifest, dataset_root=cfg.data.dataset_root, input_mode=cfg.data.input_mode)
    sample = ds[0]

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = CSNV3Loss(cfg.loss, center_halo=cfg.data.center_halo)
    tf = 1.0 if args.oracle_smoke else 0.0

    for step in range(args.steps):
        lb = sample["local_bw"].unsqueeze(0).to(device)
        cb = sample["context_bw"].unsqueeze(0).to(device)
        fb = sample["full_bw"].unsqueeze(0).to(device)
        cc = sample["crop_coords_norm"].unsqueeze(0).to(device)
        fwd = {"teacher_forcing": tf, "gt_shade_mask": sample["shade_mask"].unsqueeze(0).to(device),
               "gt_transition_mask": (sample["transition_mask"] > 127).float().unsqueeze(0).to(device)}
        if cfg.data.input_mode == "magenta" and "local_rgb" in sample:
            fwd["local_rgb"] = sample["local_rgb"].unsqueeze(0).to(device)
            fwd["local_onehot"] = sample["local_onehot"].unsqueeze(0).to(device)
        out = model(lb, cb, fb, cc, **fwd)
        targets = {k: sample[k].unsqueeze(0).to(device) if isinstance(sample[k], torch.Tensor) else sample[k] for k in sample if k not in ("letterbox_meta",)}
        targets["where_target"] = sample["where_target"].unsqueeze(0).to(device)
        targets["level_target"] = sample["level_target"].unsqueeze(0).to(device)
        targets["ordinal_target"] = sample["ordinal_target"].unsqueeze(0).to(device)
        losses = crit(out, targets, step)
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
        if step % 20 == 0:
            print(f"step={step} total={losses['total'].item():.4f} trans={losses['transition'].item():.4f} tf={tf}")

    print("OK tiny overfit complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Standalone CSN-V2 shape and gradient audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from csn_v2.config import load_config
from csn_v2.data.dataset import CSNV2TrainDataset
from csn_v2.factory import build_model
from csn_v2.training.shape_audit import ShapeAuditor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v2_overfit.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/shape_audit_csn_v2"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = Path(cfg.data.dataset_root) / cfg.data.train_crops
    ds = CSNV2TrainDataset(manifest, dataset_root=cfg.data.dataset_root)
    model = build_model(cfg)

    auditor = ShapeAuditor()
    sample = ds[0]
    fwd = auditor.audit_forward(model, sample, device)
    grad = auditor.audit_backward(model, sample, device)
    auditor.save(args.out / "shape_audit.json", fwd, grad)

    print("Forward shapes:")
    for k, v in fwd.items():
        print(f"  {k}: {v}")
    print(f"DINO backbone grad sum: {grad['dino_backbone_grad_sum']}")
    print(f"Trainable grad sum: {grad['trainable_grad_sum']}")
    if grad["dino_backbone_grad_sum"] > 1e-6:
        print("FAIL: DINO received gradients")
        return 1
    if grad["trainable_grad_sum"] < 1e-6:
        print("FAIL: no trainable gradients")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

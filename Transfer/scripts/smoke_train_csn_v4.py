#!/usr/bin/env python3
"""Smoke test: two optimizer steps with checkpoint save/resume."""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from csn_v4.config import load_config
from csn_v4.data.dataset import CSNV4TrainDataset
from csn_v4.factory import build_model
from csn_v4.losses import CSNV4Loss
from csn_v4.training.checkpointing import CheckpointManager
from scripts.train_csn_v4 import build_scheduler, collate_batch, resolve_amp, _forward_batch


def run_steps(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    cfg,
    device,
    amp_dtype,
    n_steps: int,
    start_opt_step: int = 0,
) -> int:
    opt_step = start_opt_step
    micro = start_opt_step * cfg.train.grad_accum
    target_opt = start_opt_step + n_steps
    data_iter = iter(loader)
    while opt_step < target_opt:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        out = _forward_batch(model, batch, device, amp_dtype, 0.0)
        targets = {
            k: batch[k].to(device)
            for k in batch
            if isinstance(batch[k], torch.Tensor)
            and k
            in (
                "target_class",
                "shade_mask",
                "transition_mask",
                "black_lock",
                "valid_mask",
                "where_target",
                "level_target",
                "affinity_targets",
                "affinity_valid",
                "supervision_weights",
            )
        }
        losses = criterion(out, targets, micro)
        loss = losses["total"] / cfg.train.grad_accum
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (micro + 1) % cfg.train.grad_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            print(
                f"optimizer_step={opt_step} total={losses['total'].item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        micro += 1
    return opt_step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN-V4 smoke train (2 steps + resume)")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v4_capacity.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=2, help="Optimizer steps per phase")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(cfg.data.dataset_root)
    ds = CSNV4TrainDataset(
        root / cfg.data.train_tiles,
        root / cfg.data.scenes,
        root,
        input_mode=cfg.data.input_mode,
        context_size=cfg.model.context_size,
        global_long_side=cfg.model.global_long_side,
        halo_loss_weight=cfg.data.halo_loss_weight,
    )
    if len(ds) == 0:
        print(f"Empty dataset at {root / cfg.data.train_tiles}")
        return 1

    loader = DataLoader(Subset(ds, [0]), batch_size=1, shuffle=False, collate_fn=collate_batch)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_param_groups(
            lr_new=cfg.optimizer.lr_new_modules,
            lr_stage34=cfg.optimizer.lr_convnext_stage34,
            lr_stage12=cfg.optimizer.lr_convnext_stage12,
        ),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)
    criterion = CSNV4Loss(cfg.loss)
    amp_dtype, scaler = resolve_amp(cfg, device)

    with tempfile.TemporaryDirectory(prefix="csn_v4_smoke_") as tmp:
        run_dir = Path(tmp)
        ckpt_mgr = CheckpointManager(run_dir, cfg, config_path=args.config)

        model.train()
        opt_step = run_steps(
            model, loader, optimizer, scheduler, scaler, criterion,
            cfg, device, amp_dtype, args.steps,
        )
        ckpt_mgr.save_last(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=opt_step * cfg.train.grad_accum,
            optimizer_step=opt_step,
        )

        model2 = build_model(cfg).to(device)
        optimizer2 = torch.optim.AdamW(
            model2.trainable_param_groups(
                lr_new=cfg.optimizer.lr_new_modules,
                lr_stage34=cfg.optimizer.lr_convnext_stage34,
                lr_stage12=cfg.optimizer.lr_convnext_stage12,
            ),
            weight_decay=cfg.optimizer.weight_decay,
        )
        scheduler2 = build_scheduler(optimizer2, cfg)
        _, scaler2 = resolve_amp(cfg, device)
        ckpt_mgr2 = CheckpointManager(run_dir, cfg, config_path=args.config)
        state = ckpt_mgr2.load(
            run_dir / "last.pt",
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            scaler=scaler2,
        )
        resumed_step = int(state["optimizer_step"])
        model2.train()
        final_step = run_steps(
            model2, loader, optimizer2, scheduler2, scaler2, criterion,
            cfg, device, amp_dtype, args.steps, start_opt_step=resumed_step,
        )

    print(f"Resume OK: steps {resumed_step} -> {final_step}, lr={optimizer2.param_groups[0]['lr']:.2e}")
    if final_step != resumed_step + args.steps:
        print(f"FAIL: expected {resumed_step + args.steps} steps, got {final_step}")
        return 1
    if not math.isfinite(optimizer2.param_groups[0]["lr"]):
        print("FAIL: non-finite LR after resume")
        return 1
    print("Smoke train save/resume PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

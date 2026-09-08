#!/usr/bin/env python3
"""Main training script for CarpetShadeNet-V2."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from csn_v2.config import config_to_dict, load_config
from csn_v2.dino_encoder import save_dino_load_report
from csn_v2.factory import build_model
from csn_v2.losses import CSNV2Loss
from csn_v2.data.dataset import CSNV2TrainDataset
from csn_v2.training.metrics_logger import MetricsLogger, LOSS_COLUMNS
from csn_v2.training.phases import PhaseController
from csn_v2.training.shape_audit import ShapeAuditor
from shadow_model.metrics import ConfusionMatrix
from shadow_model.optimizer_utils import audit_optimizer_coverage

LOGGER = logging.getLogger(__name__)


def resolve_amp(cfg, device):
    mode = cfg.train.amp
    if mode == "auto_bf16_else_fp16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, None
        return torch.float16, torch.amp.GradScaler()
    return torch.float16, torch.amp.GradScaler()


def build_scheduler(optimizer, cfg):
    warmup = int(cfg.train.total_steps * cfg.train.warmup_ratio)
    total = cfg.train.total_steps
    min_lr = cfg.train.min_lr
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(step):
        if step < warmup:
            return max(step / max(warmup, 1), 1e-8)
        progress = (step - warmup) / max(total - warmup, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr / base_lrs[0], cosine) if base_lrs else 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collate_batch(samples: list[dict]) -> dict:
    out = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if key in ("affinity_channel_targets", "letterbox_meta", "source_path", "full_artifact_dir"):
            out[key] = vals
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def build_eval_loader(ds: CSNV2TrainDataset, cfg) -> DataLoader:
    """Fixed small subset — no shuffle, for fast reproducible train-mIoU."""
    n = min(cfg.train.train_eval_max_samples, len(ds))
    return DataLoader(
        Subset(ds, list(range(n))),
        batch_size=cfg.train.micro_batch,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )


def _batch_global_tokens(model, batch: dict, device: torch.device, amp_dtype, cache: dict) -> torch.Tensor:
    keys = batch.get("full_artifact_dir") or batch.get("source_path")
    if isinstance(keys, str):
        keys = [keys]
    chunks: list[torch.Tensor] = []
    for b, key in enumerate(keys):
        if key not in cache:
            fb = batch["full_bw"][b : b + 1].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                full_rgb = model._prep_rgb(fb)
                cache[key] = model.dino.forward_global_tokens(full_rgb)
        chunks.append(cache[key])
    return torch.cat(chunks, dim=0)


def evaluate_miou(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    amp_dtype,
) -> float:
    model.eval()
    cm = ConfusionMatrix(num_classes=4)
    global_cache: dict[str, torch.Tensor] = {}
    t0 = time.perf_counter()
    batches_run = 0
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batches_run = i + 1
            global_tokens = _batch_global_tokens(model, batch, device, amp_dtype, global_cache)
            lb = batch["local_bw"].to(device, non_blocking=True)
            cb = batch["context_bw"].to(device, non_blocking=True)
            cc = batch["crop_coords_norm"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(
                    lb,
                    context_bw=cb,
                    crop_coords=cc,
                    global_tokens=global_tokens,
                    teacher_forcing=0.0,
                )
            pred = out["refined_logits"].argmax(dim=1).cpu().numpy()
            tgt = batch["target_class"].cpu().numpy()
            valid = (batch["valid_mask"] & ~batch["black_lock"]).cpu().numpy()
            for b in range(pred.shape[0]):
                cm.update(pred[b], tgt[b], valid[b])
    model.train()
    elapsed = time.perf_counter() - t0
    LOGGER.info(
        "Eval done: mIoU=%.4f batches=%d cached_globals=%d time=%.1fs",
        cm.macro_miou(), batches_run, len(global_cache), elapsed,
    )
    return cm.macro_miou()


def _build_checkpoint(
    model,
    optimizer,
    scheduler,
    global_step: int,
    best_miou: float,
    args,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "config_path": str(args.config),
        "best_miou": best_miou,
    }


def _load_metrics_history(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    history: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict = {}
            for k in LOSS_COLUMNS:
                if k not in row or row[k] == "":
                    continue
                parsed[k] = int(row[k]) if k == "step" else float(row[k])
            if parsed:
                history.append(parsed)
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CSN-V2")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v2_overfit.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    if args.mode:
        cfg.train.mode = args.mode
    if args.phase is not None:
        cfg.train.phase = args.phase
    if args.dataset_root:
        cfg.data.dataset_root = str(args.dataset_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resume_state: dict | None = None
    if args.resume:
        resume_path = args.resume.resolve()
        run_dir = args.run_dir.resolve() if args.run_dir else resume_path.parent
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        LOGGER.info("Resuming from %s (step=%d)", resume_path, resume_state.get("global_step", 0))
    else:
        run_dir = args.run_dir or (
            Path("runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_CSN_V2_{cfg.train.mode}"
        )
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config_to_dict(cfg), fh, indent=2)

    manifest = Path(cfg.data.dataset_root) / cfg.data.train_crops
    ds = CSNV2TrainDataset(
        manifest,
        dataset_root=cfg.data.dataset_root,
        crop_size=cfg.data.crop_size,
        context_size=cfg.data.context_size,
        global_long_side=cfg.data.global_long_side,
        allow_padded_context=cfg.data.allow_padded_context,
    )
    loader = DataLoader(ds, batch_size=cfg.train.micro_batch, shuffle=True, collate_fn=collate_batch)
    eval_loader = build_eval_loader(ds, cfg)
    eval_max_batches = min(
        cfg.train.train_eval_max_batches,
        math.ceil(cfg.train.train_eval_max_samples / max(cfg.train.micro_batch, 1)),
    )

    model = build_model(cfg).to(device)
    if resume_state is None:
        save_dino_load_report(model.dino.load_report, run_dir / "dino_load_report.json")

    phase_ctrl = PhaseController(cfg)
    s12, s34 = phase_ctrl.convnext_lrs()
    model.set_convnext_stages(s12, s34)
    optimizer = torch.optim.AdamW(
        model.trainable_param_groups(
            lr_new=cfg.optimizer.lr_new_modules,
            lr_stage34=cfg.optimizer.lr_convnext_stage34,
            lr_stage12=cfg.optimizer.lr_convnext_stage12,
        ),
        weight_decay=cfg.optimizer.weight_decay,
    )

    criterion = CSNV2Loss(cfg.loss)
    amp_dtype, scaler = resolve_amp(cfg, device)
    scheduler = build_scheduler(optimizer, cfg)

    global_step = 0
    best_miou = 0.0
    if resume_state:
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("scheduler"):
            scheduler.load_state_dict(resume_state["scheduler"])
        else:
            for _ in range(resume_state.get("global_step", 0)):
                scheduler.step()
        global_step = int(resume_state.get("global_step", 0))
        best_miou = float(resume_state.get("best_miou", 0.0))
        LOGGER.info("Restored step=%d best_miou=%.4f", global_step, best_miou)
    else:
        audit_optimizer_coverage(model, optimizer)
        if not args.skip_audit:
            auditor = ShapeAuditor()
            sample0 = ds[0]
            fwd_shapes = auditor.audit_forward(model, sample0, device)
            grad_report = auditor.audit_backward(model, sample0, device)
            auditor.save(run_dir / "shape_audit.json", fwd_shapes, grad_report)
            model.zero_grad(set_to_none=True)
            LOGGER.info(
                "Shape audit: dino_grad=%.4f trainable_grad=%.2f",
                grad_report["dino_backbone_grad_sum"],
                grad_report["trainable_grad_sum"],
            )

    metrics_logger = MetricsLogger(run_dir, cfg.train.plot_every_steps)
    metrics_logger.history = _load_metrics_history(metrics_logger.csv_path)

    model.train()
    pbar = tqdm(total=cfg.train.total_steps, initial=global_step, desc=f"CSN-V2/{cfg.train.mode}")
    while global_step < cfg.train.total_steps:
        for batch in loader:
            if global_step >= cfg.train.total_steps:
                break
            tf = phase_ctrl.teacher_forcing_prob(global_step, cfg.train.total_steps)
            lb = batch["local_bw"].to(device)
            cb = batch["context_bw"].to(device)
            fb = batch["full_bw"].to(device)
            cc = batch["crop_coords_norm"].to(device)
            gt_shade = batch["shade_mask"].float().to(device)
            gt_trans = (batch["transition_mask"] > 127).float().to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(lb, cb, fb, cc, teacher_forcing=tf, gt_shade_mask=gt_shade, gt_transition_mask=gt_trans)
                targets = {k: batch[k].to(device) if isinstance(batch[k], torch.Tensor) else batch[k] for k in batch}
                targets["where_target"] = batch["where_target"].to(device)
                targets["level_target"] = batch["level_target"].to(device)
                targets["ordinal_target"] = batch["ordinal_target"].to(device)
                losses = criterion(out, targets, global_step)

            loss = losses["total"] / cfg.train.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % cfg.train.grad_accum == 0:
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

            if global_step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                LOGGER.info(
                    "step=%d/%d total=%.4f ce=%.4f trans=%.4f aff=%.4f lr=%.2e",
                    global_step, cfg.train.total_steps,
                    losses["total"].item(), losses["final_ce"].item(),
                    losses["transition"].item(), losses["affinity"].item(), lr,
                )

            log_row = {k: float(v.item()) for k, v in losses.items()}
            log_row["lr"] = optimizer.param_groups[0]["lr"]
            metrics_logger.log(global_step, log_row)
            pbar.update(1)
            pbar.set_postfix(total=f"{losses['total'].item():.4f}")
            global_step += 1

            if global_step % cfg.train.checkpoint_every_steps == 0:
                ckpt = _build_checkpoint(model, optimizer, scheduler, global_step, best_miou, args)
                torch.save(ckpt, run_dir / "last.pt")
                LOGGER.info("Saved checkpoint at step %d", global_step)

            if (
                cfg.train.score_on_train
                and global_step % cfg.train.eval_every_steps == 0
            ):
                miou = evaluate_miou(model, eval_loader, device, eval_max_batches, amp_dtype)
                metrics_logger.log(global_step, {"train_miou": miou})
                ckpt = _build_checkpoint(model, optimizer, scheduler, global_step, best_miou, args)
                torch.save(ckpt, run_dir / "last.pt")
                if miou > best_miou:
                    best_miou = miou
                    ckpt["best_miou"] = best_miou
                    torch.save(ckpt, run_dir / "best.pt")
                    LOGGER.info("New best train mIoU: %.4f", miou)

    pbar.close()
    if device.type == "cuda":
        LOGGER.info("Peak GPU memory: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)
    LOGGER.info("Training complete. Run dir: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

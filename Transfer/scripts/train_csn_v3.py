#!/usr/bin/env python3
"""Main training script for CarpetShadeNet-V3."""
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

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from csn_v3.config import config_to_dict, load_config
from csn_v3.data.augment import apply_dihedral_csn_sample
from csn_v3.data.dataset import CSNV3TrainDataset
from csn_v3.dino_encoder import save_dino_load_report
from csn_v3.factory import build_model
from csn_v3.inference.tiled import predict_full_image_gray
from csn_v3.losses import CSNV3Loss
from csn_v3.metrics.boundary_metrics import (
    BoundaryEvalResult,
    accumulate_boundary_eval,
    compute_boundary_eval,
    compute_checkpoint_score_v3,
)
from csn_v3.training.metrics_logger import MetricsLogger, LOSS_COLUMNS
from csn_v3.training.phases import PhaseController
from csn_v3.training.shape_audit import ShapeAuditor
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
    warmup = int(cfg.train.optimizer_steps * cfg.train.warmup_ratio)
    total = cfg.train.optimizer_steps
    min_lr = cfg.train.min_lr
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(opt_step):
        if opt_step < warmup:
            return max(opt_step / max(warmup, 1), 1e-8)
        progress = (opt_step - warmup) / max(total - warmup, 1)
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


def build_eval_loader(ds: CSNV3TrainDataset, cfg, eval_mode: bool = True) -> DataLoader:
    n = min(cfg.train.train_eval_max_samples, len(ds))
    eval_ds = CSNV3TrainDataset(
        ds.manifest_path,
        dataset_root=ds.dataset_root,
        input_mode=ds.input_mode,
        magenta_guide_dir=ds.magenta_guide_dir,
        crop_size=ds.crop_size,
        context_size=ds.context_size,
        global_long_side=ds.global_long_side,
        center_halo=ds.center_halo,
        augment_dihedral=False,
        eval_mode=True,
    )
    return DataLoader(
        Subset(eval_ds, list(range(n))),
        batch_size=cfg.train.micro_batch,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )


def _batch_global_tokens(model, batch: dict, device: torch.device, amp_dtype, cache: dict) -> torch.Tensor:
    keys = batch.get("full_artifact_dir") or batch.get("source_path")
    if isinstance(keys, str):
        keys = [keys]
    chunks = []
    for b, key in enumerate(keys):
        if key not in cache:
            fb = batch["full_bw"][b : b + 1].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                full_rgb = model._prep_rgb(fb)
                cache[key] = model.dino.forward_global_tokens(full_rgb)
        chunks.append(cache[key])
    return torch.cat(chunks, dim=0)


def _forward_batch(model, batch, device, amp_dtype, tf: float):
    lb = batch["local_bw"].to(device)
    cb = batch["context_bw"].to(device)
    fb = batch["full_bw"].to(device)
    cc = batch["crop_coords_norm"].to(device)
    kwargs = {
        "teacher_forcing": tf,
        "gt_shade_mask": batch["shade_mask"].float().to(device),
        "gt_transition_mask": (batch["transition_mask"] > 127).float().to(device),
        "letterbox_meta": batch.get("letterbox_meta"),
    }
    if "local_rgb" in batch:
        kwargs["local_rgb"] = batch["local_rgb"].to(device)
        kwargs["local_onehot"] = batch["local_onehot"].to(device)
    global_tokens = _batch_global_tokens(model, batch, device, amp_dtype, {})
    kwargs["global_tokens"] = global_tokens
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        return model(lb, cb, fb, cc, **kwargs)


def evaluate_boundary_metrics(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    amp_dtype,
    boundary_dilate: int = 2,
    rotate_k: int | None = None,
) -> BoundaryEvalResult:
    model.eval()
    results: list[BoundaryEvalResult] = []
    global_cache: dict[str, torch.Tensor] = {}
    t0 = time.perf_counter()
    batches_run = 0

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batches_run = i + 1
            if rotate_k:
                batch = collate_batch([
                    apply_dihedral_csn_sample({k: batch[k][b] if isinstance(batch[k], torch.Tensor) else batch[k][b] for k in batch}, rotate_k)
                    for b in range(batch["local_bw"].shape[0])
                ])
            global_tokens = _batch_global_tokens(model, batch, device, amp_dtype, global_cache)
            lb = batch["local_bw"].to(device)
            cb = batch["context_bw"].to(device)
            fb = batch["full_bw"].to(device)
            cc = batch["crop_coords_norm"].to(device)
            fwd_kw = {"global_tokens": global_tokens, "teacher_forcing": 0.0, "letterbox_meta": batch.get("letterbox_meta")}
            if "local_rgb" in batch:
                fwd_kw["local_rgb"] = batch["local_rgb"].to(device)
                fwd_kw["local_onehot"] = batch["local_onehot"].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(lb, cb, fb, cc, **fwd_kw)
            pred = out["refined_logits"].argmax(dim=1).cpu().numpy()
            for b in range(pred.shape[0]):
                tgt = batch["target_class"][b].cpu().numpy()
                trans = batch["transition_mask"][b].cpu().numpy()
                valid = batch["valid_mask"][b].cpu().numpy()
                black = batch["black_lock"][b].cpu().numpy()
                results.append(compute_boundary_eval(
                    pred[b], tgt, trans, valid, black, boundary_dilate=boundary_dilate,
                ))

    model.train()
    agg = accumulate_boundary_eval(results)
    LOGGER.info(
        "Eval: global_miou=%.4f boundary_miou=%.4f boundary_acc=%.4f bf1=%.4f batches=%d rot=%s time=%.1fs",
        agg.global_miou, agg.boundary_miou, agg.boundary_pixel_acc, agg.bf1_2px,
        batches_run, rotate_k, time.perf_counter() - t0,
    )
    return agg


def evaluate_full_images(model, cfg, device, amp_dtype) -> dict[str, float]:
    manifest = Path(cfg.data.dataset_root) / cfg.data.eval_full_images
    if not manifest.exists():
        return {}
    results: list[BoundaryEvalResult] = []
    from csn_v3.data.targets import load_gray_bmp

    with manifest.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    for rec in records[:5]:
        src = load_gray_bmp(rec["source_bw_path"])
        gray, _ = predict_full_image_gray(
            model, src, device,
            tile_size=cfg.inference.tile_size,
            halo=cfg.inference.halo,
            stride=cfg.inference.stride,
            context_size=cfg.data.context_size,
            global_long_side=cfg.inference.global_long_side,
        )
        tgt_sem = load_gray_bmp(rec["target_semantic_path"])
        from csn_v3.masks import semantic_gray_to_class
        tgt_class = semantic_gray_to_class(torch.from_numpy(tgt_sem.astype(np.int64))).numpy()
        trans = ((tgt_class >= 1) & (tgt_class <= 3)).astype(np.uint8) * 255
        valid = np.ones_like(tgt_class, dtype=np.uint8) * 255
        black = (tgt_sem == 0).astype(np.uint8) * 255
        pred_class = np.zeros_like(tgt_class)
        for g, c in [(255, 0), (200, 1), (150, 2), (100, 3)]:
            pred_class[gray == g] = c
        results.append(compute_boundary_eval(
            pred_class, tgt_class, trans, valid, black, boundary_dilate=cfg.data.boundary_dilate,
        ))

    if not results:
        return {}
    agg = accumulate_boundary_eval(results)
    return {
        "full_global_miou": agg.global_miou,
        "full_boundary_miou": agg.boundary_miou,
        "full_bf1_2px": agg.bf1_2px,
    }


def _build_ckpt(
    model,
    optimizer,
    scheduler,
    global_step: int,
    optimizer_step: int,
    best_score: float,
    config_path: str,
    eval_metrics: dict | None = None,
) -> dict:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "best_score": best_score,
        "config_path": config_path,
    }
    if eval_metrics:
        ckpt["eval_metrics"] = eval_metrics
    return ckpt


def _save_checkpoint(path: Path, ckpt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _should_run_eval(
    global_step: int,
    cfg,
    *,
    score_on_train: bool,
) -> bool:
    return (
        score_on_train
        and global_step > 0
        and global_step % cfg.train.eval_every_steps == 0
    )


def _eval_and_save_checkpoints(
    model,
    optimizer,
    scheduler,
    *,
    global_step: int,
    optimizer_step: int,
    best_score: float,
    cfg,
    eval_loader,
    device,
    eval_max_batches: int,
    amp_dtype,
    metrics_logger: MetricsLogger,
    run_dir: Path,
    config_path: str,
) -> float:
    ev = evaluate_boundary_metrics(
        model, eval_loader, device, eval_max_batches, amp_dtype, cfg.data.boundary_dilate,
    )
    rot_results = []
    for k in cfg.train.rotated_eval_k:
        rot_results.append(evaluate_boundary_metrics(
            model, eval_loader, device, eval_max_batches, amp_dtype,
            cfg.data.boundary_dilate, rotate_k=k,
        ).boundary_miou)
    rotated_boundary_miou = float(np.mean(rot_results)) if rot_results else 0.0
    score = compute_checkpoint_score_v3(
        ev.global_miou, ev.boundary_miou, ev.bf1_2px, ev.boundary_pixel_acc,
    )
    eval_row = {
        "global_miou": ev.global_miou,
        "boundary_miou": ev.boundary_miou,
        "boundary_pixel_acc": ev.boundary_pixel_acc,
        "interior_miou": ev.interior_miou,
        "bf1_2px": ev.bf1_2px,
        "checkpoint_score_v3": score,
        "rotated_boundary_miou": rotated_boundary_miou,
    }
    ckpt = _build_ckpt(
        model, optimizer, scheduler, global_step, optimizer_step,
        best_score, config_path, eval_metrics=eval_row,
    )
    _save_checkpoint(run_dir / "last.pt", ckpt)
    if score >= best_score:
        best_score = score
        ckpt["best_score"] = best_score
        _save_checkpoint(run_dir / "best.pt", ckpt)
        LOGGER.info(
            "New best.pt score=%.4f boundary_miou=%.4f (micro_step=%d opt_step=%d)",
            score, ev.boundary_miou, global_step, optimizer_step,
        )
    LOGGER.info(
        "=== Eval @ micro_step=%d opt_step=%d === "
        "global_miou=%.4f boundary_miou=%.4f interior_miou=%.4f "
        "boundary_acc=%.4f bf1=%.4f score=%.4f rotated_boundary=%.4f",
        global_step, optimizer_step,
        ev.global_miou, ev.boundary_miou, ev.interior_miou,
        ev.boundary_pixel_acc, ev.bf1_2px, score, rotated_boundary_miou,
    )
    metrics_logger.log_eval(global_step, eval_row)
    return best_score


def _load_metrics_history(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    eval_keys = {
        "global_miou", "boundary_miou", "boundary_pixel_acc", "interior_miou", "bf1_2px",
        "checkpoint_score_v3", "rotated_boundary_miou",
    }
    history: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict = {}
            for k in LOSS_COLUMNS:
                if k not in row or row[k] == "":
                    continue
                val = int(row[k]) if k == "step" else float(row[k])
                if k in eval_keys and val == 0.0:
                    continue
                parsed[k] = val
            if parsed:
                history.append(parsed)
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CSN-V3")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v3_bw_overfit.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--input-mode", type=str, default=None, choices=["bw", "magenta"])
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
    if args.input_mode:
        cfg.data.input_mode = args.input_mode
        if args.input_mode == "magenta":
            cfg.data.train_crops = "manifests/train_crops_magenta.jsonl"
    if args.phase is not None:
        cfg.train.phase = args.phase
    if args.dataset_root:
        cfg.data.dataset_root = str(args.dataset_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resume_state = None
    if args.resume:
        resume_path = args.resume.resolve()
        run_dir = args.run_dir.resolve() if args.run_dir else resume_path.parent
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        LOGGER.info("Resuming from %s (step=%d)", resume_path, resume_state.get("global_step", 0))
    else:
        run_dir = args.run_dir or Path("runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_CSN_V3_{cfg.data.input_mode}_{cfg.train.mode}"
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config_to_dict(cfg), fh, indent=2)

    manifest = Path(cfg.data.dataset_root) / cfg.data.train_crops
    ds = CSNV3TrainDataset(
        manifest,
        dataset_root=cfg.data.dataset_root,
        input_mode=cfg.data.input_mode,
        magenta_guide_dir=cfg.data.magenta_guide_dir,
        crop_size=cfg.data.crop_size,
        context_size=cfg.data.context_size,
        global_long_side=cfg.data.global_long_side,
        center_halo=cfg.data.center_halo,
        augment_dihedral=cfg.data.augment_dihedral,
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

    criterion = CSNV3Loss(cfg.loss, center_halo=cfg.data.center_halo)
    amp_dtype, scaler = resolve_amp(cfg, device)
    scheduler = build_scheduler(optimizer, cfg)

    global_step = 0
    optimizer_step = 0
    best_score = -1.0
    if resume_state:
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("scheduler"):
            scheduler.load_state_dict(resume_state["scheduler"])
        global_step = int(resume_state.get("global_step", 0))
        optimizer_step = int(resume_state.get("optimizer_step", global_step // cfg.train.grad_accum))
        best_score = float(resume_state.get("best_score", -1.0))
        best_path = run_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_score = max(best_score, float(best_ckpt.get("best_score", -1.0)))
        elif best_score == 0.0:
            best_score = -1.0
        LOGGER.info("Restored micro=%d opt=%d best_score=%.4f", global_step, optimizer_step, best_score)
    else:
        audit_optimizer_coverage(model, optimizer)
        if not args.skip_audit:
            auditor = ShapeAuditor()
            sample0 = ds[0]
            fwd_shapes = auditor.audit_forward(model, sample0, device)
            grad_report = auditor.audit_backward(model, sample0, device)
            auditor.save(run_dir / "shape_audit.json", fwd_shapes, grad_report)
            model.zero_grad(set_to_none=True)

    metrics_logger = MetricsLogger(run_dir, cfg.train.plot_every_steps)
    metrics_logger.history = _load_metrics_history(metrics_logger.csv_path)

    if (
        resume_state
        and cfg.train.score_on_train
        and not (run_dir / "best.pt").exists()
        and global_step > 0
    ):
        LOGGER.info("best.pt missing after resume — running eval at micro_step=%d", global_step)
        model.eval()
        best_score = _eval_and_save_checkpoints(
            model, optimizer, scheduler,
            global_step=global_step,
            optimizer_step=optimizer_step,
            best_score=best_score,
            cfg=cfg,
            eval_loader=eval_loader,
            device=device,
            eval_max_batches=eval_max_batches,
            amp_dtype=amp_dtype,
            metrics_logger=metrics_logger,
            run_dir=run_dir,
            config_path=str(args.config),
        )
        model.train()

    model.train()
    pbar = tqdm(total=cfg.train.total_micro_steps, initial=global_step, desc=f"CSN-V3/{cfg.data.input_mode}")
    while global_step < cfg.train.total_micro_steps:
        for batch in loader:
            if global_step >= cfg.train.total_micro_steps:
                break
            tf = phase_ctrl.teacher_forcing_prob(global_step, cfg.train.total_micro_steps)
            out = _forward_batch(model, batch, device, amp_dtype, tf)
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
                optimizer_step += 1

            if global_step % 50 == 0:
                LOGGER.info(
                    "micro=%d opt=%d/%d total=%.4f boundary_trans=%.4f lr=%.2e",
                    global_step, optimizer_step, cfg.train.optimizer_steps,
                    losses["total"].item(), losses["transition"].item(),
                    optimizer.param_groups[0]["lr"],
                )

            log_row = {k: float(v.item()) for k, v in losses.items()}
            log_row["lr"] = optimizer.param_groups[0]["lr"]
            metrics_logger.log_loss(global_step, log_row)
            pbar.update(1)
            pbar.set_postfix(total=f"{losses['total'].item():.4f}")
            global_step += 1

            if optimizer_step > 0 and optimizer_step % cfg.train.checkpoint_every_steps == 0:
                ckpt = _build_ckpt(
                    model, optimizer, scheduler, global_step, optimizer_step,
                    best_score, str(args.config),
                )
                _save_checkpoint(run_dir / "last.pt", ckpt)
                LOGGER.info("Saved last.pt at opt_step=%d micro_step=%d", optimizer_step, global_step)

            if _should_run_eval(global_step, cfg, score_on_train=cfg.train.score_on_train):
                best_score = _eval_and_save_checkpoints(
                    model, optimizer, scheduler,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    best_score=best_score,
                    cfg=cfg,
                    eval_loader=eval_loader,
                    device=device,
                    eval_max_batches=eval_max_batches,
                    amp_dtype=amp_dtype,
                    metrics_logger=metrics_logger,
                    run_dir=run_dir,
                    config_path=str(args.config),
                )

            if (
                optimizer_step > 0
                and optimizer_step % cfg.train.full_image_eval_every_steps == 0
            ):
                full_m = evaluate_full_images(model, cfg, device, amp_dtype)
                if full_m:
                    metrics_logger.log_eval(global_step, full_m)
                    LOGGER.info("Full-image eval: %s", full_m)

    pbar.close()
    if cfg.train.score_on_train and not (run_dir / "best.pt").exists():
        LOGGER.info("No best.pt found — running final eval to save best checkpoint")
        model.eval()
        best_score = _eval_and_save_checkpoints(
            model, optimizer, scheduler,
            global_step=global_step,
            optimizer_step=optimizer_step,
            best_score=best_score,
            cfg=cfg,
            eval_loader=eval_loader,
            device=device,
            eval_max_batches=eval_max_batches,
            amp_dtype=amp_dtype,
            metrics_logger=metrics_logger,
            run_dir=run_dir,
            config_path=str(args.config),
        )
        LOGGER.info("Saved best.pt from final eval")
    LOGGER.info("Training complete. Run dir: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

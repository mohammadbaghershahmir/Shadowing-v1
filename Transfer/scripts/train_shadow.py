#!/usr/bin/env python3
"""Main training script for ShadowNet BEST_V1 profile."""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from shadow_model.checkpointing import (
    get_rng_states,
    load_checkpoint,
    manifest_hashes,
    save_checkpoint,
    set_rng_states,
)
from shadow_model.config import ShadowConfig, load_config
from shadow_model.data import GroupedBatchSampler, ShadowTrainDataset
from shadow_model.factory import build_model
from shadow_model.losses import ShadowLoss
from shadow_model.masks import build_supervision_mask
from shadow_model.metrics import ConfusionMatrix
from shadow_model.numerical_safety import NumericalGuard, save_nan_dump
from shadow_model.optimizer_utils import audit_optimizer_coverage
from shadow_model.profiles import get_profile

LOGGER = logging.getLogger(__name__)

LOSS_KEYS: tuple[str, ...] = ("total", "ce", "dice", "transition", "affinity", "boundary", "aux")


def init_loss_meter() -> dict[str, float]:
    return {key: 0.0 for key in LOSS_KEYS}


def update_loss_meter(meter: dict[str, float], losses: dict[str, torch.Tensor]) -> None:
    for key in LOSS_KEYS:
        if key in losses:
            meter[key] += float(losses[key].detach().item())


def average_loss_meter(meter: dict[str, float], count: int) -> dict[str, float]:
    denom = max(count, 1)
    return {key: value / denom for key, value in meter.items()}


def format_loss_summary(losses: dict[str, float]) -> str:
    parts = [f"{key}={losses[key]:.4f}" for key in LOSS_KEYS if losses.get(key, 0.0) != 0.0]
    return " ".join(parts) if parts else "no-loss"


def resolve_amp(cfg: ShadowConfig, device: torch.device) -> tuple[torch.dtype, torch.amp.GradScaler | None]:
    mode = cfg.train.amp
    if mode == "auto_bf16_else_fp16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, None
        return torch.float16, torch.amp.GradScaler()
    if mode == "bf16":
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler()


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: ShadowConfig) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = int(cfg.train.total_steps * cfg.train.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / max(cfg.train.total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = cfg.train.min_lr / cfg.train.lr_new
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _batch_model_kwargs(batch: dict, device: torch.device) -> dict:
    kwargs: dict = {}
    gt = batch.get("global_tokens")
    if gt is not None:
        kwargs["global_tokens"] = gt.to(device)
    gv = batch.get("global_valid_mask")
    if gv is not None:
        kwargs["global_valid_mask"] = gv.to(device)
    cb = batch.get("crop_box_normalized")
    if cb is not None:
        kwargs["crop_box"] = cb.to(device)
    lb = batch.get("letterbox_meta")
    if lb is not None:
        kwargs["letterbox_meta"] = lb
    return kwargs


@torch.no_grad()
def validate_crop(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ShadowLoss,
    device: torch.device,
    amp_dtype: torch.dtype,
    global_step: int,
    *,
    max_batches: int | None = None,
    desc: str = "val",
) -> dict:
    model.eval()
    cm = ConfusionMatrix(3)
    loss_meter = init_loss_meter()
    count = 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        rgb = batch["local_input_rgb_norm"].to(device)
        onehot = batch["local_input_onehot"].to(device)
        label = batch["target_label"].to(device)
        cand = batch["candidate_mask"].to(device)
        valid = batch["image_valid_mask"].to(device)
        center = batch["central_valid_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            outputs = model(rgb, onehot, **_batch_model_kwargs(batch, device))
            losses = criterion(outputs, label, cand, valid, center, global_step=global_step)

        update_loss_meter(loss_meter, losses)
        count += 1

        pred = outputs["class_logits"].argmax(dim=1).cpu().numpy()
        label_np = label.cpu().numpy()
        M = build_supervision_mask(cand, valid, center).cpu().numpy()
        for b in range(pred.shape[0]):
            cm.update(pred[b], label_np[b], M[b])

        avg_losses = average_loss_meter(loss_meter, count)
        pbar.set_postfix(total=f"{avg_losses['total']:.4f}", ce=f"{avg_losses['ce']:.4f}")
        if max_batches is not None and count >= max_batches:
            break

    model.train()
    return {**average_loss_meter(loss_meter, count), **cm.summary()}


def run_full_image_validation(
    model: torch.nn.Module,
    cfg: ShadowConfig,
    device: torch.device,
    split: str = "val",
    max_images: int = 2,
) -> dict:
    """Placeholder full-image validation hook (limited images for speed)."""
    from shadow_dataset.geometry import candidate_mask_from_input, class_label_map
    from shadow_dataset.image_io import load_rgb_exact
    from shadow_dataset.io_utils import read_jsonl
    from shadow_model.global_cache import load_cached_tokens
    from shadow_model.metrics import compute_small_component_miou
    from shadow_model.tiled_inference import tiled_predict

    manifest = Path(cfg.data.metadata_dir) / f"{split}_images.jsonl"
    if not manifest.exists():
        LOGGER.warning("Full-image val skipped: missing %s", manifest)
        return {"macro_miou": 0.0, "small_comp_miou": 0.0}

    records = read_jsonl(manifest)[:max_images]
    cm = ConfusionMatrix(3)
    small_ious: list[float] = []

    model.eval()
    for rec in records:
        input_rgb, _ = load_rgb_exact(rec["input_path"])
        target_rgb, _ = load_rgb_exact(rec["target_path"])
        gtokens = gvalid = letterbox_meta = None
        if cfg.dino.use_global_view and cfg.dino.cache_dir:
            tokens, pvalid, meta = load_cached_tokens(cfg.dino.cache_dir, rec["source_id"], 0, require=False)
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
        pred = np.argmax(logits, axis=0)
        candidate = candidate_mask_from_input(input_rgb)
        gt_labels = class_label_map(target_rgb, candidate)
        valid = candidate & (gt_labels >= 0)
        cm.update(pred, gt_labels.astype(np.int64), valid)
        small_ious.append(compute_small_component_miou(pred, gt_labels, candidate))

    model.train()
    summary = cm.summary()
    summary["small_comp_miou"] = float(np.mean(small_ious)) if small_ious else 0.0
    LOGGER.info(
        "Full-image val (%d imgs): mIoU=%.4f small_comp_miou=%.4f",
        len(records),
        summary["macro_miou"],
        summary["small_comp_miou"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ShadowNet BEST_V1.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", type=str, default="BEST_V1")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "overfit"],
        help="overfit: no held-out val; score train mIoU; disable early stop.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    cfg.data.fold = args.fold
    overfit = args.mode == "overfit" or cfg.train.disable_val or cfg.train.score_on_train
    if overfit:
        cfg.train.disable_val = True
        cfg.train.score_on_train = True
        cfg.train.early_stop_patience_full_image = max(
            cfg.train.early_stop_patience_full_image, 10**9
        )
        cfg.train.full_image_val_every_steps = max(
            cfg.train.full_image_val_every_steps, 10**9
        )
    profile = get_profile(args.profile)
    device = torch.device(args.device)
    amp_dtype, scaler = resolve_amp(cfg, device)
    guard = NumericalGuard(enabled=True)

    mode_tag = "overfit" if overfit else f"fold{args.fold}"
    run_dir = Path("runs") / f"{datetime.now():%Y%m%d_%H%M%S}_{args.profile}_{mode_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = cfg.to_dict()
    resolved["profile"] = profile.name
    resolved["mode"] = "overfit" if overfit else "default"
    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2)

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    use_global = profile.use_global_fusion
    train_ds = ShadowTrainDataset(
        Path(cfg.data.metadata_dir) / cfg.data.train_crops,
        global_cache_dir=cfg.dino.cache_dir if use_global else None,
        augment_dihedral=cfg.data.augment_dihedral,
        allowed_dihedral=cfg.data.allowed_dihedral,
        require_cache=cfg.dino.require_cache and use_global,
        dino_cfg=cfg.dino if use_global else None,
        seed=cfg.train.seed,
    )
    train_sampler = GroupedBatchSampler(
        train_ds.base.records,
        cfg.train.micro_batch,
        max_per_source=cfg.data.max_crops_per_source_in_batch,
        seed=cfg.train.seed,
        accum_size=cfg.train.grad_accum,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=2, pin_memory=True)

    score_loader: DataLoader | None = None
    if overfit:
        score_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.micro_batch,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        LOGGER.info(
            "Overfit mode: %d train crops, score_on_train=True, no held-out val",
            len(train_ds),
        )
    else:
        val_ds = ShadowTrainDataset(
            Path(cfg.data.metadata_dir) / cfg.data.val_crops,
            global_cache_dir=cfg.dino.cache_dir if use_global else None,
            require_cache=cfg.dino.require_cache and use_global,
            dino_cfg=cfg.dino if use_global else None,
        )
        score_loader = DataLoader(
            val_ds, batch_size=cfg.train.micro_batch, num_workers=2, pin_memory=True
        )

    model = build_model(cfg, profile).to(device)

    criterion = ShadowLoss(
        ce_weight=cfg.loss.ce.weight,
        dice_weight=cfg.loss.dice.weight,
        transition_weight=cfg.loss.transition.weight if profile.use_transition_head else 0.0,
        transition_ramp_start=cfg.loss.transition.ramp_start,
        transition_ramp_end=cfg.loss.transition.ramp_end,
        affinity_weight=cfg.loss.affinity.weight if profile.use_affinity_loss else 0.0,
        affinity_ramp_start=cfg.loss.affinity.ramp_start,
        affinity_ramp_end=cfg.loss.affinity.ramp_end,
        boundary_weight=0.1 if profile.use_main_logit_boundary_loss else 0.0,
        aux_weight=cfg.loss.aux.weight if profile.use_aux_head else 0.0,
        class_weights=cfg.loss.ce.class_weights,
        num_classes=cfg.model.num_classes,
        ignore_index=cfg.data.ignore_index,
        transition_dilate=cfg.loss.transition.dilate,
        pos_weight_clip=cfg.loss.transition.pos_weight_clip,
        affinity_offsets=cfg.loss.affinity.offsets,
        affinity_max_pairs=cfg.loss.affinity.max_pairs,
        affinity_same_component=cfg.loss.affinity.same_component,
        use_main_logit_boundary=profile.use_main_logit_boundary_loss and cfg.loss.main_logit_boundary,
        total_steps=cfg.train.total_steps,
    ).to(device)

    param_groups = model.trainable_param_groups(
        lr_new=cfg.train.lr_new,
        lr_backbone=cfg.train.lr_backbone,
        lr_decay=cfg.local_encoder.layerwise_lr_decay,
        weight_decay=cfg.train.weight_decay,
        no_decay_keywords=tuple(cfg.train.no_decay),
    )
    optimizer = torch.optim.AdamW(param_groups)
    audit_optimizer_coverage(model, optimizer)

    scheduler = build_scheduler(optimizer, cfg)
    start_step = 0
    best_score = -1.0
    patience_counter = 0
    man_hashes = manifest_hashes(Path(cfg.data.metadata_dir), cfg.data.train_crops, cfg.data.val_crops)

    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, strict=True)
        start_step = int(state.get("global_step", 0)) + 1
        best_score = state.get("best_score", -1.0)
        if state.get("rng_states"):
            set_rng_states(state["rng_states"])
        LOGGER.info("Resumed from step %d, best_score=%.4f", start_step, best_score)

    train_iter = iter(train_loader)
    model.train()
    running_meter = init_loss_meter()
    running_count = 0
    epoch = 0
    eval_every = cfg.train.checkpoint_every_steps or cfg.train.val_every_steps

    pbar = tqdm(
        range(start_step, cfg.train.total_steps),
        initial=start_step,
        total=cfg.train.total_steps,
        desc=f"train {args.profile}{'/overfit' if overfit else ''}",
        dynamic_ncols=True,
    )

    for global_step in pbar:
        t0 = time.time()
        train_ds.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        step_meter = init_loss_meter()
        batch_meta: dict = {}

        for accum_i in range(cfg.train.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_ds.set_epoch(epoch)
                train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            rgb = batch["local_input_rgb_norm"].to(device)
            onehot = batch["local_input_onehot"].to(device)
            label = batch["target_label"].to(device)
            cand = batch["candidate_mask"].to(device)
            valid_m = batch["image_valid_mask"].to(device)
            center = batch["central_valid_mask"].to(device)
            batch_meta = {"sample_id": batch.get("sample_id"), "source_id": batch.get("source_id")}

            M = build_supervision_mask(cand, valid_m, center)
            guard.check_mask(M)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs = model(rgb, onehot, **_batch_model_kwargs(batch, device))
                losses = criterion(outputs, label, cand, valid_m, center, global_step=global_step)
                loss = losses["total"] / cfg.train.grad_accum

            guard.check_losses(losses)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            update_loss_meter(step_meter, losses)

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = float(clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            cfg.train.grad_clip,
            error_if_nonfinite=True,
        ))
        guard.check_grads(model)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        try:
            guard.check_params(model)
        except (ValueError, FloatingPointError) as exc:
            dump_path = run_dir / f"nan_dump_step_{global_step}"
            save_nan_dump(dump_path, global_step, batch_meta, losses, outputs, scaler, optimizer)
            LOGGER.error("Numerical failure at step %d: %s (dump saved)", global_step, exc)
            raise

        step_avg_losses = average_loss_meter(step_meter, cfg.train.grad_accum)
        for key, value in step_avg_losses.items():
            running_meter[key] += value
        running_count += 1

        dt = time.time() - t0
        lr_current = optimizer.param_groups[-1]["lr"]
        pbar.set_postfix(
            total=f"{step_avg_losses['total']:.4f}",
            ce=f"{step_avg_losses['ce']:.4f}",
            dice=f"{step_avg_losses['dice']:.4f}",
            tr=f"{step_avg_losses['transition']:.4f}",
            aff=f"{step_avg_losses['affinity']:.4f}",
            bnd=f"{step_avg_losses['boundary']:.4f}",
            aux=f"{step_avg_losses['aux']:.4f}",
            lr=f"{lr_current:.2e}",
        )

        if global_step % 50 == 0:
            train_avg = average_loss_meter(running_meter, running_count)
            LOGGER.info(
                "TRAIN step=%d/%d %s lr=%.2e grad_norm=%.4f time=%.2fs",
                global_step,
                cfg.train.total_steps,
                format_loss_summary(train_avg),
                lr_current,
                grad_norm,
                dt,
            )

        if global_step > 0 and global_step % eval_every == 0 and score_loader is not None:
            train_avg = average_loss_meter(running_meter, running_count)
            max_batches = cfg.train.train_eval_max_batches if overfit else None
            score_metrics = validate_crop(
                model,
                score_loader,
                criterion,
                device,
                amp_dtype,
                global_step,
                max_batches=max_batches,
                desc="train-score" if overfit else "val",
            )
            score = score_metrics["macro_miou"]
            tag = "TRAIN_SCORE" if overfit else "VAL"
            LOGGER.info(
                "%s step=%d | train[%s] | eval[%s] mIoU=%.4f",
                tag,
                global_step,
                format_loss_summary(train_avg),
                format_loss_summary(score_metrics),
                score,
            )

            ckpt_kwargs = dict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                epoch=epoch,
                best_score=best_score,
                resolved_config=resolved,
                profile=profile.name,
                manifest_hash=man_hashes,
                rng_states=get_rng_states(),
            )

            if not torch.isfinite(torch.tensor(score)):
                LOGGER.error("Non-finite score; skipping checkpoint save")
            else:
                save_checkpoint(run_dir / "last.pt", **ckpt_kwargs)
                if score > best_score:
                    best_score = score
                    patience_counter = 0
                    ckpt_kwargs["best_score"] = best_score
                    save_checkpoint(run_dir / "best.pt", **ckpt_kwargs)
                    LOGGER.info(
                        "New best %s mIoU: %.4f",
                        "train" if overfit else "val",
                        best_score,
                    )
                else:
                    patience_counter += 1

            if (not overfit) and global_step % cfg.train.full_image_val_every_steps == 0:
                run_full_image_validation(model, cfg, device)

            if (not overfit) and patience_counter >= cfg.train.early_stop_patience_full_image:
                LOGGER.info("Early stopping at step %d", global_step)
                break

            model.train()
            running_meter = init_loss_meter()
            running_count = 0

    if best_score < 0:
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=max(cfg.train.total_steps - 1, start_step),
            epoch=epoch,
            best_score=best_score,
            resolved_config=resolved,
            profile=profile.name,
            manifest_hash=man_hashes,
            rng_states=get_rng_states(),
        )

    LOGGER.info("Training complete. Best score: %.4f", best_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

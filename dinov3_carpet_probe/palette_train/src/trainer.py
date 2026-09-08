"""Training and evaluation loop for palette prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dinov3_carpet_probe.palette_train.src.checkpointing import load_checkpoint, save_checkpoint
from dinov3_carpet_probe.palette_train.src.dataset import (
    PaletteTrainingDataset,
    collate_palette_batch,
    load_or_create_split_manifest,
    select_split,
    validate_dataset,
)
from dinov3_carpet_probe.palette_train.src.losses import compute_palette_losses
from dinov3_carpet_probe.palette_train.src.metrics import evaluate_palette_batch
from dinov3_carpet_probe.palette_train.src.model import PalettePredictor
from dinov3_carpet_probe.palette_train.src.transforms import build_transform
from dinov3_carpet_probe.src.io_utils import ensure_dir, write_json
from dinov3_carpet_probe.src.reproducibility import base_run_metadata, set_seed


@dataclass
class TrainingArtifacts:
    run_dir: Path
    checkpoint_last: Path
    checkpoint_best: Path
    metrics_path: Path
    manifest_path: Path


def _make_scheduler(optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int) -> LambdaLR:
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_dataloader(cfg: dict[str, Any], records: list[Any], *, training: bool) -> DataLoader:
    dataset = PaletteTrainingDataset(records, build_transform(cfg, training=training))
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=training,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=bool(cfg.get("persistent_workers", False)),
        collate_fn=collate_palette_batch,
    )


def prepare_training_state(cfg: dict[str, Any]) -> tuple[PalettePredictor, dict[str, Any], dict[str, list[Any]], Path]:
    set_seed(int(cfg["seed"]))
    records, report = validate_dataset(
        cfg["image_dir"],
        cfg["palette_dir"],
        max_num_colors_override=cfg.get("max_num_colors"),
        show_progress=True,
    )
    if report.errors:
        raise ValueError("\n".join(report.errors))
    cfg["max_num_colors"] = int(report.max_num_colors)
    manifest, manifest_path = load_or_create_split_manifest(
        records,
        manifest_dir=cfg["manifest_dir"],
        split_seed=int(cfg["split_seed"]),
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
    )
    splits = {
        "train": select_split(records, manifest, "train"),
        "val": select_split(records, manifest, "val"),
        "test": select_split(records, manifest, "test"),
    }
    model = PalettePredictor(cfg).to(cfg["device"])
    return model, manifest, splits, manifest_path


def run_eval(model: PalettePredictor, loader: DataLoader, cfg: dict[str, Any]) -> dict[str, float]:
    model.eval()
    metrics_accum: list[dict[str, float]] = []
    loss_accum: list[dict[str, float]] = []
    with torch.no_grad():
        eval_bar = tqdm(loader, desc="validation", unit="batch", dynamic_ncols=True, leave=False)
        for batch in eval_bar:
            images = batch["images"].to(cfg["device"], non_blocking=True)
            raw_images = batch["raw_images"].to(cfg["device"], non_blocking=True)
            valid_masks = batch["valid_masks"].to(cfg["device"], non_blocking=True)
            outputs = model(images, raw_images, valid_masks)
            losses, _ = compute_palette_losses(
                outputs,
                batch,
                matcher_rgb_weight=float(cfg["matcher_rgb_weight"]),
                matcher_perceptual_weight=float(cfg["matcher_perceptual_weight"]),
                matcher_presence_weight=float(cfg["matcher_presence_weight"]),
                loss_rgb_weight=float(cfg["loss_rgb"]),
                loss_perceptual_weight=float(cfg["loss_perceptual"]),
                loss_presence_weight=float(cfg["loss_presence"]),
                loss_count_weight=float(cfg["loss_count"]),
                loss_count_consistency_weight=float(cfg["loss_count_consistency"]),
                aux_loss_weight=float(cfg.get("aux_loss_weight", 0.5)),
            )
            metrics = evaluate_palette_batch(
                outputs,
                batch,
                matcher_rgb_weight=float(cfg["matcher_rgb_weight"]),
                matcher_perceptual_weight=float(cfg["matcher_perceptual_weight"]),
                matcher_presence_weight=float(cfg["matcher_presence_weight"]),
            )
            metrics_accum.append(metrics)
            loss_accum.append({key: float(val.detach().cpu().item()) for key, val in losses.items()})
            eval_bar.set_postfix(
                loss=f"{losses['loss_total'].detach().cpu().item():.4f}",
                de=f"{metrics.get('delta_e00_mean', 0.0):.3f}",
                count=f"{metrics.get('count_accuracy', 0.0):.3f}",
            )
    if not metrics_accum:
        return {}
    keys = sorted(metrics_accum[0].keys())
    merged = {f"val_{key}": sum(item[key] for item in metrics_accum) / len(metrics_accum) for key in keys}
    for key in loss_accum[0]:
        merged[f"val_{key}"] = sum(item[key] for item in loss_accum) / len(loss_accum)
    return merged


def train(
    cfg: dict[str, Any],
    *,
    resume: Path | None = None,
    limit_train_samples: int | None = None,
    limit_val_samples: int | None = None,
) -> TrainingArtifacts:
    model, manifest, splits, manifest_path = prepare_training_state(cfg)
    if limit_train_samples is not None:
        splits["train"] = splits["train"][:limit_train_samples]
    if limit_val_samples is not None:
        splits["val"] = splits["val"][:limit_val_samples]

    train_loader = _build_dataloader(cfg, splits["train"], training=True)
    val_loader = _build_dataloader(cfg, splits["val"], training=False) if splits["val"] else None

    optimizer = AdamW(model.trainable_parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    total_steps = max(1, int(cfg["epochs"]) * max(1, len(train_loader)) // max(1, int(cfg.get("grad_accum_steps", 1))))
    scheduler = _make_scheduler(
        optimizer,
        warmup_steps=int(cfg.get("warmup_steps", 500)),
        total_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("amp", True)) and str(cfg["device"]).startswith("cuda"))

    run_meta = base_run_metadata(int(cfg["seed"]), cfg)
    run_dir = ensure_dir(Path(cfg["run_root"]) / run_meta["run_id"])
    checkpoint_last = run_dir / "checkpoint_last.pt"
    checkpoint_best = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"
    write_json(run_dir / "run_meta.json", run_meta)
    write_json(run_dir / "split_manifest.json", manifest)

    start_epoch = 0
    global_step = 0
    best_metric = float("inf") if not bool(cfg.get("maximize_best_metric", False)) else float("-inf")
    patience = 0

    if resume is not None:
        checkpoint = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        if checkpoint.get("best_metric") is not None:
            best_metric = float(checkpoint["best_metric"])

    metrics_log: list[dict[str, Any]] = []
    grad_accum = max(1, int(cfg.get("grad_accum_steps", 1)))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 20))
    early_stopping_min_epochs = int(cfg.get("early_stopping_min_epochs", 0))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    for epoch in range(start_epoch, int(cfg["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[dict[str, float]] = []
        train_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch + 1}/{int(cfg['epochs'])}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_idx, batch in train_bar:
            images = batch["images"].to(cfg["device"], non_blocking=True)
            raw_images = batch["raw_images"].to(cfg["device"], non_blocking=True)
            valid_masks = batch["valid_masks"].to(cfg["device"], non_blocking=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                outputs = model(images, raw_images, valid_masks)
                losses, _ = compute_palette_losses(
                    outputs,
                    batch,
                    matcher_rgb_weight=float(cfg["matcher_rgb_weight"]),
                    matcher_perceptual_weight=float(cfg["matcher_perceptual_weight"]),
                    matcher_presence_weight=float(cfg["matcher_presence_weight"]),
                    loss_rgb_weight=float(cfg["loss_rgb"]),
                    loss_perceptual_weight=float(cfg["loss_perceptual"]),
                    loss_presence_weight=float(cfg["loss_presence"]),
                    loss_count_weight=float(cfg["loss_count"]),
                    loss_count_consistency_weight=float(cfg["loss_count_consistency"]),
                    aux_loss_weight=float(cfg.get("aux_loss_weight", 0.5)),
                )
                loss = losses["loss_total"] / grad_accum
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), max_norm=float(cfg["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            epoch_losses.append({key: float(val.detach().cpu().item()) for key, val in losses.items()})
            lr = optimizer.param_groups[0]["lr"]
            train_bar.set_postfix(
                loss=f"{losses['loss_total'].detach().cpu().item():.4f}",
                rgb=f"{losses['loss_rgb'].detach().cpu().item():.4f}",
                perc=f"{losses['loss_perceptual'].detach().cpu().item():.4f}",
                count=f"{losses['loss_count'].detach().cpu().item():.4f}",
                pres=f"{losses['loss_presence'].detach().cpu().item():.4f}",
                lr=f"{lr:.2e}",
                step=global_step,
            )

        train_summary = {
            f"train_{key}": sum(item[key] for item in epoch_losses) / max(1, len(epoch_losses))
            for key in epoch_losses[0]
        }
        summary = {"epoch": epoch, "global_step": global_step}
        summary.update(train_summary)
        print(
            (
                f"[epoch {epoch + 1}] "
                f"train_loss={summary['train_loss_total']:.4f} "
                f"rgb={summary['train_loss_rgb']:.4f} "
                f"perc={summary['train_loss_perceptual']:.4f} "
                f"count={summary['train_loss_count']:.4f} "
                f"presence={summary['train_loss_presence']:.4f}"
            ),
            flush=True,
        )
        if val_loader is not None:
            summary.update(run_eval(model, val_loader, cfg))
            print(
                (
                    f"[epoch {epoch + 1}] "
                    f"val_loss={summary.get('val_loss_total', 0.0):.4f} "
                    f"val_de00={summary.get('val_delta_e00_mean', 0.0):.4f} "
                    f"val_count_acc={summary.get('val_count_accuracy', 0.0):.4f}"
                ),
                flush=True,
            )
            metric_name = str(cfg.get("best_metric", "val_delta_e00_mean"))
            current_metric = float(summary.get(metric_name, summary.get("val_loss_total", 0.0)))
            if bool(cfg.get("maximize_best_metric", False)):
                better = current_metric > (best_metric + early_stopping_min_delta)
            else:
                better = current_metric < (best_metric - early_stopping_min_delta)
            if better:
                best_metric = current_metric
                patience = 0
                print(f"[checkpoint] new best {metric_name}={best_metric:.6f}", flush=True)
                save_checkpoint(
                    checkpoint_best,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    config=cfg,
                    manifest=manifest,
                    best_metric=best_metric,
                )
            else:
                if epoch + 1 >= early_stopping_min_epochs:
                    patience += 1
                print(
                    (
                        f"[early_stopping] patience={patience}/{early_stopping_patience} "
                        f"(min_epochs={early_stopping_min_epochs}, min_delta={early_stopping_min_delta})"
                    ),
                    flush=True,
                )
        save_checkpoint(
            checkpoint_last,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            config=cfg,
            manifest=manifest,
            best_metric=best_metric if best_metric not in (float("inf"), float("-inf")) else None,
        )
        metrics_log.append(summary)
        write_json(metrics_path, metrics_log)
        print(f"[checkpoint] saved last -> {checkpoint_last}", flush=True)
        if (epoch + 1) >= early_stopping_min_epochs and patience >= early_stopping_patience:
            print("[training] early stopping triggered", flush=True)
            break
    return TrainingArtifacts(
        run_dir=run_dir,
        checkpoint_last=checkpoint_last,
        checkpoint_best=checkpoint_best,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
    )

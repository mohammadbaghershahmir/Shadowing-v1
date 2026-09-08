#!/usr/bin/env python3
"""Train CSN-V4 — 15k-step loop with detailed tqdm, validation, plots, checkpoints."""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
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

from csn_v4.config import config_to_dict, load_config
from csn_v4.data.dataset_factory import create_tile_dataset
from csn_v3.dino_encoder import save_dino_load_report
from csn_v4.evaluation.evaluator import CSNV4Evaluator
from csn_v4.factory import build_model
from csn_v4.losses import CSNV4Loss
from csn_v4.metrics.checkpoint_score import score_from_region_results
from csn_v4.metrics.region_metrics import RegionMetrics, aggregate_region_metrics, compute_tile_region_metrics
from csn_v4.training import MetricsLogger, PhaseController, ShapeAuditor
from csn_v4.training.checkpointing import CheckpointManager
from csn_v3.data.targets import load_gray_bmp
from shadow_model.optimizer_utils import audit_optimizer_coverage

LOGGER = logging.getLogger(__name__)

VAL_SPATIAL_MANIFEST = "manifests/val_spatial_fixed_bw.jsonl"
VAL_HOLDOUT_MANIFEST = "manifests/val_scene_holdout_bw.jsonl"

LOSS_POSTFIX_KEYS = (
    "total", "final_ce", "final_dice", "where", "level_ce",
    "transition", "affinity", "base_refined_consistency",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp(cfg, device):
    mode = cfg.train.amp
    if mode == "auto_bf16_else_fp16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler()


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
        if key in ("letterbox_meta", "tile_uid", "global_cache_key"):
            out[key] = vals
        elif key in ("affinity_targets", "affinity_valid"):
            out[key] = torch.stack(vals)
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def make_tile_loader(cfg, manifest_rel: str, *, shuffle: bool) -> DataLoader | None:
    try:
        ds = create_tile_dataset(
            cfg, manifest_rel,
            split_filter="train",
            eval_mode=not shuffle,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Could not build dataset for %s: %s", manifest_rel, exc)
        return None
    if len(ds) == 0:
        return None
    return DataLoader(
        ds,
        batch_size=cfg.train.micro_batch,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=cfg.data.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.data.dataloader_workers > 0,
    )


def _forward_batch(model, batch, device, amp_dtype, tf: float):
    lb = batch["local_bw"].to(device, non_blocking=True)
    cb = batch["context_bw"].to(device, non_blocking=True)
    fb = batch["full_bw"].to(device, non_blocking=True)
    cc = batch["crop_coords_norm"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        return model(lb, cb, fb, cc, teacher_forcing=tf, letterbox_meta=batch.get("letterbox_meta"))


def evaluate_tiles(
    model,
    loader,
    device,
    amp_dtype,
    *,
    max_tiles: int | None = None,
    desc: str = "eval",
    fast_metrics: bool = True,
) -> list[RegionMetrics]:
    was_training = model.training
    model.eval()
    results: list[RegionMetrics] = []
    n_done = 0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with torch.inference_mode():
        for batch in pbar:
            out = _forward_batch(model, batch, device, amp_dtype, 0.0)
            pred = out["refined_logits"].argmax(dim=1).cpu().numpy()
            for b in range(pred.shape[0]):
                tgt = batch["target_class"][b].cpu().numpy()
                trans = batch["transition_mask"][b].cpu().numpy()
                valid = batch["valid_mask"][b].cpu().numpy()
                black = batch["black_lock"][b].cpu().numpy()
                tid = int(batch["transform_id"][b])
                results.append(compute_tile_region_metrics(
                    pred[b], tgt, trans, valid, black, transform_id=tid,
                    skip_small_component=fast_metrics,
                ))
                n_done += 1
                if max_tiles is not None and n_done >= max_tiles:
                    break
            pbar.set_postfix(n=n_done)
            if max_tiles is not None and n_done >= max_tiles:
                break
    pbar.close()
    if was_training:
        model.train()
    LOGGER.info("%s: evaluated %d tiles", desc, n_done)
    return results


def _metrics_to_dict(results: list[RegionMetrics]) -> dict[str, float]:
    agg = aggregate_region_metrics(results)
    try:
        agg["checkpoint_score_v4"] = score_from_region_results(results)
    except ValueError:
        agg["checkpoint_score_v4"] = float(agg.get("global_miou", float("nan")))
    return agg


def _finite_score(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    return v


def _kind_score(summary: dict, kind: str) -> float | None:
    fallbacks = {
        "capacity": ("score_capacity", "cap_global_miou", "cap_checkpoint_score_v4"),
        "spatial": ("score_spatial", "spat_global_miou", "spat_checkpoint_score_v4"),
        "scene_val": ("score_scene_val", "scene_global_miou", "scene_checkpoint_score_v4"),
        "overall": ("checkpoint_score_v4", "global_miou"),
    }
    for key in fallbacks.get(kind, ()):
        s = _finite_score(summary.get(key))
        if s is not None:
            return s
    if kind == "scene_val":
        sh = summary.get("scene_holdout")
        if isinstance(sh, dict):
            return _finite_score(sh.get("global_miou"))
    return None


def write_validation_report(run_dir: Path, optimizer_step: int, summary: dict, report: dict) -> None:
    """Human-readable validation snapshot for monitoring while away."""
    lines = [
        f"optimizer_step={optimizer_step}",
        f"global_miou={summary.get('global_miou', 'n/a')}",
        f"core_miou={summary.get('core_miou', 'n/a')}",
        f"halo_miou={summary.get('halo_miou', 'n/a')}",
        f"seam_band_miou={summary.get('seam_band_miou', 'n/a')}",
        f"boundary_miou={summary.get('boundary_miou', 'n/a')}",
        f"worst_orient_global_miou={summary.get('worst_orient_global_miou', 'n/a')}",
        f"checkpoint_score_v4={summary.get('checkpoint_score_v4', 'n/a')}",
        "",
        "[capacity]",
    ]
    cap = report.get("capacity") or {}
    for k in ("global_miou", "boundary_miou", "checkpoint_score_v4", "worst_orient_global_miou"):
        if k in cap:
            lines.append(f"  {k}={cap[k]}")
    lines.append("[spatial]")
    spat = report.get("spatial") or {}
    for k in ("global_miou", "boundary_miou", "checkpoint_score_v4", "worst_orient_global_miou"):
        if k in spat:
            lines.append(f"  {k}={spat[k]}")
    lines.append("[scene_holdout]")
    sh = report.get("scene_holdout") or {}
    for k in ("global_miou", "boundary_miou", "checkpoint_score_v4", "worst_orient_global_miou"):
        if k in sh:
            lines.append(f"  {k}={sh[k]}")
    text = "\n".join(lines) + "\n"
    latest = run_dir / "validation_latest.txt"
    latest.write_text(text, encoding="utf-8")
    hist = run_dir / "validation_history.jsonl"
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"optimizer_step": optimizer_step, "summary": summary, "report": report}, sort_keys=True) + "\n")


def _per_orientation_dict(results: list[RegionMetrics]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in range(8):
        subset = [r for r in results if r.transform_id == t]
        if subset:
            out[str(t)] = aggregate_region_metrics(subset)
    return out


def evaluate_scene_holdout(
    model, cfg, device, *, max_scenes: int | None = None,
    fast_metrics: bool = True,
    max_orientations: int | None = None,
) -> tuple[list[RegionMetrics], dict[str, dict]]:
    root = Path(cfg.data.dataset_root)
    manifest = root / VAL_HOLDOUT_MANIFEST
    if not manifest.is_file():
        return [], {}
    evaluator = CSNV4Evaluator(model, cfg, device)
    all_metrics: list[RegionMetrics] = []
    per_scene: dict[str, dict] = {}
    with manifest.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    seen_scenes: set[str] = set()
    scene_records: list[dict] = []
    for rec in records:
        sid = rec["scene_id"]
        if sid in seen_scenes:
            continue
        seen_scenes.add(sid)
        scene_records.append(rec)
        if max_scenes is not None and len(scene_records) >= max_scenes:
            break

    n_orient = max_orientations
    if n_orient is None:
        n_orient = int(getattr(cfg.train, "eval_holdout_max_orientations", 2))
    n_orient = max(1, min(8, n_orient))
    orientations = list(range(n_orient))
    LOGGER.info(
        "scene holdout: %d scenes × %d D4 orients (fast_metrics=%s)",
        len(scene_records), n_orient, fast_metrics,
    )

    for rec in tqdm(scene_records, desc="scene holdout", leave=True, dynamic_ncols=True):
        src = load_gray_bmp(rec["source_bw_path"])
        sem = load_gray_bmp(rec["target_semantic_path"])
        LOGGER.info(
            "scene holdout start: %s shape=%dx%d",
            rec["scene_id"], src.shape[1], src.shape[0],
        )
        full_dir = Path(rec.get("full_artifact_dir", ""))
        trans = load_gray_bmp(full_dir / "transition_mask.bmp") if (full_dir / "transition_mask.bmp").exists() else None
        valid = load_gray_bmp(full_dir / "valid_mask.bmp") if (full_dir / "valid_mask.bmp").exists() else None
        black = load_gray_bmp(full_dir / "black_lock.bmp") if (full_dir / "black_lock.bmp").exists() else None
        scene_res = evaluator.evaluate_all_orientations(
            src, sem, scene_id=rec["scene_id"],
            transition_mask=trans, valid_mask=valid, black_lock=black,
            require_all=False,
            skip_small_component=fast_metrics,
            orientations=orientations,
            show_progress=True,
        )
        all_metrics.extend(scene_res.per_orientation.values())
        per_scene[rec["scene_id"]] = scene_res.aggregate
    LOGGER.info("scene holdout: evaluated %d scenes (%d orient metrics)", len(per_scene), len(all_metrics))
    return all_metrics, per_scene


def _tile_eval_limit(cfg, *, initial: bool = False) -> int | None:
    if cfg.train.eval_all_tiles:
        return None
    n = cfg.train.eval_max_tiles
    if initial:
        n = min(n, cfg.train.initial_eval_max_tiles)
    return n


def format_tqdm_postfix(
    losses: dict,
    *,
    optimizer_step: int,
    total_opt_steps: int,
    lr: float,
    eval_metrics: dict | None = None,
    aff_valid_frac: float | None = None,
) -> dict[str, str]:
    pf: dict[str, str] = {
        "step": f"{optimizer_step}/{total_opt_steps}",
        "lr": f"{lr:.1e}",
    }
    short = {
        "total": "tot",
        "final_ce": "fce",
        "final_dice": "fdc",
        "where": "whr",
        "level_ce": "lce",
        "transition": "trn",
        "affinity": "aff",
        "base_refined_consistency": "brc",
    }
    for k in LOSS_POSTFIX_KEYS:
        v = losses.get(k)
        if v is not None:
            val = float(v.item() if hasattr(v, "item") else v)
            label = short.get(k, k[:3])
            if val == 0.0:
                pf[label] = "0"
            elif abs(val) < 0.01:
                pf[label] = f"{val:.6f}"
            elif abs(val) < 1.0:
                pf[label] = f"{val:.4f}"
            else:
                pf[label] = f"{val:.3f}"
    if aff_valid_frac is not None:
        pf["aff%"] = f"{aff_valid_frac * 100:.1f}"
    if eval_metrics:
        pf["gm"] = f"{eval_metrics.get('global_miou', 0):.4f}"
        pf["bm"] = f"{eval_metrics.get('boundary_miou', 0):.4f}"
        pf["scr"] = f"{eval_metrics.get('checkpoint_score_v4', 0):.4f}"
        pf["wd4"] = f"{eval_metrics.get('worst_orient_global_miou', 0):.4f}"
    return pf


def run_validation(
    model,
    cfg,
    device,
    amp_dtype,
    *,
    train_loader: DataLoader | None,
    spatial_loader: DataLoader | None,
    include_scene_holdout: bool = False,
    initial: bool = False,
) -> tuple[dict, dict]:
    report: dict = {"eval_tile_limit": _tile_eval_limit(cfg, initial=initial)}
    summary: dict[str, float] = {}
    max_tiles = _tile_eval_limit(cfg, initial=initial)

    cap_results: list[RegionMetrics] = []
    if train_loader is not None:
        cap_results = evaluate_tiles(
            model, train_loader, device, amp_dtype,
            max_tiles=max_tiles, desc="eval capacity",
        )
        cap_agg = _metrics_to_dict(cap_results)
        report["capacity"] = cap_agg
        report["capacity_n_tiles"] = len(cap_results)
        report["capacity_per_orientation"] = _per_orientation_dict(cap_results)
        summary.update({f"cap_{k}": v for k, v in cap_agg.items()})
        summary["score_capacity"] = cap_agg.get("checkpoint_score_v4", float("nan"))

    spat_results: list[RegionMetrics] = []
    if spatial_loader is not None:
        spat_results = evaluate_tiles(
            model, spatial_loader, device, amp_dtype,
            max_tiles=max_tiles, desc="eval spatial",
        )
        spat_agg = _metrics_to_dict(spat_results)
        report["spatial"] = spat_agg
        report["spatial_n_tiles"] = len(spat_results)
        report["spatial_per_orientation"] = _per_orientation_dict(spat_results)
        summary.update({f"spat_{k}": v for k, v in spat_agg.items()})
        summary["score_spatial"] = spat_agg.get("checkpoint_score_v4", float("nan"))

    if include_scene_holdout:
        max_scenes = None if cfg.train.eval_all_tiles else cfg.train.eval_max_holdout_scenes
        scene_results, per_scene = evaluate_scene_holdout(
            model, cfg, device,
            max_scenes=max_scenes,
            fast_metrics=True,
        )
        if scene_results:
            scene_agg = _metrics_to_dict(scene_results)
            report["scene_holdout"] = scene_agg
            report["scene_holdout_per_scene"] = per_scene
            report["scene_per_orientation"] = _per_orientation_dict(scene_results)
            summary["score_scene_val"] = scene_agg.get("checkpoint_score_v4", float("nan"))
            summary.update({f"scene_{k}": v for k, v in scene_agg.items()})
    else:
        scene_results = []

    if cfg.train.mode == "generalization" and spat_results:
        primary = spat_results
    else:
        primary = cap_results or spat_results or scene_results
    if primary:
        primary_agg = _metrics_to_dict(primary)
        summary.update(primary_agg)
        report["per_orientation"] = _per_orientation_dict(primary)
    return summary, report


def save_best_checkpoints(
    ckpt_mgr: CheckpointManager,
    eval_summary: dict,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    global_step: int,
    optimizer_step: int,
    grad_accum_pending: int,
) -> None:
    for kind in ("capacity", "spatial", "scene_val", "overall"):
        score = _kind_score(eval_summary, kind)
        if score is None:
            continue
        saved = ckpt_mgr.save_best(
            kind, score,
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            global_step=global_step, optimizer_step=optimizer_step,
            grad_accum_pending=grad_accum_pending,
            metrics=eval_summary,
        )
        if saved:
            LOGGER.info("★ New best_%s score=%.4f → %s", kind if kind != "overall" else "overall (best.pt)", score, saved)


def run_eval_step(
    *,
    model,
    cfg,
    device,
    amp_dtype,
    eval_train_loader,
    eval_spatial_loader,
    optimizer_step: int,
    global_step: int,
    run_dir: Path,
    ckpt_mgr: CheckpointManager,
    metrics_logger: MetricsLogger,
    optimizer,
    scheduler,
    scaler,
    grad_accum_pending: int,
    include_scene_holdout: bool = False,
    initial: bool = False,
) -> dict:
    LOGGER.info(
        "Validation opt=%d (tile_limit=%s, scene_holdout=%s) …",
        optimizer_step,
        _tile_eval_limit(cfg, initial=initial),
        include_scene_holdout,
    )
    eval_summary, eval_report = run_validation(
        model, cfg, device, amp_dtype,
        train_loader=eval_train_loader,
        spatial_loader=eval_spatial_loader,
        include_scene_holdout=include_scene_holdout,
        initial=initial,
    )
    eval_report["optimizer_step"] = optimizer_step
    eval_report["global_step"] = global_step
    metrics_logger.log_eval(
        global_step, eval_summary,
        optimizer_step=optimizer_step,
        report=eval_report,
    )
    write_validation_report(run_dir, optimizer_step, eval_summary, eval_report)
    ckpt_mgr.save_last(
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        global_step=global_step, optimizer_step=optimizer_step,
        grad_accum_pending=grad_accum_pending,
        metrics=eval_summary,
    )
    save_best_checkpoints(
        ckpt_mgr, eval_summary,
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        global_step=global_step, optimizer_step=optimizer_step,
        grad_accum_pending=grad_accum_pending,
    )
    LOGGER.info(
        "VALIDATION opt=%d | primary gm=%.4f bm=%.4f wd4=%.4f score=%.4f | "
        "cap=%.4f spat=%.4f scene=%.4f",
        optimizer_step,
        eval_summary.get("global_miou", float("nan")),
        eval_summary.get("boundary_miou", float("nan")),
        eval_summary.get("worst_orient_global_miou", float("nan")),
        eval_summary.get("checkpoint_score_v4", float("nan")),
        _kind_score(eval_summary, "capacity") or float("nan"),
        _kind_score(eval_summary, "spatial") or float("nan"),
        _kind_score(eval_summary, "scene_val") or float("nan"),
    )
    return eval_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CSN-V4")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v4_generalization.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="Override train.optimizer_steps")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--workers", type=int, default=None, help="Override data.dataloader_workers (use 0 on Windows)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.train.optimizer_steps = args.steps
    if args.workers is not None:
        cfg.data.dataloader_workers = max(0, int(args.workers))

    set_seed(cfg.train.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(cfg.data.dataset_root)

    run_dir: Path
    ckpt_mgr: CheckpointManager
    resume_state: dict | None = None

    if args.resume:
        resume_path = args.resume.resolve()
        run_dir = args.run_dir.resolve() if args.run_dir else resume_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_mgr = CheckpointManager(run_dir, cfg, config_path=args.config)
        resume_state = {"path": resume_path}
    else:
        run_dir = Path(args.run_dir or (
            Path("runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_CSN_V4_{cfg.data.input_mode}_{cfg.train.mode}"
        ))
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_mgr = CheckpointManager(run_dir, cfg, config_path=args.config)

    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config_to_dict(cfg), fh, indent=2)

    train_loader = make_tile_loader(cfg, cfg.data.train_tiles, shuffle=True)
    if train_loader is None:
        raise SystemExit(f"Empty/missing train manifest: {Path(cfg.data.dataset_root) / cfg.data.train_tiles}")

    from csn_v4.data.dataset_factory import materialized_index_for_manifest
    mat_idx = materialized_index_for_manifest(cfg, cfg.data.train_tiles)
    if mat_idx:
        LOGGER.info("Using materialized tiles: %s", mat_idx)
    else:
        LOGGER.info("Using runtime scene extraction (slow). Run: python scripts/materialize_csn_v4_tiles.py")

    eval_train_loader = make_tile_loader(cfg, cfg.data.train_tiles, shuffle=False)
    eval_spatial_loader = make_tile_loader(cfg, VAL_SPATIAL_MANIFEST, shuffle=False)

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
    criterion = CSNV4Loss(cfg.loss)
    amp_dtype, scaler = resolve_amp(cfg, device)
    scheduler = build_scheduler(optimizer, cfg)

    opt_total = cfg.train.optimizer_steps
    global_step = 0
    optimizer_step = 0
    grad_accum_pending = 0
    last_eval_metrics: dict | None = None
    loaded_state: dict | None = None

    if resume_state:
        loaded_state = ckpt_mgr.load(
            resume_state["path"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        global_step = int(loaded_state.get("global_step", 0))
        optimizer_step = int(loaded_state.get("optimizer_step", 0))
        grad_accum_pending = int(loaded_state.get("grad_accum_pending", 0))
        last_eval_metrics = loaded_state.get("metrics")
        LOGGER.info("Resumed from opt_step=%d micro=%d pending_accum=%d", optimizer_step, global_step, grad_accum_pending)

    if resume_state is None and not args.skip_audit:
        audit_optimizer_coverage(model, optimizer)
        auditor = ShapeAuditor()
        auditor.save(
            run_dir / "shape_audit.json",
            auditor.audit_forward(model, train_loader.dataset[0], device),
            auditor.audit_backward(model, train_loader.dataset[0], device),
        )

    metrics_logger = MetricsLogger(run_dir, cfg.train.plot_every_steps)
    model.train()

    if optimizer_step == 0 and not args.skip_initial_eval and not cfg.train.skip_initial_eval:
        LOGGER.info(
            "Running initial validation (step 0, max %d tiles/manifest, no scene holdout) …",
            cfg.train.initial_eval_max_tiles,
        )
        last_eval_metrics = run_eval_step(
            model=model, cfg=cfg, device=device, amp_dtype=amp_dtype,
            eval_train_loader=eval_train_loader, eval_spatial_loader=eval_spatial_loader,
            optimizer_step=optimizer_step, global_step=global_step, run_dir=run_dir,
            ckpt_mgr=ckpt_mgr, metrics_logger=metrics_logger,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            grad_accum_pending=grad_accum_pending,
            include_scene_holdout=False,
            initial=True,
        )
        model.train()
    elif optimizer_step == 0:
        LOGGER.info("Skipping initial validation (--skip-initial-eval or config)")

    opt_total = cfg.train.optimizer_steps

    pbar = tqdm(
        total=opt_total,
        initial=optimizer_step,
        desc=f"CSN-V4 [{cfg.train.mode}]",
        dynamic_ncols=True,
        unit="step",
    )

    data_iter = iter(train_loader)
    loss_accum: dict[str, float] = {}
    aff_frac_accum = 0.0
    while optimizer_step < opt_total:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        tf = phase_ctrl.teacher_forcing_prob(optimizer_step, opt_total)
        out = _forward_batch(model, batch, device, amp_dtype, tf)
        target_keys = (
            "target_class", "shade_mask", "transition_mask", "black_lock", "valid_mask",
            "where_target", "level_target", "affinity_targets", "affinity_valid", "supervision_weights",
        )
        targets = {
            k: batch[k].to(device, non_blocking=True) if isinstance(batch[k], torch.Tensor) else batch[k]
            for k in target_keys if k in batch
        }
        losses = criterion(out, targets, global_step)
        loss = losses["total"] / cfg.train.grad_accum
        aff_valid_frac = float((targets["affinity_valid"] > 0.5).float().mean().item())
        for k, v in losses.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + float(v.item())
        aff_frac_accum += aff_valid_frac

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_accum_pending += 1

        did_opt_step = False
        if grad_accum_pending >= cfg.train.grad_accum:
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
            grad_accum_pending = 0
            did_opt_step = True

            n_micro = max(1, cfg.train.grad_accum)
            loss_row = {k: v / n_micro for k, v in loss_accum.items()}
            loss_row["lr"] = optimizer.param_groups[0]["lr"]
            loss_row["aff_valid_frac"] = aff_frac_accum / n_micro
            metrics_logger.log_loss(optimizer_step, loss_row, optimizer_step=optimizer_step)
            loss_accum = {}
            aff_frac_accum = 0.0

            pbar.set_postfix(
                format_tqdm_postfix(
                    {k: torch.tensor(v) for k, v in loss_row.items() if k not in ("lr", "aff_valid_frac")},
                    optimizer_step=optimizer_step,
                    total_opt_steps=opt_total,
                    lr=loss_row["lr"],
                    eval_metrics=last_eval_metrics,
                    aff_valid_frac=loss_row.get("aff_valid_frac"),
                ),
                refresh=True,
            )
            pbar.update(1)

        global_step += 1

        if did_opt_step and optimizer_step > 0 and optimizer_step % cfg.train.checkpoint_every_steps == 0:
            ckpt_mgr.save_last(
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                global_step=global_step, optimizer_step=optimizer_step,
                grad_accum_pending=grad_accum_pending,
                metrics={"last_loss": loss_row},
            )

        if did_opt_step and optimizer_step > 0 and optimizer_step % cfg.train.eval_every_steps == 0:
            # capacity + spatial tiles + scene holdout (3 scenes, limited D4) every eval
            last_eval_metrics = run_eval_step(
                model=model, cfg=cfg, device=device, amp_dtype=amp_dtype,
                eval_train_loader=eval_train_loader, eval_spatial_loader=eval_spatial_loader,
                optimizer_step=optimizer_step, global_step=global_step, run_dir=run_dir,
                ckpt_mgr=ckpt_mgr, metrics_logger=metrics_logger,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                grad_accum_pending=grad_accum_pending,
                include_scene_holdout=True,
                initial=False,
            )
            model.train()

    pbar.close()

    if grad_accum_pending > 0:
        LOGGER.warning("Training ended with %d pending grad-accum micro-batches (not checkpointed).", grad_accum_pending)
        clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        grad_accum_pending = 0

    if optimizer_step > 0 and (optimizer_step % cfg.train.eval_every_steps != 0):
        LOGGER.info("Final validation at end of training …")
        last_eval_metrics = run_eval_step(
            model=model, cfg=cfg, device=device, amp_dtype=amp_dtype,
            eval_train_loader=eval_train_loader, eval_spatial_loader=eval_spatial_loader,
            optimizer_step=optimizer_step, global_step=global_step, run_dir=run_dir,
            ckpt_mgr=ckpt_mgr, metrics_logger=metrics_logger,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            grad_accum_pending=0,
        )

    ckpt_mgr.save_last(
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        global_step=global_step, optimizer_step=optimizer_step,
        grad_accum_pending=0,
        metrics=last_eval_metrics or {},
    )
    metrics_logger.plot_losses()
    metrics_logger.plot_eval()
    LOGGER.info(
        "Training complete: %d optimizer steps, run_dir=%s | best scores: %s",
        optimizer_step, run_dir, ckpt_mgr.best_scores,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

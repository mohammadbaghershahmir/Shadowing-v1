"""Periodic holdout prediction visualizations (unseen scenes only).

Saves the *raw* model prediction with the existing filenames/format, then
postprocesses an independent copy and writes sibling ``__postprocessed`` files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from csn_v3.data.targets import load_gray_bmp
from csn_v4.constants import GRAY_SHADE_100, GRAY_SHADE_150, GRAY_SHADE_200
from csn_v4.inference.tiled import predict_full_image_gray, predict_full_image_with_scores
from csn_v4.postprocess import PostprocessConfig, run_postprocess

LOGGER = logging.getLogger(__name__)

VAL_HOLDOUT_MANIFEST = "manifests/val_scene_holdout_bw.jsonl"

# Predicted shade tones only — outline/background come from the input.
_SHADE_GRAYS = frozenset({GRAY_SHADE_100, GRAY_SHADE_150, GRAY_SHADE_200})


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray)
    if g.ndim == 2:
        return np.stack([g, g, g], axis=-1).astype(np.uint8)
    if g.ndim == 3 and g.shape[-1] == 1:
        return np.repeat(g.astype(np.uint8), 3, axis=-1)
    return g.astype(np.uint8)


def _resize_max_side(rgb: np.ndarray, max_side: int = 768) -> np.ndarray:
    """Downscale for contact-sheet only. NEAREST keeps hard line/shade edges sharp."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 0.999:
        return rgb
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    return np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.NEAREST))


def _label_panel(rgb: np.ndarray, title: str) -> Image.Image:
    img = Image.fromarray(rgb)
    pad = 28
    canvas = Image.new("RGB", (img.width, img.height + pad), (32, 32, 32))
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((8, 6), title, fill=(240, 240, 240), font=font)
    return canvas


def compose_pred_with_fixed_input(source_bw: np.ndarray, pred_gray: np.ndarray) -> np.ndarray:
    """Display contract: copy input structure; only overlay predicted shades.

    Fixed from input (not re-predicted in the viz):
      - outline / black (0)
      - white background (255) unless a shade is predicted there

    Only gray tones {100,150,200} from the model are painted onto non-outline pixels.
    This keeps دورگیری visually identical to the input.
    """
    src = np.asarray(source_bw, dtype=np.uint8)
    pred = np.asarray(pred_gray, dtype=np.uint8)
    if src.shape != pred.shape:
        raise ValueError(f"shape mismatch source={src.shape} pred={pred.shape}")
    out = src.copy()
    outline = src == 0
    free = ~outline
    shade = np.isin(pred, list(_SHADE_GRAYS))
    out[free & shade] = pred[free & shade]
    # Force outline bit-exact from input
    out[outline] = 0
    return out


def _error_panel(
    pred_gray: np.ndarray,
    gt_gray: np.ndarray,
    source_bw: np.ndarray,
) -> np.ndarray:
    """Error only on free pixels (not input outline). Fixed outline shown as GT gray."""
    pred = pred_gray.astype(np.int32)
    gt = gt_gray.astype(np.int32)
    src = source_bw.astype(np.int32)
    fixed = src == 0
    comparable = ~fixed
    err = comparable & (pred != gt)
    match = comparable & (pred == gt)
    out = np.stack([gt, gt, gt], axis=-1).astype(np.uint8)
    # Fixed outline: cyan tint so it is visibly "locked", not scored
    out[fixed] = (40, 120, 180)
    out[err] = (220, 40, 40)
    out[match] = (40, 160, 40)
    return out


def _change_map_rgb(change_map: np.ndarray, source_bw: np.ndarray) -> np.ndarray:
    src = np.asarray(source_bw, dtype=np.uint8)
    base = _to_rgb(src)
    ch = np.asarray(change_map, dtype=bool)
    base[ch] = (220, 60, 40)
    return base


def build_holdout_contact_sheet(
    source_bw: np.ndarray,
    target_semantic: np.ndarray,
    pred_gray: np.ndarray,
    *,
    scene_id: str,
    step: int,
    max_side: int | None = 1024,
    title_pred: str | None = None,
) -> Image.Image:
    pred_display = compose_pred_with_fixed_input(source_bw, pred_gray)
    pred_title = title_pred or f"pred @ {step} (outline fixed)"
    panels = [
        _label_panel(_resize_max_side(_to_rgb(source_bw), max_side), f"{scene_id} | input (fixed)"),
        _label_panel(_resize_max_side(_to_rgb(target_semantic), max_side), "GT (holdout)"),
        _label_panel(
            _resize_max_side(_to_rgb(pred_display), max_side),
            pred_title,
        ),
        _label_panel(
            _resize_max_side(_error_panel(pred_display, target_semantic, source_bw), max_side),
            "error (cyan=locked outline)",
        ),
    ]
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + 6 * (len(panels) - 1)
    sheet = Image.new("RGB", (w, h), (20, 20, 20))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 6
    return sheet


def build_raw_vs_post_compare(
    source_bw: np.ndarray,
    raw_gray: np.ndarray,
    post_gray: np.ndarray,
    change_map: np.ndarray,
    *,
    scene_id: str,
    report_line: str,
    max_side: int | None = 1024,
) -> Image.Image:
    raw_d = compose_pred_with_fixed_input(source_bw, raw_gray)
    post_d = compose_pred_with_fixed_input(source_bw, post_gray)
    panels = [
        _label_panel(_resize_max_side(_to_rgb(raw_d), max_side), "Raw model output"),
        _label_panel(_resize_max_side(_to_rgb(post_d), max_side), "Postprocessed output"),
        _label_panel(
            _resize_max_side(_change_map_rgb(change_map, source_bw), max_side),
            f"change map | {report_line[:80]}",
        ),
    ]
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + 6 * (len(panels) - 1)
    sheet = Image.new("RGB", (w, h), (20, 20, 20))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 6
    return sheet


def save_fullres_panels(
    out_dir: Path,
    scene_id: str,
    source_bw: np.ndarray,
    target_semantic: np.ndarray,
    pred_gray: np.ndarray,
) -> list[Path]:
    """Save sharp full-resolution panels (no downscale)."""
    pred_display = compose_pred_with_fixed_input(source_bw, pred_gray)
    mapping = {
        f"{scene_id}__input.png": source_bw,
        f"{scene_id}__gt.png": target_semantic,
        f"{scene_id}__pred.png": pred_display,
        f"{scene_id}__error.png": _error_panel(pred_display, target_semantic, source_bw),
    }
    saved: list[Path] = []
    for name, arr in mapping.items():
        path = out_dir / name
        Image.fromarray(_to_rgb(arr)).save(path)
        saved.append(path)
    return saved


def _postprocess_cfg_from_train_cfg(cfg) -> PostprocessConfig:
    pp = getattr(cfg, "postprocess", None)
    if pp is None:
        # Optional nested dict on a plain namespace / Omega-free config
        raw = {}
        if hasattr(cfg, "__dict__"):
            raw = getattr(cfg, "__dict__", {}).get("postprocess") or {}
        if isinstance(raw, dict):
            return PostprocessConfig.from_mapping(raw)
        return PostprocessConfig(profile="conservative_cleanup")
    if isinstance(pp, PostprocessConfig):
        return pp
    if hasattr(pp, "__dict__"):
        return PostprocessConfig.from_mapping(vars(pp))
    if isinstance(pp, dict):
        return PostprocessConfig.from_mapping(pp)
    return PostprocessConfig(profile="conservative_cleanup")


def save_holdout_pred_viz(
    model,
    cfg,
    device: torch.device,
    run_dir: Path,
    *,
    optimizer_step: int,
    max_scenes: int = 2,
) -> list[Path]:
    """Predict on val_scene_holdout scenes (never used for training) and save PNGs.

    Raw outputs keep exact existing names:
      ``{sid}__t0.png``, ``{sid}__t0_pred_full.png``

    Postprocessed siblings (categorical data, not annotated overlays as source):
      ``{sid}__t0_pred_full__postprocessed.png``
      ``{sid}__t0__postprocessed.png``
      optional ``{sid}__t0_compare.png``, ``{sid}__t0_change_map.png``
      ``{sid}__t0_postprocess_report.json``
    """
    root = Path(cfg.data.dataset_root)
    manifest = root / VAL_HOLDOUT_MANIFEST
    if not manifest.is_file():
        LOGGER.warning("Holdout viz skipped: missing %s", manifest)
        return []

    with manifest.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        sid = rec["scene_id"]
        if sid in seen:
            continue
        seen.add(sid)
        unique.append(rec)
        if len(unique) >= max_scenes:
            break

    if not unique:
        LOGGER.warning("Holdout viz skipped: no scenes in %s", manifest)
        return []

    out_dir = Path(run_dir) / "pred_viz" / f"step_{optimizer_step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    pp_cfg = _postprocess_cfg_from_train_cfg(cfg)
    save_compare = True
    save_change = True
    pp_obj = getattr(cfg, "postprocess", None)
    if isinstance(pp_obj, dict):
        save_compare = bool(pp_obj.get("save_compare", True))
        save_change = bool(pp_obj.get("save_change_map", True))
    elif pp_obj is not None:
        save_compare = bool(getattr(pp_obj, "save_compare", True))
        save_change = bool(getattr(pp_obj, "save_change_map", True))

    was_training = model.training
    model.eval()
    try:
        for rec in unique:
            sid = rec["scene_id"]
            src = load_gray_bmp(rec["source_bw_path"])
            sem = load_gray_bmp(rec["target_semantic_path"])

            # Run model once — prefer scores for postprocess decisions.
            try:
                pred_gray, n_tiles, scores = predict_full_image_with_scores(
                    model,
                    src,
                    device,
                    context_size=cfg.model.context_size,
                    global_long_side=cfg.model.global_long_side,
                    transform_id=0,
                    scene_id=sid,
                    input_mode=getattr(cfg.data, "input_mode", "bw"),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("with_scores failed (%s); falling back to gray-only", exc)
                pred_gray, n_tiles = predict_full_image_gray(
                    model,
                    src,
                    device,
                    context_size=cfg.model.context_size,
                    global_long_side=cfg.model.global_long_side,
                    transform_id=0,
                    scene_id=sid,
                    input_mode=getattr(cfg.data, "input_mode", "bw"),
                )
                scores = None

            # Snapshot raw bytes for integrity checks after postprocess.
            raw_snapshot = pred_gray.copy()

            # --- RAW outputs (existing names / format) ---
            pred_display = compose_pred_with_fixed_input(src, pred_gray)
            sheet = build_holdout_contact_sheet(
                src, sem, pred_gray, scene_id=sid, step=optimizer_step,
            )
            path = out_dir / f"{sid}__t0.png"
            sheet.save(path)
            saved.append(path)
            full_path = out_dir / f"{sid}__t0_pred_full.png"
            Image.fromarray(_to_rgb(pred_display)).save(full_path)
            saved.append(full_path)

            # --- POSTPROCESS on an independent copy (categorical gray) ---
            result = run_postprocess(
                src,
                pred_gray,  # copied inside run_postprocess
                scores,
                config=pp_cfg,
            )
            # Guarantee raw array the caller still holds is bit-identical
            if not np.array_equal(pred_gray, raw_snapshot):
                raise RuntimeError("Postprocess mutated the raw prediction in place")

            post_display = compose_pred_with_fixed_input(src, result.repaired_gray)
            post_full = out_dir / f"{sid}__t0_pred_full__postprocessed.png"
            Image.fromarray(_to_rgb(post_display)).save(post_full)
            saved.append(post_full)

            post_sheet = build_holdout_contact_sheet(
                src,
                sem,
                result.repaired_gray,
                scene_id=sid,
                step=optimizer_step,
                title_pred=f"post @ {optimizer_step} [{result.report.profile}]",
            )
            post_sheet_path = out_dir / f"{sid}__t0__postprocessed.png"
            post_sheet.save(post_sheet_path)
            saved.append(post_sheet_path)

            report_path = out_dir / f"{sid}__t0_postprocess_report.json"
            report_path.write_text(
                json.dumps(result.report.to_dict(), indent=2),
                encoding="utf-8",
            )
            saved.append(report_path)

            if save_change:
                ch_path = out_dir / f"{sid}__t0_change_map.png"
                Image.fromarray(_change_map_rgb(result.change_map, src)).save(ch_path)
                saved.append(ch_path)
            if save_compare:
                cmp_path = out_dir / f"{sid}__t0_compare.png"
                build_raw_vs_post_compare(
                    src,
                    pred_gray,
                    result.repaired_gray,
                    result.change_map,
                    scene_id=sid,
                    report_line=result.report.summary_line(),
                ).save(cmp_path)
                saved.append(cmp_path)

            LOGGER.info(
                "Holdout pred viz: %s → raw %s + post %s (%d tiles) | %s",
                sid, path.name, post_full.name, n_tiles, result.report.summary_line(),
            )
    finally:
        model.train(was_training)

    latest = Path(run_dir) / "pred_viz" / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for p in saved:
        dest = latest / p.name
        dest.write_bytes(p.read_bytes())
    return saved

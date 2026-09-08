"""Region-specific and component metrics for CSN-V4."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from csn_v4.constants import CLASS_SHADE_100, CLASS_SHADE_150, CLASS_SHADE_200
from csn_v4.geometry.tile_spec import HALO, INPUT_SIZE, CORE_SIZE, core_starts, iter_core_tiles
from csn_v4.masks import as_bool_mask
from csn_v3.metrics.boundary_metrics import compute_boundary_eval
from shadow_model.metrics import ConfusionMatrix


@dataclass
class RegionMetrics:
    core_miou: float = 0.0
    core_acc: float = 0.0
    halo_miou: float = 0.0
    corner_miou: float = 0.0
    outer_border_miou: float = 0.0
    seam_band_miou: float = 0.0
    small_component_recall: float = float("nan")
    small_component_f1: float = float("nan")
    per_class_iou: list[float] = field(default_factory=list)
    per_class_presence: list[int] = field(default_factory=list)
    global_miou: float = 0.0
    global_acc: float = 0.0
    boundary_miou: float = 0.0
    bf1_2px: float = 0.0
    transform_id: int = 0

    def to_dict(self) -> dict:
        return {
            "core_miou": self.core_miou,
            "core_acc": self.core_acc,
            "halo_miou": self.halo_miou,
            "corner_miou": self.corner_miou,
            "outer_border_miou": self.outer_border_miou,
            "seam_band_miou": self.seam_band_miou,
            "small_component_recall": self.small_component_recall,
            "small_component_f1": self.small_component_f1,
            "global_miou": self.global_miou,
            "global_acc": self.global_acc,
            "boundary_miou": self.boundary_miou,
            "bf1_2px": self.bf1_2px,
            "transform_id": float(self.transform_id),
        }


def _presence_aware_miou(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, num_classes: int = 4) -> tuple[float, float, list[float], list[int]]:
    cm = ConfusionMatrix(num_classes=num_classes)
    if not mask.any():
        return 0.0, 0.0, [0.0] * num_classes, [0] * num_classes
    cm.update(pred, target, mask)
    present = [int((target[mask] == c).any()) for c in range(num_classes)]
    ious = cm.iou_per_class()
    active = [i for i, p in enumerate(present) if p]
    miou = float(np.mean([ious[i] for i in active])) if active else 0.0
    return miou, cm.pixel_accuracy(), ious, present


def _core_mask_from_specs(h: int, w: int) -> np.ndarray:
    specs = list(iter_core_tiles(w, h))
    core = np.zeros((h, w), dtype=bool)
    for spec in specs:
        cx, cy = spec.core_x, spec.core_y
        cw, ch = spec.core_w, spec.core_h
        core[cy : cy + ch, cx : cx + cw] |= spec.valid_core_mask
    return core


def _seam_mask_from_core_starts(h: int, w: int, seam_band: int = 32) -> np.ndarray:
    seam = np.zeros((h, w), dtype=bool)
    for x in core_starts(w)[1:]:
        x0 = max(0, x - seam_band)
        x1 = min(w, x + seam_band)
        seam[:, x0:x1] = True
    for y in core_starts(h)[1:]:
        y0 = max(0, y - seam_band)
        y1 = min(h, y + seam_band)
        seam[y0:y1, :] = True
    return seam


def _outer_border_mask(h: int, w: int, band: int = 64) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    b = min(band, h, w)
    m[:b, :] = True
    m[-b:, :] = True
    m[:, :b] = True
    m[:, -b:] = True
    return m


def _corner_mask(h: int, w: int, size: int = 64) -> np.ndarray:
    s = min(size, h, w)
    m = np.zeros((h, w), dtype=bool)
    m[:s, :s] = True
    m[:s, -s:] = True
    m[-s:, :s] = True
    m[-s:, -s:] = True
    return m


def _tile_halo_mask(h: int, w: int, halo: int = HALO) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    band = min(halo, h, w)
    m[:band, :] = True
    m[-band:, :] = True
    m[:, :band] = True
    m[:, -band:] = True
    return m


def _tile_core_mask(h: int, w: int, core_h: int, core_w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[HALO : HALO + core_h, HALO : HALO + core_w] = True
    return m


def _best_overlap_with_pred_components(
    pred: np.ndarray,
    cls: int,
    comp: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int]:
    pred_cls = (pred == cls) & mask
    labeled, n = ndimage.label(pred_cls)
    if n == 0:
        return 0, 0
    best_overlap = 0
    best_pred_size = 0
    for plab in range(1, n + 1):
        pcomp = labeled == plab
        overlap = int((pcomp & comp).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_pred_size = int(pcomp.sum())
    return best_overlap, best_pred_size


def _small_component_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_area: int = 256,
) -> tuple[float, float]:
    shade_classes = {CLASS_SHADE_200, CLASS_SHADE_150, CLASS_SHADE_100}
    recalls, f1s = [], []
    for cls in shade_classes:
        gt = (target == cls) & mask
        if not gt.any():
            continue
        labeled, n = ndimage.label(gt)
        for lab in range(1, n + 1):
            comp = labeled == lab
            area = int(comp.sum())
            if area > max_area:
                continue
            overlap, pred_size = _best_overlap_with_pred_components(pred, cls, comp, mask)
            rec = overlap / max(area, 1)
            prec = overlap / max(pred_size, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            recalls.append(rec)
            f1s.append(f1)
    if not recalls:
        return float("nan"), float("nan")
    return float(np.mean(recalls)), float(np.mean(f1s))


def compute_tile_region_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    transition_mask: np.ndarray,
    valid_mask: np.ndarray,
    black_lock: np.ndarray,
    *,
    core_h: int | None = None,
    core_w: int | None = None,
    transform_id: int = 0,
    boundary_dilate: int = 2,
    skip_small_component: bool = False,
) -> RegionMetrics:
    valid = as_bool_mask(valid_mask)
    black = as_bool_mask(black_lock)
    supervised = valid & ~black & (target >= 0)
    h, w = pred.shape[:2]
    ch = core_h if core_h is not None else min(CORE_SIZE, h - 2 * HALO)
    cw = core_w if core_w is not None else min(CORE_SIZE, w - 2 * HALO)

    base = compute_boundary_eval(
        pred, target, transition_mask, valid_mask, black_lock, boundary_dilate=boundary_dilate,
    )
    rm = RegionMetrics(transform_id=transform_id)
    rm.global_miou, rm.global_acc, ious, present = _presence_aware_miou(pred, target, supervised)
    rm.boundary_miou = base.boundary_miou
    rm.bf1_2px = base.bf1_2px

    core_m = _tile_core_mask(h, w, ch, cw) & supervised
    halo_m = _tile_halo_mask(h, w) & supervised

    rm.core_miou, rm.core_acc, _, _ = _presence_aware_miou(pred, target, core_m)
    rm.halo_miou, _, _, _ = _presence_aware_miou(pred, target, halo_m)
    rm.per_class_iou = ious
    rm.per_class_presence = present
    if skip_small_component:
        rm.small_component_recall = float("nan")
        rm.small_component_f1 = float("nan")
    else:
        rm.small_component_recall, rm.small_component_f1 = _small_component_metrics(pred, target, supervised)
    return rm


def compute_region_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    transition_mask: np.ndarray,
    valid_mask: np.ndarray,
    black_lock: np.ndarray,
    *,
    transform_id: int = 0,
    boundary_dilate: int = 2,
    skip_small_component: bool = False,
) -> RegionMetrics:
    valid = as_bool_mask(valid_mask)
    black = as_bool_mask(black_lock)
    supervised = valid & ~black & (target >= 0)
    h, w = pred.shape[:2]

    base = compute_boundary_eval(
        pred, target, transition_mask, valid_mask, black_lock, boundary_dilate=boundary_dilate,
    )
    rm = RegionMetrics(transform_id=transform_id)
    rm.global_miou, rm.global_acc, ious, present = _presence_aware_miou(pred, target, supervised)
    rm.boundary_miou = base.boundary_miou
    rm.bf1_2px = base.bf1_2px

    core_m = _core_mask_from_specs(h, w) & supervised
    seam_m = _seam_mask_from_core_starts(h, w) & supervised
    halo_m = np.zeros((h, w), dtype=bool)
    corner_m = _corner_mask(h, w) & supervised
    border_m = _outer_border_mask(h, w) & supervised

    rm.core_miou, rm.core_acc, _, _ = _presence_aware_miou(pred, target, core_m)
    rm.halo_miou = float("nan")
    rm.corner_miou, _, _, _ = _presence_aware_miou(pred, target, corner_m)
    rm.outer_border_miou, _, _, _ = _presence_aware_miou(pred, target, border_m)
    rm.seam_band_miou, _, _, _ = _presence_aware_miou(pred, target, seam_m)
    rm.per_class_iou = ious
    rm.per_class_presence = present
    if skip_small_component:
        rm.small_component_recall = float("nan")
        rm.small_component_f1 = float("nan")
    else:
        rm.small_component_recall, rm.small_component_f1 = _small_component_metrics(pred, target, supervised)
    return rm


def _nanmean(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    return float(np.mean(clean)) if clean else float("nan")


def aggregate_region_metrics(results: list[RegionMetrics]) -> dict[str, float]:
    if not results:
        return {}
    keys = [
        "core_miou", "halo_miou", "corner_miou", "outer_border_miou", "seam_band_miou",
        "global_miou", "boundary_miou", "bf1_2px", "small_component_recall", "small_component_f1",
    ]
    out = {k: _nanmean([getattr(r, k) for r in results]) for k in keys}
    orient_means = []
    for t in range(8):
        subset = [x.global_miou for x in results if x.transform_id == t]
        if subset:
            orient_means.append(float(np.mean(subset)))
    out["worst_orient_global_miou"] = float(min(orient_means)) if orient_means else float("nan")
    return out

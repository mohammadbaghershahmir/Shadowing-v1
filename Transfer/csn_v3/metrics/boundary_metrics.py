"""Boundary-restricted and interior metrics for CSN-V3."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from shadow_dataset.geometry import detect_class_boundaries
from shadow_model.metrics import ConfusionMatrix, compute_bf1


def dilate_binary_mask(mask: np.ndarray | torch.Tensor, radius: int = 2) -> np.ndarray:
    """Dilate bool/0-1 mask by max-pool (radius pixels)."""
    if radius <= 0:
        if isinstance(mask, torch.Tensor):
            return mask.cpu().numpy().astype(bool)
        return mask.astype(bool)
    if isinstance(mask, np.ndarray):
        t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    else:
        t = mask.float().unsqueeze(0).unsqueeze(0)
    k = 2 * radius + 1
    d = F.max_pool2d(t, k, stride=1, padding=radius)
    return (d.squeeze().numpy() > 0.5)


def _as_bool(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        return mask
    if np.issubdtype(mask.dtype, np.floating):
        return mask > 0.5
    return mask > 0


def build_boundary_eval_mask(
    transition_mask: np.ndarray,
    target_class: np.ndarray,
    valid_mask: np.ndarray,
    black_lock: np.ndarray,
    *,
    boundary_dilate: int = 2,
    use_class_boundaries: bool = True,
) -> np.ndarray:
    """Union of dilated transition mask and class boundaries, restricted to supervised pixels."""
    trans = _as_bool(transition_mask)
    valid = _as_bool(valid_mask)
    black = _as_bool(black_lock)
    supervised = valid & ~black & (target_class >= 0)

    boundary = dilate_binary_mask(trans, boundary_dilate)
    if use_class_boundaries:
        cls_b = detect_class_boundaries(target_class.astype(np.int32))
        cls_b = cls_b & (target_class >= 0)
        boundary = boundary | dilate_binary_mask(cls_b, max(1, boundary_dilate // 2))

    return boundary & supervised


@dataclass
class BoundaryEvalResult:
    global_miou: float = 0.0
    global_pixel_acc: float = 0.0
    boundary_miou: float = 0.0
    boundary_pixel_acc: float = 0.0
    interior_miou: float = 0.0
    interior_pixel_acc: float = 0.0
    bf1_2px: float = 0.0
    boundary_pixel_count: int = 0
    interior_pixel_count: int = 0
    per_class_iou_boundary: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {
            "global_miou": self.global_miou,
            "global_pixel_acc": self.global_pixel_acc,
            "boundary_miou": self.boundary_miou,
            "boundary_pixel_acc": self.boundary_pixel_acc,
            "interior_miou": self.interior_miou,
            "interior_pixel_acc": self.interior_pixel_acc,
            "bf1_2px": self.bf1_2px,
            "boundary_pixel_count": float(self.boundary_pixel_count),
            "checkpoint_score_v3": compute_checkpoint_score_v3(
                self.global_miou, self.boundary_miou, self.bf1_2px, self.boundary_pixel_acc,
            ),
        }


def _cm_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    num_classes: int = 4,
) -> tuple[float, float, list[float]]:
    cm = ConfusionMatrix(num_classes=num_classes)
    if mask.any():
        cm.update(pred, target, mask)
        return cm.macro_miou(), cm.pixel_accuracy(), cm.iou_per_class().tolist()
    return 0.0, 0.0, [0.0] * num_classes


def compute_boundary_eval(
    pred: np.ndarray,
    target: np.ndarray,
    transition_mask: np.ndarray,
    valid_mask: np.ndarray,
    black_lock: np.ndarray,
    *,
    boundary_dilate: int = 2,
    bf1_tolerance: int = 2,
    num_classes: int = 4,
) -> BoundaryEvalResult:
    """Compute global, boundary-only, and interior metrics for one sample."""
    valid = _as_bool(valid_mask)
    black = _as_bool(black_lock)
    supervised = valid & ~black & (target >= 0)

    boundary = build_boundary_eval_mask(
        transition_mask, target, valid_mask, black_lock,
        boundary_dilate=boundary_dilate,
    )
    interior = supervised & ~boundary

    result = BoundaryEvalResult()
    result.global_miou, result.global_pixel_acc, _ = _cm_metrics(pred, target, supervised, num_classes)
    result.boundary_miou, result.boundary_pixel_acc, result.per_class_iou_boundary = _cm_metrics(
        pred, target, boundary, num_classes,
    )
    result.interior_miou, result.interior_pixel_acc, _ = _cm_metrics(pred, target, interior, num_classes)
    result.boundary_pixel_count = int(boundary.sum())
    result.interior_pixel_count = int(interior.sum())

    gt_b = detect_class_boundaries(target.astype(np.int32)).astype(bool) & supervised
    pred_b = detect_class_boundaries(pred.astype(np.int32)).astype(bool) & supervised
    result.bf1_2px = compute_bf1(pred_b, gt_b, tolerance=bf1_tolerance)
    return result


def accumulate_boundary_eval(results: list[BoundaryEvalResult]) -> BoundaryEvalResult:
    """Average scalar metrics across samples (simple mean)."""
    if not results:
        return BoundaryEvalResult()
    out = BoundaryEvalResult()
    n = len(results)
    for r in results:
        out.global_miou += r.global_miou
        out.global_pixel_acc += r.global_pixel_acc
        out.boundary_miou += r.boundary_miou
        out.boundary_pixel_acc += r.boundary_pixel_acc
        out.interior_miou += r.interior_miou
        out.interior_pixel_acc += r.interior_pixel_acc
        out.bf1_2px += r.bf1_2px
        out.boundary_pixel_count += r.boundary_pixel_count
        out.interior_pixel_count += r.interior_pixel_count
    out.global_miou /= n
    out.global_pixel_acc /= n
    out.boundary_miou /= n
    out.boundary_pixel_acc /= n
    out.interior_miou /= n
    out.interior_pixel_acc /= n
    out.bf1_2px /= n
    return out


def compute_checkpoint_score_v3(
    global_miou: float,
    boundary_miou: float,
    bf1_2px: float,
    boundary_pixel_acc: float,
) -> float:
    return (
        0.35 * global_miou
        + 0.35 * boundary_miou
        + 0.20 * bf1_2px
        + 0.10 * boundary_pixel_acc
    )

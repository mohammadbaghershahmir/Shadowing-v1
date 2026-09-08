"""Categorical evaluation metrics."""
from __future__ import annotations
import numpy as np
from typing import Any


class ConfusionMatrix:
    """Accumulates a num_classes x num_classes confusion matrix."""

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        """pred, target: [H,W] int, mask: [H,W] bool."""
        valid = mask & (target >= 0) & (target < self.num_classes)
        p = pred[valid]
        t = target[valid]
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.matrix[i, j] += int(np.sum((t == i) & (p == j)))

    def reset(self) -> None:
        self.matrix[:] = 0

    def iou_per_class(self) -> np.ndarray:
        intersection = np.diag(self.matrix)
        union = self.matrix.sum(axis=1) + self.matrix.sum(axis=0) - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = intersection / np.maximum(union, 1)
        return iou

    def macro_miou(self) -> float:
        return float(self.iou_per_class().mean())

    def pixel_accuracy(self) -> float:
        correct = np.diag(self.matrix).sum()
        total = self.matrix.sum()
        return float(correct / max(total, 1))

    def precision_recall_f1(self) -> dict[str, np.ndarray]:
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = tp / np.maximum(tp + fp, 1)
            recall = tp / np.maximum(tp + fn, 1)
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        return {"precision": precision, "recall": recall, "f1": f1}

    def summary(self) -> dict[str, Any]:
        iou = self.iou_per_class()
        prf = self.precision_recall_f1()
        return {
            "macro_miou": float(iou.mean()),
            "per_class_iou": iou.tolist(),
            "pixel_accuracy": self.pixel_accuracy(),
            "macro_f1": float(prf["f1"].mean()),
            "per_class_precision": prf["precision"].tolist(),
            "per_class_recall": prf["recall"].tolist(),
            "per_class_f1": prf["f1"].tolist(),
            "confusion_matrix": self.matrix.tolist(),
        }


def compute_bf1(pred_boundary: np.ndarray, gt_boundary: np.ndarray, tolerance: int = 2) -> float:
    """Boundary F1 score at given pixel tolerance.

    pred_boundary, gt_boundary: [H, W] bool
    """
    from scipy import ndimage

    if not gt_boundary.any() and not pred_boundary.any():
        return 1.0
    if not gt_boundary.any() or not pred_boundary.any():
        return 0.0

    struct = ndimage.generate_binary_structure(2, 1)
    gt_dilated = ndimage.binary_dilation(gt_boundary, struct, iterations=tolerance)
    pred_dilated = ndimage.binary_dilation(pred_boundary, struct, iterations=tolerance)

    precision = float(np.sum(pred_boundary & gt_dilated)) / max(float(np.sum(pred_boundary)), 1)
    recall = float(np.sum(gt_boundary & pred_dilated)) / max(float(np.sum(gt_boundary)), 1)

    if precision + recall < 1e-8:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_checkpoint_score(
    macro_miou: float,
    bf1_2px: float,
    small_comp_f1: float,
    weights: tuple[float, float, float] = (0.55, 0.30, 0.15),
) -> float:
    """Weighted checkpoint selection score, all terms in [0,1]."""
    return weights[0] * macro_miou + weights[1] * bf1_2px + weights[2] * small_comp_f1


def compute_small_component_miou(
    pred: np.ndarray,
    target: np.ndarray,
    candidate: np.ndarray,
    *,
    tiny_threshold: int = 64,
) -> float:
    """Mean IoU over tiny connected components in the candidate mask."""
    from scipy import ndimage

    from shadow_dataset.geometry import find_candidate_components

    comps = find_candidate_components(candidate, connectivity=8)
    tiny = [c for c in comps if c.pixel_count <= tiny_threshold]
    if not tiny:
        return 1.0

    ious: list[float] = []
    valid = candidate & (target >= 0)
    for comp in tiny:
        mask = np.zeros(candidate.shape, dtype=bool)
        mask[comp.ys, comp.xs] = True
        region = mask & valid
        if not region.any():
            continue
        p = pred[region]
        t = target[region]
        inter = float(np.sum(p == t))
        union = float(np.sum((p != t) | (p == t)))
        ious.append(inter / max(union, 1.0))
    return float(np.mean(ious)) if ious else 1.0

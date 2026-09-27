"""Decision logic: compare model scores with a style proposal inside regions."""
from __future__ import annotations

import numpy as np

from csn_v4.constants import NUM_CLASSES
from csn_v4.postprocess.constraints import class_to_gray, gray_to_class
from csn_v4.postprocess.regions import MotifRegion


def _ensure_scores(scores: np.ndarray | None, pred_cls: np.ndarray) -> np.ndarray:
    """Return [C,H,W] float scores; synthesize one-hot from argmax if missing."""
    h, w = pred_cls.shape
    if scores is None:
        onehot = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        for c in range(NUM_CLASSES):
            onehot[c] = (pred_cls == c).astype(np.float32)
        return onehot
    s = np.asarray(scores, dtype=np.float32)
    if s.ndim != 3 or s.shape[1:] != (h, w):
        raise ValueError(f"scores shape {s.shape} incompatible with pred {(h, w)}")
    # Normalize if looks like logits (can be negative / not sum to 1)
    smax = s.max(axis=0, keepdims=True)
    exp = np.exp(s - smax)
    denom = exp.sum(axis=0, keepdims=True).clip(min=1e-8)
    probs = exp / denom
    # If already probabilities (non-negative, roughly sum 1), prefer them
    if float(s.min()) >= -1e-5 and np.allclose(s.sum(axis=0), 1.0, atol=0.05):
        return s
    return probs.astype(np.float32)


def apply_style_proposal(
    raw_gray: np.ndarray,
    scores: np.ndarray | None,
    regions: list[MotifRegion],
    proposal_cls: np.ndarray | None,
    *,
    confidence_threshold: float = 0.55,
    style_influence: float = 0.5,
    user_style_selected: bool = False,
    min_style_support: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Conservative per-pixel edits where model is uncertain and proposal is supported.

    Returns (repaired_gray, change_map, edited_labels, skipped_labels).
    """
    raw = np.asarray(raw_gray, dtype=np.uint8).copy()
    pred_cls = gray_to_class(raw)
    probs = _ensure_scores(scores, pred_cls)
    conf = probs.max(axis=0)
    out_cls = pred_cls.copy()
    change = np.zeros(pred_cls.shape, dtype=bool)
    edited: list[str] = []
    skipped: list[str] = []

    # Stronger influence when the user explicitly selected a decorative style.
    influence = float(style_influence)
    if user_style_selected:
        influence = min(1.0, influence + 0.2)
    conf_thresh = float(confidence_threshold)
    if user_style_selected:
        conf_thresh = min(0.95, conf_thresh + 0.05)

    if proposal_cls is None:
        return raw, change, edited, skipped

    prop = np.asarray(proposal_cls, dtype=np.int64)
    if prop.shape != pred_cls.shape:
        raise ValueError(f"proposal shape {prop.shape} != pred {pred_cls.shape}")

    for region in regions:
        if not region.reliable:
            skipped.append(f"{region.label}:{region.skip_reason}")
            continue
        m = region.mask
        region_changed = 0
        ys, xs = np.where(m)
        for y, x in zip(ys.tolist(), xs.tolist()):
            p_style = int(prop[y, x])
            p_model = int(pred_cls[y, x])
            if p_style == p_model:
                continue
            if p_style < 0 or p_style >= NUM_CLASSES:
                continue
            model_conf = float(conf[y, x])
            style_score = float(probs[p_style, y, x])
            # Automatic edit only when model is uncertain AND style is supported.
            supported = style_score >= min_style_support or (
                user_style_selected and influence >= 0.6 and style_score >= (min_style_support * 0.75)
            )
            uncertain = model_conf < conf_thresh
            # With explicit user style + high influence, allow slightly more assertive edits
            # still requiring style class to beat the current class score.
            assertive = (
                user_style_selected
                and influence >= 0.75
                and float(probs[p_style, y, x]) >= float(probs[p_model, y, x]) + 0.05
            )
            if (uncertain and supported) or assertive:
                out_cls[y, x] = p_style
                change[y, x] = True
                region_changed += 1
        if region_changed:
            edited.append(f"{region.label}:{region_changed}")
        else:
            skipped.append(f"{region.label}:no_confident_edits")

    repaired = class_to_gray(out_cls)
    # Preserve non-eligible pixels exactly (caller also re-applies constraints)
    return repaired, change, edited, skipped


def conservative_cleanup_edits(
    raw_gray: np.ndarray,
    scores: np.ndarray | None,
    regions: list[MotifRegion],
    *,
    confidence_threshold: float = 0.45,
    neighbor: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Fix isolated low-confidence speckles; preserve intentional small details.

    A pixel may change only if:
      - it lies in a reliable eligible region
      - model confidence is below threshold
      - a local majority of same-class neighbors exists
      - the pixel's class does not form a confident small decorative component
        (high-confidence members of the same connected shade blob nearby)
    """
    raw = np.asarray(raw_gray, dtype=np.uint8).copy()
    pred_cls = gray_to_class(raw)
    probs = _ensure_scores(scores, pred_cls)
    conf = probs.max(axis=0)
    out_cls = pred_cls.copy()
    change = np.zeros(pred_cls.shape, dtype=bool)
    edited: list[str] = []
    skipped: list[str] = []
    h, w = pred_cls.shape
    r = int(neighbor)

    for region in regions:
        if not region.reliable:
            skipped.append(f"{region.label}:{region.skip_reason}")
            continue
        m = region.mask
        # Precompute confident same-class presence inside region (protect details)
        region_changed = 0
        ys, xs = np.where(m)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if float(conf[y, x]) >= confidence_threshold:
                continue
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            window = pred_cls[y0:y1, x0:x1]
            wmask = m[y0:y1, x0:x1]
            vals = window[wmask]
            if vals.size < 5:
                continue
            # Majority excluding center
            center = int(pred_cls[y, x])
            # count classes
            counts = np.bincount(vals.astype(np.int64), minlength=NUM_CLASSES)
            counts[center] = max(0, counts[center] - 1)  # dilute center's own vote
            maj = int(np.argmax(counts))
            if maj == center or counts[maj] < max(3, vals.size // 2):
                continue
            # Protect intentional small details: if any high-confidence pixel of
            # the center class exists in the window, skip (likely decoration).
            center_conf_nearby = (window == center) & wmask & (conf[y0:y1, x0:x1] >= 0.7)
            if int(center_conf_nearby.sum()) >= 2:
                continue
            out_cls[y, x] = maj
            change[y, x] = True
            region_changed += 1
        if region_changed:
            edited.append(f"{region.label}:{region_changed}")
        else:
            skipped.append(f"{region.label}:clean")

    return class_to_gray(out_cls), change, edited, skipped

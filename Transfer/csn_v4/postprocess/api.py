"""Public postprocess API — copy-in / copy-out, never mutates raw prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from csn_v4.postprocess.constraints import apply_final_constraints, validate_result_palette
from csn_v4.postprocess.eligible import find_eligible_mask
from csn_v4.postprocess.profiles import profile_status, run_profile
from csn_v4.postprocess.regions import identify_motif_regions
from csn_v4.postprocess.types import PostprocessReport, PostprocessResult


@dataclass
class PostprocessConfig:
    profile: str = "conservative_cleanup"
    styles_root: str | Path | None = None
    confidence_threshold: float = 0.55
    style_influence: float = 0.5
    min_region_pixels: int = 16
    user_style_selected: bool = False
    # When True (decorative styles), refuse to guess editable area from BW alone.
    # Overridden automatically from the profile's requires_editable_mask.
    require_explicit_editable: bool | None = None

    @classmethod
    def from_mapping(cls, d: dict | None) -> "PostprocessConfig":
        d = d or {}
        return cls(
            profile=str(d.get("profile", "conservative_cleanup")),
            styles_root=d.get("styles_root"),
            confidence_threshold=float(d.get("confidence_threshold", 0.55)),
            style_influence=float(d.get("style_influence", 0.5)),
            min_region_pixels=int(d.get("min_region_pixels", 16)),
            user_style_selected=bool(d.get("user_style_selected", False)),
            require_explicit_editable=d.get("require_explicit_editable"),
        )


def run_postprocess(
    source_bw: np.ndarray,
    raw_pred_gray: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    editable_mask: np.ndarray | None = None,
    locked_mask: np.ndarray | None = None,
    source_index: np.ndarray | None = None,
    config: PostprocessConfig | None = None,
    profile: str | None = None,
) -> PostprocessResult:
    """Postprocess an *independent copy* of the model prediction.

    Parameters
    ----------
    source_bw
        Original input image (outline / structure). Never modified.
    raw_pred_gray
        Model categorical gray prediction. Copied immediately; caller array
        is never written to.
    scores
        Optional full-image per-pixel class scores shaped [4,H,W] (logits or
        probabilities) for classes white/200/150/100.
    editable_mask / locked_mask / source_index
        Optional region controls. Ground-truth targets must not be passed here.
    """
    cfg = config or PostprocessConfig()
    if profile is not None:
        cfg.profile = profile

    # Independent copies — raw must remain untouched for side-by-side inspection.
    raw = np.asarray(raw_pred_gray, dtype=np.uint8).copy()
    src = np.asarray(source_bw)
    if src.ndim == 3:
        src = src[..., 0]
    src = src.astype(np.uint8, copy=True)

    if raw.shape != src.shape[:2]:
        raise ValueError(f"raw pred shape {raw.shape} != source {src.shape}")

    info = profile_status(cfg.profile, styles_root=cfg.styles_root)
    require_mask = (
        cfg.require_explicit_editable
        if cfg.require_explicit_editable is not None
        else info.requires_editable_mask
    )
    # User explicitly selecting a decorative style also opts into stronger influence.
    user_selected = bool(cfg.user_style_selected) or (
        info.decorative and cfg.profile.lower() not in ("off", "conservative_cleanup", "cleanup", "conservative")
    )

    eligible, elig_info = find_eligible_mask(
        src,
        editable_mask=editable_mask,
        locked_mask=locked_mask,
        source_index=source_index,
        require_explicit_editable=bool(require_mask),
    )

    report = PostprocessReport(
        profile=info.name,
        available=info.available,
        ran=False,
        reason=info.reason,
        style_influence=float(cfg.style_influence),
        confidence_threshold=float(cfg.confidence_threshold),
        required_editable_mask=bool(require_mask),
        extras={"eligible": elig_info},
    )

    if info.name == "off":
        z = np.zeros(raw.shape, dtype=bool)
        return PostprocessResult(repaired_gray=raw.copy(), change_map=z, report=report, raw_gray=raw)

    if not info.available:
        z = np.zeros(raw.shape, dtype=bool)
        report.reason = info.reason
        return PostprocessResult(repaired_gray=raw.copy(), change_map=z, report=report, raw_gray=raw)

    if require_mask and not eligible.any():
        z = np.zeros(raw.shape, dtype=bool)
        report.reason = elig_info.get("reason") or info.reason
        report.regions_skipped = ["no_eligible_region"]
        return PostprocessResult(repaired_gray=raw.copy(), change_map=z, report=report, raw_gray=raw)

    regions = identify_motif_regions(
        src, eligible, min_region_pixels=cfg.min_region_pixels,
    )
    report.regions_total = len(regions)

    repaired, change_map, edited, skipped, info2 = run_profile(
        info.name,
        raw_gray=raw,
        scores=scores,
        regions=regions,
        source_bw=src,
        eligible=eligible,
        styles_root=cfg.styles_root,
        confidence_threshold=cfg.confidence_threshold,
        style_influence=cfg.style_influence,
        user_style_selected=user_selected,
    )
    report.available = info2.available
    report.reason = info2.reason or report.reason
    report.regions_edited = edited
    report.regions_skipped = skipped

    repaired = apply_final_constraints(
        repaired,
        source_bw=src,
        raw_gray=raw,
        editable_mask=editable_mask,
        locked_mask=locked_mask,
        eligible_used=eligible,
    )
    # change map only where we actually differ after constraints
    change_map = repaired != raw
    report.changed_pixels = int(change_map.sum())
    report.ran = True

    ok, uniq = validate_result_palette(repaired, src)
    if not ok:
        raise RuntimeError(f"Postprocess produced invalid palette/outline: uniq={sorted(uniq)}")

    # Absolute guarantee: caller's raw_pred_gray unchanged; we never held a writeable view.
    return PostprocessResult(
        repaired_gray=repaired,
        change_map=change_map,
        report=report,
        raw_gray=raw,
    )

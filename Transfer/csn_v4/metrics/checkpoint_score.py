"""Checkpoint scoring for CSN-V4 overfit experiments."""
from __future__ import annotations

from csn_v4.metrics.region_metrics import RegionMetrics

REQUIRED_ORIENTATIONS = frozenset(range(8))


def compute_checkpoint_score_v4(
    mean_miou: float,
    worst_orient_miou: float,
    mean_boundary_miou: float,
    small_component_recall: float,
    seam_band_miou: float,
) -> float:
    parts = [
        0.25 * mean_miou,
        0.25 * worst_orient_miou,
        0.20 * mean_boundary_miou,
        0.15 * small_component_recall,
        0.15 * seam_band_miou,
    ]
    return float(sum(parts))


def _nanmean(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    if not clean:
        return 0.0
    return float(sum(clean) / len(clean))


def score_from_region_results(
    results: list[RegionMetrics],
    *,
    require_all_orientations: bool = False,
) -> float:
    if not results:
        return 0.0
    present = {r.transform_id for r in results}
    if require_all_orientations:
        missing = REQUIRED_ORIENTATIONS - present
        if missing:
            raise ValueError(f"Checkpoint score requires all 8 D4 orientations; missing {sorted(missing)}")

    mean_miou = _nanmean([r.global_miou for r in results])
    mean_boundary = _nanmean([r.boundary_miou for r in results])
    mean_seam = _nanmean([r.seam_band_miou for r in results])
    small_vals = [r.small_component_recall for r in results]
    clean_small = [v for v in small_vals if v == v]
    if clean_small:
        mean_small = float(sum(clean_small) / len(clean_small))
    else:
        # Fast tile eval may skip small-component metrics; fall back to global mIoU.
        mean_small = mean_miou

    orient_means: list[float] = []
    for t in sorted(present):
        subset = [r.global_miou for r in results if r.transform_id == t]
        if subset:
            orient_means.append(float(sum(subset) / len(subset)))
    if not orient_means:
        raise ValueError("No orientation metrics available for checkpoint score")
    worst_orient = min(orient_means)
    return compute_checkpoint_score_v4(mean_miou, worst_orient, mean_boundary, mean_small, mean_seam)

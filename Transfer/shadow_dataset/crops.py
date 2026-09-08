"""Categorical class-aware deterministic crop sampling."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from shadow_dataset.constants import (
    CLASS_COUNT_KEYS,
    CROP_IOU_DEDUP_THRESHOLD,
    CROP_IOU_DEDUP_THRESHOLD_V2,
    DEFAULT_CROP_SIZE,
    DEFAULT_EDGE_BAND,
    DEFAULT_GRID_STRIDE,
    DEFAULT_MAX_CROPS_PER_IMAGE,
    DEFAULT_MIN_CROPS_PER_IMAGE,
    DEFAULT_VALID_MARGIN,
    SCHEMA_VERSION,
    STRATEGY_CANDIDATE_COMPONENT,
    STRATEGY_EDGE_CANDIDATE,
    STRATEGY_RARE_CLASS,
    STRATEGY_SHADE_BOUNDARY,
    STRATEGY_SPATIAL_GRID,
    STRATEGY_UNIFORM_CANDIDATE,
    STRATEGY_WEIGHTS,
    STRATEGY_WEIGHTS_V2,
)
from shadow_dataset.geometry import (
    ComponentInfo,
    candidate_mask_from_input,
    class_index_masks,
    class_label_map,
    detect_class_boundaries,
    find_candidate_components,
)
from shadow_dataset.types import CropRecord, CropSamplingStats, ImageRecord


@dataclass(frozen=True)
class CropBox:
    x: int
    y: int
    size: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.size)

    def iou(self, other: CropBox) -> float:
        ax2 = self.x + self.size
        ay2 = self.y + self.size
        bx2 = other.x + other.size
        by2 = other.y + other.size
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = self.size * self.size + other.size * other.size - inter
        return float(inter) / float(union) if union else 0.0


def per_image_seed(global_seed: int, source_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{source_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def desired_crop_count(
    height: int,
    width: int,
    *,
    stride: int = 384,
    min_crops: int = DEFAULT_MIN_CROPS_PER_IMAGE,
    max_crops: int = DEFAULT_MAX_CROPS_PER_IMAGE,
) -> int:
    grid_count = math.ceil(height / stride) * math.ceil(width / stride)
    return int(min(max(grid_count, min_crops), max_crops))


def _axis_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    last = length - tile_size
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def spatial_grid_boxes(
    image_height: int,
    image_width: int,
    crop_size: int,
    stride: int,
) -> list[CropBox]:
    """Dense grid including flush final and forced edge/corner origins."""
    ys = _axis_origins(image_height, crop_size, stride)
    xs = _axis_origins(image_width, crop_size, stride)
    boxes = [CropBox(x=x, y=y, size=crop_size) for y in ys for x in xs]

    last_y = max(0, image_height - crop_size) if image_height > crop_size else 0
    last_x = max(0, image_width - crop_size) if image_width > crop_size else 0
    mid_y = last_y // 2
    mid_x = last_x // 2
    forced = [
        (0, 0),
        (last_x, 0),
        (0, last_y),
        (last_x, last_y),
        (mid_x, 0),
        (mid_x, last_y),
        (0, mid_y),
        (last_x, mid_y),
    ]
    seen = {b.key for b in boxes}
    for x, y in forced:
        box = CropBox(x=int(x), y=int(y), size=crop_size)
        if box.key not in seen:
            boxes.append(box)
            seen.add(box.key)
    return boxes


def clamp_crop_origin(
    center_x: float,
    center_y: float,
    *,
    crop_size: int,
    image_width: int,
    image_height: int,
) -> CropBox:
    """Place a crop so its center is near (center_x, center_y), clamped to image."""
    return place_crop_origin(
        center_x,
        center_y,
        crop_size=crop_size,
        image_width=image_width,
        image_height=image_height,
        allow_outside=False,
    )


def place_crop_origin(
    center_x: float,
    center_y: float,
    *,
    crop_size: int,
    image_width: int,
    image_height: int,
    allow_outside: bool = False,
) -> CropBox:
    """Place crop centered on (center_x, center_y).

    When ``allow_outside`` is True, the window may overhang the image so border
    pixels land inside the supervised region (padding handled at load time).
    """
    half = crop_size / 2.0
    x = int(round(center_x - half))
    y = int(round(center_y - half))
    if allow_outside:
        x = max(1 - crop_size, min(x, image_width - 1))
        y = max(1 - crop_size, min(y, image_height - 1))
    else:
        if image_width <= crop_size:
            x = 0
        else:
            x = max(0, min(x, image_width - crop_size))
        if image_height <= crop_size:
            y = 0
        else:
            y = max(0, min(y, image_height - crop_size))
    return CropBox(x=int(x), y=int(y), size=int(crop_size))


def normalized_crop_box(
    box: CropBox,
    image_width: int,
    image_height: int,
) -> list[float]:
    """Return [x_norm, y_norm, w_norm, h_norm] relative to full image."""
    return [
        float(box.x) / float(image_width) if image_width else 0.0,
        float(box.y) / float(image_height) if image_height else 0.0,
        float(box.size) / float(image_width) if image_width else 0.0,
        float(box.size) / float(image_height) if image_height else 0.0,
    ]


def _sample_points(
    mask: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or n <= 0:
        return []
    if ys.size <= n:
        order = np.argsort(ys.astype(np.int64) * mask.shape[1] + xs.astype(np.int64))
        return [(int(xs[i]), int(ys[i])) for i in order.tolist()]
    idx = rng.choice(ys.size, size=n, replace=False)
    idx = np.sort(idx)
    return [(int(xs[i]), int(ys[i])) for i in idx.tolist()]


def _edge_band_mask(h: int, w: int, band: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    b = max(0, int(band))
    if b <= 0:
        return mask
    mask[:b, :] = True
    mask[-b:, :] = True
    mask[:, :b] = True
    mask[:, -b:] = True
    return mask


def _allocate_strategy_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate integer crop quotas from weights summing to total."""
    strategies = list(weights.keys())
    raw = [weights[s] * total for s in strategies]
    counts = [int(math.floor(v)) for v in raw]
    remainder = total - sum(counts)
    fracs = sorted(
        ((raw[i] - counts[i], strategies[i], i) for i in range(len(strategies))),
        key=lambda t: (-t[0], t[1]),
    )
    for k in range(remainder):
        counts[fracs[k][2]] += 1
    return {strategies[i]: counts[i] for i in range(len(strategies))}


def _component_centers(
    component: ComponentInfo, crop_size: int
) -> list[tuple[float, float]]:
    """One or more centers covering a component."""
    bw = component.max_x - component.min_x + 1
    bh = component.max_y - component.min_y + 1
    centers: list[tuple[float, float]] = [
        (component.centroid_x, component.centroid_y)
    ]
    if bw > crop_size or bh > crop_size:
        step = max(crop_size // 2, 1)
        for y in range(component.min_y, component.max_y + 1, step):
            for x in range(component.min_x, component.max_x + 1, step):
                centers.append((float(x), float(y)))
    return centers


def compute_crop_stats(
    candidate: np.ndarray,
    target_rgb: np.ndarray,
    boundary: np.ndarray,
    box: CropBox,
    *,
    valid_margin: int,
) -> dict[str, object]:
    """Compute candidate/class/boundary stats for a crop window (no padding)."""
    h, w = candidate.shape
    x0, y0 = box.x, box.y
    x1, y1 = x0 + box.size, y0 + box.size

    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(w, x1), min(h, y1)

    empty_counts = {k: 0 for k in CLASS_COUNT_KEYS}
    if ix0 >= ix1 or iy0 >= iy1:
        return {
            "candidate_pixels": 0,
            "candidate_pixels_in_valid_center": 0,
            "class_counts": empty_counts,
            "boundary_pixels": 0,
        }

    cand_crop = candidate[iy0:iy1, ix0:ix1]
    cand_count = int(cand_crop.sum())

    margin = valid_margin
    local_y0 = iy0 - y0
    local_x0 = ix0 - x0
    yy, xx = np.indices(cand_crop.shape)
    local_ys = yy + local_y0
    local_xs = xx + local_x0
    center_mask = (
        (local_ys >= margin)
        & (local_ys < box.size - margin)
        & (local_xs >= margin)
        & (local_xs < box.size - margin)
    )
    cand_center = int((cand_crop & center_mask).sum())

    tg = target_rgb[iy0:iy1, ix0:ix1]
    class_masks = class_index_masks(tg, cand_crop)
    class_counts = {k: int(v.sum()) for k, v in class_masks.items()}
    boundary_pixels = int(boundary[iy0:iy1, ix0:ix1].sum())

    return {
        "candidate_pixels": cand_count,
        "candidate_pixels_in_valid_center": cand_center,
        "class_counts": class_counts,
        "boundary_pixels": boundary_pixels,
    }


def _accept_crop(
    box: CropBox,
    candidate: np.ndarray,
    *,
    accepted: list[CropBox],
    accepted_priority: list[bool],
    stats: CropSamplingStats,
    force_keep: bool,
    iou_threshold: float,
) -> bool:
    """Deduplicate and IoU-filter a crop candidate."""
    stats.generated += 1
    h, w = candidate.shape
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(w, box.x + box.size), min(h, box.y + box.size)
    if x0 >= x1 or y0 >= y1:
        stats.rejected_no_candidate += 1
        return False
    if not np.any(candidate[y0:y1, x0:x1]):
        stats.rejected_no_candidate += 1
        return False

    key = box.key
    if any(a.key == key for a in accepted):
        stats.rejected_duplicate += 1
        return False

    for other, other_priority in zip(accepted, accepted_priority):
        if box.iou(other) >= iou_threshold:
            if force_keep and other_priority:
                continue
            if force_keep and not other_priority:
                continue
            stats.rejected_high_iou += 1
            return False

    accepted.append(box)
    accepted_priority.append(force_keep)
    stats.accepted += 1
    return True


def sample_crops_for_image(
    image: ImageRecord,
    input_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    global_seed: int,
    crop_size: int = DEFAULT_CROP_SIZE,
    valid_margin: int = DEFAULT_VALID_MARGIN,
    min_crops: int = DEFAULT_MIN_CROPS_PER_IMAGE,
    max_crops: int = DEFAULT_MAX_CROPS_PER_IMAGE,
    rare_class_keys: Sequence[str] | None = None,
    fixed_count: int | None = None,
    include_spatial_grid: bool = False,
    allow_padded_origins: bool = False,
    grid_stride: int = DEFAULT_GRID_STRIDE,
    edge_band: int = DEFAULT_EDGE_BAND,
    iou_dedup_threshold: float | None = None,
    strategy_weights: dict[str, float] | None = None,
    bake_dihedral: bool = False,
    dihedral_variants: Sequence[int] | None = None,
) -> tuple[list[CropRecord], CropSamplingStats, dict[str, object]]:
    """Generate deterministic class-aware crops for one full image."""
    stats = CropSamplingStats()
    h, w = input_rgb.shape[:2]
    assert (h, w) == (image.height, image.width)

    candidate = candidate_mask_from_input(input_rgb)
    if not np.any(candidate):
        return [], stats, {"components": 0, "boundary_pixels": 0}

    class_labels = class_label_map(target_rgb, candidate)
    boundary = detect_class_boundaries(class_labels)
    components = find_candidate_components(candidate, connectivity=8)
    class_masks = class_index_masks(target_rgb, candidate)

    weights = strategy_weights or (
        STRATEGY_WEIGHTS_V2 if include_spatial_grid else STRATEGY_WEIGHTS
    )
    iou_thr = (
        iou_dedup_threshold
        if iou_dedup_threshold is not None
        else (
            CROP_IOU_DEDUP_THRESHOLD_V2
            if include_spatial_grid
            else CROP_IOU_DEDUP_THRESHOLD
        )
    )
    stride_for_count = grid_stride if include_spatial_grid else 384
    n_crops = (
        fixed_count
        if fixed_count is not None
        else desired_crop_count(
            h, w, stride=stride_for_count, min_crops=min_crops, max_crops=max_crops
        )
    )
    quotas = _allocate_strategy_counts(n_crops, weights)
    rng = np.random.default_rng(per_image_seed(global_seed, image.source_id))

    accepted_boxes: list[CropBox] = []
    accepted_priority: list[bool] = []
    meta: list[dict[str, object]] = []

    def try_add_box(
        box: CropBox,
        strategy: str,
        *,
        force_keep: bool = False,
        is_tiny: bool = False,
        component_count: int = len(components),
    ) -> bool:
        if not _accept_crop(
            box,
            candidate,
            accepted=accepted_boxes,
            accepted_priority=accepted_priority,
            stats=stats,
            force_keep=force_keep,
            iou_threshold=iou_thr,
        ):
            return False
        crop_stats = compute_crop_stats(
            candidate, target_rgb, boundary, box, valid_margin=valid_margin
        )
        meta.append(
            {
                "box": box,
                "strategy": strategy,
                "is_tiny": is_tiny,
                "component_count": component_count,
                "stats": crop_stats,
            }
        )
        stats.by_strategy[strategy] = stats.by_strategy.get(strategy, 0) + 1
        if is_tiny:
            stats.tiny_component_crops += 1
        return True

    def try_add(
        cx: float,
        cy: float,
        strategy: str,
        *,
        force_keep: bool = False,
        is_tiny: bool = False,
        component_count: int = len(components),
    ) -> bool:
        box = place_crop_origin(
            cx,
            cy,
            crop_size=crop_size,
            image_width=w,
            image_height=h,
            allow_outside=allow_padded_origins,
        )
        return try_add_box(
            box,
            strategy,
            force_keep=force_keep,
            is_tiny=is_tiny,
            component_count=component_count,
        )

    if include_spatial_grid and STRATEGY_SPATIAL_GRID in quotas:
        n_grid = quotas[STRATEGY_SPATIAL_GRID]
        added = 0
        for box in spatial_grid_boxes(h, w, crop_size, grid_stride):
            if added >= n_grid:
                break
            if try_add_box(box, STRATEGY_SPATIAL_GRID, force_keep=True):
                added += 1

    if STRATEGY_EDGE_CANDIDATE in quotas and quotas[STRATEGY_EDGE_CANDIDATE] > 0:
        n_edge = quotas[STRATEGY_EDGE_CANDIDATE]
        edge_mask = candidate & _edge_band_mask(h, w, edge_band)
        pts = _sample_points(edge_mask, rng, max(n_edge * 4, n_edge))
        added = 0
        for cx, cy in pts:
            if added >= n_edge:
                break
            if try_add(cx, cy, STRATEGY_EDGE_CANDIDATE, force_keep=True):
                added += 1
        if added == 0 and n_edge > 0:
            ys, xs = np.nonzero(candidate)
            if ys.size:
                for cx, cy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
                    if added >= n_edge:
                        break
                    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
                    i = int(np.argmin(d2))
                    if try_add(
                        float(xs[i]), float(ys[i]), STRATEGY_EDGE_CANDIDATE, force_keep=True
                    ):
                        added += 1

    n_boundary = quotas.get(STRATEGY_SHADE_BOUNDARY, 0)
    boundary_pts = _sample_points(boundary, rng, max(n_boundary * 3, n_boundary))
    added = 0
    for cx, cy in boundary_pts:
        if added >= n_boundary:
            break
        if try_add(cx, cy, STRATEGY_SHADE_BOUNDARY, force_keep=True):
            added += 1

    n_comp = quotas.get(STRATEGY_CANDIDATE_COMPONENT, 0)
    added = 0
    tiny_first = sorted(
        components,
        key=lambda c: (
            0 if c.is_tiny else 1,
            -c.pixel_count,
            c.centroid_y,
            c.centroid_x,
        ),
    )
    for comp in tiny_first:
        if added >= n_comp:
            break
        for cx, cy in _component_centers(comp, crop_size):
            if added >= n_comp:
                break
            if try_add(
                cx,
                cy,
                STRATEGY_CANDIDATE_COMPONENT,
                force_keep=comp.is_tiny,
                is_tiny=comp.is_tiny,
            ):
                added += 1

    n_rare = quotas.get(STRATEGY_RARE_CLASS, 0)
    rare_keys = list(rare_class_keys) if rare_class_keys else list(CLASS_COUNT_KEYS)
    added = 0
    for key in rare_keys:
        if added >= n_rare:
            break
        mask = class_masks.get(key)
        if mask is None or not np.any(mask):
            continue
        pts = _sample_points(mask, rng, max((n_rare - added) * 2, 1))
        for cx, cy in pts:
            if added >= n_rare:
                break
            if try_add(cx, cy, STRATEGY_RARE_CLASS):
                added += 1

    n_uniform = quotas.get(STRATEGY_UNIFORM_CANDIDATE, 0)
    pts = _sample_points(candidate, rng, max(n_uniform * 3, n_uniform))
    added = 0
    for cx, cy in pts:
        if added >= n_uniform:
            break
        if try_add(cx, cy, STRATEGY_UNIFORM_CANDIDATE):
            added += 1

    remaining = n_crops - len(accepted_boxes)
    if remaining > 0:
        pts = _sample_points(candidate, rng, remaining * 5)
        for cx, cy in pts:
            if len(accepted_boxes) >= n_crops:
                break
            try_add(cx, cy, STRATEGY_UNIFORM_CANDIDATE)

    if include_spatial_grid and len(accepted_boxes) < n_crops:
        for box in spatial_grid_boxes(h, w, crop_size, max(64, grid_stride // 2)):
            if len(accepted_boxes) >= n_crops:
                break
            try_add_box(box, STRATEGY_SPATIAL_GRID, force_keep=False)

    records: list[CropRecord] = []
    ordered = sorted(
        meta,
        key=lambda m: (
            m["box"].y,  # type: ignore[attr-defined]
            m["box"].x,  # type: ignore[attr-defined]
            str(m["strategy"]),
        ),
    )
    if bake_dihedral:
        variants = list(dihedral_variants) if dihedral_variants is not None else list(range(8))
    else:
        variants = [0]

    for item in ordered:
        box: CropBox = item["box"]  # type: ignore[assignment]
        crop_stats = item["stats"]  # type: ignore[assignment]
        for k in variants:
            sample_id = f"{image.source_id}__x{box.x}_y{box.y}_k{k}"
            records.append(
                CropRecord(
                    schema_version=SCHEMA_VERSION,
                    sample_id=sample_id,
                    source_id=image.source_id,
                    input_path=image.input_path,
                    target_path=image.target_path,
                    split=image.split,
                    image_width=image.width,
                    image_height=image.height,
                    x=box.x,
                    y=box.y,
                    crop_size=crop_size,
                    valid_margin=valid_margin,
                    sampling_strategy=str(item["strategy"]),
                    candidate_pixels=int(crop_stats["candidate_pixels"]),  # type: ignore[index]
                    candidate_pixels_in_valid_center=int(
                        crop_stats["candidate_pixels_in_valid_center"]  # type: ignore[index]
                    ),
                    class_counts={
                        k_cls: int(v)
                        for k_cls, v in crop_stats["class_counts"].items()  # type: ignore[index]
                    },
                    boundary_pixels=int(crop_stats["boundary_pixels"]),  # type: ignore[index]
                    component_count=int(item["component_count"]),  # type: ignore[arg-type]
                    normalized_crop_box=normalized_crop_box(box, w, h),
                    is_tiny_component=bool(item["is_tiny"]),
                    group_id=image.group_id,
                    transform_id=int(k),
                )
            )

    extras = {
        "components": len(components),
        "boundary_pixels": int(boundary.sum()),
        "component_sizes": [c.pixel_count for c in components],
        "spatial_boxes": len(accepted_boxes),
        "has_corner_crop": any(
            (b.x <= 0 or b.y <= 0 or b.x + b.size >= w or b.y + b.size >= h)
            for b in accepted_boxes
        ),
    }
    return records, stats, extras


def compute_train_rare_class_order(
    train_images: Sequence[ImageRecord],
) -> list[str]:
    """Order class keys from rarest to most common using train images only."""
    totals = {k: 0 for k in CLASS_COUNT_KEYS}
    for img in train_images:
        for key in totals:
            totals[key] += int(img.class_counts.get(key, 0))
    rank = {k: i for i, k in enumerate(CLASS_COUNT_KEYS)}
    return sorted(totals.keys(), key=lambda k: (totals[k], rank[k]))

"""Motif-region grouping using input geometry and eligible mask."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from csn_v4.constants import GRAY_BLACK


@dataclass
class MotifRegion:
    region_id: int
    label: str
    mask: np.ndarray  # bool HxW
    pixel_count: int
    bbox: tuple[int, int, int, int]  # y0,x0,y1,x1
    reliable: bool
    skip_reason: str = ""


def _touches_image_border(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _open_outline_gap(
    source_bw: np.ndarray,
    region_mask: np.ndarray,
) -> bool:
    """Heuristic: region reaches image border through non-black pixels → open outline."""
    if not _touches_image_border(region_mask):
        return False
    # Dilate region slightly; if dilated region still on border via non-black, treat as open.
    src = np.asarray(source_bw)
    if src.ndim == 3:
        src = src[..., 0]
    free = src != GRAY_BLACK
    # Region already constrained to eligible; border touch + free path ⇒ unreliable.
    return bool(region_mask.any() and free[region_mask].all() is not False and _touches_image_border(region_mask))


def identify_motif_regions(
    source_bw: np.ndarray,
    eligible: np.ndarray,
    *,
    min_region_pixels: int = 16,
    connectivity: int = 1,
) -> list[MotifRegion]:
    """Group eligible pixels into independent local regions.

    Black outline pixels act as hard barriers (not part of eligible). Regions that
    cannot be identified reliably (too small, open outline / border spill) are
    marked ``reliable=False`` with a skip reason — callers must leave them unchanged.
    """
    eligible = np.asarray(eligible, dtype=bool)
    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled, n = ndimage.label(eligible, structure=structure)
    regions: list[MotifRegion] = []
    for rid in range(1, n + 1):
        mask = labeled == rid
        count = int(mask.sum())
        ys, xs = np.where(mask)
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
        reliable = True
        reason = ""
        if count < min_region_pixels:
            reliable = False
            reason = f"too_small({count}<{min_region_pixels})"
        elif _open_outline_gap(source_bw, mask):
            reliable = False
            reason = "open_outline_or_border_spill"
        regions.append(
            MotifRegion(
                region_id=rid,
                label=f"R{rid}",
                mask=mask,
                pixel_count=count,
                bbox=bbox,
                reliable=reliable,
                skip_reason=reason,
            )
        )
    return regions

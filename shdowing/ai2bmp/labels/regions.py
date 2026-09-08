"""4-connected region labeling per original palette index."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage


MAX_DETAILED_REGION_RECORDS = 256
MIN_REGION_AREA_FOR_METADATA = 64


@dataclass
class RegionRecord:
    region_id: int
    original_palette_index: int
    contiguous_class_id: int
    resolved_rgb: list[int]
    area: int
    bbox: list[int]
    centroid: list[float]
    perimeter: int
    perimeter_contact: float
    border_contact: float
    adjacent_region_count: int
    shared_boundary_lengths: dict[str, int]


def label_regions(
    index_map: np.ndarray,
    contiguous_used_index_map: np.ndarray,
    palette: np.ndarray,
    contiguous_mapping: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    region_id_map = np.zeros(index_map.shape, dtype=np.int32)
    records: list[RegionRecord] = []
    next_id = 1

    orig_to_cont = {int(k): int(v) for k, v in contiguous_mapping["original_to_contiguous"].items()}
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    for orig_idx in sorted(orig_to_cont.keys()):
        mask = index_map == orig_idx
        labeled, num = ndimage.label(mask, structure=structure)
        if num == 0:
            continue
        active = labeled > 0
        region_id_map[active] = labeled[active] + next_id - 1

        cont_id = orig_to_cont[orig_idx]
        rgb = palette[orig_idx].tolist()
        counts = np.bincount(labeled.ravel())
        candidate_ids = np.where(counts[1:] >= MIN_REGION_AREA_FOR_METADATA)[0] + 1
        if len(candidate_ids) > MAX_DETAILED_REGION_RECORDS:
            areas = counts[candidate_ids]
            candidate_ids = candidate_ids[np.argsort(areas)[-MAX_DETAILED_REGION_RECORDS:]]

        slices = ndimage.find_objects(labeled)
        for comp_idx in candidate_ids.tolist():
            slc = slices[int(comp_idx) - 1]
            if slc is None:
                continue
            local = labeled[slc] == int(comp_idx)
            area = int(counts[int(comp_idx)])
            ys, xs = np.nonzero(local)
            y0, x0 = int(slc[0].start), int(slc[1].start)
            ys_abs = ys + y0
            xs_abs = xs + x0
            x1, y1 = int(xs_abs.max()) + 1, int(ys_abs.max()) + 1
            cy, cx = float(ys_abs.mean()), float(xs_abs.mean())
            eroded = ndimage.binary_erosion(local, structure=np.ones((3, 3), dtype=bool))
            perimeter = int((local & ~eroded).sum())
            rid = int(comp_idx) + next_id - 1
            records.append(
                RegionRecord(
                    region_id=rid,
                    original_palette_index=int(orig_idx),
                    contiguous_class_id=int(cont_id),
                    resolved_rgb=rgb,
                    area=area,
                    bbox=[x0, y0, x1, y1],
                    centroid=[cx, cy],
                    perimeter=perimeter,
                    perimeter_contact=float(perimeter / max(1, area)),
                    border_contact=0.0,
                    adjacent_region_count=0,
                    shared_boundary_lengths={},
                )
            )

        next_id += int(num)

    return region_id_map, [asdict(r) for r in records]

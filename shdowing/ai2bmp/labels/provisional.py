"""Provisional role/shade suggestions (not authoritative GT)."""

from __future__ import annotations

from typing import Any

import numpy as np

from shdowing.ai2bmp.color import srgb_uint8_to_lab
from shdowing.ai2bmp.constants import DEFAULT_TASK_AVAILABILITY
from shdowing.ai2bmp.io_utils import save_rgb_png, write_json


def generate_provisional_labels(
    index_map: np.ndarray,
    palette: np.ndarray,
    region_records: list[dict[str, Any]],
    provisional_dir,
) -> dict[str, Any]:
    h, w = index_map.shape
    used = np.unique(index_map)
    total = index_map.size

    # background suggestions by coverage + border contact
    role_suggestions = []
    index_stats: dict[int, dict[str, float]] = {}
    for idx in used.tolist():
        idx = int(idx)
        mask = index_map == idx
        area = int(mask.sum())
        border = (
            mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum()
        )
        index_stats[idx] = {
            "coverage": area / total,
            "border_contact": float(border / max(1, area)),
        }

    if index_stats:
        max_cov_idx = max(index_stats, key=lambda k: index_stats[k]["coverage"])
        role_suggestions.append(
            {
                "role": "background_candidate",
                "original_palette_index": int(max_cov_idx),
                "confidence": min(0.95, 0.5 + index_stats[max_cov_idx]["coverage"]),
                "evidence": index_stats[max_cov_idx],
                "approved_for_training": False,
            }
        )

    # outline suggestions from thin regions
    thin_regions = [r for r in region_records if r["area"] <= 64 or r["perimeter_contact"] > 0.8]
    for r in thin_regions[:5]:
        role_suggestions.append(
            {
                "role": "outline_candidate",
                "region_id": r["region_id"],
                "original_palette_index": r["original_palette_index"],
                "confidence": 0.4,
                "evidence": {"area": r["area"], "perimeter_contact": r["perimeter_contact"]},
                "approved_for_training": False,
            }
        )

    # shade-family suggestions via Lab hue similarity + adjacency
    shade_groups = []
    labs = {int(i): srgb_uint8_to_lab(palette[int(i)].reshape(1, 1, 3))[0, 0] for i in used}
    sorted_by_l = sorted(labs.items(), key=lambda kv: kv[1][0])
    if len(sorted_by_l) >= 3:
        group = [sorted_by_l[0][0], sorted_by_l[1][0], sorted_by_l[2][0]]
        shade_groups.append(
            {
                "group_id": 0,
                "original_indices": group,
                "confidence": 0.35,
                "evidence": {"method": "ordered_lightness_adjacent", "lab_l": [float(labs[i][0]) for i in group]},
                "approved_for_training": False,
            }
        )

    write_json(provisional_dir / "role_suggestions.json", {"suggestions": role_suggestions})
    write_json(provisional_dir / "shade_family_suggestions.json", {"groups": shade_groups})

    preview = palette[np.sort(used)].reshape(1, -1, 3).repeat(32, axis=0)
    save_rgb_png(provisional_dir / "role_suggestions_preview.png", preview.astype(np.uint8))
    save_rgb_png(provisional_dir / "shade_family_preview.png", preview.astype(np.uint8))

    return {
        "task_availability": dict(DEFAULT_TASK_AVAILABILITY),
        "role_suggestion_count": len(role_suggestions),
        "shade_group_count": len(shade_groups),
    }

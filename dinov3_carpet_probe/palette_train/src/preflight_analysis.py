"""Preflight analysis for palette cardinality and target-color observability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from dinov3_carpet_probe.palette_train.src.color import ciede2000_from_uint8
from dinov3_carpet_probe.palette_train.src.dataset import PaletteSampleRecord, summarize_num_colors, validate_dataset
from dinov3_carpet_probe.palette_train.src.transforms import build_transform, load_rgb_jpg


def _visibility_stats(min_distances: list[float]) -> dict[str, float]:
    if not min_distances:
        return {"de_le_1": 0.0, "de_le_2": 0.0, "de_le_5": 0.0, "de_le_10": 0.0}
    arr = np.asarray(min_distances, dtype=np.float64)
    return {
        "de_le_1": float(np.mean(arr <= 1.0)),
        "de_le_2": float(np.mean(arr <= 2.0)),
        "de_le_5": float(np.mean(arr <= 5.0)),
        "de_le_10": float(np.mean(arr <= 10.0)),
    }


def _min_delta_to_pixels(target_rgb: list[int], image_rgb: np.ndarray) -> float:
    flat = image_rgb.reshape(-1, 3).astype(np.uint8)
    target = np.asarray(target_rgb, dtype=np.uint8)[None, :]
    distances = ciede2000_from_uint8(np.repeat(target, flat.shape[0], axis=0), flat)
    return float(distances.min()) if distances.size else float("inf")


def run_preflight(
    cfg: dict[str, Any],
    *,
    limit_samples: int | None = None,
) -> dict[str, Any]:
    records, report = validate_dataset(
        cfg["image_dir"],
        cfg["palette_dir"],
        max_num_colors_override=cfg.get("max_num_colors"),
        show_progress=True,
    )
    transform = build_transform(cfg, training=False)
    raw_counts = report.num_colors_stats
    example_over_12 = report.excluded_over_cap[:20]
    raw_all_records, raw_report = validate_dataset(
        cfg["image_dir"],
        cfg["palette_dir"],
        max_num_colors_override=None,
        show_progress=False,
    )
    raw_num_colors = [record.num_colors for record in raw_all_records]

    original_mins: list[float] = []
    resized_mins: list[float] = []
    sampled_records = records[:limit_samples] if limit_samples is not None else records
    per_sample_examples: list[dict[str, Any]] = []
    for record in tqdm(sampled_records, desc="preflight_visibility", unit="image", dynamic_ncols=True):
        image = load_rgb_jpg(record.image_path)
        image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
        _, meta = transform(record.image_path)
        resized_np = (
            meta["raw_tensor"].permute(1, 2, 0).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()
        )
        sample_original: list[float] = []
        sample_resized: list[float] = []
        for color in record.rgb:
            d0 = _min_delta_to_pixels(color, image_np)
            d1 = _min_delta_to_pixels(color, resized_np)
            original_mins.append(d0)
            resized_mins.append(d1)
            sample_original.append(d0)
            sample_resized.append(d1)
    return {
        "raw_num_colors_stats": summarize_num_colors(raw_num_colors),
        "counts_over_thresholds": {
            "gt_16": int(sum(v > 16 for v in raw_num_colors)),
            "gt_32": int(sum(v > 32 for v in raw_num_colors)),
            "gt_64": int(sum(v > 64 for v in raw_num_colors)),
            "gt_128": int(sum(v > 128 for v in raw_num_colors)),
        },
        "excluded_over_12_count": len(report.excluded_over_cap),
        "excluded_over_12_examples": example_over_12,
        "trainable_count": report.trainable_count,
        "visibility_original": _visibility_stats(original_mins),
        "visibility_resized_512": _visibility_stats(resized_mins),
        "sampled_images_for_visibility": len(sampled_records),
        "high_cardinality_examples": [
            {"stem": record.stem, "num_colors": record.num_colors}
            for record in raw_all_records
            if record.num_colors > 12
        ][:20],
    }

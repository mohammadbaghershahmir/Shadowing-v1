"""Decode palette predictions into deterministic JSON outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dinov3_carpet_probe.palette_train.src.color import ciede2000_np, srgb_uint8_to_lab


def rgb_uint8_to_hex(rgb: list[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _sort_palette_rgb(rgb_colors: list[list[int]]) -> list[list[int]]:
    if not rgb_colors:
        return []
    arr = np.asarray(rgb_colors, dtype=np.uint8)
    lab = srgb_uint8_to_lab(arr)
    order = np.lexsort((lab[:, 2], lab[:, 1], lab[:, 0]))
    return [rgb_colors[idx] for idx in order.tolist()]


@dataclass
class DecodedPalette:
    sample_id: int
    stem: str
    num_colors: int
    rgb: list[list[int]]
    hex: list[str]
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "sample_id": self.sample_id,
            "num_colors": self.num_colors,
            "rgb": self.rgb,
            "hex": self.hex,
        }
        if self.warning:
            data["warning"] = self.warning
        return data


def decode_palette_prediction(
    *,
    stem: str,
    pred_rgb: torch.Tensor,
    presence_logits: torch.Tensor,
    count_logits: torch.Tensor,
) -> DecodedPalette:
    pred_count = int(torch.argmax(count_logits).item())
    ranked = torch.argsort(torch.sigmoid(presence_logits), descending=True)
    unique: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for idx in ranked.tolist():
        rgb = pred_rgb[idx].clamp(0.0, 1.0)
        rgb_uint8 = torch.round(rgb * 255.0).to(torch.int64).tolist()
        rgb_key = tuple(int(x) for x in rgb_uint8)
        if rgb_key in seen:
            continue
        seen.add(rgb_key)
        unique.append([int(x) for x in rgb_uint8])
        if len(unique) >= pred_count:
            break
    warning = None
    if len(unique) < pred_count:
        warning = (
            f"Predicted count {pred_count}, but only {len(unique)} unique RGB colors were available "
            "after duplicate filtering."
        )
    unique = _sort_palette_rgb(unique)
    return DecodedPalette(
        sample_id=int(stem) if stem.isdigit() else -1,
        stem=stem,
        num_colors=len(unique),
        rgb=unique,
        hex=[rgb_uint8_to_hex(color) for color in unique],
        warning=warning,
    )


def output_path_for_stem(output_dir: Path | str, stem: str) -> Path:
    return Path(output_dir) / f"{stem}.json"

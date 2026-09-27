"""Contact sheet generation for Dataset V2 inspection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset_v2.artifacts import SampleArtifacts
from dataset_v2.bmp_io import save_rgb_bmp


def _to_rgb_preview(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    return arr


def _thumbnail(arr: np.ndarray, max_side: int = 256) -> np.ndarray:
    rgb = _to_rgb_preview(arr)
    h, w = rgb.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale >= 1.0:
        return rgb
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((nw, nh), resample=Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def build_contact_sheet(
    artifacts: SampleArtifacts,
    labels: list[tuple[str, np.ndarray]] | None = None,
) -> np.ndarray:
    if labels is None:
        labels = [
            ("target_semantic", artifacts.target_semantic),
            ("source_bw", artifacts.source_bw),
            ("source_oracle", artifacts.source_oracle),
            ("shade_mask", artifacts.shade_mask),
            ("shade_level_preview", artifacts.shade_level_preview),
            ("transition_mask", artifacts.transition_mask),
            ("black_lock", artifacts.black_lock),
            ("reconstructed_target", artifacts.reconstructed_target),
            ("roundtrip_diff", artifacts.roundtrip_diff),
        ]

    thumbs: list[tuple[str, np.ndarray]] = [(name, _thumbnail(arr)) for name, arr in labels]
    label_h = 18
    pad = 8
    cols = 3
    rows = int(np.ceil(len(thumbs) / cols))
    cell_w = max(t[1].shape[1] for t in thumbs) + pad * 2
    cell_h = max(t[1].shape[0] for t in thumbs) + label_h + pad * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (name, thumb) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        ox = c * cell_w + pad
        oy = r * cell_h + pad + label_h
        draw.text((c * cell_w + pad, r * cell_h + pad), name, fill=(220, 220, 220), font=font)
        im = Image.fromarray(thumb, mode="RGB")
        sheet.paste(im, (ox, oy))

    return np.asarray(sheet, dtype=np.uint8)


def save_contact_sheet(path: Path, artifacts: SampleArtifacts) -> None:
    sheet = build_contact_sheet(artifacts)
    save_rgb_bmp(path, sheet)

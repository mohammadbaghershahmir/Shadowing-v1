"""Deterministic QA preview generation and HTML contact sheet."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from shadow_dataset.constants import (
    CLASS_TO_RGB,
    IGNORE_INDEX,
    OUTLINE,
    TARGET_REGION,
    UNSHADED,
)
from shadow_dataset.crops import CropBox
from shadow_dataset.dataset import central_valid_mask, extract_padded_crop
from shadow_dataset.geometry import (
    candidate_mask_from_input,
    class_label_map,
    detect_class_boundaries,
)
from shadow_dataset.image_io import load_rgb_exact
from shadow_dataset.types import CropRecord
from shadow_dataset.validation import encode_target_labels

LOGGER = logging.getLogger(__name__)


def _to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb.astype(np.uint8))


def _overlay_rect(
    rgb: np.ndarray,
    box: CropBox,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    width: int = 3,
) -> np.ndarray:
    img = _to_pil(rgb.copy())
    draw = ImageDraw.Draw(img)
    x0, y0 = box.x, box.y
    x1, y1 = box.x + box.size - 1, box.y + box.size - 1
    for i in range(width):
        draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i], outline=color)
    return np.asarray(img, dtype=np.uint8)


def _mask_to_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    out[mask] = color
    return out


def _labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    out = np.full((*labels.shape, 3), 40, dtype=np.uint8)  # ignore -> neutral gray
    out[labels == IGNORE_INDEX] = (40, 40, 40)
    for label, rgb in CLASS_TO_RGB.items():
        value = rgb[0]
        out[labels == label] = (value, value, value)
    return out


def _boundary_overlay(local_input: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    out = local_input.copy()
    # Draw boundary in cyan for visibility on magenta/white/black.
    out[boundary] = (0, 255, 255)
    return out


def _draw_center_rect(
    rgb: np.ndarray,
    valid_margin: int,
    *,
    color: tuple[int, int, int] = (255, 165, 0),
) -> np.ndarray:
    h, w = rgb.shape[:2]
    img = _to_pil(rgb.copy())
    draw = ImageDraw.Draw(img)
    m = valid_margin
    draw.rectangle([m, m, w - m - 1, h - m - 1], outline=color, width=2)
    return np.asarray(img, dtype=np.uint8)


def render_preview_panel(crop: CropRecord) -> np.ndarray:
    """Compose a multi-panel QA preview image for one crop."""
    input_full, _ = load_rgb_exact(crop.input_path)
    target_full, _ = load_rgb_exact(crop.target_path)

    box = CropBox(x=crop.x, y=crop.y, size=crop.crop_size)
    panel1 = _overlay_rect(input_full, box)

    local_input, image_valid = extract_padded_crop(
        input_full, x=crop.x, y=crop.y, crop_size=crop.crop_size, pad_value=UNSHADED
    )
    local_target, _ = extract_padded_crop(
        target_full, x=crop.x, y=crop.y, crop_size=crop.crop_size, pad_value=UNSHADED
    )
    labels_full = encode_target_labels(input_full, target_full)
    local_labels, _ = extract_padded_crop(
        labels_full,
        x=crop.x,
        y=crop.y,
        crop_size=crop.crop_size,
        pad_value=IGNORE_INDEX,
    )
    local_labels = local_labels.copy()
    local_labels[~image_valid] = IGNORE_INDEX

    candidate = candidate_mask_from_input(local_input)
    class_labels = class_label_map(local_target, candidate)
    boundary = detect_class_boundaries(class_labels)

    panel2 = local_input
    panel3 = local_target
    panel4 = _mask_to_rgb(candidate, TARGET_REGION)
    # Dim non-candidate.
    panel4[~candidate] = (30, 30, 30)
    panel5 = _labels_to_rgb(local_labels)
    panel6 = _boundary_overlay(local_input, boundary)
    panel7 = _draw_center_rect(local_input, crop.valid_margin)

    # Downscale full image panel to crop size for a uniform contact sheet row.
    full_pil = _to_pil(panel1)
    full_resized = full_pil.resize(
        (crop.crop_size, crop.crop_size), resample=Image.Resampling.NEAREST
    )
    panels = [
        np.asarray(full_resized, dtype=np.uint8),
        panel2,
        panel3,
        panel4,
        panel5,
        panel6,
        panel7,
    ]
    # Add thin separators.
    gap = 4
    sep = np.full((crop.crop_size, gap, 3), 200, dtype=np.uint8)
    row_parts: list[np.ndarray] = []
    for i, p in enumerate(panels):
        row_parts.append(p)
        if i < len(panels) - 1:
            row_parts.append(sep)
    row = np.concatenate(row_parts, axis=1)

    # Caption bar with labels.
    labels = [
        "full+box",
        "local_in",
        "local_gt",
        "candidate",
        "labels",
        "boundary",
        "center",
    ]
    caption_h = 28
    caption = np.full((caption_h, row.shape[1], 3), 245, dtype=np.uint8)
    cap_img = _to_pil(caption)
    draw = ImageDraw.Draw(cap_img)
    x = 4
    cell_w = crop.crop_size + gap
    for i, text in enumerate(labels):
        draw.text((x + i * cell_w, 6), text, fill=(0, 0, 0))
    caption = np.asarray(cap_img, dtype=np.uint8)
    return np.concatenate([caption, row], axis=0)


def select_preview_crops(
    crops: list[CropRecord],
    count: int,
    seed: int,
) -> list[CropRecord]:
    """Deterministically select up to ``count`` crops for previews."""
    if count <= 0 or not crops:
        return []
    # Prefer diversity: round-robin by strategy, then stable hash.
    by_strategy: dict[str, list[CropRecord]] = {}
    for crop in crops:
        by_strategy.setdefault(crop.sampling_strategy, []).append(crop)
    for key in by_strategy:
        by_strategy[key].sort(
            key=lambda c: (
                hashlib.sha256(f"{seed}:{c.sample_id}".encode()).hexdigest(),
                c.sample_id,
            )
        )

    selected: list[CropRecord] = []
    strategies = sorted(by_strategy.keys())
    idxs = {s: 0 for s in strategies}
    while len(selected) < count:
        progressed = False
        for s in strategies:
            if idxs[s] < len(by_strategy[s]):
                selected.append(by_strategy[s][idxs[s]])
                idxs[s] += 1
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return selected


def write_previews(
    crops: list[CropRecord],
    output_dir: Path,
    *,
    count: int,
    seed: int,
) -> list[Path]:
    """Write preview PNGs and an HTML index; return written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_preview_crops(crops, count, seed)
    written: list[Path] = []

    for i, crop in enumerate(selected):
        panel = render_preview_panel(crop)
        name = f"{i:04d}_{crop.sample_id}.png"
        # Sanitize Windows-illegal filename chars lightly.
        safe = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in name)
        path = output_dir / safe
        _to_pil(panel).save(path)
        written.append(path)
        LOGGER.debug("Wrote preview %s", path)

    _write_html_index(output_dir, written, selected)
    return written


def _write_html_index(
    output_dir: Path,
    paths: list[Path],
    crops: list[CropRecord],
) -> None:
    rows: list[str] = []
    for path, crop in zip(paths, crops):
        rel = path.name
        rows.append(
            "<div class='card'>"
            f"<a href='{rel}'><img src='{rel}' loading='lazy'/></a>"
            f"<div class='meta'><code>{crop.sample_id}</code><br/>"
            f"{crop.sampling_strategy} | cand={crop.candidate_pixels} | "
            f"boundary={crop.boundary_pixels}</div></div>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Shadow dataset previews</title>
<style>
body {{ font-family: Segoe UI, sans-serif; margin: 16px; background: #f0f0f0; }}
h1 {{ font-size: 18px; }}
.grid {{ display: flex; flex-direction: column; gap: 16px; }}
.card {{ background: #fff; padding: 8px; border: 1px solid #ccc; }}
img {{ max-width: 100%; height: auto; image-rendering: pixelated; }}
.meta {{ font-size: 12px; margin-top: 6px; color: #333; }}
</style>
</head>
<body>
<h1>Dataset QA previews ({len(paths)})</h1>
<p>Panels: full+box | local input | local GT | candidate | labels | boundary | center</p>
<div class="grid">
{"".join(rows)}
</div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")

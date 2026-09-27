"""Deterministic crop selection with integral-image statistics."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from dataset_v2.constants import CROP_CATEGORIES


@dataclass(frozen=True)
class CropBox:
    x: int
    y: int
    width: int
    height: int
    category: str
    crop_id: str

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


def integral_image(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float64)
    return np.pad(m, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)


def window_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return float(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1])


def _content_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _iou(a: CropBox, b: CropBox) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.width, b.x + b.width)
    iy2 = min(a.y + a.height, b.y + b.height)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / max(union, 1)


def _candidate_origins(length: int, size: int, stride: int) -> list[int]:
    if length < size:
        return []
    last = length - size
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def score_windows(
    image_w: int,
    image_h: int,
    crop_size: int,
    *,
    transition: np.ndarray,
    shade: np.ndarray,
    dark: np.ndarray,
    line_like: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, list[tuple[float, int, int]]]:
    """Return scored (score, x, y) lists per category."""
    if image_w < crop_size or image_h < crop_size:
        return {cat: [] for cat in CROP_CATEGORIES}

    stride = max(crop_size // 4, 32)
    ii_trans = integral_image(transition > 0)
    ii_shade = integral_image(shade > 0)
    ii_dark = integral_image(dark > 0)
    ii_line = integral_image(line_like > 0)
    area = crop_size * crop_size

    scored: dict[str, list[tuple[float, int, int]]] = {cat: [] for cat in CROP_CATEGORIES}
    for y in _candidate_origins(image_h, crop_size, stride):
        for x in _candidate_origins(image_w, crop_size, stride):
            t = window_sum(ii_trans, x, y, crop_size, crop_size) / area
            s = window_sum(ii_shade, x, y, crop_size, crop_size) / area
            d = window_sum(ii_dark, x, y, crop_size, crop_size) / area
            ln = window_sum(ii_line, x, y, crop_size, crop_size) / area
            scored["transition_rich"].append((t, x, y))
            scored["shade_rich"].append((s, x, y))
            scored["dark_rare"].append((d, x, y))
            if s <= 0.001 and ln > 0.01:
                scored["negative"].append((ln, x, y))
            scored["mixed_random"].append((float(rng.random()), x, y))

    for cat in ("transition_rich", "shade_rich", "dark_rare", "negative"):
        scored[cat].sort(key=lambda t: (-t[0], t[1], t[2]))
    scored["mixed_random"].sort(key=lambda t: (t[0], t[1], t[2]))
    return scored


def select_crops(
    sample_id: str,
    image_w: int,
    image_h: int,
    crop_sizes: list[int],
    *,
    transition: np.ndarray,
    shade: np.ndarray,
    shade_level_id: np.ndarray,
    source_bw: np.ndarray,
    rng: np.random.Generator,
    iou_threshold: float = 0.85,
) -> list[CropBox]:
    dark = (shade_level_id == 2).astype(np.uint8) * 255
    line_like = (source_bw == 0).astype(np.uint8) * 255

    selected: list[CropBox] = []
    seen_hashes: set[str] = set()

    for size in sorted(crop_sizes):
        if image_w < size or image_h < size:
            continue
        scored = score_windows(
            image_w, image_h, size,
            transition=transition,
            shade=shade,
            dark=dark,
            line_like=line_like,
            rng=rng,
        )
        for category in CROP_CATEGORIES:
            candidates = scored.get(category, [])
            if not candidates:
                continue
            for score, x, y in candidates[:32]:
                box = CropBox(x=x, y=y, width=size, height=size, category=category, crop_id="")
                # Dedup by overlap with accepted boxes of same size.
                if any(_iou(box, other) >= iou_threshold for other in selected if other.width == size):
                    continue
                crop_id = f"{sample_id}__{size}x{size}__{category}__x{x}_y{y}"
                box = CropBox(
                    x=x, y=y, width=size, height=size, category=category, crop_id=crop_id,
                )
                # Content hash from transition+shade window.
                th = transition[y : y + size, x : x + size]
                sh = shade[y : y + size, x : x + size]
                ch = _content_hash(th, sh)
                if ch in seen_hashes:
                    continue
                seen_hashes.add(ch)
                selected.append(box)
                break
    return selected

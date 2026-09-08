"""Global semantic guide letterboxing (nearest-neighbour, exact colours)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from shadow_dataset.constants import (
    DEFAULT_LETTERBOX_SIZE,
    OUTLINE,
    TARGET_REGION,
    UNSHADED,
)
from shadow_dataset.geometry import candidate_mask_from_input


@dataclass(frozen=True)
class LetterboxResult:
    """Letterboxed semantic guide and validity mask."""

    rgb: np.ndarray  # uint8 [H, W, 3]
    valid_mask: np.ndarray  # bool [H, W]
    scale: float
    offset_x: int
    offset_y: int
    content_width: int
    content_height: int


def _resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a boolean mask with nearest-neighbour interpolation."""
    if mask.dtype != np.uint8:
        arr = mask.astype(np.uint8) * 255
    else:
        arr = mask
    img = Image.fromarray(arr)
    resized = img.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) >= 128


def letterbox_semantic_guide(
    input_rgb: np.ndarray,
    *,
    size: int = DEFAULT_LETTERBOX_SIZE,
    pad_color: tuple[int, int, int] = UNSHADED,
) -> LetterboxResult:
    """Produce a square letterboxed semantic guide from a full input image.

    Aspect ratio is preserved. Semantic masks (outline / unshaded / TARGET_REGION)
    are resized with nearest-neighbour, then the RGB guide is reconstructed so
    black, white, and magenta values remain exact. Padding uses ``pad_color``
    (default white/unshaded).
    """
    if input_rgb.ndim != 3 or input_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got shape {input_rgb.shape}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    h, w = input_rgb.shape[:2]
    scale = min(size / h, size / w)
    content_h = max(1, int(round(h * scale)))
    content_w = max(1, int(round(w * scale)))
    # Ensure content fits.
    content_h = min(content_h, size)
    content_w = min(content_w, size)

    outline = (
        (input_rgb[:, :, 0] == OUTLINE[0])
        & (input_rgb[:, :, 1] == OUTLINE[1])
        & (input_rgb[:, :, 2] == OUTLINE[2])
    )
    candidate = candidate_mask_from_input(input_rgb)
    # Unshaded = white pixels (everything that is not outline/candidate is
    # treated carefully: only exact white maps to unshaded channel).
    unshaded = (
        (input_rgb[:, :, 0] == UNSHADED[0])
        & (input_rgb[:, :, 1] == UNSHADED[1])
        & (input_rgb[:, :, 2] == UNSHADED[2])
    )

    outline_r = _resize_mask_nearest(outline, content_w, content_h)
    candidate_r = _resize_mask_nearest(candidate, content_w, content_h)
    unshaded_r = _resize_mask_nearest(unshaded, content_w, content_h)

    # Priority: outline > candidate > unshaded (reconstruct exact colours).
    content = np.empty((content_h, content_w, 3), dtype=np.uint8)
    content[:, :] = pad_color
    content[unshaded_r] = UNSHADED
    content[candidate_r] = TARGET_REGION
    content[outline_r] = OUTLINE

    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:, :] = pad_color
    offset_y = (size - content_h) // 2
    offset_x = (size - content_w) // 2
    canvas[offset_y : offset_y + content_h, offset_x : offset_x + content_w] = content

    valid = np.zeros((size, size), dtype=bool)
    valid[offset_y : offset_y + content_h, offset_x : offset_x + content_w] = True

    return LetterboxResult(
        rgb=canvas,
        valid_mask=valid,
        scale=float(scale),
        offset_x=int(offset_x),
        offset_y=int(offset_y),
        content_width=int(content_w),
        content_height=int(content_h),
    )

"""Materialize crop folders for pilot visual QA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from shdowing.ai2bmp.io_utils import ensure_dir, save_rgb_png


def materialize_crop(
    crop: dict[str, Any],
    aligned_rgb: np.ndarray,
    target_rgb: np.ndarray,
    boundary_mask: np.ndarray,
    out_dir: Path,
) -> None:
    x, y, size = int(crop["x"]), int(crop["y"]), int(crop["crop_size"])
    crop_dir = ensure_dir(out_dir / crop["sample_id"])
    save_rgb_png(crop_dir / "input_aligned.png", aligned_rgb[y : y + size, x : x + size])
    save_rgb_png(crop_dir / "target_rgb.png", target_rgb[y : y + size, x : x + size])
    Image.fromarray(boundary_mask[y : y + size, x : x + size], mode="L").save(
        crop_dir / "boundary_mask.bmp", format="BMP"
    )

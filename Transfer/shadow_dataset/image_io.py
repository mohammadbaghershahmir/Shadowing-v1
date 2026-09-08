"""Lossless image loading with exact RGB decoding (no colour correction)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from shadow_dataset.constants import SUPPORTED_EXTENSIONS


class ImageLoadError(ValueError):
    """Raised when an image cannot be decoded under the pixel contract."""


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_rgb_exact(path: Path | str) -> tuple[np.ndarray, bool]:
    """Load an image as uint8 RGB without colour management.

    Palette / indexed images are expanded via the embedded palette so each
    index maps to its exact stored RGB triple.

    Returns
    -------
    rgb :
        Array of shape (H, W, 3), dtype uint8.
    had_alpha :
        True if the source contained an alpha channel (caller must validate).
    """
    path = Path(path)
    if not is_supported_image(path):
        raise ImageLoadError(
            f"Unsupported extension for {path}: expected one of "
            f"{sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Avoid Pillow colour-management conversions; decode palette manually.
    with Image.open(path) as img:
        img.load()
        mode = img.mode
        width, height = img.size

        if mode == "RGBA":
            arr = np.asarray(img, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 4:
                raise ImageLoadError(f"Unexpected RGBA array shape for {path}: {arr.shape}")
            rgb = arr[:, :, :3].copy()
            alpha = arr[:, :, 3]
            had_alpha = True
            if not np.all(alpha == 255):
                # Still return rgb; validation will flag alpha_problem.
                return rgb, True
            return rgb, True

        if mode == "RGB":
            rgb = np.asarray(img, dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ImageLoadError(f"Unexpected RGB array shape for {path}: {rgb.shape}")
            return rgb.copy(), False

        if mode == "P":
            # Expand palette without dithering or colour profile transforms.
            palette = img.getpalette()
            if palette is None:
                raise ImageLoadError(f"Indexed image has no palette: {path}")
            indices = np.asarray(img, dtype=np.uint8)
            # Palette is a flat list of RGB triples (up to 256).
            pal = np.asarray(palette, dtype=np.uint8)
            n_entries = len(pal) // 3
            pal_rgb = pal[: n_entries * 3].reshape(n_entries, 3)
            # Transparency chunk (if any) means alpha is present.
            transparency = img.info.get("transparency")
            had_alpha = transparency is not None
            if indices.max() >= n_entries:
                raise ImageLoadError(
                    f"Palette index out of range in {path}: "
                    f"max={int(indices.max())}, palette_entries={n_entries}"
                )
            rgb = pal_rgb[indices]
            if had_alpha:
                # Build alpha from transparency info for validation.
                alpha = np.full(indices.shape, 255, dtype=np.uint8)
                if isinstance(transparency, int):
                    alpha[indices == transparency] = 0
                elif isinstance(transparency, bytes):
                    # Per-index alpha table.
                    for idx, a in enumerate(transparency):
                        if idx < n_entries and a < 255:
                            alpha[indices == idx] = a
                if not np.all(alpha == 255):
                    return rgb.copy(), True
            return rgb.copy(), had_alpha

        if mode == "L":
            gray = np.asarray(img, dtype=np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
            return rgb, False

        if mode == "LA":
            arr = np.asarray(img, dtype=np.uint8)
            gray = arr[:, :, 0]
            alpha = arr[:, :, 1]
            rgb = np.stack([gray, gray, gray], axis=-1)
            return rgb, True

        if mode in {"1", "I", "I;16", "F"}:
            # Convert without colour management by going through L/RGB raw.
            converted = img.convert("RGB")
            rgb = np.asarray(converted, dtype=np.uint8)
            return rgb.copy(), False

        # Fallback: convert to RGBA to detect alpha, then strip.
        converted = img.convert("RGBA")
        arr = np.asarray(converted, dtype=np.uint8)
        rgb = arr[:, :, :3].copy()
        alpha = arr[:, :, 3]
        had_alpha = True
        return rgb, True


def load_rgb_and_validate_alpha(path: Path | str) -> tuple[np.ndarray, bool]:
    """Load RGB and report whether any non-opaque alpha was present."""
    rgb, had_alpha = load_rgb_exact(path)
    if not had_alpha:
        return rgb, False

    # Re-check alpha for modes that returned had_alpha=True with opaque alpha.
    path = Path(path)
    with Image.open(path) as img:
        img.load()
        if img.mode == "RGBA":
            alpha = np.asarray(img, dtype=np.uint8)[:, :, 3]
            return rgb, bool(not np.all(alpha == 255))
        if img.mode == "LA":
            alpha = np.asarray(img, dtype=np.uint8)[:, :, 1]
            return rgb, bool(not np.all(alpha == 255))
        if img.mode == "P" and img.info.get("transparency") is not None:
            transparency = img.info["transparency"]
            indices = np.asarray(img, dtype=np.uint8)
            if isinstance(transparency, int):
                return rgb, bool(np.any(indices == transparency))
            if isinstance(transparency, bytes):
                for idx, a in enumerate(transparency):
                    if a < 255 and np.any(indices == idx):
                        return rgb, True
                return rgb, False
        return rgb, False

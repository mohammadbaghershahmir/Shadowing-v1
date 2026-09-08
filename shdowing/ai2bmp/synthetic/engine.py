"""Apply corruption operations to RGB images."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from shdowing.ai2bmp.color import delta_e_stats, lab_to_srgb_uint8, srgb_uint8_to_lab


def _gaussian_field(shape: tuple[int, int], rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.standard_normal(shape)
    return ndimage.gaussian_filter(noise, sigma=sigma)


def apply_palette_shift(
    rgb: np.ndarray,
    index_map: np.ndarray,
    palette: np.ndarray,
    rng: np.random.Generator,
    *,
    delta_e_band: str = "mild",
    chroma_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    band_map = {"mild": 6, "moderate": 18, "strong": 40, "stress": 70}
    magnitude = band_map.get(delta_e_band, 10)
    out = rgb.copy()
    modified = np.zeros(rgb.shape[:2], dtype=bool)
    used = np.unique(index_map)
    for idx in used.tolist():
        mask = index_map == int(idx)
        base = palette[int(idx)].astype(np.float64)
        lab = srgb_uint8_to_lab(base.reshape(1, 1, 3))[0, 0]
        delta = rng.normal(0, magnitude / 3, size=3)
        if chroma_only:
            delta[0] *= 0.2
        else:
            delta[1:] *= 1.0
        new_lab = lab + delta
        new_rgb, _ = lab_to_srgb_uint8(new_lab.reshape(1, 1, 3))
        out[mask] = new_rgb.reshape(3)
        modified[mask] = True
    return out.astype(np.uint8), modified


def apply_intra_region_variation(
    rgb: np.ndarray,
    index_map: np.ndarray,
    rng: np.random.Generator,
    *,
    band: str = "easy",
) -> tuple[np.ndarray, np.ndarray]:
    band_sigma = {"easy": 8.0, "medium": 16.0, "hard": 24.0, "extreme": 40.0}
    sigma = band_sigma.get(band, 12.0)
    h, w = rgb.shape[:2]
    field = _gaussian_field((h, w), rng, sigma)
    field = (field - field.min()) / max(1e-8, field.max() - field.min())
    scale = {"easy": 8, "medium": 18, "hard": 35, "extreme": 55}.get(band, 10)
    out = rgb.astype(np.float64)
    modified = np.zeros((h, w), dtype=bool)
    for idx in np.unique(index_map).tolist():
        mask = index_map == int(idx)
        delta = (field[mask][:, None] - 0.5) * scale
        lab = srgb_uint8_to_lab(out[mask].astype(np.uint8))
        lab = lab + np.stack([delta[:, 0], delta[:, 0] * 0.5, delta[:, 0] * 0.3], axis=1)
        new_rgb, _ = lab_to_srgb_uint8(lab.reshape(-1, 1, 3))
        out[mask] = new_rgb.reshape(-1, 3)
        modified[mask] = True
    return np.clip(out, 0, 255).astype(np.uint8), modified


def apply_boundary_stress(
    rgb: np.ndarray,
    boundary_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    kind: str = "blur",
) -> tuple[np.ndarray, np.ndarray]:
    out = rgb.copy()
    modified = np.zeros(rgb.shape[:2], dtype=bool)
    b = boundary_mask > 0
    if kind == "antialias":
        img = Image.fromarray(out, mode="RGB")
        blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=0.6)))
        out[b] = blurred[b]
        modified[b] = True
    elif kind == "bleed":
        for c in range(3):
            ch = out[:, :, c].astype(np.float32)
            dilated = ndimage.grey_dilation(ch, size=(3, 3))
            out[:, :, c] = np.where(b, (0.6 * ch + 0.4 * dilated), ch).astype(np.uint8)
        modified[b] = True
    elif kind == "blur":
        img = Image.fromarray(out, mode="RGB")
        blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=1.0)))
        expanded = ndimage.binary_dilation(b, iterations=1)
        out[expanded] = blurred[expanded]
        modified[expanded] = True
    return out, modified


def apply_resolution_native(
    rgb: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = rgb.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    native = np.array(Image.fromarray(rgb, mode="RGB").resize((nw, nh), Image.Resampling.LANCZOS))
    aligned = np.array(Image.fromarray(native, mode="RGB").resize((w, h), Image.Resampling.LANCZOS))
    meta = {
        "source_width": w,
        "source_height": h,
        "native_width": nw,
        "native_height": nh,
        "aligned_width": w,
        "aligned_height": h,
        "scale_x": nw / w,
        "scale_y": nh / h,
        "alignment_method": "LANCZOS",
    }
    modified = np.any(native != np.array(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.NEAREST)), axis=-1)
    # mark all pixels modified when resolution changes
    modified_mask = np.ones((h, w), dtype=bool)
    return aligned, native, modified_mask, meta


def apply_jpeg_roundtrip(rgb: np.ndarray, *, quality: int = 50) -> tuple[np.ndarray, np.ndarray]:
    import io

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert("RGB"))
    if decoded.shape != rgb.shape:
        decoded = np.array(Image.fromarray(decoded).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.LANCZOS))
    modified = np.any(decoded != rgb, axis=-1)
    return decoded, modified


def apply_recipe(
    rgb: np.ndarray,
    index_map: np.ndarray,
    palette: np.ndarray,
    boundary_mask: np.ndarray,
    recipe_ops: list[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
    out = rgb.copy()
    modified = np.zeros(rgb.shape[:2], dtype=bool)
    native = None
    meta: dict[str, Any] = {}

    if not recipe_ops:
        return out, out.copy(), modified, meta

    for op in recipe_ops:
        name = op.get("op")
        if name == "palette_shift":
            out, m = apply_palette_shift(
                out,
                index_map,
                palette,
                rng,
                delta_e_band=op.get("delta_e_band", "mild"),
                chroma_only=bool(op.get("chroma_only", False)),
            )
            modified |= m
        elif name == "intra_region":
            out, m = apply_intra_region_variation(out, index_map, rng, band=op.get("band", "easy"))
            modified |= m
        elif name == "boundary_stress":
            out, m = apply_boundary_stress(out, boundary_mask, rng, kind=op.get("kind", "blur"))
            modified |= m
        elif name == "shade_family":
            out, m = apply_palette_shift(out, index_map, palette, rng, delta_e_band="moderate")
            modified |= m
        elif name == "resolution":
            out, native, m, rmeta = apply_resolution_native(out, scale=float(op.get("scale", 0.445)))
            modified |= m
            meta.update(rmeta)
        elif name == "jpeg":
            out, m = apply_jpeg_roundtrip(out, quality=int(op.get("quality", 50)))
            modified |= m

    if native is None:
        native = out.copy()

    meta["delta_e"] = delta_e_stats(rgb, out, modified if modified.any() else None)
    meta["changed_pixel_fraction"] = float(modified.mean())
    return out, native, modified, meta

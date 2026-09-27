"""BMP-only load/save with palette-resolved RGB and exact pixel preservation."""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_v2.constants import EXPECTED_RGB


class BmpLoadError(ValueError):
    pass


@dataclass
class BmpInspectResult:
    path: Path
    relative_path: str
    width: int
    height: int
    pillow_mode: str
    bit_depth: int | None
    color_type: str
    unique_rgb: dict[tuple[int, int, int], int]
    sha256: str
    had_alpha: bool
    status: str = "pending"
    errors: list[str] = field(default_factory=list)
    unexpected_rgb: dict[tuple[int, int, int], int] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_color_type(mode: str) -> str:
    if mode == "P":
        return "indexed"
    if mode in {"L", "LA", "1"}:
        return "grayscale"
    if mode in {"RGB", "RGBA"}:
        return "rgb"
    return "other"


def _infer_bit_depth(img: Image.Image) -> int | None:
    mode = img.mode
    if mode == "1":
        return 1
    if mode == "L":
        return 8
    if mode == "P":
        pal = img.getpalette()
        if pal is None:
            return 8
        return 8
    if mode == "RGB":
        return 24
    if mode == "RGBA":
        return 32
    return None


def resolve_bmp_rgb(path: Path | str) -> tuple[np.ndarray, BmpInspectResult]:
    """Load BMP and expand to exact uint8 RGB via embedded palette when indexed."""
    path = Path(path)
    if path.suffix.lower() != ".bmp":
        raise BmpLoadError(f"Not a BMP file: {path}")

    with Image.open(path) as img:
        img.load()
        mode = img.mode
        width, height = img.size
        had_alpha = False

        if mode == "RGB":
            rgb = np.asarray(img, dtype=np.uint8).copy()
        elif mode == "P":
            palette = img.getpalette()
            if palette is None:
                raise BmpLoadError(f"Indexed BMP missing palette: {path}")
            indices = np.asarray(img, dtype=np.uint8)
            pal = np.asarray(palette, dtype=np.uint8)
            n_entries = len(pal) // 3
            pal_rgb = pal[: n_entries * 3].reshape(n_entries, 3)
            if int(indices.max(initial=0)) >= n_entries:
                raise BmpLoadError(
                    f"Palette index out of range in {path}: max={int(indices.max())}"
                )
            rgb = pal_rgb[indices]
            if img.info.get("transparency") is not None:
                had_alpha = True
        elif mode == "L":
            gray = np.asarray(img, dtype=np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif mode == "RGBA":
            arr = np.asarray(img, dtype=np.uint8)
            rgb = arr[:, :, :3].copy()
            had_alpha = not np.all(arr[:, :, 3] == 255)
        elif mode == "LA":
            arr = np.asarray(img, dtype=np.uint8)
            gray = arr[:, :, 0]
            rgb = np.stack([gray, gray, gray], axis=-1)
            had_alpha = not np.all(arr[:, :, 1] == 255)
        else:
            raise BmpLoadError(f"Unsupported BMP mode {mode!r} in {path}")

        if rgb.shape != (height, width, 3):
            raise BmpLoadError(f"Unexpected RGB shape {rgb.shape} for {path}")

        meta = BmpInspectResult(
            path=path,
            relative_path="",
            width=width,
            height=height,
            pillow_mode=mode,
            bit_depth=_infer_bit_depth(img),
            color_type=_infer_color_type(mode),
            unique_rgb={},
            sha256=sha256_file(path),
            had_alpha=had_alpha,
        )
        return rgb, meta


def count_unique_rgb(rgb: np.ndarray) -> dict[tuple[int, int, int], int]:
    flat = rgb.reshape(-1, 3)
    if flat.size == 0:
        return {}
    packed = flat[:, 0].astype(np.uint32) << 16 | flat[:, 1].astype(np.uint32) << 8 | flat[:, 2]
    uniq, counts = np.unique(packed, return_counts=True)
    out: dict[tuple[int, int, int], int] = {}
    for val, cnt in zip(uniq.tolist(), counts.tolist(), strict=True):
        r = (val >> 16) & 0xFF
        g = (val >> 8) & 0xFF
        b = val & 0xFF
        out[(int(r), int(g), int(b))] = int(cnt)
    return out


def validate_five_colors(
    rgb: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int, int], int], list[str]]:
    """Return valid_mask (255=expected), unexpected counts, error messages."""
    h, w, _ = rgb.shape
    valid = np.zeros((h, w), dtype=np.uint8)
    unexpected: dict[tuple[int, int, int], int] = {}
    errors: list[str] = []

    counts = count_unique_rgb(rgb)
    total = h * w
    for color, cnt in counts.items():
        pct = 100.0 * cnt / max(total, 1)
        if color in EXPECTED_RGB:
            pass
        else:
            unexpected[color] = cnt
            errors.append(f"unexpected RGB {color}: count={cnt} ({pct:.4f}%)")

    # Build valid mask pixel-wise (exact triple match).
    for color in EXPECTED_RGB:
        r, g, b = color
        mask = (rgb[:, :, 0] == r) & (rgb[:, :, 1] == g) & (rgb[:, :, 2] == b)
        valid[mask] = 255

    if unexpected:
        errors.insert(0, f"found {len(unexpected)} unexpected RGB value(s)")
    return valid, unexpected, errors


def save_gray_bmp(path: Path | str, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        raise ValueError("save_gray_bmp expects uint8 array")
    if arr.ndim != 2:
        raise ValueError(f"save_gray_bmp expects HxW, got {arr.shape}")
    Image.fromarray(arr, mode="L").save(path, format="BMP")


def save_rgb_bmp(path: Path | str, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        raise ValueError("save_rgb_bmp expects uint8 array")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"save_rgb_bmp expects HxWx3, got {arr.shape}")
    Image.fromarray(arr, mode="RGB").save(path, format="BMP")


def copy_original_bmp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_unexpected_colors_diagnostic(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    unexpected: dict[tuple[int, int, int], int],
) -> np.ndarray:
    """Dim valid pixels; highlight unexpected in red."""
    out = (rgb.astype(np.float32) * 0.25).astype(np.uint8)
    if not unexpected:
        return out
    bad = valid_mask == 0
    out[bad] = np.array([255, 0, 0], dtype=np.uint8)
    return out

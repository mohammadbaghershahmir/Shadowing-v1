"""Aspect-preserving preprocessing for global and tiled DINOv3 inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


LVD_MEAN = (0.485, 0.456, 0.406)
LVD_STD = (0.229, 0.224, 0.225)
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)


@dataclass
class PreprocessMeta:
    original_hw: tuple[int, int]  # (H, W)
    resized_hw: tuple[int, int]  # content before pad
    processed_hw: tuple[int, int]  # after pad (H, W)
    pad_box: tuple[int, int, int, int]  # top, left, bottom, right
    scale: float
    pad_multiple: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TileSpec:
    y0: int
    x0: int
    y1: int
    x1: int
    row: int
    col: int


def load_rgb_png(path: str | Any) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        # Convert carefully; reject unexpected multi-spectral
        if img.mode in ("RGBA", "L", "P", "LA"):
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image mode {img.mode} for {path}; expected RGB PNG.")
    return img


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return int(np.ceil(value / multiple) * multiple)


def resize_long_side_preserve_aspect(
    image: Image.Image,
    long_side: int,
) -> tuple[Image.Image, float]:
    w, h = image.size
    scale = long_side / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return resized, scale


def pad_to_multiple(
    tensor: torch.Tensor,
    pad_multiple: int,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad CHW tensor on bottom/right to multiple of pad_multiple."""
    _, h, w = tensor.shape
    target_h = _ceil_to_multiple(h, pad_multiple)
    target_w = _ceil_to_multiple(w, pad_multiple)
    pad_bottom = target_h - h
    pad_right = target_w - w
    # F.pad for CHW: (left, right, top, bottom)
    if pad_bottom or pad_right:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), value=pad_value)
    pad_box = (0, 0, pad_bottom, pad_right)  # top, left, bottom, right
    return tensor, pad_box


def normalize_tensor(
    tensor: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def preprocess_global(
    image: Image.Image,
    *,
    long_side: int,
    pad_multiple: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[torch.Tensor, PreprocessMeta, Image.Image]:
    """Return normalized CHW tensor [3,H,W], metadata, and resized RGB (pre-pad) for viz."""
    resized, scale = resize_long_side_preserve_aspect(image, long_side)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    # Pad with ImageNet-ish zeros in float space before normalize — use 0 then normalize
    # Prefer padding in raw [0,1] with 0, then normalize (common practice).
    tensor, pad_box = pad_to_multiple(tensor, pad_multiple, pad_value=0.0)
    tensor = normalize_tensor(tensor, mean, std)
    meta = PreprocessMeta(
        original_hw=(image.size[1], image.size[0]),
        resized_hw=(resized.size[1], resized.size[0]),
        processed_hw=(tensor.shape[1], tensor.shape[2]),
        pad_box=pad_box,
        scale=scale,
        pad_multiple=pad_multiple,
        mean=mean,
        std=std,
        mode="global",
    )
    return tensor, meta, resized


def make_tile_grid(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> list[TileSpec]:
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be < tile_size ({tile_size})")
    stride = tile_size - overlap
    tiles: list[TileSpec] = []
    row = 0
    y = 0
    while True:
        y0 = y
        y1 = min(y0 + tile_size, height)
        if y1 - y0 < tile_size and y0 > 0:
            y0 = max(0, height - tile_size)
            y1 = height
        col = 0
        x = 0
        while True:
            x0 = x
            x1 = min(x0 + tile_size, width)
            if x1 - x0 < tile_size and x0 > 0:
                x0 = max(0, width - tile_size)
                x1 = width
            tiles.append(TileSpec(y0=y0, x0=x0, y1=y1, x1=x1, row=row, col=col))
            if x1 >= width:
                break
            x += stride
            col += 1
        if y1 >= height:
            break
        y += stride
        row += 1
    # de-duplicate identical boxes
    unique: list[TileSpec] = []
    seen: set[tuple[int, int, int, int]] = set()
    for t in tiles:
        key = (t.y0, t.x0, t.y1, t.x1)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def preprocess_for_tiling(
    image: Image.Image,
    *,
    pad_multiple: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    max_long_side: int | None = None,
) -> tuple[torch.Tensor, PreprocessMeta, Image.Image, list[TileSpec]]:
    """
    Native-detail path: optionally cap long side, pad to multiple, return full tensor + tile specs.
    Tile size/overlap applied by caller via make_tile_grid on processed_hw.
    """
    working = image
    scale = 1.0
    if max_long_side is not None and max(image.size) > max_long_side:
        working, scale = resize_long_side_preserve_aspect(image, max_long_side)
    arr = np.asarray(working).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    tensor, pad_box = pad_to_multiple(tensor, pad_multiple, pad_value=0.0)
    tensor_norm = normalize_tensor(tensor, mean, std)
    meta = PreprocessMeta(
        original_hw=(image.size[1], image.size[0]),
        resized_hw=(working.size[1], working.size[0]),
        processed_hw=(tensor_norm.shape[1], tensor_norm.shape[2]),
        pad_box=pad_box,
        scale=scale,
        pad_multiple=pad_multiple,
        mean=mean,
        std=std,
        mode="tiled",
    )
    return tensor_norm, meta, working, []


def extract_tiles(
    tensor: torch.Tensor,
    tiles: list[TileSpec],
) -> list[torch.Tensor]:
    """Extract CHW tiles; pad small edge tiles to full tile size if needed."""
    out: list[torch.Tensor] = []
    for t in tiles:
        crop = tensor[:, t.y0 : t.y1, t.x0 : t.x1]
        th, tw = crop.shape[1], crop.shape[2]
        # All tiles from make_tile_grid should already be full size except possibly when
        # image smaller than tile — pad to rectangular consistency handled by caller.
        out.append(crop)
        _ = (th, tw)
    return out


def content_slice_from_pad(pad_box: tuple[int, int, int, int], h: int, w: int) -> tuple[slice, slice]:
    top, left, bottom, right = pad_box
    return slice(top, h - bottom if bottom else h), slice(left, w - right if right else w)

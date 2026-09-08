"""Synthetic BMP builders for unit tests only."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _row_stride(width: int, bpp: int) -> int:
    return ((width * bpp + 31) // 32) * 4


def _write_bmp_8bit(
    path: Path,
    index_map: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
    *,
    clr_used: int | None = None,
) -> None:
    h, w = index_map.shape
    bpp = 8
    stride = _row_stride(w, bpp)
    declared = len(palette_rgb) if clr_used is None else int(clr_used)
    if clr_used == 0:
        declared = 0

    dib_size = 40
    palette_count = max(len(palette_rgb), declared if declared else 256)
    if declared == 0:
        palette_count = 256
    palette_bytes = 4 * palette_count
    pixel_offset = 14 + dib_size + palette_bytes
    pixel_size = stride * h

    out = bytearray()
    out.extend(b"BM")
    # placeholder file size; patched after payload write
    out.extend(struct.pack("<IHHI", 0, 0, 0, pixel_offset))
    out.extend(struct.pack("<I", dib_size))
    out.extend(
        struct.pack(
            "<iiHHIIiiII",
            w,
            -h,
            1,
            bpp,
            0,
            pixel_size,
            0,
            0,
            declared,
            0,
        )
    )

    for i in range(palette_count):
        if i < len(palette_rgb):
            r, g, b = palette_rgb[i]
        else:
            r, g, b = 0, 0, 0
        out.extend(bytes([b, g, r, 0]))

    for y in range(h):
        row = np.zeros(stride, dtype=np.uint8)
        row[:w] = index_map[y]
        out.extend(row.tobytes())

    struct.pack_into("<I", out, 2, len(out))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def _write_bmp_4bit(
    path: Path,
    index_map: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
    *,
    clr_used: int | None = None,
) -> None:
    h, w = index_map.shape
    bpp = 4
    stride = _row_stride(w, bpp)
    declared = len(palette_rgb) if clr_used is None else int(clr_used)

    dib_size = 40
    palette_count = max(len(palette_rgb), declared)
    palette_bytes = 4 * palette_count
    pixel_offset = 14 + dib_size + palette_bytes
    pixel_size = stride * h

    out = bytearray()
    out.extend(b"BM")
    out.extend(struct.pack("<IHHI", 0, 0, 0, pixel_offset))
    out.extend(struct.pack("<I", dib_size))
    out.extend(
        struct.pack(
            "<iiHHIIiiII",
            w,
            -h,
            1,
            bpp,
            0,
            pixel_size,
            0,
            0,
            declared,
            0,
        )
    )

    for i in range(palette_count):
        if i < len(palette_rgb):
            r, g, b = palette_rgb[i]
        else:
            r, g, b = 0, 0, 0
        out.extend(bytes([b, g, r, 0]))

    for y in range(h):
        row = bytearray(stride)
        for x in range(w):
            val = int(index_map[y, x]) & 0x0F
            byte_idx = x // 2
            if x % 2 == 0:
                row[byte_idx] = (val << 4) | (row[byte_idx] & 0x0F)
            else:
                row[byte_idx] = (row[byte_idx] & 0xF0) | val
        out.extend(row)

    struct.pack_into("<I", out, 2, len(out))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def make_palette_8_indices() -> list[tuple[int, int, int]]:
    return [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]


def make_index_map_checkerboard(size: int = 64, indices: list[int] | None = None) -> np.ndarray:
    idx = indices or [0, 1, 2, 3]
    out = np.zeros((size, size), dtype=np.uint8)
    block = max(4, size // 8)
    for y in range(size):
        for x in range(size):
            out[y, x] = idx[((y // block) + (x // block)) % len(idx)]
    return out


def write_test_bmp_8(
    path: Path,
    index_map: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
    *,
    clr_used: int | None = None,
) -> Path:
    _write_bmp_8bit(path, index_map, palette_rgb, clr_used=clr_used)
    return path


def write_test_bmp_4(
    path: Path,
    index_map: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
    *,
    clr_used: int | None = None,
) -> Path:
    _write_bmp_4bit(path, index_map, palette_rgb, clr_used=clr_used)
    return path

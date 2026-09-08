"""Decode indexed BMP raster without re-quantization."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from shdowing.ai2bmp.bmp.header import BI_RGB, BmpHeaderInfo


def _row_stride(width: int, bpp: int) -> int:
    bits = width * bpp
    return ((bits + 31) // 32) * 4


def decode_index_raster(data: bytes, header: BmpHeaderInfo) -> np.ndarray:
    if header.compression != BI_RGB:
        raise ValueError(f"Unsupported compression {header.compression_name}")

    bpp = header.bits_per_pixel
    if bpp not in (4, 8):
        raise ValueError(f"Unsupported bpp {bpp}; only 4 and 8 indexed BMP supported")

    w, h = header.width, header.absolute_height
    stride = _row_stride(w, bpp)
    expected = stride * h
    pixel_data = data[header.pixel_data_offset : header.pixel_data_offset + expected]
    if len(pixel_data) < expected:
        raise ValueError(
            f"Pixel data truncated: expected {expected} bytes, got {len(pixel_data)}"
        )

    out = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        src_row = row if header.top_down else (h - 1 - row)
        row_off = src_row * stride
        row_bytes = pixel_data[row_off : row_off + stride]
        if bpp == 8:
            out[row, :w] = np.frombuffer(row_bytes[:w], dtype=np.uint8)
        else:
            indices = np.zeros(w, dtype=np.uint8)
            for col in range(w):
                byte_idx = col // 2
                nibble = col % 2
                b = row_bytes[byte_idx]
                indices[col] = (b >> 4) if nibble == 0 else (b & 0x0F)
            out[row] = indices
    return out


def smallest_uint_dtype(max_index: int) -> np.dtype:
    if max_index <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_index <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def index_histogram(index_map: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(index_map, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def used_indices_from_raster(index_map: np.ndarray) -> np.ndarray:
    return np.sort(np.unique(index_map))


def index_to_rgb(index_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Exact palette lookup -> HxWx3 uint8 RGB."""
    return palette[index_map.astype(np.int64)]

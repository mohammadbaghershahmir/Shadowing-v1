"""BMP palette table parsing (BGR/BGRA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from shdowing.ai2bmp.bmp.header import BmpHeaderInfo


@dataclass(frozen=True)
class PaletteInfo:
    entries_rgb: list[tuple[int, int, int]]
    palette_entry_byte_width: int
    palette_offset: int
    declared_palette_entry_count: int
    physical_palette_entry_count: int | None
    physical_palette_detection_status: str


def palette_offset_bytes(header: BmpHeaderInfo) -> int:
    return 14 + header.dib_header_size


def parse_palette(data: bytes, header: BmpHeaderInfo) -> PaletteInfo:
    if header.bits_per_pixel > 8:
        return PaletteInfo(
            entries_rgb=[],
            palette_entry_byte_width=0,
            palette_offset=0,
            declared_palette_entry_count=0,
            physical_palette_entry_count=None,
            physical_palette_detection_status="not_applicable",
        )

    offset = palette_offset_bytes(header)
    entry_bytes = 4 if header.dib_header_size >= 40 else 3
    available = max(0, header.pixel_data_offset - offset)
    if available <= 0:
        physical_count = None
        status = "ambiguous"
    elif available % entry_bytes != 0:
        physical_count = None
        status = "ambiguous"
    else:
        physical_count = available // entry_bytes
        status = "detected"

    declared = header.declared_palette_entry_count
    if physical_count is not None:
        read_count = min(declared, physical_count)
    else:
        read_count = declared

    entries: list[tuple[int, int, int]] = []
    pos = offset
    for _ in range(read_count):
        if pos + entry_bytes > len(data):
            break
        b, g, r = data[pos], data[pos + 1], data[pos + 2]
        entries.append((int(r), int(g), int(b)))
        pos += entry_bytes

    return PaletteInfo(
        entries_rgb=entries,
        palette_entry_byte_width=entry_bytes,
        palette_offset=offset,
        declared_palette_entry_count=declared,
        physical_palette_entry_count=physical_count,
        physical_palette_detection_status=status,
    )


def palette_table_array(entries_rgb: list[tuple[int, int, int]]) -> np.ndarray:
    """Return [N, 3] uint8 RGB palette table."""
    if not entries_rgb:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.array(entries_rgb, dtype=np.uint8)


def rgb_for_index(palette: np.ndarray, index: int) -> tuple[int, int, int]:
    if index < 0 or index >= len(palette):
        raise IndexError(f"Palette index {index} out of range [0, {len(palette)})")
    r, g, b = palette[index]
    return int(r), int(g), int(b)


def find_duplicate_rgb_indices(
    used_indices: np.ndarray, palette: np.ndarray
) -> list[dict[str, Any]]:
    rgb_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for idx in used_indices.tolist():
        rgb = tuple(int(x) for x in palette[int(idx)])
        rgb_to_indices.setdefault(rgb, []).append(int(idx))
    dups = []
    for rgb, indices in rgb_to_indices.items():
        if len(indices) > 1:
            dups.append({"rgb": list(rgb), "indices": sorted(indices)})
    return dups

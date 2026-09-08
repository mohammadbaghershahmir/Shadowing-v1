"""Strict BMP validation with Pillow cross-check."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from shdowing.ai2bmp.bmp.header import parse_bmp_header
from shdowing.ai2bmp.bmp.palette import (
    find_duplicate_rgb_indices,
    palette_table_array,
    parse_palette,
    rgb_for_index,
)
from shdowing.ai2bmp.bmp.raster import (
    decode_index_raster,
    index_histogram,
    index_to_rgb,
    smallest_uint_dtype,
    used_indices_from_raster,
)
from shdowing.ai2bmp.io_utils import sha256_file


@dataclass
class BmpAuditRecord:
    filename: str
    relative_path: str
    source_path: str
    sha256: str
    file_size: int
    signature: str
    dib_header_type: str
    dib_header_size: int
    width: int
    signed_height: int
    absolute_height: int
    top_down: bool
    planes: int
    bits_per_pixel: int
    compression: int
    compression_name: str
    pixel_data_offset: int
    image_size_field: int
    clr_used: int
    maximum_index_capacity: int
    declared_palette_entry_count: int
    physical_palette_entry_count: int | None
    physical_palette_detection_status: str
    palette_entry_byte_width: int
    pillow_mode: str
    pillow_format: str
    used_original_indices: list[int] = field(default_factory=list)
    used_index_count: int = 0
    used_index_histogram: dict[str, int] = field(default_factory=dict)
    used_index_percentages: dict[str, float] = field(default_factory=dict)
    maximum_used_index: int = 0
    resolved_rgb_by_index: dict[str, list[int]] = field(default_factory=dict)
    unique_used_rgb_colors: list[list[int]] = field(default_factory=list)
    unique_used_rgb_count: int = 0
    duplicate_rgb_indices: list[dict[str, Any]] = field(default_factory=list)
    unused_declared_palette_entries: list[int] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    status: str = "invalid"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pillow_crosscheck(path: Path, index_map: np.ndarray, palette: np.ndarray) -> list[str]:
    errors: list[str] = []
    with Image.open(path) as im:
        if im.mode not in ("P", "PA"):
            errors.append(f"Pillow mode {im.mode!r} is not indexed palette mode")
            return errors
        pillow_arr = np.array(im)
        if pillow_arr.shape != index_map.shape:
            errors.append(
                f"Pillow raster shape {pillow_arr.shape} != decoded shape {index_map.shape}"
            )
            return errors
        if not np.array_equal(pillow_arr, index_map):
            mismatch = int(np.sum(pillow_arr != index_map))
            errors.append(f"Pillow index raster differs from byte decode in {mismatch} pixels")

        pal = im.getpalette()
        if pal is None:
            errors.append("Pillow palette missing")
            return errors
        pillow_palette = np.array(pal, dtype=np.uint8).reshape(-1, 3)
        compare_len = min(len(palette), len(pillow_palette))
        if compare_len > 0:
            diff = np.any(palette[:compare_len] != pillow_palette[:compare_len])
            if diff:
                errors.append("Pillow palette RGB disagrees with explicit BGR palette parse")

        rgb_from_pillow = pillow_palette[pillow_arr]
        rgb_from_parse = palette[pillow_arr.astype(np.int64)]
        if not np.array_equal(rgb_from_pillow, rgb_from_parse):
            mismatch = int(np.sum(rgb_from_pillow != rgb_from_parse))
            errors.append(f"Pillow-resolved RGB disagrees with explicit palette in {mismatch} pixels")
    return errors


def validate_bmp_file(path: Path, *, relative_to: Path | None = None) -> BmpAuditRecord:
    path = Path(path)
    rel = str(path.relative_to(relative_to)).replace("\\", "/") if relative_to else path.name
    record = BmpAuditRecord(
        filename=path.name,
        relative_path=rel,
        source_path=str(path.resolve()),
        sha256=sha256_file(path),
        file_size=0,
        signature="",
        dib_header_type="",
        dib_header_size=0,
        width=0,
        signed_height=0,
        absolute_height=0,
        top_down=False,
        planes=0,
        bits_per_pixel=0,
        compression=0,
        compression_name="",
        pixel_data_offset=0,
        image_size_field=0,
        clr_used=0,
        maximum_index_capacity=0,
        declared_palette_entry_count=0,
        physical_palette_entry_count=None,
        physical_palette_detection_status="unknown",
        palette_entry_byte_width=0,
        pillow_mode="",
        pillow_format="",
    )

    errors: list[str] = []
    warnings: list[str] = []

    if path.suffix.lower() != ".bmp":
        errors.append("Not a BMP file")

    try:
        data = path.read_bytes()
        record.file_size = len(data)
        header = parse_bmp_header(path, relative_to=relative_to)
        record.signature = header.signature
        record.dib_header_type = header.dib_header_type
        record.dib_header_size = header.dib_header_size
        record.width = header.width
        record.signed_height = header.signed_height
        record.absolute_height = header.absolute_height
        record.top_down = header.top_down
        record.planes = header.planes
        record.bits_per_pixel = header.bits_per_pixel
        record.compression = header.compression
        record.compression_name = header.compression_name
        record.pixel_data_offset = header.pixel_data_offset
        record.image_size_field = header.image_size_field
        record.clr_used = header.clr_used
        record.maximum_index_capacity = header.maximum_index_capacity
        record.declared_palette_entry_count = header.declared_palette_entry_count

        if header.width <= 0 or header.absolute_height <= 0:
            errors.append("Invalid dimensions")

        if header.bits_per_pixel > 8:
            errors.append(f"Not indexed BMP: bpp={header.bits_per_pixel}")
        elif header.bits_per_pixel == 0:
            errors.append("Invalid bits_per_pixel=0")

        palette_info = parse_palette(data, header)
        record.physical_palette_entry_count = palette_info.physical_palette_entry_count
        record.physical_palette_detection_status = palette_info.physical_palette_detection_status
        record.palette_entry_byte_width = palette_info.palette_entry_byte_width

        palette = palette_table_array(palette_info.entries_rgb)

        with Image.open(path) as im:
            record.pillow_mode = im.mode
            record.pillow_format = im.format or "BMP"
            if im.mode in ("RGBA", "PA"):
                alpha = np.array(im.getchannel("A"))
                if np.any(alpha != 255):
                    errors.append("RGBA image contains non-opaque alpha values")

        index_map = decode_index_raster(data, header)
        used = used_indices_from_raster(index_map)
        record.used_original_indices = used.tolist()
        record.used_index_count = int(len(used))
        record.maximum_used_index = int(used.max()) if len(used) else 0

        hist = index_histogram(index_map)
        total = int(index_map.size)
        record.used_index_histogram = {str(k): v for k, v in hist.items()}
        record.used_index_percentages = {
            str(k): round(100.0 * v / total, 6) for k, v in hist.items()
        }

        resolved: dict[str, list[int]] = {}
        for idx in used:
            idx_int = int(idx)
            if idx_int >= len(palette):
                errors.append(f"Used index {idx_int} exceeds palette length {len(palette)}")
                continue
            resolved[str(idx_int)] = list(rgb_for_index(palette, idx_int))
        record.resolved_rgb_by_index = resolved

        unique_rgbs = sorted({tuple(v) for v in resolved.values()})
        record.unique_used_rgb_colors = [list(rgb) for rgb in unique_rgbs]
        record.unique_used_rgb_count = len(unique_rgbs)
        record.duplicate_rgb_indices = find_duplicate_rgb_indices(used, palette)

        declared_indices = set(range(header.declared_palette_entry_count))
        record.unused_declared_palette_entries = sorted(
            declared_indices - set(int(x) for x in used.tolist())
        )

        errors.extend(_pillow_crosscheck(path, index_map, palette))

        rgb = index_to_rgb(index_map, palette)
        reconstructed = index_to_rgb(index_map, palette)
        if not np.array_equal(rgb, reconstructed):
            errors.append("Canonical RGB reconstruction is not exact")

    except Exception as exc:
        errors.append(str(exc))

    record.validation_errors = errors
    record.validation_warnings = warnings

    if errors:
        record.status = "invalid"
        record.reason = errors[0]
    elif warnings:
        record.status = "warning"
        record.reason = warnings[0]
    else:
        record.status = "valid"
        record.reason = "ok"

    return record


def load_validated_index_and_palette(
    path: Path, *, relative_to: Path | None = None
) -> tuple[BmpAuditRecord, np.ndarray, np.ndarray]:
    record = validate_bmp_file(path, relative_to=relative_to)
    if record.status == "invalid":
        raise ValueError(f"Invalid BMP {path.name}: {record.reason}")

    data = path.read_bytes()
    header = parse_bmp_header(path, relative_to=relative_to)
    palette_info = parse_palette(data, header)
    palette = palette_table_array(palette_info.entries_rgb)
    index_map = decode_index_raster(data, header)
    dtype = smallest_uint_dtype(int(index_map.max()))
    return record, index_map.astype(dtype), palette

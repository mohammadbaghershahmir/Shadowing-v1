"""Explicit BMP/DIB header parsing."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


BI_RGB = 0
BI_RLE8 = 1
BI_RLE4 = 2
BI_BITFIELDS = 3


@dataclass(frozen=True)
class BmpHeaderInfo:
    filename: str
    relative_path: str
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


def _compression_name(code: int) -> str:
    return {
        BI_RGB: "BI_RGB",
        BI_RLE8: "BI_RLE8",
        BI_RLE4: "BI_RLE4",
        BI_BITFIELDS: "BI_BITFIELDS",
    }.get(code, f"UNKNOWN_{code}")


def parse_bmp_header(path: Path, *, relative_to: Path | None = None) -> BmpHeaderInfo:
    data = path.read_bytes()
    if len(data) < 14:
        raise ValueError(f"BMP too small: {path}")

    signature = data[0:2].decode("ascii", errors="replace")
    if signature != "BM":
        raise ValueError(f"Invalid BMP signature {signature!r} in {path.name}")

    file_size, _reserved1, _reserved2, pixel_offset = struct.unpack_from("<IHHI", data, 2)
    dib_start = 14
    dib_size = struct.unpack_from("<I", data, dib_start)[0]

    if dib_size == 12:
        # BITMAPINFOHEADER core (OS/2 1.x style)
        width, height, planes, bpp, compression = struct.unpack_from("<ihHHII", data, dib_start + 4)
        image_size = 0
        clr_used = 0
        dib_type = "BITMAPCOREHEADER"
    elif dib_size >= 40:
        width, height, planes, bpp, compression, image_size, _, _, clr_used, _ = struct.unpack_from(
            "<iiHHIIiiII", data, dib_start + 4
        )
        dib_type = "BITMAPINFOHEADER" if dib_size == 40 else f"BITMAPINFOHEADER_V{dib_size}"
    else:
        raise ValueError(f"Unsupported DIB header size {dib_size} in {path.name}")

    top_down = height < 0
    abs_height = abs(height)
    rel = str(path.relative_to(relative_to)) if relative_to else path.name

    if clr_used == 0 and bpp <= 8:
        declared = 1 << bpp
    else:
        declared = clr_used

    return BmpHeaderInfo(
        filename=path.name,
        relative_path=rel.replace("\\", "/"),
        file_size=len(data),
        signature=signature,
        dib_header_type=dib_type,
        dib_header_size=dib_size,
        width=int(width),
        signed_height=int(height),
        absolute_height=int(abs_height),
        top_down=top_down,
        planes=int(planes),
        bits_per_pixel=int(bpp),
        compression=int(compression),
        compression_name=_compression_name(int(compression)),
        pixel_data_offset=int(pixel_offset),
        image_size_field=int(image_size),
        clr_used=int(clr_used),
        maximum_index_capacity=int(1 << bpp) if bpp <= 8 else 0,
        declared_palette_entry_count=int(declared),
    )

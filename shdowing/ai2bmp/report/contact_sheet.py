"""Pilot contact sheet generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shdowing.ai2bmp.io_utils import ensure_dir


def build_contact_sheet(images: list[tuple[str, np.ndarray]], out_path: Path, thumb: int = 256) -> None:
    cols = min(4, max(1, len(images)))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 24)), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)
    for i, (label, arr) in enumerate(images):
        r, c = divmod(i, cols)
        img = Image.fromarray(arr)
        img = img.resize((thumb, thumb), Image.Resampling.NEAREST)
        sheet.paste(img, (c * thumb, r * (thumb + 24)))
        draw.text((c * thumb + 4, r * (thumb + 24) + thumb + 2), label[:40], fill=(0, 0, 0))
    ensure_dir(out_path.parent)
    sheet.save(out_path, format="PNG")

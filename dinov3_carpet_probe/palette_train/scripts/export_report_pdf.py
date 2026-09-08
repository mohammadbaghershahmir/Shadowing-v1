"""Export the palette pipeline markdown report to PDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "dinov3_carpet_probe" / "palette_train" / "PALETTE_PIPELINE_REPORT.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dinov3_carpet_probe" / "palette_train" / "PALETTE_PIPELINE_REPORT.pdf",
    )
    return parser.parse_args()


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _wrap_line(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_pdf(input_path: Path, output_path: Path) -> None:
    text = input_path.read_text(encoding="utf-8").splitlines()
    page_size = (1654, 2339)  # A4-ish at ~150 dpi
    margin = 110
    title_font = _font(30)
    heading_font = _font(24)
    body_font = _font(18)
    code_font = _font(17)

    pages: list[Image.Image] = []
    page = Image.new("RGB", page_size, "white")
    draw = ImageDraw.Draw(page)
    y = margin
    max_width = page_size[0] - 2 * margin

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        p = Image.new("RGB", page_size, "white")
        return p, ImageDraw.Draw(p), margin

    for raw_line in text:
        stripped = raw_line.rstrip()
        if stripped.startswith("# "):
            font = title_font
            line_gap = 18
            content = stripped[2:]
        elif stripped.startswith("## "):
            font = heading_font
            line_gap = 14
            content = stripped[3:]
        elif stripped.startswith("```") or stripped.startswith("- "):
            font = code_font if stripped.startswith("```") else body_font
            line_gap = 10
            content = stripped
        else:
            font = body_font
            line_gap = 10
            content = stripped if stripped else " "

        wrapped = _wrap_line(draw, content, font, max_width)
        for line in wrapped:
            bbox = draw.textbbox((0, 0), line, font=font)
            height = (bbox[3] - bbox[1]) + line_gap
            if y + height > page_size[1] - margin:
                pages.append(page)
                page, draw, y = new_page()
            draw.text((margin, y), line, fill="black", font=font)
            y += height
        if not stripped:
            y += 6

    pages.append(page)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output_path, save_all=True, append_images=pages[1:])


def main() -> int:
    args = parse_args()
    render_pdf(args.input, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

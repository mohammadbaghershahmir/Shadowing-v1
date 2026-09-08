"""Input / feature-grid / tile-boundary overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from dinov3_carpet_probe.src.io_utils import ensure_dir
from dinov3_carpet_probe.src.preprocessing import TileSpec


UPSAMPLE_WARNING = "Upsampled feature overlays are not true pixel-level predictions."


def save_input_grid_visualization(
    original: Image.Image,
    *,
    processed_hw: tuple[int, int],
    stride: int,
    out_path: Path | str,
    tiles: list[TileSpec] | None = None,
    title: str = "",
    pad_box: tuple[int, int, int, int] | None = None,
) -> str:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, 2 if tiles else 1, figsize=(10 if tiles else 5, 5), dpi=130)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].imshow(original)
    oh, ow = original.size[1], original.size[0]
    # Draw feature grid scaled to original
    gh = max(1, processed_hw[0] // stride)
    gw = max(1, processed_hw[1] // stride)
    for i in range(1, gh):
        y = i / gh * oh
        axes[0].axhline(y, color="white", alpha=0.15, linewidth=0.4)
    for j in range(1, gw):
        x = j / gw * ow
        axes[0].axvline(x, color="white", alpha=0.15, linewidth=0.4)
    axes[0].set_title(
        f"{title}\norig={ow}x{oh} processed={processed_hw[1]}x{processed_hw[0]} "
        f"stride={stride} grid={gw}x{gh}\n{UPSAMPLE_WARNING}",
        fontsize=7,
    )
    axes[0].axis("off")

    if tiles is not None and len(axes) > 1:
        # Show tiles on processed-space diagram scaled to original via processed dims
        axes[1].imshow(original)
        scale_x = ow / processed_hw[1]
        scale_y = oh / processed_hw[0]
        for t in tiles:
            rect = patches.Rectangle(
                (t.x0 * scale_x, t.y0 * scale_y),
                (t.x1 - t.x0) * scale_x,
                (t.y1 - t.y0) * scale_y,
                linewidth=0.6,
                edgecolor="lime",
                facecolor="none",
                alpha=0.7,
            )
            axes[1].add_patch(rect)
        axes[1].set_title(f"Tile boundaries (n={len(tiles)})", fontsize=8)
        axes[1].axis("off")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)

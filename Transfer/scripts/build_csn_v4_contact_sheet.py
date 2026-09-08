#!/usr/bin/env python3
"""Build visual contact sheet for CSN-V4 dataset review."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from csn_v4.data.targets import load_gray_bmp
from csn_v4.geometry.d4 import D4Transform
from csn_v4.geometry.tile_spec import HALO, INPUT_SIZE, build_tile_spec


SEM_COLORS = {
    0: (0, 0, 0),
    100: (255, 0, 0),
    150: (0, 255, 0),
    200: (0, 0, 255),
    255: (220, 220, 220),
}


def _colorize_sem(sem: np.ndarray) -> np.ndarray:
    out = np.zeros((*sem.shape, 3), dtype=np.uint8)
    for g, rgb in SEM_COLORS.items():
        out[sem == g] = rgb
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("E:/Shadowing/Dataset_V4"))
    parser.add_argument("--scene-id", type=str, default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.dataset_root
    scenes = []
    with (root / "manifests/scenes.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                scenes.append(json.loads(line))
    scene = next(s for s in scenes if s["scene_id"] == args.scene_id) if args.scene_id else scenes[0]

    src = load_gray_bmp(scene["source_bw_path"])
    sem = load_gray_bmp(scene["target_semantic_path"])
    full_dir = Path(scene["full_artifact_dir"])

    panels: list[tuple[str, np.ndarray]] = []
    panels.append(("source_bw", np.stack([src] * 3, axis=-1)))
    panels.append(("target_semantic", _colorize_sem(sem)))

    for name in ("shade_mask", "transition_mask", "black_lock", "valid_mask"):
        p = full_dir / f"{name}.bmp"
        if p.exists():
            arr = load_gray_bmp(p)
            panels.append((name, np.stack([arr] * 3, axis=-1)))

    tbw = D4Transform(3).apply(src)
    tsem = D4Transform(3).apply(sem)
    panels.append(("d4_3_bw", np.stack([tbw] * 3, axis=-1)))
    panels.append(("d4_3_sem", _colorize_sem(tsem)))

    fh, fw = tbw.shape
    spec = build_tile_spec(fw, fh, HALO, HALO, transform_id=3, scene_id=scene["scene_id"])
    tile_bw = np.full((INPUT_SIZE, INPUT_SIZE), 255, dtype=np.uint8)
    tile_sem = np.full((INPUT_SIZE, INPUT_SIZE), 255, dtype=np.uint8)
    ix, iy = spec.input_x, spec.input_y
    for sy in range(INPUT_SIZE):
        for sx in range(INPUT_SIZE):
            gy, gx = iy + sy, ix + sx
            if 0 <= gy < fh and 0 <= gx < fw:
                tile_bw[sy, sx] = tbw[gy, gx]
                tile_sem[sy, sx] = tsem[gy, gx]
    overlay = _colorize_sem(tile_sem)
    overlay[:HALO, :, 0] = 255
    overlay[-HALO:, :, 0] = 255
    overlay[:, :HALO, 1] = 255
    overlay[:, -HALO:, 1] = 255
    ys, xs = spec.model_core_slices()
    overlay[ys, xs, 2] = 255
    panels.append(("tile+d4_3+halo/core", overlay))

    n = len(panels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"CSN-V4 contact sheet — {scene['scene_id']}", fontsize=12)
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_review.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

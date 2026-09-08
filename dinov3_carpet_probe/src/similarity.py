"""Cosine similarity maps and Gradio click UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dinov3_carpet_probe.src.io_utils import ensure_dir, read_json, save_npy, write_json


ANCHOR_LABELS = [
    "background",
    "motif_fill",
    "shade",
    "border",
    "outline",
    "rare_detail",
]


def normalized_to_feature_index(
    u: float,
    v: float,
    grid_hw: tuple[int, int],
) -> tuple[int, int]:
    """u,v in [0,1] relative to original image width/height -> feature (row, col)."""
    h, w = grid_hw
    col = int(np.clip(np.floor(u * w), 0, w - 1))
    row = int(np.clip(np.floor(v * h), 0, h - 1))
    return row, col


def cosine_similarity_map(
    features_hwc: np.ndarray,
    row: int,
    col: int,
) -> np.ndarray:
    h, w, c = features_hwc.shape
    feats = features_hwc.reshape(-1, c).astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-6)
    feats_n = feats / norms
    q = feats_n[row * w + col]
    sims = feats_n @ q
    return sims.reshape(h, w)


def save_similarity_visuals(
    image: Image.Image,
    sim_map: np.ndarray,
    out_dir: Path | str,
    stem: str,
    *,
    anchor: dict[str, Any] | None = None,
) -> dict[str, str]:
    out_dir = ensure_dir(out_dir)
    save_npy(Path(out_dir) / f"{stem}_similarity.npy", sim_map.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    axes[0].imshow(image)
    axes[0].set_title("Image")
    if anchor is not None:
        x = anchor["u"] * image.size[0]
        y = anchor["v"] * image.size[1]
        axes[0].scatter([x], [y], c="red", s=40, marker="x")
        axes[0].set_title(f"Anchor: {anchor.get('label', '')}")
    axes[0].axis("off")

    im1 = axes[1].imshow(sim_map, cmap="magma", vmin=-1, vmax=1)
    axes[1].set_title("Cosine similarity (feature grid)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    # Overlay: upsample sim to image size
    sim_img = Image.fromarray(
        ((sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8) * 255).astype(np.uint8)
    ).resize(image.size, Image.Resampling.NEAREST)
    axes[2].imshow(image)
    axes[2].imshow(np.asarray(sim_img), cmap="magma", alpha=0.45)
    if anchor is not None:
        axes[2].scatter(
            [anchor["u"] * image.size[0]],
            [anchor["v"] * image.size[1]],
            c="cyan",
            s=40,
            marker="x",
        )
    axes[2].set_title("Heatmap overlay (upsampled; not pixel-precise)")
    axes[2].axis("off")
    png = Path(out_dir) / f"{stem}_similarity.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {"npy": str(Path(out_dir) / f"{stem}_similarity.npy"), "png": str(png)}


def default_anchors() -> list[dict[str, Any]]:
    """Reasonable default probe anchors in normalized coords (user should refine via UI)."""
    presets = [
        ("background", 0.08, 0.08),
        ("motif_fill", 0.50, 0.45),
        ("shade", 0.35, 0.55),
        ("border", 0.50, 0.08),
        ("outline", 0.62, 0.40),
        ("rare_detail", 0.75, 0.70),
    ]
    return [{"label": lab, "u": u, "v": v} for lab, u, v in presets]


def load_or_create_anchors(path: Path | str | None) -> list[dict[str, Any]]:
    if path and Path(path).exists():
        data = read_json(path)
        if isinstance(data, dict) and "anchors" in data:
            return data["anchors"]
        return data
    return default_anchors()


def save_anchors(path: Path | str, anchors: list[dict[str, Any]]) -> None:
    write_json(path, {"anchors": anchors, "coords": "normalized_uv_original_image"})


def build_similarity_gradio(
    image: Image.Image,
    features_hwc: np.ndarray,
    *,
    out_dir: Path | str | None = None,
):
    import gradio as gr

    out_dir = ensure_dir(out_dir) if out_dir else None
    anchors: list[dict[str, Any]] = []

    def on_select(evt: gr.SelectData, label: str):
        # evt.index is (x, y) pixel coords for Image
        x, y = evt.index
        u = x / max(image.size[0] - 1, 1)
        v = y / max(image.size[1] - 1, 1)
        row, col = normalized_to_feature_index(u, v, features_hwc.shape[:2])
        sim = cosine_similarity_map(features_hwc, row, col)
        anchor = {"label": label or "custom", "u": float(u), "v": float(v), "feat_row": row, "feat_col": col}
        anchors.append(anchor)
        # Build overlay preview
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.imshow(image)
        sim_img = Image.fromarray(
            ((sim - sim.min()) / (sim.max() - sim.min() + 1e-8) * 255).astype(np.uint8)
        ).resize(image.size, Image.Resampling.NEAREST)
        ax.imshow(np.asarray(sim_img), cmap="magma", alpha=0.45)
        ax.scatter([x], [y], c="cyan", s=50, marker="x")
        ax.set_title(f"{anchor['label']} cosine similarity")
        ax.axis("off")
        if out_dir:
            save_similarity_visuals(
                image, sim, out_dir, f"interactive_{len(anchors)}_{anchor['label']}", anchor=anchor
            )
            save_anchors(Path(out_dir) / "anchors_interactive.json", anchors)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return buf, anchors

    with gr.Blocks(title="DINOv3 carpet cosine similarity") as demo:
        gr.Markdown(
            "Click a location on the carpet image. Cosine similarity uses **dense DINO features** "
            "(patch/stage grid). Upsampled overlays are **not** pixel-level predictions."
        )
        label = gr.Dropdown(choices=ANCHOR_LABELS + ["custom"], value="motif_fill", label="Anchor label")
        inp = gr.Image(value=np.asarray(image), label="Click image", type="numpy", interactive=True)
        out = gr.Image(label="Similarity overlay")
        state = gr.JSON(label="Anchors (normalized u,v)")
        inp.select(on_select, inputs=[label], outputs=[out, state])
    return demo

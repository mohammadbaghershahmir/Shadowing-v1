"""Deterministic DINO-only K-means clustering (structural, not palettes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans

from dinov3_carpet_probe.src.io_utils import ensure_dir, save_npy, write_json

CLUSTER_WARNING = (
    "These are structural DINO feature clusters, NOT color palettes."
)


def connected_components(label_map: np.ndarray, label: int) -> int:
    """Fast 4-connected component count via scipy."""
    mask = label_map == label
    if not mask.any():
        return 0
    _, n = ndimage.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int))
    return int(n)


def cluster_features(
    features_hwc: np.ndarray,
    k: int,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w, c = features_hwc.shape
    x = features_hwc.reshape(-1, c).astype(np.float32)
    finite = np.isfinite(x).all(axis=1)
    n_samples = int(finite.sum())
    # MiniBatchKMeans keeps large tiled maps tractable on CPU post-GPU inference
    batch_size = int(min(4096, max(256, n_samples // 10)))
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=3,
        batch_size=batch_size,
        max_iter=100,
    )
    labels = np.full(h * w, -1, dtype=np.int32)
    labels[finite] = km.fit_predict(x[finite])
    label_map = labels.reshape(h, w)
    stats = []
    for cid in range(k):
        area = int((label_map == cid).sum())
        comps = connected_components(label_map, cid) if area else 0
        stats.append(
            {
                "cluster": cid,
                "area_patches": area,
                "area_fraction": float(area / (h * w)),
                "n_components": comps,
            }
        )
    meta = {
        "k": k,
        "random_state": random_state,
        "inertia": float(km.inertia_),
        "clusters": stats,
        "warning": CLUSTER_WARNING,
        "grid_hw": [h, w],
        "algorithm": "MiniBatchKMeans",
    }
    return label_map, meta


def save_clustering_outputs(
    features_hwc: np.ndarray,
    image: Image.Image,
    out_viz: Path | str,
    out_raw: Path | str,
    stem: str,
    k_values: list[int],
    *,
    random_state: int = 0,
) -> dict[int, dict[str, Any]]:
    out_viz = ensure_dir(out_viz)
    out_raw = ensure_dir(out_raw)
    results = {}
    cmap = plt.get_cmap("tab20")
    for k in k_values:
        label_map, meta = cluster_features(features_hwc, k, random_state=random_state)
        save_npy(Path(out_raw) / f"{stem}_clusters_k{k}.npy", label_map)
        write_json(Path(out_raw) / f"{stem}_clusters_k{k}_stats.json", meta)

        color = cmap(label_map % 20)[..., :3]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        axes[0].imshow(color)
        axes[0].set_title(f"K={k} cluster map\n{CLUSTER_WARNING}", fontsize=8)
        axes[0].axis("off")
        overlay = Image.fromarray((color * 255).astype(np.uint8)).resize(
            image.size, Image.Resampling.NEAREST
        )
        axes[1].imshow(image)
        axes[1].imshow(np.asarray(overlay), alpha=0.45)
        axes[1].set_title("Overlay (upsampled; not pixel-precise)", fontsize=8)
        axes[1].axis("off")
        png = Path(out_viz) / f"{stem}_clusters_k{k}.png"
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        meta["png"] = str(png)
        results[k] = meta
    return results

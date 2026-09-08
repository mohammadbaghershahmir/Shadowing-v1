"""Deterministic PCA feature visualizations (artificial colors)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from dinov3_carpet_probe.src.io_utils import ensure_dir, save_npy, write_json


PCA_WARNING = (
    "PCA colors are artificial visualizations of feature variance — "
    "they are NOT actual carpet colors."
)


def fit_transform_pca(
    features_hwc: np.ndarray,
    *,
    n_components: int = 3,
    percentiles: tuple[float, float] = (1.0, 99.0),
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    features_hwc: [H,W,C]
    returns rgb [H,W,3] in 0..1 and metadata
    """
    h, w, c = features_hwc.shape
    x = features_hwc.reshape(-1, c).astype(np.float64)
    finite = np.isfinite(x).all(axis=1)
    if finite.sum() < n_components:
        raise ValueError("Not enough finite feature vectors for PCA")
    pca = PCA(n_components=n_components, random_state=random_state, svd_solver="full")
    pca.fit(x[finite])
    proj = pca.transform(x)
    proj = proj.reshape(h, w, n_components)
    lo, hi = np.percentile(proj[np.isfinite(proj).all(axis=-1)], percentiles, axis=0)
    # avoid zero range
    hi = np.where(np.abs(hi - lo) < 1e-8, lo + 1e-8, hi)
    norm = (proj - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    if n_components == 3:
        rgb = norm
    else:
        rgb = np.zeros((h, w, 3), dtype=np.float64)
        rgb[..., :n_components] = norm
    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "percentiles": list(percentiles),
        "percentile_low": lo.tolist(),
        "percentile_high": hi.tolist(),
        "n_components": n_components,
        "random_state": random_state,
        "warning": PCA_WARNING,
        "shape_hw": [h, w],
        "channels_in": c,
    }
    return rgb.astype(np.float32), meta


def save_pca_outputs(
    features_hwc: np.ndarray,
    out_dir: Path | str,
    stem: str,
    *,
    percentiles: tuple[float, float] = (1.0, 99.0),
    random_state: int = 0,
    title_suffix: str = "",
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    rgb, meta = fit_transform_pca(
        features_hwc, percentiles=percentiles, random_state=random_state
    )
    raw_path = Path(out_dir).parent / "raw" if (Path(out_dir).name == "viz") else Path(out_dir)
    # Caller passes viz dir; also write meta beside
    save_npy(Path(out_dir) / f"{stem}_pca_rgb.npy", rgb)
    write_json(Path(out_dir) / f"{stem}_pca_meta.json", meta)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(rgb)
    ax.set_title(
        f"PCA feature map {title_suffix}\n{PCA_WARNING}",
        fontsize=8,
    )
    ax.axis("off")
    png_path = Path(out_dir) / f"{stem}_pca.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    meta["png"] = str(png_path)
    return meta

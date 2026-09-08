"""Patch correspondence with spatial non-maximum suppression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dinov3_carpet_probe.src.io_utils import ensure_dir, save_npy, write_json
from dinov3_carpet_probe.src.similarity import cosine_similarity_map, normalized_to_feature_index


def top_n_with_nms(
    sim_map: np.ndarray,
    *,
    top_n: int = 8,
    nms_radius: int = 3,
    exclude_rc: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    h, w = sim_map.shape
    sims = sim_map.copy()
    if exclude_rc is not None:
        er, ec = exclude_rc
        r0, r1 = max(0, er - nms_radius), min(h, er + nms_radius + 1)
        c0, c1 = max(0, ec - nms_radius), min(w, ec + nms_radius + 1)
        sims[r0:r1, c0:c1] = -np.inf

    results = []
    for _ in range(top_n):
        idx = int(np.argmax(sims))
        score = float(sims.flat[idx])
        if not np.isfinite(score):
            break
        r, c = divmod(idx, w)
        results.append({"row": r, "col": c, "score": score})
        r0, r1 = max(0, r - nms_radius), min(h, r + nms_radius + 1)
        c0, c1 = max(0, c - nms_radius), min(w, c + nms_radius + 1)
        sims[r0:r1, c0:c1] = -np.inf
    return results


def feature_to_image_xy(
    row: int,
    col: int,
    grid_hw: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    """Map feature cell center to original image pixel xy (width, height order for scatter)."""
    gh, gw = grid_hw
    iw, ih = image_size
    x = (col + 0.5) / gw * iw
    y = (row + 0.5) / gh * ih
    return x, y


def run_correspondence_for_anchors(
    features_hwc: np.ndarray,
    image: Image.Image,
    anchors: list[dict[str, Any]],
    out_viz: Path | str,
    out_raw: Path | str,
    stem: str,
    *,
    top_n: int = 8,
    nms_radius: int = 3,
) -> dict[str, Any]:
    out_viz = ensure_dir(out_viz)
    out_raw = ensure_dir(out_raw)
    grid_hw = features_hwc.shape[:2]
    all_results: dict[str, Any] = {}

    for i, anchor in enumerate(anchors):
        label = anchor.get("label", f"anchor_{i}")
        row, col = normalized_to_feature_index(anchor["u"], anchor["v"], grid_hw)
        sim = cosine_similarity_map(features_hwc, row, col)
        matches = top_n_with_nms(
            sim, top_n=top_n, nms_radius=nms_radius, exclude_rc=(row, col)
        )
        payload = {
            "anchor": {**anchor, "feat_row": row, "feat_col": col},
            "matches": matches,
            "top_n": top_n,
            "nms_radius_patches": nms_radius,
            "note": "Tests repeated motifs / symmetry via feature cosine similarity.",
        }
        write_json(Path(out_raw) / f"{stem}_corr_{label}.json", payload)
        save_npy(Path(out_raw) / f"{stem}_corr_{label}_sim.npy", sim.astype(np.float32))

        fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
        ax.imshow(image)
        ax.set_title(
            f"Correspondence: {label}\n(top-{top_n} with spatial NMS; feature-grid matches)",
            fontsize=8,
        )
        ax_x, ax_y = feature_to_image_xy(row, col, grid_hw, image.size)
        ax.scatter([ax_x], [ax_y], c="cyan", s=80, marker="x", label="anchor")
        for m in matches:
            mx, my = feature_to_image_xy(m["row"], m["col"], grid_hw, image.size)
            ax.scatter([mx], [my], c="yellow", s=40, marker="o")
            ax.plot([ax_x, mx], [ax_y, my], c="yellow", alpha=0.4, linewidth=0.8)
        ax.legend(loc="upper right", fontsize=7)
        ax.axis("off")
        png = Path(out_viz) / f"{stem}_correspondence_{label}.png"
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        payload["png"] = str(png)
        all_results[label] = payload
    return all_results

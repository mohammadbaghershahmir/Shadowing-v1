"""Three-model comparison contact sheets and go/no-go checklist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dinov3_carpet_probe.src.io_utils import ensure_dir, write_json


GO_NOGO_TEMPLATE = """# DINOv3 Carpet Probe — Go / No-Go Checklist

Run: `{run_id}`
Image: `{image_stem}`

Review the comparison sheet and per-model outputs, then check criteria.

## Proceed to Lab-fusion palette pipeline only if ≥2 of these hold

- [ ] Similarity maps consistently retrieve repeated / symmetric motifs
- [ ] PCA/clustering separates field vs border vs motif regions coherently
- [ ] Similar-RGB but structurally different regions get different features
- [ ] Global vs tiled organization is stable (tiled adds detail without chaos)
- [ ] Fused intermediate/stage features are more useful than final-only

## Rejection signals

- [ ] Outputs mainly follow irrelevant texture
- [ ] Clusters are highly fragmented / patchy
- [ ] Similarity fails to retrieve repeated motifs
- [ ] Thin structures disappear with no tiled recovery
- [ ] LVD and SAT add no useful structure beyond RGB/Lab
- [ ] Cost ≫ visible benefit

## Decision

- [ ] **GO** — proceed to future palette pipeline fusion experiments
- [ ] **NO-GO** — do not force DINOv3 into the final architecture

Notes:

_______________________________________________

Comparison sheet: `{comparison_png}`
"""


def build_comparison_sheet(
    image_stem: str,
    model_order: list[str],
    model_display: dict[str, str],
    artifacts: dict[str, dict[str, Any]],
    out_path: Path | str,
    *,
    metrics: dict[str, dict[str, Any]] | None = None,
) -> str:
    """
    artifacts[model_key] should contain paths:
      pca_global_final, sim_motif_fill (optional), clusters_k8, clusters_k12, correspondence
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    n = len(model_order)
    rows = ["PCA global final", "Sim motif_fill", "Clusters K=8", "Clusters K=12", "Correspondence"]
    keys = [
        "pca_global_final",
        "sim_motif_fill",
        "clusters_k8",
        "clusters_k12",
        "correspondence",
    ]
    fig, axes = plt.subplots(len(rows), n, figsize=(4 * n, 3.2 * len(rows)), dpi=120)
    if n == 1:
        axes = np.array(axes).reshape(len(rows), 1)

    for col, mk in enumerate(model_order):
        art = artifacts.get(mk, {})
        for row, (rname, k) in enumerate(zip(rows, keys)):
            ax = axes[row, col]
            path = art.get(k)
            if path and Path(path).exists():
                ax.imshow(Image.open(path))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")
            if row == 0:
                title = model_display.get(mk, mk)
                if metrics and mk in metrics:
                    m = metrics[mk]
                    title += (
                        f"\n{m.get('runtime_s', '?')}s | "
                        f"VRAM {m.get('peak_vram_mb', '?')}MB | "
                        f"grid {m.get('grid_hw', '?')}"
                    )
                ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(rname, fontsize=8)

    fig.suptitle(
        f"Three-model comparison — {image_stem}\n"
        "Compare spatial organization only; do not compare PCA colors across fits.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def write_go_nogo_checklist(
    out_path: Path | str,
    *,
    run_id: str,
    image_stem: str,
    comparison_png: str,
) -> str:
    text = GO_NOGO_TEMPLATE.format(
        run_id=run_id, image_stem=image_stem, comparison_png=comparison_png
    )
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(text, encoding="utf-8")
    return str(out_path)


def write_summary_json(out_path: Path | str, summary: dict[str, Any]) -> None:
    write_json(out_path, summary)

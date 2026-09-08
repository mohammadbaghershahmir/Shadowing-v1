#!/usr/bin/env python
"""Build / refresh comparison report from an existing run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dinov3_carpet_probe.src.comparison_report import (
    build_comparison_sheet,
    write_go_nogo_checklist,
    write_summary_json,
)
from dinov3_carpet_probe.src.io_utils import ensure_dir, load_models_config, read_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=str)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    models_cfg = load_models_config()
    model_keys = list(models_cfg["models"].keys())
    display = {k: v["display_name"] for k, v in models_cfg["models"].items()}
    report_dir = ensure_dir(run_dir / "report")
    summary = {"run_dir": str(run_dir), "images": {}}

    skip = {"report", "anchors", "_env"}
    for image_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name not in skip):
        stem = image_dir.name
        if not (image_dir / "input.png").exists() and not any(image_dir.glob("*/viz")):
            continue
        artifacts = {}
        metrics = {}
        for mk in model_keys:
            mdir = image_dir / mk
            if not mdir.exists():
                continue
            viz = mdir / "viz"
            meta = mdir / "meta" / "run_meta.json"
            arts = {
                "pca_global_final": str(viz / "global_final_pca.png")
                if (viz / "global_final_pca.png").exists()
                else None,
                "sim_motif_fill": str(viz / "global_final_sim_motif_fill_similarity.png")
                if (viz / "global_final_sim_motif_fill_similarity.png").exists()
                else None,
                "clusters_k8": str(viz / "global_final_clusters_k8.png")
                if (viz / "global_final_clusters_k8.png").exists()
                else None,
                "clusters_k12": str(viz / "global_final_clusters_k12.png")
                if (viz / "global_final_clusters_k12.png").exists()
                else None,
                "correspondence": str(viz / "global_final_correspondence_motif_fill.png")
                if (viz / "global_final_correspondence_motif_fill.png").exists()
                else None,
            }
            artifacts[mk] = arts
            if meta.exists():
                metrics[mk] = read_json(meta)
        if not artifacts:
            continue
        cmp = build_comparison_sheet(
            stem,
            [m for m in model_keys if m in artifacts],
            display,
            artifacts,
            report_dir / f"comparison_{stem}.png",
            metrics=metrics,
        )
        write_go_nogo_checklist(
            report_dir / f"GO_NOGO_CHECKLIST_{stem}.md",
            run_id=run_dir.name,
            image_stem=stem,
            comparison_png=cmp,
        )
        summary["images"][stem] = {"comparison": cmp, "models": list(artifacts.keys())}

    write_summary_json(report_dir / "summary.json", summary)
    print(f"Wrote report under {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

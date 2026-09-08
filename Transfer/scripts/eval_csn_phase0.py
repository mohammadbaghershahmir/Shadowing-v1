#!/usr/bin/env python3
"""Phase-0 baseline evaluation: V2/V3/V4 crop vs tiled vs D4 orientations."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from csn_v4.data.targets import load_gray_bmp
from csn_v4.geometry.d4 import D4Transform
from csn_v4.metrics.region_metrics import compute_region_metrics
from csn_v3.masks import semantic_gray_to_class


def _pred_to_class(gray: np.ndarray) -> np.ndarray:
    from csn_v4.constants import SEMANTIC_GRAY_TO_CLASS
    out = np.zeros_like(gray, dtype=np.int64)
    for g, c in SEMANTIC_GRAY_TO_CLASS.items():
        out[gray == g] = c
    return out


def _load_artifacts(artifact_dir: Path) -> dict:
    sem = load_gray_bmp(artifact_dir / "target_semantic.bmp")
    valid = load_gray_bmp(artifact_dir / "valid_mask.bmp") if (artifact_dir / "valid_mask.bmp").exists() else np.ones_like(sem) * 255
    black = load_gray_bmp(artifact_dir / "black_lock.bmp") if (artifact_dir / "black_lock.bmp").exists() else np.zeros_like(sem)
    trans = load_gray_bmp(artifact_dir / "transition_mask.bmp") if (artifact_dir / "transition_mask.bmp").exists() else np.zeros_like(sem)
    tgt = semantic_gray_to_class(torch.from_numpy(sem.astype(np.int64))).numpy()
    return {"target_class": tgt, "valid_mask": valid, "black_lock": black, "transition_mask": trans, "semantic": sem}


def eval_v2_v3_crop(model, loader_fn, device, samples, version: str) -> list[dict]:
    results = []
    model.eval()
    for sample in samples:
        # Direct crop eval delegated to version-specific loader
        pass
    return results


def eval_full_tiled(model, version: str, device, scene: dict, transform_ids: list[int]) -> list[dict]:
    from csn_v4.inference.tiled import predict_full_image_gray as v4_predict
    src = load_gray_bmp(scene["source_bw_path"])
    art = _load_artifacts(Path(scene["artifact_dir"]))
    rows = []
    for k in transform_ids:
        tsrc = D4Transform(k).apply(src)
        tsem = D4Transform(k).apply(art["semantic"])
        tvalid = D4Transform(k).apply(art["valid_mask"])
        tblack = D4Transform(k).apply(art["black_lock"])
        ttrans = D4Transform(k).apply(art["transition_mask"])
        tgt = semantic_gray_to_class(torch.from_numpy(tsem.astype(np.int64))).numpy()

        if version == "v4":
            gray, _ = v4_predict(model, tsrc, device, transform_id=k, scene_id=scene["scene_id"])
            pred = _pred_to_class(gray)
        elif version == "v3":
            from csn_v3.inference.tiled import predict_full_image_gray
            gray, _ = predict_full_image_gray(model, tsrc, device)
            pred = _pred_to_class(gray)
        elif version == "v2":
            from csn_v2.inference.tiled import predict_full_image_gray
            gray, _ = predict_full_image_gray(model, tsrc, device)
            pred = _pred_to_class(gray)
        else:
            raise ValueError(version)

        rm = compute_region_metrics(
            pred, tgt, ttrans, tvalid, tblack, transform_id=k,
        )
        row = {"scene_id": scene["scene_id"], "transform_id": k, "mode": "tiled_full", **rm.to_dict()}
        rows.append(row)
    return rows


def save_error_map(pred: np.ndarray, tgt: np.ndarray, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        err = (pred != tgt).astype(np.uint8) * 255
        plt.imsave(path, err, cmap="gray")
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN Phase-0 baseline eval")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_phase0"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--v2-checkpoint", type=Path, default=Path("runs/20260801_172727_CSN_V2_overfit/best.pt"))
    parser.add_argument("--v3-checkpoint", type=Path, default=Path("runs/20260802_094525_CSN_V3_bw_overfit/best.pt"))
    parser.add_argument("--v4-checkpoint", type=Path, default=None)
    parser.add_argument("--scenes-manifest", type=Path, default=Path("E:/Shadowing/Dataset_V4/manifests/scenes.jsonl"))
    parser.add_argument("--dataset-v3-fallback", type=Path, default=Path("E:/Shadowing/Dataset_V3/manifests/eval_full_images.jsonl"))
    parser.add_argument("--orientations", type=str, default="all", help="all or comma-separated k values")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes: list[dict] = []
    if args.scenes_manifest.exists():
        with args.scenes_manifest.open(encoding="utf-8") as fh:
            scenes = [json.loads(line) for line in fh if line.strip()]
    elif args.dataset_v3_fallback.exists():
        with args.dataset_v3_fallback.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    scenes.append({
                        "scene_id": rec["sample_id"],
                        "source_bw_path": rec["source_bw_path"],
                        "target_semantic_path": rec["target_semantic_path"],
                        "artifact_dir": str(Path(rec["target_semantic_path"]).parent),
                    })

    if args.orientations == "all":
        orientations = list(range(8))
    else:
        orientations = [int(x) for x in args.orientations.split(",")]

    summary: dict = {"timestamp": datetime.now().isoformat(), "models": {}}

    def run_model(name: str, ckpt: Path | None, version: str) -> None:
        if ckpt is None or not ckpt.exists():
            summary["models"][name] = {"status": "skipped", "reason": "checkpoint missing"}
            return
        if version == "v2":
            from csn_v2.factory import load_model_from_checkpoint
            model, _ = load_model_from_checkpoint(ckpt, device=device)
        elif version == "v3":
            from csn_v3.factory import load_model_from_checkpoint
            model, _ = load_model_from_checkpoint(ckpt, device=device)
        else:
            from csn_v4.factory import load_model_from_checkpoint
            model, _ = load_model_from_checkpoint(ckpt, device=device)

        all_rows = []
        for scene in scenes:
            rows = eval_full_tiled(model, version, device, scene, orientations)
            all_rows.extend(rows)
            if rows:
                worst = min(rows, key=lambda r: r["global_miou"])
                map_dir = out_dir / name / "maps" / scene["scene_id"]
                map_dir.mkdir(parents=True, exist_ok=True)
                with (map_dir / f"worst_k{worst['transform_id']}.json").open("w", encoding="utf-8") as fh:
                    json.dump(worst, fh, indent=2)

        model_dir = out_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "per_sample.csv").open("w", encoding="utf-8") as fh:
            if all_rows:
                fh.write(",".join(all_rows[0].keys()) + "\n")
                for r in all_rows:
                    fh.write(",".join(str(r[k]) for k in all_rows[0].keys()) + "\n")

        if all_rows:
            metrics = {k: float(np.mean([r[k] for r in all_rows])) for k in ("global_miou", "boundary_miou", "seam_band_miou", "corner_miou")}
            by_k = {k: float(np.mean([r["global_miou"] for r in all_rows if r["transform_id"] == k])) for k in orientations}
            metrics["worst_orientation_miou"] = min(by_k.values()) if by_k else 0.0
            metrics["per_orientation"] = by_k
            summary["models"][name] = metrics
        else:
            summary["models"][name] = {"status": "no_results"}

    run_model("csn_v2", args.v2_checkpoint, "v2")
    run_model("csn_v3", args.v3_checkpoint, "v3")
    run_model("csn_v4", args.v4_checkpoint, "v4")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

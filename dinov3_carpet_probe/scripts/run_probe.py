#!/usr/bin/env python
"""Run zero-shot DINOv3 carpet-map feature probe."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo roots on path when executed as a script
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from PIL import Image

from dinov3_carpet_probe.src.clustering import save_clustering_outputs
from dinov3_carpet_probe.src.comparison_report import (
    build_comparison_sheet,
    write_go_nogo_checklist,
    write_summary_json,
)
from dinov3_carpet_probe.src.correspondence import run_correspondence_for_anchors
from dinov3_carpet_probe.src.environment_check import run_environment_check
from dinov3_carpet_probe.src.feature_adapters import (
    extract_features,
    extract_features_transformers_fallback,
)
from dinov3_carpet_probe.src.io_utils import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_WEIGHTS_DIR,
    REPO_ROOT,
    ensure_dir,
    load_experiment_config,
    load_models_config,
    resolve_input_images,
    save_npy,
    sha256_file,
    write_json,
)
from dinov3_carpet_probe.src.model_loader import (
    load_model,
    peak_vram_mb,
    unload_model,
    validate_weights,
)
from dinov3_carpet_probe.src.pca_visualizer import save_pca_outputs
from dinov3_carpet_probe.src.preprocessing import (
    load_rgb_png,
    make_tile_grid,
    preprocess_for_tiling,
    preprocess_global,
)
from dinov3_carpet_probe.src.reproducibility import base_run_metadata, set_seed
from dinov3_carpet_probe.src.similarity import (
    cosine_similarity_map,
    load_or_create_anchors,
    normalized_to_feature_index,
    save_anchors,
    save_similarity_visuals,
)
from dinov3_carpet_probe.src.tiled_inference import run_tiled_inference
from dinov3_carpet_probe.src.viz_overlays import save_input_grid_visualization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv3 carpet zero-shot feature probe")
    p.add_argument("--input", type=str, default=None, help="PNG file or directory (1-2 PNGs)")
    p.add_argument("--config", type=str, default=None, help="experiment.yaml path")
    p.add_argument("--models", nargs="*", default=None, help="Subset of model keys")
    p.add_argument("--modes", nargs="*", default=None, choices=["global", "tiled"])
    p.add_argument("--global-long-side", type=int, default=None)
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--tile-batch-size", type=int, default=None)
    p.add_argument("--anchors-json", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--weights-dir", type=str, default=None)
    p.add_argument("--loader", type=str, default=None, choices=["torch_hub", "transformers"])
    p.add_argument("--validate-weights-only", action="store_true")
    p.add_argument("--skip-env-check", action="store_true")
    return p.parse_args()


def _mean_std(cfg: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return tuple(cfg["mean"]), tuple(cfg["std"])


def analyze_feature_map(
    *,
    features_hwc: np.ndarray,
    image: Image.Image,
    out_raw: Path,
    out_viz: Path,
    stem: str,
    anchors: list,
    k_values: list[int],
    top_n: int,
    nms_radius: int,
    pca_percentiles: tuple[float, float],
    seed: int,
) -> dict:
    artifacts = {}
    # PCA
    pca_meta = save_pca_outputs(
        features_hwc,
        out_viz,
        stem,
        percentiles=pca_percentiles,
        random_state=seed,
        title_suffix=stem,
    )
    artifacts["pca"] = pca_meta.get("png")

    # Similarity per anchor
    for anchor in anchors:
        label = anchor["label"]
        row, col = normalized_to_feature_index(anchor["u"], anchor["v"], features_hwc.shape[:2])
        sim = cosine_similarity_map(features_hwc, row, col)
        paths = save_similarity_visuals(
            image,
            sim,
            out_viz,
            f"{stem}_sim_{label}",
            anchor={**anchor, "feat_row": row, "feat_col": col},
        )
        save_npy(out_raw / f"{stem}_sim_{label}.npy", sim.astype(np.float32))
        artifacts[f"sim_{label}"] = paths["png"]

    # Clustering
    cluster_meta = save_clustering_outputs(
        features_hwc,
        image,
        out_viz,
        out_raw,
        stem,
        k_values,
        random_state=seed,
    )
    for k, meta in cluster_meta.items():
        artifacts[f"clusters_k{k}"] = meta.get("png")

    # Correspondence
    corr = run_correspondence_for_anchors(
        features_hwc,
        image,
        anchors,
        out_viz,
        out_raw,
        stem,
        top_n=top_n,
        nms_radius=nms_radius,
    )
    # Prefer motif_fill correspondence for comparison sheet
    if "motif_fill" in corr:
        artifacts["correspondence"] = corr["motif_fill"].get("png")
    elif corr:
        artifacts["correspondence"] = next(iter(corr.values())).get("png")

    return artifacts


def run_one_model_on_image(
    *,
    loaded,
    image: Image.Image,
    image_path: Path,
    exp: dict,
    out_model: Path,
    anchors: list,
    modes: list[str],
) -> dict:
    cfg = loaded.config
    mean, std = _mean_std(cfg)
    pad_multiple = int(cfg["pad_multiple"])
    device = exp.get("device", "cuda")
    autocast_dtype = exp.get("autocast_dtype", "bfloat16")
    seed = int(exp.get("seed", 0))
    k_values = list(exp.get("k_values", [4, 8, 12, 16]))
    top_n = int(exp.get("top_n_correspondence", 8))
    nms_radius = int(exp.get("nms_radius_patches", 3))
    pca_pct = tuple(exp.get("pca_percentiles", [1, 99]))

    out_raw = ensure_dir(out_model / "raw")
    out_viz = ensure_dir(out_model / "viz")
    out_meta = ensure_dir(out_model / "meta")

    extract_fn = extract_features
    if loaded.source == "transformers":
        extract_fn = extract_features_transformers_fallback

    metrics: dict = {
        "model_key": loaded.key,
        "source": loaded.source,
        "param_count": loaded.param_count,
        "checkpoint": loaded.checkpoint_path,
        "input_sha256": sha256_file(image_path),
    }
    artifacts_for_sheet: dict = {}
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    if "global" in modes:
        t0 = time.perf_counter()
        tensor, pre_meta, resized = preprocess_global(
            image,
            long_side=int(exp["global_long_side"]),
            pad_multiple=pad_multiple,
            mean=mean,
            std=std,
        )
        batch = tensor.unsqueeze(0)
        final_b, fused_b = extract_fn(
            loaded, batch, device=device, autocast_dtype=autocast_dtype
        )
        runtime = time.perf_counter() - t0
        metrics["global_runtime_s"] = round(runtime, 3)
        metrics["global_peak_vram_mb"] = peak_vram_mb()
        metrics["global_grid_hw"] = list(final_b.grid_hw)
        metrics["global_stride"] = final_b.stride
        metrics["preprocess_global"] = pre_meta.to_dict()

        save_npy(out_raw / "features_global_final.npy", final_b.features[0].numpy())
        save_npy(out_raw / "features_global_fused.npy", fused_b.features[0].numpy())
        write_json(
            out_meta / "features_global_meta.json",
            {
                "final": {
                    "shape": list(final_b.features.shape),
                    "stride": final_b.stride,
                    "channels": final_b.channels,
                    "layers": final_b.layer_or_stage_ids,
                    "extras": final_b.extras,
                },
                "fused": {
                    "shape": list(fused_b.features.shape),
                    "stride": fused_b.stride,
                    "channels": fused_b.channels,
                    "layers": fused_b.layer_or_stage_ids,
                    "extras": fused_b.extras,
                },
            },
        )

        save_input_grid_visualization(
            image,
            processed_hw=pre_meta.processed_hw,
            stride=final_b.stride,
            out_path=out_viz / "input_grid_global.png",
            title=f"{loaded.key} global",
            pad_box=pre_meta.pad_box,
        )

        for kind, bundle in (("final", final_b), ("fused", fused_b)):
            feats = bundle.features[0].numpy()
            arts = analyze_feature_map(
                features_hwc=feats,
                image=image,
                out_raw=out_raw,
                out_viz=out_viz,
                stem=f"global_{kind}",
                anchors=anchors,
                k_values=k_values,
                top_n=top_n,
                nms_radius=nms_radius,
                pca_percentiles=pca_pct,
                seed=seed,
            )
            if kind == "final":
                artifacts_for_sheet["pca_global_final"] = arts.get("pca")
                artifacts_for_sheet["sim_motif_fill"] = arts.get("sim_motif_fill")
                artifacts_for_sheet["clusters_k8"] = arts.get("clusters_k8")
                artifacts_for_sheet["clusters_k12"] = arts.get("clusters_k12")
                artifacts_for_sheet["correspondence"] = arts.get("correspondence")

    if "tiled" in modes:
        t0 = time.perf_counter()
        # Use native resolution capped mildly for 8GB: allow up to 1536 long side for tiles
        max_long = max(int(exp["global_long_side"]), int(exp.get("tile_native_long_side", 1536)))
        full_tensor, pre_meta, working, _ = preprocess_for_tiling(
            image,
            pad_multiple=pad_multiple,
            mean=mean,
            std=std,
            max_long_side=max_long,
        )
        tiles_preview = make_tile_grid(
            pre_meta.processed_hw[0],
            pre_meta.processed_hw[1],
            tile_size=int(exp["tile_size"]),
            overlap=int(exp["overlap"]),
        )
        final_b, fused_b, tiles, tile_meta = run_tiled_inference(
            loaded,
            full_tensor,
            tile_size=int(exp["tile_size"]),
            overlap=int(exp["overlap"]),
            tile_batch_size=int(exp["tile_batch_size"]),
            device=device,
            autocast_dtype=autocast_dtype,
            extract_fn=extract_fn,
        )
        runtime = time.perf_counter() - t0
        metrics["tiled_runtime_s"] = round(runtime, 3)
        metrics["tiled_peak_vram_mb"] = peak_vram_mb()
        metrics["tiled_grid_hw"] = list(final_b.grid_hw)
        metrics["tiled_stride"] = final_b.stride
        metrics["preprocess_tiled"] = pre_meta.to_dict()
        metrics["tile_meta"] = tile_meta

        save_npy(out_raw / "features_tiled_final.npy", final_b.features[0].numpy())
        save_npy(out_raw / "features_tiled_fused.npy", fused_b.features[0].numpy())
        write_json(out_meta / "features_tiled_meta.json", tile_meta)

        save_input_grid_visualization(
            image,
            processed_hw=pre_meta.processed_hw,
            stride=final_b.stride,
            out_path=out_viz / "input_grid_tiled.png",
            tiles=tiles_preview,
            title=f"{loaded.key} tiled",
            pad_box=pre_meta.pad_box,
        )

        for kind, bundle in (("final", final_b), ("fused", fused_b)):
            analyze_feature_map(
                features_hwc=bundle.features[0].numpy(),
                image=image,
                out_raw=out_raw,
                out_viz=out_viz,
                stem=f"tiled_{kind}",
                anchors=anchors,
                k_values=k_values,
                top_n=top_n,
                nms_radius=nms_radius,
                pca_percentiles=pca_pct,
                seed=seed,
            )

    metrics["artifacts"] = artifacts_for_sheet
    metrics["runtime_s"] = metrics.get("global_runtime_s", metrics.get("tiled_runtime_s"))
    metrics["peak_vram_mb"] = metrics.get("global_peak_vram_mb") or metrics.get("tiled_peak_vram_mb")
    metrics["grid_hw"] = metrics.get("global_grid_hw") or metrics.get("tiled_grid_hw")
    write_json(out_meta / "run_meta.json", metrics)
    return metrics


def main() -> int:
    args = parse_args()
    exp = load_experiment_config(args.config)
    if args.seed is not None:
        exp["seed"] = args.seed
    if args.global_long_side is not None:
        exp["global_long_side"] = args.global_long_side
    if args.tile_size is not None:
        exp["tile_size"] = args.tile_size
    if args.overlap is not None:
        exp["overlap"] = args.overlap
    if args.tile_batch_size is not None:
        exp["tile_batch_size"] = args.tile_batch_size
    if args.loader is not None:
        exp["loader"] = args.loader
    if args.modes is not None:
        exp["modes"] = args.modes
    if args.models is not None:
        exp["models_order"] = args.models

    set_seed(int(exp.get("seed", 0)))
    output_root = Path(args.output_root) if args.output_root else (
        Path(exp["output_root"]) if exp.get("output_root") else DEFAULT_OUTPUT_ROOT
    )
    weights_dir = Path(args.weights_dir) if args.weights_dir else (
        Path(exp["weights_dir"]) if exp.get("weights_dir") else DEFAULT_WEIGHTS_DIR
    )
    ensure_dir(output_root)
    ensure_dir(weights_dir)

    if not args.skip_env_check:
        run_environment_check(output_root=output_root, require_cuda=True)

    status = validate_weights(
        weights_dir=weights_dir,
        output_json=ensure_dir(output_root / "_env") / "weights_status.json",
        attempt_load=False,
        repo_dir=REPO_ROOT,
    )
    print("Weights status:", status["all_found"])
    for k, v in status["models"].items():
        print(f"  {k}: {v['status']} local={v['local_pth']} hf={v['hf_cache']}")

    if args.validate_weights_only:
        if not status["all_found"]:
            print("\nHARD STOP: one or more official weights are MISSING.")
            print(status["access_procedure"])
            return 2
        # Optional dry load
        status2 = validate_weights(
            weights_dir=weights_dir,
            output_json=ensure_dir(output_root / "_env") / "weights_status_loaded.json",
            attempt_load=True,
            repo_dir=REPO_ROOT,
            device=exp.get("device", "cuda"),
        )
        print("Load validation all_found:", status2["all_found"])
        return 0 if status2["all_found"] else 2

    if args.input is None:
        print("ERROR: --input is required for a full probe run.")
        return 2

    models_cfg = load_models_config()
    model_keys = list(exp.get("models_order", list(models_cfg["models"].keys())))
    modes = list(exp.get("modes", ["global", "tiled"]))

    # Hard stop before inference if none of the requested official weights are available.
    requested_status = [status["models"].get(k, {}).get("status") for k in model_keys]
    if all(s != "FOUND" for s in requested_status):
        print("\nHARD STOP: requested official DINOv3 weights are MISSING/gated.")
        print("Configured input will not be processed until weights are available.")
        print(f"Input was: {args.input}")
        print(status["access_procedure"])
        return 2


    run_meta = base_run_metadata(int(exp.get("seed", 0)), exp)
    run_id = run_meta["run_id"]
    run_dir = ensure_dir(output_root / run_id)
    write_json(run_dir / "run_meta.json", run_meta)

    images = resolve_input_images(args.input)
    anchors = load_or_create_anchors(args.anchors_json)
    save_anchors(ensure_dir(run_dir / "anchors") / "shared_anchors.json", anchors)

    display_names = {k: models_cfg["models"][k]["display_name"] for k in model_keys}
    summary: dict = {"run_id": run_id, "images": {}, "weights_status": status}

    for image_path in images:
        image = load_rgb_png(image_path)
        stem = image_path.stem
        image_dir = ensure_dir(run_dir / stem)
        image.save(image_dir / "input.png")
        per_model_metrics = {}
        per_model_artifacts = {}

        for mk in model_keys:
            print(f"\n=== Loading model {mk} (one at a time) ===")
            try:
                loaded = load_model(
                    mk,
                    device=exp.get("device", "cuda"),
                    repo_dir=REPO_ROOT,
                    weights_dir=weights_dir,
                    prefer=exp.get("loader", "torch_hub"),
                )
            except Exception as exc:
                print(f"FAILED to load {mk}: {exc}")
                print(status["access_procedure"])
                summary.setdefault("errors", []).append({"model": mk, "error": str(exc)})
                # Hard rule: do not continue with substitutes; skip this model
                continue

            print(
                f"Loaded {mk} via {loaded.source}; params={loaded.param_count}; "
                f"ckpt={loaded.checkpoint_path}; device={loaded.device}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                print(
                    f"[gpu] {torch.cuda.get_device_name(0)} | "
                    f"allocated={torch.cuda.memory_allocated()/1024**2:.0f}MB",
                    flush=True,
                )
            out_model = ensure_dir(image_dir / mk)
            try:
                print(f"=== Inference on GPU: {mk} modes={modes} ===", flush=True)
                metrics = run_one_model_on_image(
                    loaded=loaded,
                    image=image,
                    image_path=image_path,
                    exp=exp,
                    out_model=out_model,
                    anchors=anchors,
                    modes=modes,
                )
                per_model_metrics[mk] = metrics
                per_model_artifacts[mk] = metrics.get("artifacts", {})
            finally:
                unload_model(loaded.model)
                print(f"Unloaded {mk}")

        # Comparison sheet
        report_dir = ensure_dir(run_dir / "report")
        if per_model_artifacts:
            cmp_path = build_comparison_sheet(
                stem,
                [m for m in model_keys if m in per_model_artifacts],
                display_names,
                per_model_artifacts,
                report_dir / f"comparison_{stem}.png",
                metrics=per_model_metrics,
            )
            write_go_nogo_checklist(
                report_dir / f"GO_NOGO_CHECKLIST_{stem}.md",
                run_id=run_id,
                image_stem=stem,
                comparison_png=cmp_path,
            )
            summary["images"][stem] = {
                "comparison": cmp_path,
                "models": per_model_metrics,
            }

    write_summary_json(run_dir / "report" / "summary.json", summary)
    print(f"\nDone. Outputs: {run_dir}")
    if summary.get("errors") and not summary.get("images"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

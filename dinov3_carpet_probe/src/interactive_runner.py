"""Reusable interactive runner for the Gradio carpet-probe UI."""

from __future__ import annotations

import io
import time
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dinov3_carpet_probe.src.clustering import cluster_features
from dinov3_carpet_probe.src.correspondence import (
    feature_to_image_xy,
    run_correspondence_for_anchors,
    top_n_with_nms,
)
from dinov3_carpet_probe.src.feature_adapters import extract_features
from dinov3_carpet_probe.src.io_utils import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_WEIGHTS_DIR,
    REPO_ROOT,
    ensure_dir,
    save_npy,
    write_json,
)
from dinov3_carpet_probe.src.model_loader import (
    LoadedModel,
    list_available_models,
    load_model,
    peak_vram_mb,
    unload_model,
)
from dinov3_carpet_probe.src.pca_visualizer import fit_transform_pca
from dinov3_carpet_probe.src.preprocessing import (
    make_tile_grid,
    preprocess_for_tiling,
    preprocess_global,
)
from dinov3_carpet_probe.src.reproducibility import set_seed
from dinov3_carpet_probe.src.similarity import (
    ANCHOR_LABELS,
    cosine_similarity_map,
    default_anchors,
    normalized_to_feature_index,
    save_anchors,
)
from dinov3_carpet_probe.src.tiled_inference import run_tiled_inference
from dinov3_carpet_probe.src.viz_overlays import save_input_grid_visualization


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _rgb_array_to_pil(rgb: np.ndarray) -> Image.Image:
    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def default_params() -> dict[str, Any]:
    return {
        "device": "cuda",
        "autocast_dtype": "float32",
        "loader": "torch_hub",
        "global_long_side": 1024,
        "tile_native_long_side": 1536,
        "modes": ["global", "tiled"],
        "tile_size": 512,
        "overlap": 128,
        "tile_batch_size": 1,
        "feature_kind": "final",
        "fusion_layers": "5,11,17,23",
        "fusion_stages": "1,2,3",
        "final_stage": 3,
        "pca_lo": 1.0,
        "pca_hi": 99.0,
        "k_values": [4, 8, 12, 16],
        "top_n_correspondence": 8,
        "nms_radius_patches": 3,
        "seed": 0,
        "save_outputs": True,
    }


def parse_int_list(text: str | list[int] | None, fallback: list[int]) -> list[int]:
    if text is None:
        return list(fallback)
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return list(fallback)
    return [int(p) for p in parts]


def gpu_status_text() -> str:
    if not torch.cuda.is_available():
        return "GPU: unavailable"
    alloc = torch.cuda.memory_allocated() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    return (
        f"GPU: {torch.cuda.get_device_name(0)} | "
        f"alloc={alloc:.0f}MB peak={peak:.0f}MB total={total:.0f}MB"
    )


@dataclass
class FeatureCache:
    mode: str  # global | tiled
    final: np.ndarray
    fused: np.ndarray
    stride_final: int
    stride_fused: int
    processed_hw: tuple[int, int]
    pad_box: tuple[int, int, int, int]
    tiles: list | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    loaded: LoadedModel | None = None
    image: Image.Image | None = None
    image_stem: str = "upload"
    caches: dict[str, FeatureCache] = field(default_factory=dict)
    anchors: list[dict[str, Any]] = field(default_factory=default_anchors)
    run_dir: Path | None = None
    last_params: dict[str, Any] = field(default_factory=default_params)
    last_metrics: dict[str, Any] = field(default_factory=dict)
    last_suite: dict[str, Any] = field(default_factory=dict)


class InteractiveRunner:
    def __init__(
        self,
        weights_dir: Path | str | None = None,
        output_root: Path | str | None = None,
        repo_dir: Path | str | None = None,
    ):
        self.weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
        self.output_root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
        self.repo_dir = Path(repo_dir) if repo_dir else REPO_ROOT
        self.state = SessionState()

    def available_model_choices(self) -> list[tuple[str, str]]:
        """Return Gradio choices as (label, value)."""
        items = list_available_models(self.weights_dir)
        return [
            (f"{m['display_name']} (~{m.get('approx_params_m', '?')}M)", m["key"])
            for m in items
        ]

    def unload(self) -> str:
        if self.state.loaded is not None:
            unload_model(self.state.loaded.model)
            self.state.loaded = None
        self.state.caches.clear()
        return f"Model unloaded. {gpu_status_text()}"

    def load_selected_model(self, model_key: str, params: dict[str, Any] | None = None) -> str:
        params = {**default_params(), **(params or {})}
        if self.state.loaded is not None and self.state.loaded.key == model_key:
            return f"Already loaded: {model_key}. {gpu_status_text()}"
        if self.state.loaded is not None:
            unload_model(self.state.loaded.model)
            self.state.loaded = None
            self.state.caches.clear()
        device = params.get("device", "cuda")
        loaded = load_model(
            model_key,
            device=device,
            repo_dir=self.repo_dir,
            weights_dir=self.weights_dir,
            prefer=params.get("loader", "torch_hub"),
        )
        self.state.loaded = loaded
        self.state.caches.clear()
        return (
            f"Loaded {loaded.key} via {loaded.source} "
            f"({loaded.param_count/1e6:.1f}M params). {gpu_status_text()}"
        )

    def set_image(self, image: Image.Image | np.ndarray | str | Path | None, name: str | None = None) -> str:
        if image is None:
            return "No image"
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
            stem = Path(image).stem
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            stem = name or "upload"
        else:
            img = image.convert("RGB")
            stem = name or getattr(image, "filename", None) or "upload"
            if isinstance(stem, str) and ("/" in stem or "\\" in stem):
                stem = Path(stem).stem
        self.state.image = img
        self.state.image_stem = str(stem)
        self.state.caches.clear()
        self.state.last_suite.clear()
        return f"Image set: {img.size[0]}x{img.size[1]} ({self.state.image_stem})"

    def _apply_fusion_overrides(self, params: dict[str, Any]) -> dict[str, Any]:
        assert self.state.loaded is not None
        cfg = deepcopy(self.state.loaded.config)
        if cfg["architecture"] == "vit":
            default_layers = cfg.get("fusion_layers", [5, 11, 17, 23])
            cfg["fusion_layers"] = parse_int_list(params.get("fusion_layers"), default_layers)
        else:
            default_stages = cfg.get("fusion_stages", [1, 2, 3])
            cfg["fusion_stages"] = parse_int_list(params.get("fusion_stages"), default_stages)
            cfg["final_stage"] = int(params.get("final_stage", cfg.get("final_stage", 3)))
        return cfg

    def extract(self, params: dict[str, Any] | None = None, progress: Callable | None = None) -> dict[str, Any]:
        params = {**default_params(), **(params or {})}
        self.state.last_params = params
        if self.state.loaded is None:
            raise RuntimeError("Load a model first")
        if self.state.image is None:
            raise RuntimeError("Upload an image first")

        set_seed(int(params.get("seed", 0)))
        cfg = self._apply_fusion_overrides(params)
        # Temporarily patch config for extract_features
        original_cfg = self.state.loaded.config
        self.state.loaded.config = cfg

        mean = tuple(cfg["mean"])
        std = tuple(cfg["std"])
        pad_multiple = int(cfg["pad_multiple"])
        device = params.get("device", "cuda")
        autocast_dtype = params.get("autocast_dtype", "float32")
        modes = params.get("modes") or ["global"]
        if isinstance(modes, str):
            modes = [modes]

        if params.get("save_outputs", True):
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.state.run_dir = ensure_dir(
                self.output_root / f"ui_{ts}" / self.state.image_stem / self.state.loaded.key
            )
            ensure_dir(self.state.run_dir / "raw")
            ensure_dir(self.state.run_dir / "viz")
            ensure_dir(self.state.run_dir / "meta")
            self.state.image.save(self.state.run_dir.parent.parent / "input.png")
            # also next to model
            self.state.image.save(self.state.run_dir.parent / "input.png")

        metrics: dict[str, Any] = {"model_key": self.state.loaded.key, "modes": modes}
        self.state.caches.clear()

        try:
            if "global" in modes:
                if progress:
                    progress(0.1, desc="Global inference on GPU…")
                t0 = time.perf_counter()
                tensor, pre_meta, _ = preprocess_global(
                    self.state.image,
                    long_side=int(params["global_long_side"]),
                    pad_multiple=pad_multiple,
                    mean=mean,
                    std=std,
                )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                final_b, fused_b = extract_features(
                    self.state.loaded,
                    tensor.unsqueeze(0),
                    device=device,
                    autocast_dtype=autocast_dtype,
                )
                self.state.caches["global"] = FeatureCache(
                    mode="global",
                    final=final_b.features[0].numpy(),
                    fused=fused_b.features[0].numpy(),
                    stride_final=final_b.stride,
                    stride_fused=fused_b.stride,
                    processed_hw=pre_meta.processed_hw,
                    pad_box=pre_meta.pad_box,
                    meta={
                        "preprocess": pre_meta.to_dict(),
                        "final_extras": final_b.extras,
                        "fused_extras": fused_b.extras,
                    },
                )
                metrics["global_runtime_s"] = round(time.perf_counter() - t0, 3)
                metrics["global_peak_vram_mb"] = peak_vram_mb()
                metrics["global_grid_hw"] = list(final_b.grid_hw)
                if self.state.run_dir:
                    save_npy(self.state.run_dir / "raw" / "features_global_final.npy", self.state.caches["global"].final)
                    save_npy(self.state.run_dir / "raw" / "features_global_fused.npy", self.state.caches["global"].fused)

            if "tiled" in modes:
                if progress:
                    progress(0.45, desc="Tiled inference on GPU…")
                t0 = time.perf_counter()
                max_long = max(int(params["global_long_side"]), int(params["tile_native_long_side"]))
                full_tensor, pre_meta, _, _ = preprocess_for_tiling(
                    self.state.image,
                    pad_multiple=pad_multiple,
                    mean=mean,
                    std=std,
                    max_long_side=max_long,
                )
                tiles_preview = make_tile_grid(
                    pre_meta.processed_hw[0],
                    pre_meta.processed_hw[1],
                    tile_size=int(params["tile_size"]),
                    overlap=int(params["overlap"]),
                )
                final_b, fused_b, tiles, tile_meta = run_tiled_inference(
                    self.state.loaded,
                    full_tensor,
                    tile_size=int(params["tile_size"]),
                    overlap=int(params["overlap"]),
                    tile_batch_size=int(params["tile_batch_size"]),
                    device=device,
                    autocast_dtype=autocast_dtype,
                )
                self.state.caches["tiled"] = FeatureCache(
                    mode="tiled",
                    final=final_b.features[0].numpy(),
                    fused=fused_b.features[0].numpy(),
                    stride_final=final_b.stride,
                    stride_fused=fused_b.stride,
                    processed_hw=pre_meta.processed_hw,
                    pad_box=pre_meta.pad_box,
                    tiles=tiles_preview,
                    meta={"preprocess": pre_meta.to_dict(), "tile_meta": tile_meta},
                )
                metrics["tiled_runtime_s"] = round(time.perf_counter() - t0, 3)
                metrics["tiled_peak_vram_mb"] = peak_vram_mb()
                metrics["tiled_grid_hw"] = list(final_b.grid_hw)
                metrics["n_tiles"] = len(tiles)
                if self.state.run_dir:
                    save_npy(self.state.run_dir / "raw" / "features_tiled_final.npy", self.state.caches["tiled"].final)
                    save_npy(self.state.run_dir / "raw" / "features_tiled_fused.npy", self.state.caches["tiled"].fused)
                    write_json(self.state.run_dir / "meta" / "tile_meta.json", tile_meta)
        finally:
            self.state.loaded.config = original_cfg

        self.state.last_metrics = metrics
        if self.state.run_dir:
            write_json(self.state.run_dir / "meta" / "run_meta.json", metrics)
            save_anchors(self.state.run_dir / "anchors.json", self.state.anchors)

        status = (
            f"Features cached | Model: {self.state.loaded.key} | "
            f"Modes: {','.join(modes)} | {gpu_status_text()}"
        )
        return {"status": status, "metrics": metrics, "modes": list(self.state.caches.keys())}

    def _pick_features(self, mode: str, kind: str) -> tuple[np.ndarray, FeatureCache]:
        if mode not in self.state.caches:
            raise RuntimeError(f"No cached features for mode={mode}. Run Extract first.")
        cache = self.state.caches[mode]
        feats = cache.final if kind == "final" else cache.fused
        return feats, cache

    def make_grid_overlay(self, mode: str = "global") -> Image.Image:
        if self.state.image is None or mode not in self.state.caches:
            raise RuntimeError("Need image + extracted features")
        cache = self.state.caches[mode]
        out = self.state.run_dir / "viz" / f"input_grid_{mode}.png" if self.state.run_dir else None
        if out is None:
            import tempfile

            out = Path(tempfile.mkdtemp()) / f"grid_{mode}.png"
        save_input_grid_visualization(
            self.state.image,
            processed_hw=cache.processed_hw,
            stride=cache.stride_final,
            out_path=out,
            tiles=cache.tiles if mode == "tiled" else None,
            title=f"{self.state.loaded.key if self.state.loaded else ''} {mode}",
            pad_box=cache.pad_box,
        )
        return Image.open(out).convert("RGB")

    def run_pca(self, mode: str = "global", kind: str = "final", params: dict[str, Any] | None = None) -> tuple[Image.Image, dict]:
        params = {**self.state.last_params, **(params or {})}
        feats, _ = self._pick_features(mode, kind)
        rgb, meta = fit_transform_pca(
            feats,
            percentiles=(float(params["pca_lo"]), float(params["pca_hi"])),
            random_state=int(params.get("seed", 0)),
        )
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.imshow(rgb)
        ax.set_title(f"PCA {mode}/{kind}\n(artificial colors ≠ carpet colors)", fontsize=8)
        ax.axis("off")
        pil = _fig_to_pil(fig)
        if self.state.run_dir:
            path = self.state.run_dir / "viz" / f"{mode}_{kind}_pca.png"
            pil.save(path)
            write_json(self.state.run_dir / "viz" / f"{mode}_{kind}_pca_meta.json", meta)
        return pil, meta

    def run_similarity_at(
        self,
        u: float,
        v: float,
        label: str = "custom",
        mode: str = "global",
        kind: str = "final",
        add_anchor: bool = True,
    ) -> tuple[Image.Image, dict]:
        if self.state.image is None:
            raise RuntimeError("No image")
        feats, _ = self._pick_features(mode, kind)
        row, col = normalized_to_feature_index(u, v, feats.shape[:2])
        sim = cosine_similarity_map(feats, row, col)
        anchor = {"label": label or "custom", "u": float(u), "v": float(v), "feat_row": row, "feat_col": col}
        if add_anchor:
            # replace same label if present
            self.state.anchors = [a for a in self.state.anchors if a.get("label") != label] + [anchor]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=110)
        axes[0].imshow(self.state.image)
        axes[0].scatter([u * self.state.image.size[0]], [v * self.state.image.size[1]], c="red", s=50, marker="x")
        axes[0].set_title(f"Anchor: {label}")
        axes[0].axis("off")
        im1 = axes[1].imshow(sim, cmap="magma", vmin=-1, vmax=1)
        axes[1].set_title("Cosine similarity")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        sim_img = Image.fromarray(
            ((sim - sim.min()) / (sim.max() - sim.min() + 1e-8) * 255).astype(np.uint8)
        ).resize(self.state.image.size, Image.Resampling.NEAREST)
        axes[2].imshow(self.state.image)
        axes[2].imshow(np.asarray(sim_img), cmap="magma", alpha=0.45)
        axes[2].scatter([u * self.state.image.size[0]], [v * self.state.image.size[1]], c="cyan", s=50, marker="x")
        axes[2].set_title("Overlay (not pixel-precise)")
        axes[2].axis("off")
        pil = _fig_to_pil(fig)
        if self.state.run_dir:
            pil.save(self.state.run_dir / "viz" / f"{mode}_{kind}_sim_{label}.png")
            save_npy(self.state.run_dir / "raw" / f"{mode}_{kind}_sim_{label}.npy", sim.astype(np.float32))
            save_anchors(self.state.run_dir / "anchors.json", self.state.anchors)
        return pil, {"anchor": anchor, "sim_min": float(sim.min()), "sim_max": float(sim.max())}

    def run_clusters(
        self,
        mode: str = "global",
        kind: str = "final",
        params: dict[str, Any] | None = None,
    ) -> tuple[list[Image.Image], dict]:
        params = {**self.state.last_params, **(params or {})}
        if self.state.image is None:
            raise RuntimeError("No image")
        feats, _ = self._pick_features(mode, kind)
        k_values = parse_int_list(params.get("k_values"), [4, 8, 12, 16])
        images = []
        metas = {}
        cmap = plt.get_cmap("tab20")
        for k in k_values:
            label_map, meta = cluster_features(feats, k, random_state=int(params.get("seed", 0)))
            color = cmap(label_map % 20)[..., :3]
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=110)
            axes[0].imshow(color)
            axes[0].set_title(f"K={k} (structural clusters, not palette)", fontsize=8)
            axes[0].axis("off")
            overlay = Image.fromarray((color * 255).astype(np.uint8)).resize(
                self.state.image.size, Image.Resampling.NEAREST
            )
            axes[1].imshow(self.state.image)
            axes[1].imshow(np.asarray(overlay), alpha=0.45)
            axes[1].set_title("Overlay", fontsize=8)
            axes[1].axis("off")
            pil = _fig_to_pil(fig)
            images.append(pil)
            metas[k] = meta
            if self.state.run_dir:
                pil.save(self.state.run_dir / "viz" / f"{mode}_{kind}_clusters_k{k}.png")
                save_npy(self.state.run_dir / "raw" / f"{mode}_{kind}_clusters_k{k}.npy", label_map)
        return images, metas

    def run_correspondence(
        self,
        mode: str = "global",
        kind: str = "final",
        params: dict[str, Any] | None = None,
    ) -> tuple[list[Image.Image], dict]:
        params = {**self.state.last_params, **(params or {})}
        if self.state.image is None:
            raise RuntimeError("No image")
        feats, _ = self._pick_features(mode, kind)
        if self.state.run_dir:
            out_viz = self.state.run_dir / "viz"
            out_raw = self.state.run_dir / "raw"
        else:
            import tempfile

            tmp = Path(tempfile.mkdtemp())
            out_viz, out_raw = tmp / "viz", tmp / "raw"
        results = run_correspondence_for_anchors(
            feats,
            self.state.image,
            self.state.anchors,
            out_viz,
            out_raw,
            f"{mode}_{kind}",
            top_n=int(params.get("top_n_correspondence", 8)),
            nms_radius=int(params.get("nms_radius_patches", 3)),
        )
        images = []
        for label, payload in results.items():
            png = payload.get("png")
            if png and Path(png).exists():
                images.append(Image.open(png).convert("RGB"))
        return images, results

    def run_full_suite(
        self,
        params: dict[str, Any] | None = None,
        progress: Callable | None = None,
    ) -> dict[str, Any]:
        params = {**default_params(), **(params or {})}
        extract_info = self.extract(params, progress=progress)
        mode = "global" if "global" in self.state.caches else next(iter(self.state.caches))
        kind = params.get("feature_kind", "final")

        if progress:
            progress(0.6, desc="PCA…")
        pca_img, pca_meta = self.run_pca(mode, kind, params)
        grid_img = self.make_grid_overlay(mode)

        if progress:
            progress(0.7, desc="Similarity for anchors…")
        sim_images = []
        for a in self.state.anchors:
            sim_pil, _ = self.run_similarity_at(
                a["u"], a["v"], label=a.get("label", "anchor"), mode=mode, kind=kind, add_anchor=False
            )
            sim_images.append(sim_pil)

        if progress:
            progress(0.85, desc="Clustering…")
        cluster_images, cluster_meta = self.run_clusters(mode, kind, params)

        if progress:
            progress(0.95, desc="Correspondence…")
        corr_images, corr_meta = self.run_correspondence(mode, kind, params)

        suite = {
            "status": extract_info["status"],
            "metrics": extract_info["metrics"],
            "mode": mode,
            "kind": kind,
            "grid": grid_img,
            "pca": pca_img,
            "pca_meta": pca_meta,
            "similarity": sim_images,
            "clusters": cluster_images,
            "cluster_meta": cluster_meta,
            "correspondence": corr_images,
            "corr_meta": corr_meta,
            "anchors": self.state.anchors,
            "run_dir": str(self.state.run_dir) if self.state.run_dir else None,
            "input": self.state.image,
        }
        self.state.last_suite = suite
        if progress:
            progress(1.0, desc="Done")
        return suite

    def zip_current_run(self) -> str | None:
        if not self.state.run_dir or not self.state.run_dir.exists():
            return None
        # zip parent ui_timestamp folder for convenience
        root = self.state.run_dir.parent.parent
        zip_path = root.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in root.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(root.parent))
        return str(zip_path)

    def compare_models(
        self,
        model_keys: list[str],
        params: dict[str, Any] | None = None,
        progress: Callable | None = None,
    ) -> tuple[Image.Image, dict]:
        """Run suite on up to 3 models with same image/params/anchors; return contact strip."""
        params = {**default_params(), **(params or {})}
        if self.state.image is None:
            raise RuntimeError("Upload an image first")
        keys = [k for k in model_keys if k][:3]
        if len(keys) < 2:
            raise RuntimeError("Select at least 2 models to compare")

        saved_anchors = list(self.state.anchors)
        panels: list[Image.Image] = []
        metrics_all: dict[str, Any] = {}
        for i, key in enumerate(keys):
            if progress:
                progress(i / len(keys), desc=f"Compare: {key}")
            self.load_selected_model(key, params)
            self.state.anchors = list(saved_anchors)
            # Prefer global-only for compare speed unless user asked tiled
            p = dict(params)
            if "global" not in (p.get("modes") or ["global"]):
                p["modes"] = ["global"]
            else:
                p["modes"] = ["global"]  # compare uses global for VRAM safety
            suite = self.run_full_suite(p)
            # compose column: pca + first cluster + first corr
            col_imgs = [suite["pca"]]
            if suite["clusters"]:
                col_imgs.append(suite["clusters"][0])
            if suite["correspondence"]:
                col_imgs.append(suite["correspondence"][0])
            # stack vertically
            w = max(im.size[0] for im in col_imgs)
            resized = []
            for im in col_imgs:
                ratio = w / im.size[0]
                resized.append(im.resize((w, int(im.size[1] * ratio)), Image.Resampling.BILINEAR))
            h = sum(im.size[1] for im in resized)
            col = Image.new("RGB", (w, h), (20, 20, 20))
            y = 0
            for im in resized:
                col.paste(im, (0, y))
                y += im.size[1]
            # title bar
            fig, ax = plt.subplots(figsize=(w / 100, 0.6), dpi=100)
            ax.text(0.5, 0.5, key, ha="center", va="center", color="white", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            fig.patch.set_facecolor("#222")
            title = _fig_to_pil(fig).resize((w, 40))
            titled = Image.new("RGB", (w, 40 + col.size[1]), (20, 20, 20))
            titled.paste(title, (0, 0))
            titled.paste(col, (0, 40))
            panels.append(titled)
            metrics_all[key] = suite["metrics"]

        # side by side
        h = max(p.size[1] for p in panels)
        total_w = sum(p.size[0] for p in panels) + 8 * (len(panels) - 1)
        sheet = Image.new("RGB", (total_w, h), (12, 12, 12))
        x = 0
        for p in panels:
            sheet.paste(p, (x, 0))
            x += p.size[0] + 8

        if self.state.run_dir:
            out = self.state.run_dir.parent.parent / "report"
            ensure_dir(out)
            sheet.save(out / "compare_sheet.png")
            write_json(out / "compare_metrics.json", metrics_all)
        return sheet, metrics_all

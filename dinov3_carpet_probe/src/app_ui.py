"""Professional Gradio UI for DINOv3 carpet probe."""

from __future__ import annotations

from typing import Any

import gradio as gr

from dinov3_carpet_probe.src.interactive_runner import (
    InteractiveRunner,
    default_params,
    gpu_status_text,
)
from dinov3_carpet_probe.src.similarity import ANCHOR_LABELS


def _collect_params(
    global_long_side,
    tile_native_long_side,
    modes,
    tile_size,
    overlap,
    tile_batch_size,
    feature_kind,
    fusion_layers,
    fusion_stages,
    final_stage,
    autocast_dtype,
    pca_lo,
    pca_hi,
    k_values,
    top_n,
    nms_radius,
    seed,
    save_outputs,
) -> dict[str, Any]:
    # Gradio CheckboxGroup may return list; Dropdown of modes similarly
    if modes is None:
        modes = ["global"]
    if isinstance(k_values, str):
        k_list = [int(x.strip()) for x in k_values.split(",") if x.strip()]
    else:
        k_list = [int(x) for x in (k_values or [4, 8, 12, 16])]
    return {
        **default_params(),
        "global_long_side": int(global_long_side),
        "tile_native_long_side": int(tile_native_long_side),
        "modes": list(modes),
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "tile_batch_size": int(tile_batch_size),
        "feature_kind": feature_kind,
        "fusion_layers": fusion_layers,
        "fusion_stages": fusion_stages,
        "final_stage": int(final_stage),
        "autocast_dtype": autocast_dtype,
        "pca_lo": float(pca_lo),
        "pca_hi": float(pca_hi),
        "k_values": k_list,
        "top_n_correspondence": int(top_n),
        "nms_radius_patches": int(nms_radius),
        "seed": int(seed),
        "save_outputs": bool(save_outputs),
        "device": "cuda",
        "loader": "torch_hub",
    }


def build_app(runner: InteractiveRunner | None = None, server_name: str = "127.0.0.1") -> gr.Blocks:
    runner = runner or InteractiveRunner()
    choices = runner.available_model_choices()
    if not choices:
        raise RuntimeError(
            "No local DINOv3 checkpoints found under dinov3_carpet_probe/weights/. "
            "Place official .pth files there first."
        )
    default_key = choices[0][1]
    choice_labels = [c[0] for c in choices]
    label_to_key = {c[0]: c[1] for c in choices}
    key_to_label = {c[1]: c[0] for c in choices}

    app_css = """
    .gradio-container { max-width: 1400px !important; }
    #status-box textarea { font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
    """

    with gr.Blocks(title="DINOv3 Carpet Probe") as demo:
        gr.Markdown(
            """
# DINOv3 Carpet Probe Studio
**Zero-shot dense-feature diagnostics** (not palette extraction).

- Upsampled maps are **not** pixel-level predictions  
- PCA colors are artificial — **not** carpet colors  
- Clusters are **structural**, not color palettes  
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                model_dd = gr.Dropdown(
                    choices=choice_labels,
                    value=key_to_label[default_key],
                    label="مدل / Model",
                    info="Only checkpoints present in weights/",
                )
                compare_models = gr.CheckboxGroup(
                    choices=choice_labels,
                    value=choice_labels[: min(2, len(choice_labels))],
                    label="مقایسه چند مدل / Compare models",
                )
                image_in = gr.Image(type="pil", label="آپلود تصویر / Upload RGB", height=280)
                status = gr.Textbox(label="وضعیت / Status", lines=3, elem_id="status-box")
                gpu_box = gr.Textbox(value=gpu_status_text(), label="GPU", interactive=False)

                with gr.Accordion("پارامترها / Parameters", open=True):
                    modes = gr.CheckboxGroup(
                        choices=["global", "tiled"],
                        value=["global", "tiled"],
                        label="Inference modes",
                    )
                    feature_kind = gr.Radio(
                        choices=["final", "fused"],
                        value="final",
                        label="Feature kind (for viz)",
                    )
                    global_long_side = gr.Slider(512, 1536, value=1024, step=64, label="global_long_side")
                    tile_native_long_side = gr.Slider(
                        512, 2048, value=1536, step=64, label="tile_native_long_side"
                    )
                    tile_size = gr.Slider(256, 768, value=512, step=64, label="tile_size")
                    overlap = gr.Slider(32, 384, value=128, step=16, label="overlap")
                    tile_batch_size = gr.Slider(1, 4, value=1, step=1, label="tile_batch_size")
                    autocast_dtype = gr.Dropdown(
                        choices=["float32", "bfloat16", "float16"],
                        value="float32",
                        label="autocast_dtype",
                    )
                    fusion_layers = gr.Textbox(
                        value="5,11,17,23",
                        label="ViT fusion_layers (ViT-B default 2,5,8,11)",
                    )
                    fusion_stages = gr.Textbox(value="1,2,3", label="ConvNeXt fusion_stages")
                    final_stage = gr.Slider(0, 3, value=3, step=1, label="ConvNeXt final_stage")
                    pca_lo = gr.Slider(0, 10, value=1, step=0.5, label="PCA percentile low")
                    pca_hi = gr.Slider(90, 100, value=99, step=0.5, label="PCA percentile high")
                    k_values = gr.CheckboxGroup(
                        choices=["4", "8", "12", "16"],
                        value=["4", "8", "12", "16"],
                        label="K-means K values",
                    )
                    top_n = gr.Slider(1, 32, value=8, step=1, label="correspondence top_n")
                    nms_radius = gr.Slider(1, 16, value=3, step=1, label="nms_radius_patches")
                    seed = gr.Number(value=0, precision=0, label="seed")
                    save_outputs = gr.Checkbox(value=True, label="Save under outputs/ui_*")

                with gr.Row():
                    btn_load = gr.Button("Load model (GPU)", variant="secondary")
                    btn_unload = gr.Button("Unload")
                btn_extract = gr.Button("Extract features", variant="secondary")
                btn_suite = gr.Button("▶ Run full suite", variant="primary")
                btn_compare = gr.Button("Compare selected models")
                btn_refresh_viz = gr.Button("Refresh viz (cached features)")
                btn_zip = gr.Button("Export ZIP")
                zip_file = gr.File(label="Download ZIP")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Overview"):
                        overview_input = gr.Image(label="Input", type="pil")
                        overview_grid = gr.Image(label="Feature grid / tiles", type="pil")
                        overview_meta = gr.JSON(label="Metrics")

                    with gr.Tab("PCA"):
                        pca_global_final = gr.Image(label="global / final", type="pil")
                        pca_global_fused = gr.Image(label="global / fused", type="pil")
                        pca_tiled_final = gr.Image(label="tiled / final", type="pil")
                        pca_tiled_fused = gr.Image(label="tiled / fused", type="pil")

                    with gr.Tab("Similarity"):
                        gr.Markdown("کلیک روی تصویر برای cosine similarity روی فیچرهای کش‌شده")
                        sim_label = gr.Dropdown(
                            choices=ANCHOR_LABELS + ["custom"],
                            value="motif_fill",
                            label="Anchor label",
                        )
                        sim_click_img = gr.Image(
                            label="Click image",
                            type="pil",
                            interactive=True,
                        )
                        sim_out = gr.Image(label="Similarity result", type="pil")
                        anchors_json = gr.JSON(label="Anchors (normalized u,v)")

                    with gr.Tab("Clustering"):
                        cluster_gallery = gr.Gallery(label="Clusters", columns=2, height=480)

                    with gr.Tab("Correspondence"):
                        corr_gallery = gr.Gallery(label="Top-N matches", columns=2, height=480)

                    with gr.Tab("Compare"):
                        compare_img = gr.Image(label="Comparison sheet", type="pil")
                        compare_meta = gr.JSON(label="Per-model metrics")

                    with gr.Tab("Exports"):
                        export_info = gr.Textbox(label="Run directory", lines=2)

        param_components = [
            global_long_side,
            tile_native_long_side,
            modes,
            tile_size,
            overlap,
            tile_batch_size,
            feature_kind,
            fusion_layers,
            fusion_stages,
            final_stage,
            autocast_dtype,
            pca_lo,
            pca_hi,
            k_values,
            top_n,
            nms_radius,
            seed,
            save_outputs,
        ]

        def _key_from_label(label: str) -> str:
            return label_to_key.get(label, default_key)

        def on_load(model_label, *param_args):
            params = _collect_params(*param_args)
            # Auto-set fusion layers for ViT-B when loading that model
            key = _key_from_label(model_label)
            if key == "vitb16_lvd" and params.get("fusion_layers") in ("5,11,17,23", "5, 11, 17, 23"):
                params["fusion_layers"] = "2,5,8,11"
            msg = runner.load_selected_model(key, params)
            return msg, gpu_status_text()

        def on_unload():
            msg = runner.unload()
            return msg, gpu_status_text()

        def on_image(img):
            if img is None:
                return "No image", None, None
            msg = runner.set_image(img)
            return msg, img, img

        def on_extract(model_label, *param_args, progress=gr.Progress(track_tqdm=False)):
            params = _collect_params(*param_args)
            key = _key_from_label(model_label)
            if runner.state.loaded is None or runner.state.loaded.key != key:
                runner.load_selected_model(key, params)
            info = runner.extract(params, progress=progress)
            grid = None
            mode = "global" if "global" in runner.state.caches else next(iter(runner.state.caches), None)
            if mode:
                grid = runner.make_grid_overlay(mode)
            return (
                info["status"],
                gpu_status_text(),
                runner.state.image,
                grid,
                info["metrics"],
                str(runner.state.run_dir) if runner.state.run_dir else "",
            )

        def _pca_safe(mode, kind, params):
            if mode not in runner.state.caches:
                return None
            try:
                img, _ = runner.run_pca(mode, kind, params)
                return img
            except Exception:
                return None

        def on_suite(model_label, *param_args, progress=gr.Progress(track_tqdm=False)):
            params = _collect_params(*param_args)
            key = _key_from_label(model_label)
            if runner.state.loaded is None or runner.state.loaded.key != key:
                progress(0.02, desc=f"Loading {key}")
                runner.load_selected_model(key, params)
            suite = runner.run_full_suite(params, progress=progress)

            # Extra PCA variants if cached
            params2 = params
            pca_gf = _pca_safe("global", "final", params2)
            pca_gu = _pca_safe("global", "fused", params2)
            pca_tf = _pca_safe("tiled", "final", params2)
            pca_tu = _pca_safe("tiled", "fused", params2)

            return (
                suite["status"],
                gpu_status_text(),
                suite.get("input"),
                suite.get("grid"),
                suite.get("metrics"),
                pca_gf or suite.get("pca"),
                pca_gu,
                pca_tf,
                pca_tu,
                suite.get("input"),
                suite.get("similarity", [None])[0] if suite.get("similarity") else None,
                runner.state.anchors,
                suite.get("clusters") or [],
                suite.get("correspondence") or [],
                str(runner.state.run_dir) if runner.state.run_dir else "",
            )

        def on_refresh(*param_args):
            params = _collect_params(*param_args)
            if not runner.state.caches:
                raise gr.Error("No cached features — run Extract or Full suite first")
            mode = "global" if "global" in runner.state.caches else next(iter(runner.state.caches))
            kind = params["feature_kind"]
            pca_img, _ = runner.run_pca(mode, kind, params)
            grid = runner.make_grid_overlay(mode)
            clusters, _ = runner.run_clusters(mode, kind, params)
            corrs, _ = runner.run_correspondence(mode, kind, params)
            return (
                f"Viz refreshed from cache ({mode}/{kind}). {gpu_status_text()}",
                gpu_status_text(),
                grid,
                _pca_safe("global", "final", params),
                _pca_safe("global", "fused", params),
                _pca_safe("tiled", "final", params),
                _pca_safe("tiled", "fused", params),
                clusters,
                corrs,
            )

        def on_click(label, feature_kind_v, evt: gr.SelectData, *param_args):
            if runner.state.image is None:
                raise gr.Error("Upload an image and extract features first")
            if not runner.state.caches:
                raise gr.Error("Extract features first")
            params = _collect_params(*param_args)
            mode = "global" if "global" in runner.state.caches else next(iter(runner.state.caches))
            x, y = evt.index
            w, h = runner.state.image.size
            u = float(x) / max(w - 1, 1)
            v = float(y) / max(h - 1, 1)
            pil, meta = runner.run_similarity_at(
                u, v, label=label or "custom", mode=mode, kind=feature_kind_v, add_anchor=True
            )
            return pil, runner.state.anchors, f"Anchor {meta['anchor']}"

        def on_compare(model_labels, *param_args, progress=gr.Progress(track_tqdm=False)):
            params = _collect_params(*param_args)
            keys = [_key_from_label(lbl) for lbl in (model_labels or [])]
            sheet, metrics = runner.compare_models(keys, params, progress=progress)
            return (
                f"Compared: {', '.join(keys)}. {gpu_status_text()}",
                gpu_status_text(),
                sheet,
                metrics,
            )

        def on_zip():
            path = runner.zip_current_run()
            if not path:
                raise gr.Error("No run directory to zip — run suite with Save enabled")
            return path, f"ZIP: {path}"

        btn_load.click(
            on_load,
            inputs=[model_dd, *param_components],
            outputs=[status, gpu_box],
        )
        btn_unload.click(on_unload, outputs=[status, gpu_box])
        image_in.change(on_image, inputs=[image_in], outputs=[status, overview_input, sim_click_img])

        btn_extract.click(
            on_extract,
            inputs=[model_dd, *param_components],
            outputs=[status, gpu_box, overview_input, overview_grid, overview_meta, export_info],
        )
        btn_suite.click(
            on_suite,
            inputs=[model_dd, *param_components],
            outputs=[
                status,
                gpu_box,
                overview_input,
                overview_grid,
                overview_meta,
                pca_global_final,
                pca_global_fused,
                pca_tiled_final,
                pca_tiled_fused,
                sim_click_img,
                sim_out,
                anchors_json,
                cluster_gallery,
                corr_gallery,
                export_info,
            ],
        )
        btn_refresh_viz.click(
            on_refresh,
            inputs=[*param_components],
            outputs=[
                status,
                gpu_box,
                overview_grid,
                pca_global_final,
                pca_global_fused,
                pca_tiled_final,
                pca_tiled_fused,
                cluster_gallery,
                corr_gallery,
            ],
        )
        sim_click_img.select(
            on_click,
            inputs=[sim_label, feature_kind, *param_components],
            outputs=[sim_out, anchors_json, status],
        )
        btn_compare.click(
            on_compare,
            inputs=[compare_models, *param_components],
            outputs=[status, gpu_box, compare_img, compare_meta],
        )
        btn_zip.click(on_zip, outputs=[zip_file, status])

    # Gradio 6: theme/css go on launch(); attach for launch_app
    demo.app_css = app_css  # type: ignore[attr-defined]
    return demo

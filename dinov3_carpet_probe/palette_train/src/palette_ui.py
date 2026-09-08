"""Minimal Gradio UI for palette checkpoint inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image
import torch

from dinov3_carpet_probe.palette_train.src.checkpointing import load_checkpoint
from dinov3_carpet_probe.palette_train.src.decode import DecodedPalette, decode_palette_prediction
from dinov3_carpet_probe.palette_train.src.model import PalettePredictor
from dinov3_carpet_probe.palette_train.src.transforms import build_transform


def _palette_html(decoded: DecodedPalette) -> str:
    if not decoded.rgb:
        return "<div>No palette colors predicted.</div>"
    blocks = []
    for rgb, hex_value in zip(decoded.rgb, decoded.hex):
        text_color = "#000000" if sum(rgb) > 382 else "#FFFFFF"
        blocks.append(
            (
                f"<div style='width:120px;height:90px;border-radius:12px;"
                f"background:{hex_value};display:flex;flex-direction:column;"
                f"justify-content:flex-end;padding:8px;box-sizing:border-box;"
                f"color:{text_color};font-family:Consolas,monospace;'>"
                f"<div>{hex_value}</div>"
                f"<div>{rgb}</div>"
                f"</div>"
            )
        )
    warning = f"<p><b>Warning:</b> {decoded.warning}</p>" if decoded.warning else ""
    return (
        f"<div style='display:flex;gap:12px;flex-wrap:wrap'>{''.join(blocks)}</div>"
        f"<p><b>num_colors:</b> {decoded.num_colors}</p>"
        f"{warning}"
    )


class PaletteUIRunner:
    def __init__(self) -> None:
        self._checkpoint_path: str | None = None
        self._cfg: dict[str, Any] | None = None
        self._model: PalettePredictor | None = None
        self._transform: Any = None

    @staticmethod
    def _clean_path(value: str) -> str:
        return value.strip().strip('"').strip("'")

    def _load(self, checkpoint_path: str) -> str:
        checkpoint_path = self._clean_path(checkpoint_path)
        ckpt_path = str(Path(checkpoint_path).resolve())
        if self._checkpoint_path == ckpt_path and self._model is not None:
            return f"Checkpoint already loaded: {ckpt_path}"
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = dict(checkpoint["config"])
        model = PalettePredictor(cfg).to(cfg["device"])
        load_checkpoint(ckpt_path, model=model, optimizer=None, scheduler=None, scaler=None, restore_rng=False)
        model.eval()
        self._checkpoint_path = ckpt_path
        self._cfg = cfg
        self._model = model
        self._transform = build_transform(cfg, training=False)
        return (
            f"Loaded checkpoint: {ckpt_path}\n"
            f"model_key={cfg['model_key']} | max_num_colors={cfg['max_num_colors']} | device={cfg['device']}"
        )

    def predict(self, checkpoint_path: str, image: Image.Image | None) -> tuple[str, str, dict[str, Any]]:
        checkpoint_path = self._clean_path(checkpoint_path)
        if not checkpoint_path:
            raise gr.Error("Checkpoint path is required.")
        if image is None:
            raise gr.Error("Input image is required.")
        status = self._load(checkpoint_path)
        tensor, meta = self._transform.transform_image(image, apply_flips=False)
        with torch.no_grad():
            outputs = self._model(
                tensor.unsqueeze(0).to(self._cfg["device"]),
                meta["raw_tensor"].unsqueeze(0).to(self._cfg["device"]),
                meta["valid_mask"].unsqueeze(0).to(self._cfg["device"]),
            )
        decoded = decode_palette_prediction(
            stem="uploaded_image",
            pred_rgb=outputs["pred_rgb"][0].detach().cpu(),
            presence_logits=outputs["presence_logits"][0].detach().cpu(),
            count_logits=outputs["count_logits"][0].detach().cpu(),
        )
        return status, _palette_html(decoded), decoded.to_dict()


def build_palette_ui() -> gr.Blocks:
    runner = PaletteUIRunner()
    css = ".gradio-container { max-width: 1100px !important; }"
    with gr.Blocks(title="Palette Checkpoint UI") as demo:
        gr.Markdown(
            """
# Palette Checkpoint UI
Load a trained palette checkpoint, upload one JPG/RGB image, and inspect the predicted palette as color swatches and JSON.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                checkpoint_path = gr.Textbox(
                    label="Checkpoint path",
                    placeholder=r"E:\Dino\dinov3\dinov3_carpet_probe\palette_train\runs\<run_id>\checkpoint_best.pt",
                )
                image_in = gr.Image(type="pil", label="Input image", height=360)
                run_btn = gr.Button("Predict palette", variant="primary")
                status = gr.Textbox(label="Status", lines=4, interactive=False)
            with gr.Column(scale=1):
                palette_html = gr.HTML(label="Palette colors")
                palette_json = gr.JSON(label="Output JSON")
        run_btn.click(
            fn=runner.predict,
            inputs=[checkpoint_path, image_in],
            outputs=[status, palette_html, palette_json],
        )
    demo.app_css = css
    return demo

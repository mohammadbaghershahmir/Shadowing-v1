#!/usr/bin/env python3
"""Simple Gradio UI: upload a semantic guide image → shaded categorical output."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shadow_dataset.constants import TARGET_REGION
from shadow_model.config import load_config
from shadow_model.factory import load_model_from_checkpoint
from shadow_model.global_cache import encode_global_view
from shadow_model.render import render_prediction, validate_output_colors
from shadow_model.tiled_inference import tiled_predict

LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("runs/20260729_210532_BEST_V1_fold0/best.pt")
DEFAULT_CONFIG = Path("configs/shadow_best.yaml")


def _to_uint8_rgb(image: np.ndarray | Image.Image) -> np.ndarray:
    """Convert Gradio/PIL input to HxWx3 uint8 RGB."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).round()
            image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    return np.ascontiguousarray(image)


@lru_cache(maxsize=1)
def load_model(checkpoint: str, config_path: str, device: str) -> tuple[torch.nn.Module, object]:
    """Load BEST_V1 checkpoint once and cache it."""
    cfg = load_config(config_path)
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, state = load_model_from_checkpoint(checkpoint, torch_device)
    model.eval()
    LOGGER.info(
        "Loaded checkpoint %s (step=%s, best_score=%s) on %s",
        checkpoint,
        state.get("global_step"),
        state.get("best_score"),
        torch_device,
    )
    return model, cfg


def shade_image(
    image: np.ndarray | Image.Image | None,
    checkpoint: str,
    config_path: str,
    device: str,
) -> tuple[np.ndarray | None, str]:
    """Run tiled inference and return shaded RGB + status text."""
    if image is None:
        return None, "لطفاً یک تصویر ورودی آپلود کنید."

    try:
        input_rgb = _to_uint8_rgb(image)
    except Exception as exc:  # noqa: BLE001
        return None, f"خطا در خواندن تصویر: {exc}"

    magenta = (
        (input_rgb[:, :, 0] == TARGET_REGION[0])
        & (input_rgb[:, :, 1] == TARGET_REGION[1])
        & (input_rgb[:, :, 2] == TARGET_REGION[2])
    )
    n_target = int(magenta.sum())
    if n_target == 0:
        return input_rgb, (
            "هیچ پیکسل TARGET_REGION=(255,0,255) پیدا نشد. "
            "خروجی همان ورودی است."
        )

    model, cfg = load_model(checkpoint, config_path, device)
    torch_device = next(model.parameters()).device

    gtokens = gvalid = letterbox_meta = None
    if getattr(model, "use_global_fusion", False):
        if model.dino_encoder is None:
            return None, "مدل global fusion دارد ولی dino_encoder لود نشده."
        gtokens, gvalid, letterbox_meta = encode_global_view(
            input_rgb,
            model.dino_encoder,
            input_size=cfg.dino.input_size,
            patch_size=cfg.dino.patch_size,
            device=torch_device,
        )

    t0 = time.time()
    with torch.inference_mode():
        logits = tiled_predict(
            model,
            input_rgb,
            tile_size=cfg.eval.tile,
            halo=cfg.eval.halo,
            stride=cfg.eval.stride,
            device=str(torch_device),
            global_tokens=gtokens,
            global_valid_mask=gvalid,
            letterbox_meta=letterbox_meta,
        )
    output_rgb = render_prediction(input_rgb, logits)
    color_check = validate_output_colors(output_rgb)
    dt = time.time() - t0

    h, w = input_rgb.shape[:2]
    status = (
        f"OK | size={w}x{h} | target_px={n_target:,} | "
        f"invalid_colors={color_check['invalid_color_count']} | "
        f"time={dt:.2f}s | device={torch_device} | ckpt={Path(checkpoint).name}"
    )
    return output_rgb, status


def build_ui(checkpoint: Path, config_path: Path, device: str):
    import gradio as gr

    with gr.Blocks(title="Shadowing BEST_V1") as demo:
        gr.Markdown(
            """
            # Shadowing — BEST_V1
            تصویر ورودی معنایی را آپلود کنید (سیاه / سفید / magenta).
            مدل فقط پیکسل‌های magenta را با کلاس‌های `(200,200,200)` / `(150,150,150)` / `(100,100,100)` جایگزین می‌کند.
            """
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="ورودی (semantic guide)", type="numpy", image_mode="RGB")
                run_btn = gr.Button("سایه بزن", variant="primary")
            with gr.Column():
                out = gr.Image(label="خروجی سایه‌زده", type="numpy")
                status = gr.Textbox(label="وضعیت", lines=2)

        with gr.Accordion("تنظیمات", open=False):
            ckpt = gr.Textbox(label="Checkpoint", value=str(checkpoint))
            cfg = gr.Textbox(label="Config", value=str(config_path))
            dev = gr.Textbox(label="Device", value=device)

        run_btn.click(
            fn=shade_image,
            inputs=[inp, ckpt, cfg, dev],
            outputs=[out, status],
        )
        inp.change(
            fn=shade_image,
            inputs=[inp, ckpt, cfg, dev],
            outputs=[out, status],
        )

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shadowing Gradio UI")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    # Warm-load so first UI click is faster
    load_model(str(args.checkpoint), str(args.config), args.device)

    demo = build_ui(args.checkpoint, args.config, args.device)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

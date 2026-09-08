#!/usr/bin/env python3
"""Gradio UI for CSN-V2: upload BW carpet → 512×512 tiled inference → stitched BMP."""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v2.config import load_config
from csn_v2.factory import load_model_from_checkpoint
from csn_v2.inference.tiled import predict_full_image_gray
from csn_v2.renderer import save_output_bmp, validate_palette

LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("runs/20260801_172727_CSN_V2_overfit/best.pt")
DEFAULT_CONFIG = Path("configs/csn_v2_overfit.yaml")
UI_OUTPUT_DIR = Path(tempfile.gettempdir()) / "csn_v2_ui_outputs"


def _to_uint8_gray(image: np.ndarray | Image.Image) -> np.ndarray:
    """Convert Gradio/PIL input to HxW uint8 grayscale."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("L"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[-1] >= 3:
                image = (
                    0.299 * image[:, :, 0]
                    + 0.587 * image[:, :, 1]
                    + 0.114 * image[:, :, 2]
                )
            else:
                image = image[:, :, 0]
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).round()
            image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    return np.ascontiguousarray(image)


def _gray_to_rgb_preview(gray: np.ndarray) -> np.ndarray:
    """Repeat gray channel for Gradio RGB preview."""
    return np.stack([gray, gray, gray], axis=-1)


@lru_cache(maxsize=1)
def load_model(checkpoint: str, config_path: str, device: str) -> tuple[torch.nn.Module, object]:
    """Load CSN-V2 checkpoint once and cache it."""
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, state = load_model_from_checkpoint(checkpoint, config_path, device=torch_device)
    cfg = load_config(config_path)
    model.eval()
    LOGGER.info(
        "Loaded checkpoint %s (step=%s, best_miou=%s) on %s",
        checkpoint,
        state.get("global_step"),
        state.get("best_miou"),
        torch_device,
    )
    return model, cfg


def shade_image(
    image: np.ndarray | Image.Image | None,
    checkpoint: str,
    config_path: str,
    device: str,
) -> tuple[np.ndarray | None, str | None, str]:
    """512×512 tiled inference, stitch full image, save BMP download."""
    if image is None:
        return None, None, "لطفاً یک تصویر BW آپلود کنید."

    try:
        source_bw = _to_uint8_gray(image)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"خطا در خواندن تصویر: {exc}"

    n_white = int((source_bw == 255).sum())
    if n_white == 0:
        return _gray_to_rgb_preview(source_bw), None, "هیچ پیکسل سفید (255) پیدا نشد. خروجی همان ورودی است."

    model, cfg = load_model(checkpoint, config_path, device)
    torch_device = next(model.parameters()).device

    t0 = time.perf_counter()
    with torch.inference_mode():
        gray, n_tiles = predict_full_image_gray(
            model,
            source_bw,
            device=torch_device,
            tile_size=cfg.inference.tile_size,
            halo=cfg.inference.halo,
            stride=cfg.inference.stride,
            context_size=cfg.data.context_size,
            global_long_side=cfg.inference.global_long_side,
        )
    ok, uniq = validate_palette(gray)
    dt = time.perf_counter() - t0

    UI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bmp_path = UI_OUTPUT_DIR / f"shaded_{uuid.uuid4().hex[:8]}.bmp"
    save_output_bmp(bmp_path, gray)

    h, w = source_bw.shape[:2]
    status = (
        f"OK | size={w}x{h} | tiles={n_tiles} (512×512, stride={cfg.inference.stride}, "
        f"halo={cfg.inference.halo}) | white_px={n_white:,} | "
        f"palette_ok={ok} | unique={sorted(uniq)} | "
        f"time={dt:.2f}s | device={torch_device} | bmp={bmp_path.name}"
    )
    return _gray_to_rgb_preview(gray), str(bmp_path), status


def build_ui(checkpoint: Path, config_path: Path, device: str):
    import gradio as gr

    with gr.Blocks(title="Shadowing CSN-V2") as demo:
        gr.Markdown(
            """
            # Shadowing — CSN-V2
            تصویر BW فرش را آپلود کنید. inference به **تایل‌های 512×512** با overlap
            (stride/halo) تقسیم می‌شود، هر تایل جدا پردازش می‌شود و سپس با blending به هم
            چسبانده می‌شود. خروجی: BMP grayscale (`255/200/150/100/0`).
            """
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="ورودی (source_bw)", type="numpy", image_mode="RGB")
                run_btn = gr.Button("سایه بزن", variant="primary")
            with gr.Column():
                out = gr.Image(label="پیش‌نمایش خروجی", type="numpy")
                bmp_out = gr.File(label="دانلود BMP", file_types=[".bmp"])
                status = gr.Textbox(label="وضعیت", lines=4)

        with gr.Accordion("تنظیمات", open=False):
            ckpt = gr.Textbox(label="Checkpoint", value=str(checkpoint))
            cfg = gr.Textbox(label="Config", value=str(config_path))
            dev = gr.Textbox(label="Device", value=device)

        run_btn.click(
            fn=shade_image,
            inputs=[inp, ckpt, cfg, dev],
            outputs=[out, bmp_out, status],
        )

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN-V2 Gradio UI")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    load_model(str(args.checkpoint), str(args.config), args.device)

    demo = build_ui(args.checkpoint, args.config, args.device)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

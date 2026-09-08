#!/usr/bin/env python3
"""Gradio UI for CSN-V3 with BMP output."""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v3.config import load_config
from csn_v3.factory import load_model_from_checkpoint
from csn_v3.inference.tiled import predict_full_image_gray
from csn_v3.renderer import save_output_bmp, validate_palette

UI_OUTPUT_DIR = Path(tempfile.gettempdir()) / "csn_v3_ui_outputs"


def _to_uint8_gray(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("L"))
    elif image.ndim == 3:
        image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)
    return np.ascontiguousarray(image.astype(np.uint8))


def _gray_to_rgb_preview(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN-V3 Gradio UI")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v3_bw_overfit.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    @lru_cache(maxsize=1)
    def _load():
        import torch
        dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, _ = load_model_from_checkpoint(args.checkpoint, args.config, device=dev)
        return model, dev

    def shade(image):
        if image is None:
            return None, None, "No input"
        model, dev = _load()
        src = _to_uint8_gray(image)
        t0 = time.perf_counter()
        gray, n_tiles = predict_full_image_gray(
            model, src, dev,
            tile_size=cfg.inference.tile_size,
            halo=cfg.inference.halo,
            stride=cfg.inference.stride,
            context_size=cfg.data.context_size,
            global_long_side=cfg.inference.global_long_side,
        )
        ok, uniq = validate_palette(gray)
        UI_OUTPUT_DIR.mkdir(exist_ok=True)
        bmp = UI_OUTPUT_DIR / f"shaded_{uuid.uuid4().hex[:8]}.bmp"
        save_output_bmp(bmp, gray)
        status = f"OK tiles={n_tiles} palette={ok} uniq={sorted(uniq)} time={time.perf_counter()-t0:.2f}s"
        return _gray_to_rgb_preview(gray), str(bmp), status

    import gradio as gr
    with gr.Blocks(title="CSN-V3") as demo:
        gr.Markdown("# CSN-V3 — 512×512 tiled inference + BMP download")
        inp = gr.Image(type="numpy")
        out = gr.Image(type="numpy")
        bmp = gr.File(file_types=[".bmp"])
        st = gr.Textbox()
        btn = gr.Button("Run")
        btn.click(shade, [inp], [out, bmp, st])
    demo.launch(server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

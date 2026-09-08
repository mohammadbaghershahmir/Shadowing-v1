"""Core-paste tiled inference for CSN-V4."""
from __future__ import annotations

import numpy as np
import torch

from csn_v4.constants import CLASS_TO_GRAY
from csn_v4.geometry.extract import global_cache_key
from csn_v4.geometry.tile_spec import HALO, INPUT_SIZE, iter_core_tiles, paste_core_predictions
from csn_v4.geometry.d4 import D4Transform
from csn_v3.data.multiscale import build_global_token_padding_mask, letterbox_full_bw
from csn_v4.geometry.extract import extract_local_bw, extract_context_bw
from csn_v3.renderer import apply_black_lock
from csn_v3.data.targets import load_gray_bmp


def _class_to_gray_array(pred: np.ndarray) -> np.ndarray:
    out = np.full_like(pred, 255, dtype=np.uint8)
    for cls, gray in CLASS_TO_GRAY.items():
        out[pred == cls] = gray
    return out


def tiled_predict_classes(
    model,
    source_bw: np.ndarray,
    device: torch.device,
    *,
    context_size: int = 1024,
    global_long_side: int = 1024,
    transform_id: int = 0,
    scene_id: str = "infer",
    local_rgb: np.ndarray | None = None,
    local_onehot: np.ndarray | None = None,
    input_mode: str = "bw",
) -> np.ndarray:
    H, W = source_bw.shape[:2]
    if input_mode == "magenta" and local_rgb is None:
        raise RuntimeError("CSN-V4 magenta inference requires local_rgb guide per tile; BW fallback disabled.")

    full_thumb, letterbox_meta = letterbox_full_bw(source_bw, global_long_side)
    gf = torch.from_numpy(full_thumb.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    cache_key = global_cache_key(scene_id, transform_id)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full_rgb = model._prep_rgb(gf)
            global_tokens = model.dino.forward_global_tokens(full_rgb)
            global_pad = build_global_token_padding_mask(
                letterbox_meta, global_tokens.shape[1], model.cfg.model.dino.patch_size, device,
            )

            canvas = np.zeros((H, W), dtype=np.int64)
            specs = list(iter_core_tiles(W, H, transform_id=transform_id, scene_id=scene_id))

            for spec in specs:
                local = extract_local_bw(source_bw, spec)
                context = extract_context_bw(source_bw, spec, context_size=context_size)
                lb = torch.from_numpy(local.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                cb = torch.from_numpy(context.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                cc = torch.from_numpy(spec.input_bbox_norm).unsqueeze(0).to(device)

                fwd_kw = dict(
                    global_tokens=global_tokens,
                    letterbox_meta=letterbox_meta,
                    teacher_forcing=0.0,
                )
                if input_mode == "magenta" and local_rgb is not None:
                    ix, iy = spec.input_x, spec.input_y
                    s = spec.input_size
                    guide_crop = np.full((s, s, 3), 255, dtype=np.uint8)
                    for sy in range(s):
                        for sx in range(s):
                            gy, gx = iy + sy, ix + sx
                            if 0 <= gy < H and 0 <= gx < W:
                                guide_crop[sy, sx] = local_rgb[gy, gx]
                    fwd_kw["local_rgb"] = torch.from_numpy(guide_crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

                out = model(lb, cb, gf, cc, **fwd_kw)
                logits = out["refined_logits"].squeeze(0)
                ys, xs = spec.model_core_slices()
                core_pred = logits.argmax(dim=0)[ys, xs].cpu().numpy()
                paste_core_predictions(canvas, spec, core_pred)
    finally:
        model.train(was_training)
    return canvas


def predict_full_image_gray(
    model,
    source_bw: np.ndarray,
    device: torch.device,
    **kwargs,
) -> tuple[np.ndarray, int]:
    input_mode = kwargs.get("input_mode", getattr(model.cfg.data, "input_mode", "bw"))
    transform_id = kwargs.get("transform_id", 0)
    pred = tiled_predict_classes(
        model, source_bw, device,
        context_size=kwargs.get("context_size", 1024),
        global_long_side=kwargs.get("global_long_side", 1024),
        transform_id=transform_id,
        scene_id=kwargs.get("scene_id", "infer"),
        local_rgb=kwargs.get("local_rgb"),
        input_mode=input_mode,
    )
    gray = _class_to_gray_array(pred)
    gray = apply_black_lock(gray, source_bw)
    n_tiles = len(list(iter_core_tiles(source_bw.shape[1], source_bw.shape[0])))
    return gray, n_tiles


def predict_full_image_bmp(model, source_bw_path: str, device: torch.device, **kwargs) -> np.ndarray:
    src = load_gray_bmp(source_bw_path)
    gray, _ = predict_full_image_gray(model, src, device, **kwargs)
    return gray


# Public alias used by evaluation and training scripts.
tiled_predict = tiled_predict_classes

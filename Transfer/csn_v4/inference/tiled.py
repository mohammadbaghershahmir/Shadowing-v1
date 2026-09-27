"""Core-paste and optional soft-overlap tiled inference for CSN-V4."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from csn_v4.constants import CLASS_TO_GRAY
from csn_v4.geometry.d4 import D4Transform
from csn_v4.geometry.extract import global_cache_key
from csn_v4.geometry.tile_spec import (
    HALO,
    INPUT_SIZE,
    iter_core_tiles,
)
from csn_v4.geometry.extract import extract_local_bw, extract_context_bw
from csn_v3.data.multiscale import build_global_token_padding_mask, letterbox_full_bw
from csn_v3.renderer import apply_black_lock
from csn_v3.data.targets import load_gray_bmp
from csn_v4.indexed.contract import allowed_mask, compose_output
from csn_v4.indexed.schema import INDEX_ALLOWED, INDEX_BACKGROUND, INDEX_OUTLINE


def _class_to_gray_array(pred: np.ndarray) -> np.ndarray:
    out = np.full_like(pred, 255, dtype=np.uint8)
    for cls, gray in CLASS_TO_GRAY.items():
        out[pred == cls] = gray
    return out


def _smooth_window(tile_h: int, tile_w: int, halo: int) -> np.ndarray:
    """Raised-cosine spatial weights: full weight in core, fade in halo."""
    w = np.ones((tile_h, tile_w), dtype=np.float32)
    if halo <= 0 or halo * 2 >= min(tile_h, tile_w):
        return w
    ramp = np.linspace(0.0, 1.0, halo, dtype=np.float32)
    w[:halo, :] *= ramp[:, None]
    w[-halo:, :] *= ramp[::-1][:, None]
    w[:, :halo] *= ramp[None, :]
    w[:, -halo:] *= ramp[::-1][None, :]
    return w


def _extract_index_crop(
    source_index: np.ndarray,
    spec,
    *,
    pad_value: int = INDEX_BACKGROUND,
) -> np.ndarray:
    """Nearest-neighbor categorical crop (no interpolation of indices)."""
    H, W = source_index.shape[:2]
    s = spec.input_size
    out = np.full((s, s), pad_value, dtype=np.int64)
    ix, iy = spec.input_x, spec.input_y
    for sy in range(s):
        for sx in range(s):
            gy, gx = iy + sy, ix + sx
            if 0 <= gy < H and 0 <= gx < W:
                out[sy, sx] = int(source_index[gy, gx])
    return out


def _apply_eligible_constraints(
    pred: np.ndarray,
    source_bw: np.ndarray,
    *,
    eligible_mask: np.ndarray | None = None,
    source_index: np.ndarray | None = None,
) -> np.ndarray:
    """Lock outline/background; constrain predictions outside eligible region.

    Never derives the editable region from ground-truth Y — only from the
    supplied source index map or eligible_mask.
    """
    out = pred.copy()
    # Black outline lock from source BW (always).
    outline = source_bw == 0
    out[outline] = 0  # CLASS_BACKGROUND / unshaded gray after render; class 0

    if source_index is not None:
        m = allowed_mask(source_index)
        # Outside M: force unshaded (class 0); outline already locked above.
        out[~m] = 0
        out[source_index == INDEX_OUTLINE] = 0
    elif eligible_mask is not None:
        elig = eligible_mask.astype(bool)
        out[~elig] = 0
    return out


def _forward_tile_logits(
    model,
    *,
    local_bw: np.ndarray,
    context_bw: np.ndarray,
    full_thumb: np.ndarray,
    letterbox_meta: dict,
    crop_coords: torch.Tensor,
    global_tokens: torch.Tensor,
    device: torch.device,
    input_mode: str,
    local_index: np.ndarray | None = None,
    local_rgb: np.ndarray | None = None,
) -> torch.Tensor:
    lb = torch.from_numpy(local_bw.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    cb = torch.from_numpy(context_bw.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    gf = torch.from_numpy(full_thumb.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    cc = crop_coords.unsqueeze(0).to(device) if crop_coords.ndim == 1 else crop_coords.to(device)

    fwd_kw: dict = dict(
        global_tokens=global_tokens,
        letterbox_meta=letterbox_meta,
        teacher_forcing=0.0,
    )
    if input_mode == "indexed_guided":
        if local_index is None:
            raise RuntimeError(
                "indexed_guided inference requires source_index / local_index; "
                "do not derive it from ground truth."
            )
        fwd_kw["local_index"] = torch.from_numpy(local_index.astype(np.int64)).unsqueeze(0).to(device)
    if input_mode == "magenta" and local_rgb is not None:
        fwd_kw["local_rgb"] = (
            torch.from_numpy(local_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

    out = model(lb, cb, gf, cc, **fwd_kw)
    return out["refined_logits"].squeeze(0)


def tiled_predict_logits(
    model,
    source_bw: np.ndarray,
    device: torch.device,
    *,
    context_size: int = 1024,
    global_long_side: int = 1024,
    transform_id: int = 0,
    scene_id: str = "infer",
    local_rgb: np.ndarray | None = None,
    input_mode: str = "bw",
    source_index: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
    blend_mode: str = "core_paste",
    num_classes: int | None = None,
) -> np.ndarray:
    """Return [C,H,W] float32 logits (or class probabilities after soft blend)."""
    H, W = source_bw.shape[:2]
    if input_mode == "magenta" and local_rgb is None:
        raise RuntimeError("CSN-V4 magenta inference requires local_rgb guide; BW fallback disabled.")
    if input_mode == "indexed_guided" and source_index is None:
        raise RuntimeError(
            "indexed_guided requires an input index map X∈{0,1,2} supplied by the caller "
            "(not inferred from ground truth)."
        )
    if source_index is not None and source_index.shape[:2] != (H, W):
        raise ValueError(f"source_index shape {source_index.shape} != source_bw {(H, W)}")
    if eligible_mask is not None and eligible_mask.shape[:2] != (H, W):
        raise ValueError(f"eligible_mask shape {eligible_mask.shape} != source_bw {(H, W)}")

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
            _ = global_pad  # mask is re-derived inside model forward when needed
            _ = cache_key

            specs = list(iter_core_tiles(W, H, transform_id=transform_id, scene_id=scene_id))
            # Probe class count from first forward if needed
            n_cls = num_classes
            if n_cls is None:
                n_cls = 3 if input_mode == "indexed_guided" else 4

            if blend_mode == "soft_overlap":
                accum = np.zeros((n_cls, H, W), dtype=np.float32)
                weight = np.zeros((H, W), dtype=np.float32)
                base_w = _smooth_window(INPUT_SIZE, INPUT_SIZE, HALO)

                for spec in specs:
                    local = extract_local_bw(source_bw, spec)
                    context = extract_context_bw(source_bw, spec, context_size=context_size)
                    local_idx = (
                        _extract_index_crop(source_index, spec)
                        if source_index is not None
                        else None
                    )
                    guide = None
                    if input_mode == "magenta" and local_rgb is not None:
                        guide = np.full((spec.input_size, spec.input_size, 3), 255, dtype=np.uint8)
                        ix, iy = spec.input_x, spec.input_y
                        for sy in range(spec.input_size):
                            for sx in range(spec.input_size):
                                gy, gx = iy + sy, ix + sx
                                if 0 <= gy < H and 0 <= gx < W:
                                    guide[sy, sx] = local_rgb[gy, gx]

                    logits = _forward_tile_logits(
                        model,
                        local_bw=local,
                        context_bw=context,
                        full_thumb=full_thumb,
                        letterbox_meta=letterbox_meta,
                        crop_coords=torch.from_numpy(spec.input_bbox_norm),
                        global_tokens=global_tokens,
                        device=device,
                        input_mode=input_mode,
                        local_index=local_idx,
                        local_rgb=guide,
                    )
                    # Softmax for stable blending across tiles
                    probs = F.softmax(logits, dim=0).cpu().numpy()
                    ix, iy = spec.input_x, spec.input_y
                    s = spec.input_size
                    # Clip to image bounds
                    y0, x0 = max(0, iy), max(0, ix)
                    y1, x1 = min(H, iy + s), min(W, ix + s)
                    ty0, tx0 = y0 - iy, x0 - ix
                    ty1, tx1 = ty0 + (y1 - y0), tx0 + (x1 - x0)
                    tw = base_w[ty0:ty1, tx0:tx1].copy()
                    # Keep full weight on image borders
                    if iy <= 0:
                        tw[:HALO, :] = 1.0
                    if ix <= 0:
                        tw[:, :HALO] = 1.0
                    if iy + s >= H:
                        tw[-HALO:, :] = 1.0
                    if ix + s >= W:
                        tw[:, -HALO:] = 1.0
                    # Only blend real (non-pad) pixels
                    valid = spec.valid_input_mask[ty0:ty1, tx0:tx1]
                    tw = tw * valid.astype(np.float32)
                    accum[:, y0:y1, x0:x1] += probs[:, ty0:ty1, tx0:tx1] * tw[None, :, :]
                    weight[y0:y1, x0:x1] += tw

                # Uncovered pixels (should not happen with full coverage) → zeros
                dens = weight.clip(min=1e-6)
                return accum / dens[None, :, :]

            # Default: core-paste logits (argmax later)
            canvas = np.zeros((n_cls, H, W), dtype=np.float32)
            for spec in specs:
                local = extract_local_bw(source_bw, spec)
                context = extract_context_bw(source_bw, spec, context_size=context_size)
                local_idx = (
                    _extract_index_crop(source_index, spec)
                    if source_index is not None
                    else None
                )
                guide = None
                if input_mode == "magenta" and local_rgb is not None:
                    guide = np.full((spec.input_size, spec.input_size, 3), 255, dtype=np.uint8)
                    ix, iy = spec.input_x, spec.input_y
                    for sy in range(spec.input_size):
                        for sx in range(spec.input_size):
                            gy, gx = iy + sy, ix + sx
                            if 0 <= gy < H and 0 <= gx < W:
                                guide[sy, sx] = local_rgb[gy, gx]

                logits = _forward_tile_logits(
                    model,
                    local_bw=local,
                    context_bw=context,
                    full_thumb=full_thumb,
                    letterbox_meta=letterbox_meta,
                    crop_coords=torch.from_numpy(spec.input_bbox_norm),
                    global_tokens=global_tokens,
                    device=device,
                    input_mode=input_mode,
                    local_index=local_idx,
                    local_rgb=guide,
                )
                ys, xs = spec.model_core_slices()
                core_logits = logits[:, ys, xs].cpu().numpy()
                cx, cy = spec.core_origin_xy
                cw, ch = spec.core_size_xy
                mask = spec.valid_core_mask
                region = canvas[:, cy : cy + ch, cx : cx + cw]
                region[:, mask] = core_logits[:, mask]
            return canvas
    finally:
        model.train(was_training)


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
    source_index: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
    blend_mode: str = "core_paste",
    d4_tta: bool = False,
    d4_tta_orientations: list[int] | None = None,
) -> np.ndarray:
    """Predict class map [H,W]. Optional soft-overlap blend and D4 TTA."""
    del local_onehot  # reserved; not used at inference
    H, W = source_bw.shape[:2]
    orients = list(d4_tta_orientations) if d4_tta_orientations is not None else list(range(8))

    def _one(src_bw, src_idx, elig, tid: int) -> np.ndarray:
        logits = tiled_predict_logits(
            model,
            src_bw,
            device,
            context_size=context_size,
            global_long_side=global_long_side,
            transform_id=tid,
            scene_id=scene_id,
            local_rgb=local_rgb,
            input_mode=input_mode,
            source_index=src_idx,
            eligible_mask=elig,
            blend_mode=blend_mode,
        )
        return logits

    if d4_tta:
        # Average inverse-mapped probabilities over orientations.
        n_cls = 3 if input_mode == "indexed_guided" else 4
        accum = np.zeros((n_cls, H, W), dtype=np.float32)
        for k in orients:
            tf = D4Transform(k)
            src_t = tf.apply(source_bw)
            idx_t = tf.apply(source_index) if source_index is not None else None
            elig_t = tf.apply(eligible_mask.astype(np.uint8)).astype(bool) if eligible_mask is not None else None
            logits_t = _one(src_t, idx_t, elig_t, k)
            # Softmax then inverse-transform each channel with nearest-neighbor semantics via D4.
            probs_t = torch.from_numpy(logits_t)
            if blend_mode != "soft_overlap":
                probs_t = F.softmax(probs_t, dim=0)
            else:
                # soft_overlap path already returns probabilities
                pass
            probs_np = probs_t.numpy() if isinstance(probs_t, torch.Tensor) else probs_t
            inv = tf.inverse
            for c in range(probs_np.shape[0]):
                accum[c] += inv.apply(probs_np[c])
        accum /= max(len(orients), 1)
        pred = accum.argmax(axis=0).astype(np.int64)
    else:
        logits = _one(source_bw, source_index, eligible_mask, transform_id)
        if blend_mode == "soft_overlap":
            pred = logits.argmax(axis=0).astype(np.int64)
        else:
            pred = logits.argmax(axis=0).astype(np.int64)

    if input_mode == "indexed_guided" and source_index is not None:
        # Compose full 0..4 index map from X + internal preds; then map to BW classes for metrics path.
        composed = compose_output(source_index, pred)
        # Convert indices {0,1,2,3,4} → BW class ids {0,1,2,3} for gray render path:
        # 0,1 → class 0 (bg/outline locked), 2→0 unshaded, 3→1, 4→2 (approx shade mapping)
        # Prefer returning internal-compatible class for BW metrics when using indexed.
        # Callers that need the index map should use predict_full_image_index.
        out = np.zeros_like(pred)
        out[composed == INDEX_ALLOWED] = 0
        out[composed == 3] = 1
        out[composed == 4] = 2
        # class 3 (darkest shade) unused in 3-class indexed; keep 0 outside
        pred = out
    else:
        pred = _apply_eligible_constraints(
            pred, source_bw, eligible_mask=eligible_mask, source_index=source_index,
        )
    return pred


def predict_full_image_index(
    model,
    source_index: np.ndarray,
    device: torch.device,
    **kwargs,
) -> np.ndarray:
    """Indexed-guided inference → full Y∈{0,1,2,3,4} index map. Never uses GT."""
    # Build a BW proxy for tiling geometry (outline=0, else white).
    source_bw = np.where(source_index == INDEX_OUTLINE, 0, 255).astype(np.uint8)
    kwargs = dict(kwargs)
    kwargs["input_mode"] = "indexed_guided"
    kwargs["source_index"] = source_index
    logits = tiled_predict_logits(
        model,
        source_bw,
        device,
        context_size=kwargs.get("context_size", 1024),
        global_long_side=kwargs.get("global_long_side", 1024),
        transform_id=kwargs.get("transform_id", 0),
        scene_id=kwargs.get("scene_id", "infer"),
        input_mode="indexed_guided",
        source_index=source_index,
        blend_mode=kwargs.get("blend_mode", "core_paste"),
        num_classes=3,
    )
    pred_internal = logits.argmax(axis=0).astype(np.int64)
    return compose_output(source_index, pred_internal)


def predict_full_image_gray(
    model,
    source_bw: np.ndarray,
    device: torch.device,
    **kwargs,
) -> tuple[np.ndarray, int]:
    input_mode = kwargs.get("input_mode", getattr(model.cfg.data, "input_mode", "bw"))
    transform_id = kwargs.get("transform_id", 0)
    infer_cfg = getattr(model.cfg, "inference", None)
    blend_mode = kwargs.get(
        "blend_mode",
        getattr(infer_cfg, "blend_mode", "core_paste") if infer_cfg else "core_paste",
    )
    d4_tta = kwargs.get(
        "d4_tta",
        getattr(infer_cfg, "d4_tta", False) if infer_cfg else False,
    )
    d4_orients = kwargs.get(
        "d4_tta_orientations",
        getattr(infer_cfg, "d4_tta_orientations", None) if infer_cfg else None,
    )

    t0 = time.perf_counter()
    pred = tiled_predict_classes(
        model, source_bw, device,
        context_size=kwargs.get("context_size", 1024),
        global_long_side=kwargs.get("global_long_side", 1024),
        transform_id=transform_id,
        scene_id=kwargs.get("scene_id", "infer"),
        local_rgb=kwargs.get("local_rgb"),
        input_mode=input_mode,
        source_index=kwargs.get("source_index"),
        eligible_mask=kwargs.get("eligible_mask"),
        blend_mode=blend_mode,
        d4_tta=d4_tta,
        d4_tta_orientations=d4_orients,
    )
    _ = time.perf_counter() - t0
    gray = _class_to_gray_array(pred)
    gray = apply_black_lock(gray, source_bw)
    n_tiles = len(list(iter_core_tiles(source_bw.shape[1], source_bw.shape[0])))
    return gray, n_tiles


def predict_full_image_with_scores(
    model,
    source_bw: np.ndarray,
    device: torch.device,
    **kwargs,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Backward-compatible extension: gray prediction + per-class scores [4,H,W].

    Existing ``predict_full_image_gray`` is unchanged. Scores are probabilities
    (softmax of blended logits when ``blend_mode=soft_overlap``, else softmax of
    core-pasted logits). D4-TTA averages inverse-mapped probabilities.
    """
    input_mode = kwargs.get("input_mode", getattr(model.cfg.data, "input_mode", "bw"))
    if input_mode == "indexed_guided":
        # Indexed head is 3-class; pad to 4 channels (white unused / folded).
        gray, n_tiles = predict_full_image_gray(model, source_bw, device, **kwargs)
        from csn_v4.constants import NUM_CLASSES, SEMANTIC_GRAY_TO_CLASS

        h, w = gray.shape[:2]
        scores = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        for gv, cls in SEMANTIC_GRAY_TO_CLASS.items():
            scores[cls] = (gray == gv).astype(np.float32)
        return gray, n_tiles, scores

    transform_id = kwargs.get("transform_id", 0)
    infer_cfg = getattr(model.cfg, "inference", None)
    blend_mode = kwargs.get(
        "blend_mode",
        getattr(infer_cfg, "blend_mode", "core_paste") if infer_cfg else "core_paste",
    )
    d4_tta = kwargs.get(
        "d4_tta",
        getattr(infer_cfg, "d4_tta", False) if infer_cfg else False,
    )
    d4_orients = kwargs.get(
        "d4_tta_orientations",
        getattr(infer_cfg, "d4_tta_orientations", None) if infer_cfg else None,
    )
    context_size = kwargs.get("context_size", 1024)
    global_long_side = kwargs.get("global_long_side", 1024)
    scene_id = kwargs.get("scene_id", "infer")

    H, W = source_bw.shape[:2]
    orients = list(d4_orients) if d4_orients is not None else list(range(8))

    def _probs_one(src_bw, src_idx, elig, tid: int) -> np.ndarray:
        logits = tiled_predict_logits(
            model,
            src_bw,
            device,
            context_size=context_size,
            global_long_side=global_long_side,
            transform_id=tid,
            scene_id=scene_id,
            local_rgb=kwargs.get("local_rgb"),
            input_mode=input_mode,
            source_index=src_idx,
            eligible_mask=elig,
            blend_mode=blend_mode,
            num_classes=4,
        )
        if blend_mode == "soft_overlap":
            return logits.astype(np.float32)
        t = torch.from_numpy(logits)
        return F.softmax(t, dim=0).numpy().astype(np.float32)

    if d4_tta:
        accum = np.zeros((4, H, W), dtype=np.float32)
        for k in orients:
            tf = D4Transform(k)
            src_t = tf.apply(source_bw)
            idx_t = tf.apply(kwargs["source_index"]) if kwargs.get("source_index") is not None else None
            elig_t = (
                tf.apply(kwargs["eligible_mask"].astype(np.uint8)).astype(bool)
                if kwargs.get("eligible_mask") is not None
                else None
            )
            probs_t = _probs_one(src_t, idx_t, elig_t, k)
            inv = tf.inverse
            for c in range(probs_t.shape[0]):
                accum[c] += inv.apply(probs_t[c])
        scores = accum / max(len(orients), 1)
    else:
        scores = _probs_one(
            source_bw,
            kwargs.get("source_index"),
            kwargs.get("eligible_mask"),
            transform_id,
        )

    pred = scores.argmax(axis=0).astype(np.int64)
    # Re-apply eligible / black constraints consistently with tiled_predict_classes
    pred = _apply_eligible_constraints(
        pred,
        source_bw,
        eligible_mask=kwargs.get("eligible_mask"),
        source_index=kwargs.get("source_index"),
    )
    gray = _class_to_gray_array(pred)
    gray = apply_black_lock(gray, source_bw)
    n_tiles = len(list(iter_core_tiles(W, H)))
    return gray, n_tiles, scores


def predict_full_image_bmp(model, source_bw_path: str, device: torch.device, **kwargs) -> np.ndarray:
    src = load_gray_bmp(source_bw_path)
    gray, _ = predict_full_image_gray(model, src, device, **kwargs)
    return gray


# Public alias used by evaluation and training scripts.
tiled_predict = tiled_predict_classes

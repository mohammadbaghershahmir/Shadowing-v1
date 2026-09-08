"""Unified dense feature adapters -> [B, H, W, C]."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass
class FeatureBundle:
    features: torch.Tensor  # [B, H, W, C] float32 on CPU typically
    kind: str  # final | fused
    stride: int
    grid_hw: tuple[int, int]
    channels: int
    layer_or_stage_ids: list[int] = field(default_factory=list)
    dtype_str: str = "float32"
    extras: dict[str, Any] = field(default_factory=dict)


def _assert_finite(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} contains NaN/Inf")


def _l2_normalize_hwc(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [B,H,W,C]
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def _nchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def _autocast_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


@torch.inference_mode()
def extract_vit_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    fusion_layers: Sequence[int] = (5, 11, 17, 23),
    device: str = "cuda",
    autocast_dtype: str = "float32",
) -> tuple[FeatureBundle, FeatureBundle]:
    """
    images: [B,3,H,W] already normalized.
    Notebook pattern (pca.ipynb / dense_sparse_matching.ipynb):
      model.cuda(); image.cuda();
      get_intermediate_layers(..., reshape=True, norm=True) under torch.autocast('cuda').
    Single forward extracts fusion layers; final = last fusion layer (23 for ViT-L).
    Returns (final, fused) FeatureBundles on CPU float32.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("GPU required: set device=cuda (matching official DINOv3 notebooks).")
    model.eval()
    x = images.to(device, non_blocking=True)
    dtype = _autocast_dtype(autocast_dtype)
    patch = getattr(model, "patch_size", 16)
    if isinstance(patch, tuple):
        patch = patch[0]

    layer_ids = list(fusion_layers)
    # One GPU forward (notebook style), not two full passes
    with torch.autocast(device_type="cuda", dtype=dtype):
        mid = model.get_intermediate_layers(
            x, n=layer_ids, reshape=True, norm=True
        )
    final_t = mid[-1]
    final_bhwc = _nchw_to_bhwc(final_t.float())
    _assert_finite(final_bhwc, "vit_final")
    layers_bhwc = [_nchw_to_bhwc(t.float()) for t in mid]
    for i, t in enumerate(layers_bhwc):
        _assert_finite(t, f"vit_layer_{layer_ids[i]}")
        layers_bhwc[i] = _l2_normalize_hwc(t)
    fused = torch.cat(layers_bhwc, dim=-1)
    _assert_finite(fused, "vit_fused")

    b, h, w, c = final_bhwc.shape
    final_bundle = FeatureBundle(
        features=final_bhwc.cpu(),
        kind="final",
        stride=int(patch),
        grid_hw=(h, w),
        channels=c,
        layer_or_stage_ids=[layer_ids[-1]],
        dtype_str=str(final_bhwc.dtype),
        extras={
            "n_storage_tokens": int(getattr(model, "n_storage_tokens", 0)),
            "notebook_api": "get_intermediate_layers(reshape=True, norm=True)",
        },
    )
    bf, hf, wf, cf = fused.shape
    fused_bundle = FeatureBundle(
        features=fused.cpu(),
        kind="fused",
        stride=int(patch),
        grid_hw=(hf, wf),
        channels=cf,
        layer_or_stage_ids=list(layer_ids),
        dtype_str=str(fused.dtype),
        extras={"fusion": "l2norm_per_layer_then_concat", "layers": list(layer_ids)},
    )
    return final_bundle, fused_bundle


@torch.inference_mode()
def extract_convnext_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    final_stage: int = 3,
    fusion_stages: Sequence[int] = (1, 2, 3),
    device: str = "cuda",
    autocast_dtype: str = "float32",
) -> tuple[FeatureBundle, FeatureBundle]:
    """
    ConvNeXt stages are not ViT layers. Keep native spatial resolutions;
    fuse by upsampling to the finest selected stage.
    Uses one CUDA forward via get_intermediate_layers (same API as notebooks).
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("GPU required: set device=cuda (matching official DINOv3 notebooks).")
    model.eval()
    x = images.to(device, non_blocking=True)
    dtype = _autocast_dtype(autocast_dtype)
    # Ensure no ViT-grid forced resize
    if getattr(model, "patch_size", None) is not None:
        model.patch_size = None

    stage_ids = sorted(set(list(fusion_stages) + [final_stage]))
    with torch.autocast(device_type="cuda", dtype=dtype):
        outs = model.get_intermediate_layers(
            x, n=stage_ids, reshape=True, norm=True
        )
    # outs: tuple of NCHW tensors (patch maps)
    id_to_map = {sid: _nchw_to_bhwc(t.float()) for sid, t in zip(stage_ids, outs)}
    for sid, t in id_to_map.items():
        _assert_finite(t, f"convnext_stage_{sid}")

    final_map = id_to_map[final_stage]
    b, h, w, c = final_map.shape
    # Effective stride of final stage ~ 32 for standard ConvNeXt
    in_h, in_w = images.shape[-2], images.shape[-1]
    stride_final = max(1, in_h // h)

    final_bundle = FeatureBundle(
        features=final_map.cpu(),
        kind="final",
        stride=int(stride_final),
        grid_hw=(h, w),
        channels=c,
        layer_or_stage_ids=[final_stage],
        dtype_str=str(final_map.dtype),
        extras={"architecture": "convnext"},
    )

    # Fuse: upsample to finest among fusion_stages
    fusion_maps = [id_to_map[s] for s in fusion_stages]
    target_h = max(m.shape[1] for m in fusion_maps)
    target_w = max(m.shape[2] for m in fusion_maps)
    normed = []
    for m in fusion_maps:
        if m.shape[1] != target_h or m.shape[2] != target_w:
            nchw = m.permute(0, 3, 1, 2)
            nchw = F.interpolate(nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
            m = nchw.permute(0, 2, 3, 1).contiguous()
        normed.append(_l2_normalize_hwc(m))
    fused = torch.cat(normed, dim=-1)
    _assert_finite(fused, "convnext_fused")
    stride_fused = max(1, in_h // target_h)
    fused_bundle = FeatureBundle(
        features=fused.cpu(),
        kind="fused",
        stride=int(stride_fused),
        grid_hw=(target_h, target_w),
        channels=fused.shape[-1],
        layer_or_stage_ids=list(fusion_stages),
        dtype_str=str(fused.dtype),
        extras={
            "fusion": "upsample_to_finest_l2norm_concat",
            "stages": list(fusion_stages),
            "note": "ConvNeXt stages are not identical to ViT layers",
        },
    )
    return final_bundle, fused_bundle


@torch.inference_mode()
def extract_features(
    loaded_model,
    images: torch.Tensor,
    *,
    device: str = "cuda",
    autocast_dtype: str = "float32",
) -> tuple[FeatureBundle, FeatureBundle]:
    cfg = loaded_model.config
    arch = cfg["architecture"]
    if arch == "vit":
        return extract_vit_features(
            loaded_model.model,
            images,
            fusion_layers=cfg.get("fusion_layers", [5, 11, 17, 23]),
            device=device,
            autocast_dtype=autocast_dtype,
        )
    if arch == "convnext":
        return extract_convnext_features(
            loaded_model.model,
            images,
            final_stage=cfg.get("final_stage", 3),
            fusion_stages=cfg.get("fusion_stages", [1, 2, 3]),
            device=device,
            autocast_dtype=autocast_dtype,
        )
    raise ValueError(f"Unsupported architecture: {arch}")


@torch.inference_mode()
def extract_features_transformers_fallback(
    loaded_model,
    images: torch.Tensor,
    *,
    device: str = "cuda",
) -> tuple[FeatureBundle, FeatureBundle]:
    """
    Best-effort HF fallback for final-layer dense tokens.
    Multi-layer fusion may be limited depending on transformers version.
    """
    model = loaded_model.model
    cfg = loaded_model.config
    x = images.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        outputs = model(pixel_values=x, output_hidden_states=True)
    # Prefer last_hidden_state excluding special tokens when possible
    hidden = outputs.last_hidden_state  # [B, N, C]
    b, n, c = hidden.shape
    h_in, w_in = images.shape[-2:]
    patch = cfg.get("stride_final", 16)
    hp, wp = h_in // patch, w_in // patch
    # Heuristic: drop CLS (+ registers if present)
    n_special = n - hp * wp
    if n_special < 0:
        raise RuntimeError(
            f"HF fallback token count mismatch: N={n}, expected grid {hp}x{wp}. "
            "Use Torch Hub loader instead."
        )
    patches = hidden[:, n_special:, :].float()
    feats = patches.reshape(b, hp, wp, c)
    _assert_finite(feats, "hf_final")
    final = FeatureBundle(
        features=feats.cpu(),
        kind="final",
        stride=patch,
        grid_hw=(hp, wp),
        channels=c,
        layer_or_stage_ids=[-1],
        extras={"source": "transformers_fallback", "n_special": n_special},
    )
    # Fused: use last 4 hidden states if available
    if getattr(outputs, "hidden_states", None) is not None and len(outputs.hidden_states) >= 24:
        layer_ids = cfg.get("fusion_layers", [5, 11, 17, 23])
        maps = []
        for lid in layer_ids:
            hs = outputs.hidden_states[lid + 1]  # +1 if 0 is embeddings
            # Try both offsets
            for offset in (1, 0):
                idx = lid + offset
                if 0 <= idx < len(outputs.hidden_states):
                    hs = outputs.hidden_states[idx]
                    break
            p = hs[:, n_special:, :].float().reshape(b, hp, wp, -1)
            maps.append(_l2_normalize_hwc(p))
        fused_t = torch.cat(maps, dim=-1)
        fused = FeatureBundle(
            features=fused_t.cpu(),
            kind="fused",
            stride=patch,
            grid_hw=(hp, wp),
            channels=fused_t.shape[-1],
            layer_or_stage_ids=list(layer_ids),
            extras={"source": "transformers_fallback"},
        )
    else:
        fused = FeatureBundle(
            features=feats.cpu(),
            kind="fused",
            stride=patch,
            grid_hw=(hp, wp),
            channels=c,
            layer_or_stage_ids=[-1],
            extras={"source": "transformers_fallback", "note": "fused_equals_final"},
        )
    return final, fused

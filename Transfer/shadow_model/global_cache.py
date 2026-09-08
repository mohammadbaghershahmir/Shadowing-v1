"""Precompute and load frozen DINO global tokens with metadata and stale detection."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shadow_dataset.image_io import load_rgb_exact
from shadow_dataset.io_utils import read_jsonl
from shadow_dataset.letterbox import letterbox_semantic_guide
from shadow_model.global_dino import (
    FrozenDinoEncoder,
    compute_patch_valid_mask,
    normalize_for_dino,
)

LOGGER = logging.getLogger(__name__)

DIHEDRAL_TRANSFORMS = {
    0: lambda x: x,
    1: lambda x: np.flip(x, axis=1).copy(),
    2: lambda x: np.flip(x, axis=0).copy(),
    3: lambda x: np.flip(np.flip(x, 0), 1).copy(),
    4: lambda x: np.rot90(x, 1).copy(),
    5: lambda x: np.rot90(x, 2).copy(),
    6: lambda x: np.rot90(x, 3).copy(),
    7: lambda x: np.flip(np.rot90(x, 1), axis=1).copy(),
}


def source_hash(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]


def checkpoint_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_paths(cache_dir: Path, sid: str, variant: int) -> tuple[Path, Path]:
    sid_hash = source_hash(sid)
    base = cache_dir / f"{sid_hash}_k{variant}"
    return base.with_suffix(".tokens.npy"), base.with_suffix(".meta.json")


def build_metadata(
    rec: dict,
    variant: int,
    lb,
    dino_cfg: Any,
    ckpt_hash: str,
) -> dict:
    return {
        "source_id": rec["source_id"],
        "stem": rec.get("stem", ""),
        "transform_id": variant,
        "model_id": dino_cfg.model_name,
        "hub_entry": dino_cfg.hub_entry,
        "checkpoint_sha256": ckpt_hash,
        "feature_layer": dino_cfg.feature_layer,
        "input_size": dino_cfg.input_size,
        "patch_size": dino_cfg.patch_size,
        "token_dim": dino_cfg.token_dim,
        "image_width": int(rec.get("width", lb.rgb.shape[1])),
        "image_height": int(rec.get("height", lb.rgb.shape[0])),
        "scale": lb.scale,
        "offset_x": lb.offset_x,
        "offset_y": lb.offset_y,
        "content_width": lb.content_width,
        "content_height": lb.content_height,
    }


def validate_cache_entry(meta_path: Path, dino_cfg: Any, ckpt_hash: str) -> None:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("checkpoint_sha256") != ckpt_hash:
        raise RuntimeError(f"Stale cache (checkpoint changed): {meta_path}")
    if meta.get("model_id") != dino_cfg.model_name:
        raise RuntimeError(f"Stale cache (model changed): {meta_path}")
    if meta.get("feature_layer") != dino_cfg.feature_layer:
        raise RuntimeError(f"Stale cache (layer changed): {meta_path}")


def cache_global_tokens(
    metadata_dir: str | Path,
    cache_dir: str | Path,
    dino_cfg: Any,
    dihedral_variants: list[int] | None = None,
    device: str = "cuda",
) -> None:
    metadata_dir = Path(metadata_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_hash = checkpoint_sha256(dino_cfg.checkpoint)

    all_records: list[dict] = []
    for name in ["train_images.jsonl", "val_images.jsonl", "test_images.jsonl"]:
        p = metadata_dir / name
        if p.exists():
            all_records.extend(read_jsonl(p))

    seen: set[str] = set()
    unique: list[dict] = []
    for rec in all_records:
        if rec["source_id"] not in seen:
            seen.add(rec["source_id"])
            unique.append(rec)

    encoder = FrozenDinoEncoder.get_shared(dino_cfg).to(device)
    variants = dihedral_variants or list(range(8))

    for idx, rec in enumerate(unique):
        sid = rec["source_id"]
        input_rgb, _ = load_rgb_exact(rec["input_path"])
        LOGGER.info("[%d/%d] cache %s", idx + 1, len(unique), sid)
        for k in variants:
            tok_path, meta_path = cache_paths(cache_dir, sid, k)
            if tok_path.exists() and meta_path.exists():
                try:
                    validate_cache_entry(meta_path, dino_cfg, ckpt_hash)
                    continue
                except RuntimeError:
                    LOGGER.warning("Refreshing stale cache %s", tok_path)

            transformed = DIHEDRAL_TRANSFORMS[k](input_rgb)
            lb = letterbox_semantic_guide(transformed, size=dino_cfg.input_size)
            x = normalize_for_dino(lb.rgb).to(device)
            tokens = encoder.forward_tokens(x).cpu().half().numpy().squeeze(0)
            if tokens.shape != (1024, dino_cfg.token_dim):
                raise ValueError(f"Unexpected token shape {tokens.shape} for {sid}")
            patch_valid = compute_patch_valid_mask(lb.valid_mask, dino_cfg.patch_size)
            meta = build_metadata(rec, k, lb, dino_cfg, ckpt_hash)
            valid_path = tok_path.parent / tok_path.name.replace(".tokens.npy", ".valid.npy")
            np.save(tok_path, tokens)
            np.save(valid_path, patch_valid)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def encode_global_view(
    input_rgb: np.ndarray,
    encoder: FrozenDinoEncoder,
    *,
    input_size: int = 512,
    patch_size: int = 16,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Letterbox + encode a full semantic guide for global fusion at inference.

    Returns:
        tokens: [1, N, D] float32 on device
        patch_valid: [1, N] bool on device
        letterbox_meta: dict with scale/offsets and original image size
    """
    lb = letterbox_semantic_guide(input_rgb, size=input_size)
    x = normalize_for_dino(lb.rgb).to(device)
    with torch.no_grad():
        tokens = encoder.forward_tokens(x).float()
    patch_valid = compute_patch_valid_mask(lb.valid_mask, patch_size)
    letterbox_meta = {
        "scale": float(lb.scale),
        "offset_x": float(lb.offset_x),
        "offset_y": float(lb.offset_y),
        "content_width": int(lb.content_width),
        "content_height": int(lb.content_height),
        "image_width": int(input_rgb.shape[1]),
        "image_height": int(input_rgb.shape[0]),
    }
    valid_t = torch.from_numpy(patch_valid).unsqueeze(0).to(device=device)
    return tokens, valid_t, letterbox_meta


def load_cached_tokens(
    cache_dir: str | Path,
    source_id: str,
    variant: int = 0,
    dino_cfg: Any | None = None,
    require: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cache_dir = Path(cache_dir)
    tok_path, meta_path = cache_paths(cache_dir, source_id, variant)
    if not tok_path.exists():
        if require:
            raise FileNotFoundError(f"Missing global token cache: {tok_path}")
        raise FileNotFoundError(tok_path)
    if dino_cfg is not None:
        validate_cache_entry(meta_path, dino_cfg, checkpoint_sha256(dino_cfg.checkpoint))
    tokens = np.load(tok_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    valid_path = tok_path.parent / tok_path.name.replace(".tokens.npy", ".valid.npy")
    if not valid_path.exists():
        raise FileNotFoundError(f"Missing patch_valid cache: {valid_path}")
    patch_valid = np.load(valid_path)
    if tokens.shape[0] != patch_valid.shape[0]:
        raise ValueError(f"Token/mask mismatch {tokens.shape} vs {patch_valid.shape}")
    if tokens.shape != (1024, meta.get("token_dim", 1024)):
        raise ValueError(f"Unexpected token shape {tokens.shape}")
    return tokens.astype(np.float32), patch_valid.astype(bool), meta

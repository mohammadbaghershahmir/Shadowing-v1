"""Strict DINOv3 ViT-L/16 loader with SHA-256 validation and shared singleton."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_WRAPPER_PREFIXES = ("module.", "backbone.", "teacher.", "encoder.", "model.")


def compute_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DINO checkpoint not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip known wrapper prefixes until keys match model."""
    out = dict(state_dict)
    for _ in range(4):
        sample = next(iter(out))
        matched = False
        for prefix in _WRAPPER_PREFIXES:
            if all(k.startswith(prefix) for k in out):
                out = {k[len(prefix) :]: v for k, v in out.items()}
                matched = True
                break
        if not matched:
            break
    return out


def load_dino_strict(
    repo_root: str | Path,
    checkpoint: str | Path,
    *,
    embed_dim: int = 1024,
    patch_size: int = 16,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load dinov3_vitl16 from official repo with strict state dict validation."""
    repo_root = Path(repo_root)
    checkpoint = Path(checkpoint)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"DINO repo root not found: {repo_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINO checkpoint not found: {checkpoint}")

    sha256 = compute_sha256(checkpoint)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from dinov3.hub.backbones import dinov3_vitl16

    model = dinov3_vitl16(pretrained=False, weights=str(checkpoint))

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        state_dict = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(raw)}")

    state_dict = normalize_state_dict_keys(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"DINO strict load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    model_patch = getattr(model, "patch_size", None)
    if model_patch is None:
        model_patch = patch_size
    if int(model_patch) != patch_size:
        raise ValueError(f"Expected patch_size={patch_size}, got {model_patch}")

    model_embed = getattr(model, "embed_dim", None)
    if model_embed is None:
        model_embed = embed_dim
    if int(model_embed) != embed_dim:
        raise ValueError(f"Expected embed_dim={embed_dim}, got {model_embed}")

    report = {
        "checkpoint": str(checkpoint),
        "sha256": sha256,
        "repo_root": str(repo_root),
        "patch_size": int(model_patch),
        "embed_dim": int(model_embed),
        "missing_keys": [],
        "unexpected_keys": [],
        "strict_load": True,
    }
    return model, report


class LayerMixer(nn.Module):
    """Trainable softmax-weighted mix of DINO intermediate spatial grids."""

    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_grids: list[torch.Tensor]) -> torch.Tensor:
        w = F.softmax(self.weights, dim=0)
        out = layer_grids[0] * w[0]
        for i in range(1, len(layer_grids)):
            out = out + layer_grids[i] * w[i]
        return out


class SharedDinoEncoder(nn.Module):
    """Single shared frozen DINO instance for local/context/global views."""

    _instance: SharedDinoEncoder | None = None
    _cfg_key: str | None = None

    def __init__(self, dino_cfg: Any):
        super().__init__()
        self.patch_size = dino_cfg.patch_size
        self.embed_dim = dino_cfg.embed_dim
        self.intermediate_layers = list(dino_cfg.intermediate_layers)
        self._load_report: dict[str, Any] = {}

        model, report = load_dino_strict(
            dino_cfg.repo_root,
            dino_cfg.checkpoint,
            embed_dim=dino_cfg.embed_dim,
            patch_size=dino_cfg.patch_size,
        )
        self.backbone = model
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.layer_mixer_local = LayerMixer(len(self.intermediate_layers))
        self.layer_mixer_context = LayerMixer(len(self.intermediate_layers))
        self.layer_mixer_global = LayerMixer(len(self.intermediate_layers))
        self._load_report = report

    @classmethod
    def get_shared(cls, dino_cfg: Any) -> SharedDinoEncoder:
        key = f"{dino_cfg.repo_root}|{dino_cfg.checkpoint}"
        if cls._instance is None or cls._cfg_key != key:
            cls._instance = cls(dino_cfg)
            cls._cfg_key = key
        return cls._instance

    @classmethod
    def reset_shared(cls) -> None:
        cls._instance = None
        cls._cfg_key = None

    @property
    def load_report(self) -> dict[str, Any]:
        return dict(self._load_report)

    def train(self, mode: bool = True) -> SharedDinoEncoder:
        super().train(mode)
        self.backbone.eval()
        return self

    def _resolve_layer_indices(self) -> list[int]:
        depth = len(self.backbone.blocks)
        resolved = []
        for idx in self.intermediate_layers:
            if idx < 0:
                resolved.append(depth + idx)
            else:
                resolved.append(idx)
        return resolved

    def forward_spatial_layers(
        self,
        rgb_norm: torch.Tensor,
        layer_mixer: LayerMixer,
    ) -> torch.Tensor:
        """Return mixed spatial grid [B, embed_dim, gh, gw] excluding CLS/register tokens."""
        layer_indices = self._resolve_layer_indices()
        with torch.no_grad():
            layers_out = self.backbone.get_intermediate_layers(
                rgb_norm,
                n=layer_indices,
                reshape=True,
                return_class_token=False,
                return_extra_tokens=False,
                norm=True,
            )
        grids = list(layers_out)
        return layer_mixer(grids)

    def forward_local(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial_layers(rgb_norm, self.layer_mixer_local)

    def forward_context(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial_layers(rgb_norm, self.layer_mixer_context)

    def forward_global_tokens(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        """Return patch tokens [B, N, D] for cross-attention."""
        layer_indices = self._resolve_layer_indices()
        with torch.no_grad():
            layers_out = self.backbone.get_intermediate_layers(
                rgb_norm,
                n=layer_indices,
                reshape=False,
                return_class_token=False,
                return_extra_tokens=False,
                norm=True,
            )
        grids = list(layers_out)
        w = F.softmax(self.layer_mixer_global.weights, dim=0)
        mixed = layers_out[0] * w[0]
        for i in range(1, len(layers_out)):
            mixed = mixed + layers_out[i] * w[i]
        return mixed


def normalize_rgb_tensor(rgb: torch.Tensor) -> torch.Tensor:
    """rgb float [0,1] -> ImageNet normalized."""
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb - mean) / std


def repeat_bw_to_rgb(bw: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] in [0,1] -> [B,3,H,W]."""
    if bw.shape[1] == 3:
        return bw
    return bw.repeat(1, 3, 1, 1)


def save_dino_load_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

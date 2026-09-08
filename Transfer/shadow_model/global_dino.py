"""Shared frozen vitl16_lvd encoder for local and global views."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenDinoEncoder(nn.Module):
    """Single shared frozen vitl16_lvd instance."""

    _instance: "FrozenDinoEncoder | None" = None

    def __init__(
        self,
        repo_dir: str,
        checkpoint: str,
        hub_entry: str = "dinov3_vitl16",
        feature_layer: int = 23,
        patch_size: int = 16,
        token_dim: int = 1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.token_dim = token_dim
        self.feature_layer = feature_layer
        repo = Path(repo_dir)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        self.model = torch.hub.load(
            str(repo),
            hub_entry,
            source="local",
            weights=str(checkpoint),
            trust_repo=True,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def get_shared(cls, dino_cfg: Any) -> "FrozenDinoEncoder":
        if cls._instance is None:
            cls._instance = cls(
                repo_dir=dino_cfg.repo_dir,
                checkpoint=dino_cfg.checkpoint,
                hub_entry=dino_cfg.hub_entry,
                feature_layer=dino_cfg.feature_layer,
                patch_size=dino_cfg.patch_size,
                token_dim=dino_cfg.token_dim,
            )
        return cls._instance

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B,3,H,W] ImageNet-normalized -> [B, N, token_dim]."""
        with torch.no_grad():
            layers = self.model.get_intermediate_layers(
                images,
                n=[self.feature_layer],
                reshape=False,
                return_class_token=True,
                norm=True,
            )
        return layers[0][0]

    def tokens_to_grid(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[B,N,D] -> [B,D,h,w] assuming square grid."""
        B, N, D = tokens.shape
        gh = h // self.patch_size
        gw = w // self.patch_size
        assert gh * gw == N, f"Token count {N} != grid {gh}x{gw}"
        return tokens.transpose(1, 2).reshape(B, D, gh, gw)


def normalize_rgb_tensor(rgb: torch.Tensor) -> torch.Tensor:
    """rgb float [0,1] or uint8-like -> ImageNet normalized."""
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    return (rgb - mean) / std


def normalize_for_dino(rgb_uint8: np.ndarray) -> torch.Tensor:
    x = rgb_uint8.astype(np.float32) / 255.0
    for c in range(3):
        x[:, :, c] = (x[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def compute_patch_valid_mask(valid_mask: np.ndarray, patch_size: int = 16) -> np.ndarray:
    H, W = valid_mask.shape
    gh, gw = H // patch_size, W // patch_size
    mask_f = valid_mask[: gh * patch_size, : gw * patch_size].astype(np.float32)
    patches = mask_f.reshape(gh, patch_size, gw, patch_size)
    patch_means = patches.mean(axis=(1, 3))
    flat = (patch_means > 0.5).reshape(-1)
    assert flat.shape[0] == gh * gw
    return flat

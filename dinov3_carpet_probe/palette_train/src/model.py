"""Frozen DINOv3 backbone plus trainable palette head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dinov3_carpet_probe.palette_train.src.backbone_features import FrozenDinoFeatureExtractor
from dinov3_carpet_probe.palette_train.src.color import srgb_to_oklab
from dinov3_carpet_probe.palette_train.src.head import PaletteSetHead


class PalettePredictor(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FrozenDinoFeatureExtractor(
            model_key=str(cfg["model_key"]),
            device=str(cfg["device"]),
            repo_dir=cfg.get("repo_dir"),
            weights_dir=cfg.get("weights_dir"),
            checkpoint_path=cfg.get("checkpoint_path"),
            fusion_layers=cfg.get("fusion_layers", (5, 11, 17, 23)),
        )
        dummy = torch.zeros(1, 3, int(cfg["input_long_side"]), int(cfg["input_long_side"]))
        with torch.no_grad():
            feat = self.feature_extractor(dummy)
        self.patch_size = self.feature_extractor.patch_size
        self.raw_color_dim = 15
        self.head = PaletteSetHead(
            fused_dim=feat.fused_dim,
            raw_color_dim=self.raw_color_dim,
            cls_dim=feat.cls_token.shape[-1],
            max_num_colors=int(cfg["max_num_colors"]),
            spare_queries=int(cfg.get("spare_queries", 4)),
            head_dim=int(cfg.get("head_dim", 256)),
            decoder_layers=int(cfg.get("decoder_layers", 4)),
            decoder_heads=int(cfg.get("decoder_heads", 8)),
            decoder_ffn_dim=int(cfg.get("decoder_ffn_dim", 1024)),
            dropout=float(cfg.get("dropout", 0.1)),
            aux_loss=bool(cfg.get("aux_loss", True)),
        )

    def _raw_patch_features(self, raw_images: torch.Tensor, valid_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patch_size
        unfold = torch.nn.functional.unfold(raw_images, kernel_size=patch, stride=patch)
        mask_unfold = torch.nn.functional.unfold(valid_masks, kernel_size=patch, stride=patch)
        bsz, _, num_patches = unfold.shape
        pixels_per_patch = patch * patch
        rgb = unfold.view(bsz, 3, pixels_per_patch, num_patches).permute(0, 3, 2, 1)
        mask = mask_unfold.view(bsz, 1, pixels_per_patch, num_patches).permute(0, 3, 2, 1).squeeze(-1)
        patch_valid = mask.mean(dim=2) > 0.5
        weights = mask.unsqueeze(-1)
        denom = weights.sum(dim=2).clamp_min(1.0)
        mean_rgb = (rgb * weights).sum(dim=2) / denom
        centered = (rgb - mean_rgb.unsqueeze(2)) * weights
        std_rgb = torch.sqrt((centered.pow(2).sum(dim=2) / denom).clamp_min(1e-8))
        rgb_for_min = rgb.masked_fill(~mask.unsqueeze(-1).bool(), 1.0)
        rgb_for_max = rgb.masked_fill(~mask.unsqueeze(-1).bool(), 0.0)
        min_rgb = rgb_for_min.min(dim=2).values
        max_rgb = rgb_for_max.max(dim=2).values
        mean_oklab = srgb_to_oklab(mean_rgb)
        raw_features = torch.cat([mean_rgb, std_rgb, min_rgb, max_rgb, mean_oklab], dim=-1)
        return raw_features, patch_valid

    @property
    def checkpoint_path(self) -> str | None:
        return self.feature_extractor.checkpoint_path

    def forward(self, images: torch.Tensor, raw_images: torch.Tensor, valid_masks: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            feat = self.feature_extractor(images)
        raw_features, patch_valid_mask = self._raw_patch_features(raw_images.to(images.device), valid_masks.to(images.device))
        return self.head(
            feat.patch_tokens,
            raw_features,
            feat.cls_token,
            grid_hw=feat.grid_hw,
            patch_valid_mask=patch_valid_mask,
        )

    def trainable_parameters(self):
        return self.head.parameters()

"""Frozen DINOv3 ViT-L/16 feature extraction for palette training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from dinov3_carpet_probe.src.io_utils import REPO_ROOT, load_models_config
from dinov3_carpet_probe.src.model_loader import LoadedModel, freeze_model, load_model


@dataclass
class BackboneFeatureBatch:
    patch_tokens: torch.Tensor
    cls_token: torch.Tensor
    grid_hw: tuple[int, int]
    patch_dim: int
    fused_dim: int
    layer_ids: list[int]
    checkpoint_path: str | None


class FrozenDinoFeatureExtractor(nn.Module):
    """Public-API-only frozen DINOv3 feature extractor."""

    def __init__(
        self,
        *,
        model_key: str,
        device: str,
        repo_dir: Path | str | None,
        weights_dir: Path | str | None,
        checkpoint_path: str | None = None,
        fusion_layers: Sequence[int] = (5, 11, 17, 23),
    ) -> None:
        super().__init__()
        if model_key != "vitl16_lvd":
            raise ValueError(f"This training pipeline is locked to vitl16_lvd, got {model_key}.")
        if checkpoint_path is not None and not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Configured checkpoint_path not found: {checkpoint_path}")
        self.model_key = model_key
        self.device_name = device
        self.fusion_layers = list(fusion_layers)
        if checkpoint_path is not None:
            cfg = load_models_config()["models"][model_key]
            repo_root = Path(repo_dir) if repo_dir else REPO_ROOT
            model = torch.hub.load(
                str(repo_root),
                cfg["hub_entry"],
                source="local",
                weights=str(checkpoint_path),
                trust_repo=True,
            )
            model = freeze_model(model)
            if device.startswith("cuda"):
                model = model.cuda()
            else:
                model = model.to(device)
            self.loaded = LoadedModel(
                key=model_key,
                model=model,
                config=cfg,
                source="torch_hub",
                checkpoint_path=str(checkpoint_path),
                param_count=sum(p.numel() for p in model.parameters()),
                device=device,
            )
        else:
            self.loaded = load_model(
                model_key,
                device=device,
                repo_dir=repo_dir,
                weights_dir=weights_dir,
                prefer="torch_hub",
            )
        self.backbone = self.loaded.model
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    @property
    def checkpoint_path(self) -> str | None:
        return self.loaded.checkpoint_path

    @property
    def patch_size(self) -> int:
        patch = getattr(self.backbone, "patch_size", 16)
        return int(patch[0] if isinstance(patch, tuple) else patch)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> BackboneFeatureBatch:
        images = images.to(self.device_name, non_blocking=True)
        layers = self.backbone.get_intermediate_layers(
            images,
            n=self.fusion_layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        patch_tokens = [patch for patch, _ in layers]
        cls_tokens = [cls for _, cls in layers]
        patch_fused = torch.cat(patch_tokens, dim=-1)
        cls_final = cls_tokens[-1]
        batch_size, num_patches, patch_dim = patch_tokens[0].shape
        side_h = images.shape[-2] // self.patch_size
        side_w = images.shape[-1] // self.patch_size
        if side_h * side_w != num_patches:
            raise ValueError(
                f"Patch token count mismatch: {num_patches} != {side_h}*{side_w}. "
                "Input resolution must remain divisible by patch size."
            )
        return BackboneFeatureBatch(
            patch_tokens=patch_fused,
            cls_token=cls_final,
            grid_hw=(side_h, side_w),
            patch_dim=patch_dim,
            fused_dim=patch_fused.shape[-1],
            layer_ids=list(self.fusion_layers),
            checkpoint_path=self.loaded.checkpoint_path,
        )

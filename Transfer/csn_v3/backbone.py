"""ConvNeXt V2-Base local encoder with staged freezing."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import timm

LOGGER = logging.getLogger(__name__)


class ConvNeXtV2Backbone(nn.Module):
    """ConvNeXt V2-Base multiscale features at strides 4/8/16/32."""

    def __init__(
        self,
        model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
    ):
        super().__init__()
        if not pretrained:
            self.encoder = timm.create_model(
                model_name,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                pretrained=False,
            )
            self.pretrained_loaded = False
            return
        try:
            self.encoder = timm.create_model(
                model_name,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                pretrained=True,
            )
            self.pretrained_loaded = True
        except Exception as exc:
            LOGGER.error("ConvNeXt pretrained load failed for %s: %s", model_name, exc)
            raise RuntimeError(
                f"ConvNeXt pretrained weights required but failed to load: {model_name}"
            ) from exc

        self.feature_channels = list(self.encoder.feature_info.channels())
        self.feature_strides = list(self.encoder.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)

    def set_stage_trainable(self, stage12_lr: float, stage34_lr: float) -> None:
        """Freeze/unfreeze stages by LR: stage12_lr=0 freezes stages 0-1."""
        for pname, param in self.encoder.named_parameters():
            if "stages_0" in pname or "stages_1" in pname:
                param.requires_grad_(stage12_lr > 0)
            elif "stages_2" in pname or "stages_3" in pname:
                param.requires_grad_(stage34_lr > 0)
            elif "stem" in pname:
                param.requires_grad_(stage12_lr > 0)

    def param_groups(
        self,
        lr_stage12: float,
        lr_stage34: float,
        lr_decay: float = 0.875,
    ) -> list[dict]:
        groups = []
        stage_map = [
            ("stages_3", lr_stage34),
            ("stages_2", lr_stage34 * lr_decay),
            ("stages_1", lr_stage12 * lr_decay),
            ("stages_0", lr_stage12 * lr_decay**2),
        ]
        assigned = set()
        for stage_key, lr in stage_map:
            if lr <= 0:
                continue
            params = []
            for pname, param in self.encoder.named_parameters():
                if stage_key in pname and param.requires_grad:
                    params.append(param)
                    assigned.add(pname)
            if params:
                groups.append({"params": params, "lr": lr})
        stem_params = [
            p for n, p in self.encoder.named_parameters()
            if "stem" in n and p.requires_grad and n not in assigned
        ]
        if stem_params and lr_stage12 > 0:
            groups.append({"params": stem_params, "lr": lr_stage12 * lr_decay**3})
        return groups

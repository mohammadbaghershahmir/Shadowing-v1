"""Trainable ConvNeXt V2-Base local feature extractor via timm."""
from __future__ import annotations
import torch
import torch.nn as nn
import timm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LocalBackbone(nn.Module):
    """ConvNeXt V2-Base with features_only=True.

    Returns 4 feature maps at strides 4, 8, 16, 32
    with channels [128, 256, 512, 1024].
    """

    def __init__(
        self,
        model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
        )
        self.feature_channels = self.encoder.feature_info.channels()  # [128, 256, 512, 1024]
        self.feature_strides = self.encoder.feature_info.reduction()  # [4, 8, 16, 32]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)

    def param_groups(self, lr_backbone: float, lr_decay: float = 0.875) -> list[dict]:
        """Build parameter groups with layer-wise LR decay.

        ConvNeXt V2-Base has 4 stages. Apply decay from last to first:
        stage3 (stride32): lr_backbone * decay^0
        stage2 (stride16): lr_backbone * decay^1
        stage1 (stride8):  lr_backbone * decay^2
        stage0 (stride4):  lr_backbone * decay^3
        Stem gets same LR as stage0.
        """
        groups = []
        stage_names = ["stages_3", "stages_2", "stages_1", "stages_0"]
        for i, name in enumerate(stage_names):
            lr = lr_backbone * (lr_decay ** i)
            params = []
            for pname, param in self.encoder.named_parameters():
                stage_key = f"stages.{3 - i}"
                if stage_key in pname:
                    params.append(param)
            if params:
                groups.append({"params": params, "lr": lr})

        stem_lr = lr_backbone * (lr_decay ** len(stage_names))
        stem_params = []
        for pname, param in self.encoder.named_parameters():
            if "stem" in pname:
                stem_params.append(param)
        if stem_params:
            groups.append({"params": stem_params, "lr": stem_lr})

        return groups

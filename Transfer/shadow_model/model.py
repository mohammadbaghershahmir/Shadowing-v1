"""ShadowNet: dual-DINO BEST_V1 model assembly."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from shadow_model.backbone_local import LocalBackbone
from shadow_model.decoder import UPerDecoder
from shadow_model.fusion import GlobalDinoFusion
from shadow_model.global_dino import FrozenDinoEncoder, normalize_rgb_tensor
from shadow_model.heads import AuxHead, ClassHead, TransitionHead
from shadow_model.local_dino_fusion import LocalDinoFusion
from shadow_model.profiles import ExperimentProfile
from shadow_model.stem import StructuralStem


class ShadowNet(nn.Module):
    """Dense three-class categorical segmentation with dual frozen DINO views."""

    def __init__(
        self,
        profile: ExperimentProfile,
        *,
        local_model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        num_classes: int = 3,
        decoder_dim: int = 256,
        stem_channels: list[int] | None = None,
        refine_channels: int = 64,
        dino_dim: int = 1024,
        dino_cfg: Any | None = None,
        cross_attn_blocks: int = 1,
        cross_attn_heads: int = 4,
        gate_init_local: float = 0.05,
        gate_init_global: float = 0.05,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()
        if stem_channels is None:
            stem_channels = [32, 64, 96]

        self.profile = profile
        self.backbone = LocalBackbone(local_model_name, pretrained)
        self.use_structural_stem = profile.use_structural_stem
        self.stem = StructuralStem(in_channels=3, channels=stem_channels) if self.use_structural_stem else None

        backbone_channels = self.backbone.feature_channels
        stem_512 = stem_channels[0] if self.use_structural_stem else 0
        stem_128 = stem_channels[2] if self.use_structural_stem else 0

        self.decoder = UPerDecoder(
            backbone_channels=backbone_channels,
            stem_channels_512=stem_512,
            stem_channels_128=stem_128,
            decoder_dim=decoder_dim,
            refine_channels=refine_channels,
        )

        self.class_head = ClassHead(refine_channels, num_classes)
        self.use_transition_head = profile.use_transition_head
        self.transition_head = TransitionHead(refine_channels) if profile.use_transition_head else None
        self.use_aux_head = profile.use_aux_head
        self.aux_head = AuxHead(decoder_dim, num_classes) if profile.use_aux_head else None

        self.use_local_dino = profile.use_local_dino
        self.use_global_fusion = profile.use_global_fusion
        self.dino_encoder: FrozenDinoEncoder | None = None
        if dino_cfg is not None and (profile.use_local_dino or profile.use_global_fusion):
            self.dino_encoder = FrozenDinoEncoder.get_shared(dino_cfg)

        self.local_dino_fusion = (
            LocalDinoFusion(decoder_dim, dino_dim, gate_init_local) if profile.use_local_dino else None
        )
        self.global_fusion = (
            GlobalDinoFusion(
                decoder_dim, dino_dim, cross_attn_blocks, cross_attn_heads, gate_init_global,
                global_size=getattr(dino_cfg, "input_size", 512) if dino_cfg else 512,
            )
            if profile.use_global_fusion
            else None
        )
        self.lateral_c3 = nn.Sequential(
            nn.Conv2d(backbone_channels[2], decoder_dim, 1, bias=False),
            nn.GroupNorm(32, decoder_dim),
        )

    def _extract_local_dino(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        assert self.dino_encoder is not None
        return self.dino_encoder.forward_tokens(rgb_norm)

    def forward(
        self,
        rgb: torch.Tensor,
        onehot: torch.Tensor,
        global_tokens: torch.Tensor | None = None,
        crop_box: torch.Tensor | None = None,
        global_valid_mask: torch.Tensor | None = None,
        letterbox_meta: dict | None = None,
        local_dino_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(rgb)
        c3 = features[2]
        l3_baseline = self.lateral_c3(c3)

        if self.use_structural_stem and self.stem is not None:
            stem_512, stem_128 = self.stem(onehot)
        else:
            b, _, h, w = rgb.shape
            stem_512 = torch.zeros(b, 0, h, w, device=rgb.device, dtype=rgb.dtype)
            stem_128 = torch.zeros(b, 0, h // 4, w // 4, device=rgb.device, dtype=rgb.dtype)

        h_feat = l3_baseline
        if self.use_local_dino and self.local_dino_fusion is not None:
            if local_dino_tokens is None and self.dino_encoder is not None:
                with torch.no_grad():
                    local_dino_tokens = self._extract_local_dino(rgb)
            if local_dino_tokens is not None:
                _, _, H, W = h_feat.shape
                h_feat = self.local_dino_fusion(h_feat, local_dino_tokens, H, W)

        l3_for_decoder = h_feat
        if self.use_global_fusion and self.global_fusion is not None:
            if global_tokens is None or crop_box is None or letterbox_meta is None:
                raise ValueError("BEST_V1 global fusion requires global_tokens, crop_box, letterbox_meta")
            l3_for_decoder, _ = self.global_fusion(
                h_feat, global_tokens, crop_box, letterbox_meta, global_valid_mask
            )

        refined, aux_feat = self.decoder(features, stem_512, stem_128, l3_for_decoder)

        outputs: dict[str, torch.Tensor] = {"class_logits": self.class_head(refined)}
        if self.use_transition_head and self.transition_head is not None:
            outputs["transition_logits"] = self.transition_head(refined)
        if self.use_aux_head and self.aux_head is not None:
            outputs["aux_logits"] = self.aux_head(aux_feat)
        outputs["l3_baseline"] = l3_baseline
        return outputs

    def trainable_param_groups(
        self,
        lr_new: float = 1.25e-4,
        lr_backbone: float = 1.5e-5,
        lr_decay: float = 0.875,
        weight_decay: float = 0.05,
        no_decay_keywords: tuple[str, ...] = ("bias", "norm"),
    ) -> list[dict]:
        """Build optimizer groups; every trainable param exactly once."""
        groups: list[dict] = []
        covered: set[int] = set()

        for grp in self.backbone.param_groups(lr_backbone, lr_decay):
            decay, no_decay = [], []
            for p in grp["params"]:
                name = next((n for n, pp in self.backbone.named_parameters() if pp is p), "")
                (no_decay if any(k in name for k in no_decay_keywords) else decay).append(p)
                covered.add(id(p))
            if decay:
                groups.append({"params": decay, "lr": grp["lr"], "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": grp["lr"], "weight_decay": 0.0})

        new_decay, new_no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in covered:
                continue
            if "dino_encoder" in name:
                continue
            (new_no_decay if any(k in name for k in no_decay_keywords) else new_decay).append(p)
            covered.add(id(p))

        if new_decay:
            groups.append({"params": new_decay, "lr": lr_new, "weight_decay": weight_decay})
        if new_no_decay:
            groups.append({"params": new_no_decay, "lr": lr_new, "weight_decay": 0.0})
        return groups

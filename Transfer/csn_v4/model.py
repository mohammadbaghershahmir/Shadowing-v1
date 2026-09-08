"""CarpetShadeNetV4 full model assembly."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from csn_v3.adapters import CropCoordEmbedding
from csn_v3.conditioners import MaskEncoder, TransitionEncoder
from csn_v3.decoder import MultiscaleGatedDecoder
from csn_v3.dino_encoder import SharedDinoEncoder, normalize_rgb_tensor, repeat_bw_to_rgb
from csn_v3.factorized_logits import compute_base_logits
from csn_v3.heads import (
    AffinityHead,
    JointDecisionHead,
    LineAwareRefiner,
    TransitionHead,
    WhereHead,
)
from csn_v3.stem import HighResStructuralStem
from csn_v4.backbone import ConvNeXtV2Backbone
from csn_v4.config import CSNV4Config
from csn_v4.fusion import FourBranchFusion
from csn_v4.heads import CategoricalLevelHead
from csn_v3.data.multiscale import build_global_token_padding_mask


def _ensure_mask_b1hw(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mask.dtype == torch.bool:
        x = mask.float()
    elif mask.is_floating_point():
        x = mask
    else:
        x = (mask > 127).float()
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 4 and x.shape[1] != 1:
        x = x[:, :1]
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1, -1, -1)
    return x


class CarpetShadeNetV4(nn.Module):
    def __init__(self, cfg: CSNV4Config):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        stem_ch = tuple(m.structural_stem.channels)
        self.stem = HighResStructuralStem(list(stem_ch))
        self.backbone = ConvNeXtV2Backbone(m.convnext.model_name, m.convnext.pretrained)
        self.dino = SharedDinoEncoder.get_shared(m.dino)
        self.fusion = FourBranchFusion(
            fusion_dim=m.fusion.dim,
            convnext_dim=m.convnext.output_dims[2],
            dino_dim=m.dino.embed_dim,
            num_heads=m.fusion.cross_attention_heads,
            num_blocks=m.fusion.cross_attention_blocks,
            zero_init_residual=m.fusion.zero_init_residual,
        )
        self.decoder = MultiscaleGatedDecoder(
            backbone_channels=self.backbone.feature_channels,
            stem_channels=stem_ch,
            decoder_channels=m.decoder.channels,
            fusion_dim=m.fusion.dim,
        )
        feat_ch = m.decoder.channels[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_feat_proj = nn.Linear(m.fusion.dim, 256)
        self.where_head = WhereHead(feat_ch)
        self.transition_head = TransitionHead(feat_ch)
        self.affinity_head = AffinityHead(feat_ch)
        self.mask_encoder = MaskEncoder()
        self.transition_encoder = TransitionEncoder()
        self.level_head = CategoricalLevelHead(feat_ch, feat_ch, 256)
        self.joint_head = JointDecisionHead(feat_ch)
        self.refiner = LineAwareRefiner(feat_ch)
        # Joint/refiner deltas use ZeroInitConv2d (exact zeros) so refined==base and
        # base_refined_consistency stays 0 until those weights move. Seed a tiny
        # residual so brc is observable from step 0 without destabilizing training.
        self._seed_residual_deltas(std=1e-2)
        self._training_mode = True

    def _seed_residual_deltas(self, std: float = 1e-2) -> None:
        for delta in (self.joint_head.delta, self.refiner.delta):
            conv = getattr(delta, "conv", delta)
            if isinstance(conv, nn.Conv2d):
                nn.init.normal_(conv.weight, mean=0.0, std=std)

    def train(self, mode: bool = True) -> CarpetShadeNetV4:
        super().train(mode)
        self._training_mode = mode
        self.dino.eval()
        for p in self.dino.backbone.parameters():
            p.requires_grad_(False)
        return self

    def _prep_rgb(self, bw: torch.Tensor) -> torch.Tensor:
        return normalize_rgb_tensor(repeat_bw_to_rgb(bw))

    def _global_vec(self, f16: torch.Tensor) -> torch.Tensor:
        return self.global_feat_proj(self.global_pool(f16).flatten(1))

    def _build_global_pad_mask(
        self,
        letterbox_meta: dict | list | None,
        num_tokens: int,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor | None:
        if letterbox_meta is None:
            return None
        metas = letterbox_meta if isinstance(letterbox_meta, list) else [letterbox_meta]
        masks = []
        for i in range(batch_size):
            meta = metas[i] if i < len(metas) else metas[0]
            if isinstance(meta, dict):
                masks.append(
                    build_global_token_padding_mask(
                        meta, num_tokens, self.cfg.model.dino.patch_size, device,
                    ).squeeze(0)
                )
        if not masks:
            return None
        return torch.stack(masks, dim=0)

    def forward(
        self,
        local_bw: torch.Tensor,
        context_bw: torch.Tensor | None = None,
        full_bw: torch.Tensor | None = None,
        crop_coords: torch.Tensor | None = None,
        global_tokens: torch.Tensor | None = None,
        letterbox_meta: dict | list | None = None,
        local_rgb: torch.Tensor | None = None,
        local_onehot: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
        gt_shade_mask: torch.Tensor | None = None,
        gt_transition_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = local_bw.shape[0]
        input_mode = getattr(self.cfg.data, "input_mode", "bw")

        if input_mode == "magenta":
            if local_rgb is None:
                raise RuntimeError("Magenta mode requires local_rgb; BW fallback is disabled in CSN-V4.")
            line_mask = local_onehot[:, 0:1].float() if local_onehot is not None else (1.0 - local_bw)
            local_rgb_norm = normalize_rgb_tensor(local_rgb)
        else:
            line_mask = 1.0 - local_bw
            local_rgb_norm = self._prep_rgb(local_bw)

        h0, h1, h2 = self.stem(line_mask)
        features = self.backbone(local_rgb_norm)
        dino_local = self.dino.forward_local(local_rgb_norm)

        if context_bw is not None:
            ctx_rgb = self._prep_rgb(context_bw)
            ctx_rgb_in = F.interpolate(ctx_rgb, size=local_rgb_norm.shape[-2:], mode="nearest") if context_bw.shape[-1] != local_bw.shape[-1] else ctx_rgb
            dino_context = self.dino.forward_context(ctx_rgb_in)
        else:
            dino_context = dino_local

        if global_tokens is None and full_bw is not None:
            global_tokens = self.dino.forward_global_tokens(self._prep_rgb(full_bw))
        elif global_tokens is None:
            global_tokens = self.dino.forward_global_tokens(local_rgb_norm)

        global_pad_mask = self._build_global_pad_mask(
            letterbox_meta, global_tokens.shape[1], local_bw.device, batch_size,
        )

        if crop_coords is None:
            crop_coords = torch.tensor(
                [[0.0, 0.0, 1.0, 1.0, 0.5, 0.5]], device=local_bw.device, dtype=local_bw.dtype,
            ).expand(batch_size, -1)

        f16, gate_maps = self.fusion(
            features[2], dino_local, dino_context, global_tokens, crop_coords,
            local_size=local_bw.shape[-1],
            global_key_padding_mask=global_pad_mask,
        )
        dec = self.decoder(features, f16, h0, h1, h2)
        feat = dec["F"]
        global_vec = self._global_vec(f16)

        where_logits = self.where_head(feat)
        transition_logits = self.transition_head(feat)
        affinity_logits = self.affinity_head(feat)
        where_prob = torch.sigmoid(where_logits)

        if teacher_forcing >= 1.0 and gt_shade_mask is not None:
            where_cond_in = _ensure_mask_b1hw(gt_shade_mask, batch_size)
        elif teacher_forcing <= 0.0:
            where_cond_in = where_prob
        else:
            gt_w = _ensure_mask_b1hw(gt_shade_mask, batch_size) if gt_shade_mask is not None else where_prob
            where_cond_in = teacher_forcing * gt_w + (1 - teacher_forcing) * where_prob

        trans_prob = torch.sigmoid(transition_logits)
        if teacher_forcing >= 1.0 and gt_transition_mask is not None:
            trans_cond_in = _ensure_mask_b1hw(gt_transition_mask, batch_size)
        elif teacher_forcing <= 0.0:
            trans_cond_in = trans_prob
        else:
            gt_t = _ensure_mask_b1hw(gt_transition_mask, batch_size) if gt_transition_mask is not None else trans_prob
            trans_cond_in = teacher_forcing * gt_t + (1 - teacher_forcing) * trans_prob

        where_enc = self.mask_encoder(where_cond_in)
        trans_enc = self.transition_encoder(trans_cond_in)
        level_logits = self.level_head(feat, where_enc, trans_enc, global_vec)
        base_logits = compute_base_logits(where_logits, level_logits)
        delta, corr_gate = self.joint_head(
            feat, where_logits, transition_logits, affinity_logits,
            level_logits, base_logits, global_vec,
        )
        joint_logits = base_logits + torch.sigmoid(corr_gate) * delta
        refine_delta, refine_gate = self.refiner(
            joint_logits, feat, line_mask, where_logits, transition_logits, affinity_logits,
        )
        refined_logits = joint_logits + torch.sigmoid(refine_gate) * refine_delta

        out: dict[str, Any] = {
            "where_logits": where_logits,
            "transition_logits": transition_logits,
            "affinity_logits": affinity_logits,
            "level_logits": level_logits,
            "base_logits": base_logits,
            "joint_logits": joint_logits,
            "refined_logits": refined_logits,
        }
        if not self._training_mode:
            out["pred_class"] = refined_logits.argmax(dim=1)
            out["branch_gate_maps"] = gate_maps
        return out

    def trainable_param_groups(self, lr_new: float = 2e-4, lr_stage34: float = 2e-5, lr_stage12: float = 0.0) -> list[dict]:
        new_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad and not n.startswith("dino.backbone") and not n.startswith("backbone.")
        ]
        groups = [{"params": new_params, "lr": lr_new}]
        groups.extend(self.backbone.param_groups(lr_stage12, lr_stage34))
        return groups

    def set_convnext_stages(self, stage12_lr: float, stage34_lr: float) -> None:
        self.backbone.set_stage_trainable(stage12_lr, stage34_lr)

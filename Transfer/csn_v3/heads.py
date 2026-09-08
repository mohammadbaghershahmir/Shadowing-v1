"""Prediction heads for CSN-V3."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from csn_v3.adapters import ZeroInitConv2d
from csn_v3.constants import AFFINITY_NUM_CHANNELS, NUM_CLASSES, NUM_LEVEL_CLASSES, NUM_ORDINAL_THRESHOLDS


class WhereHead(nn.Module):
    def __init__(self, in_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransitionHead(nn.Module):
    def __init__(self, in_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffinityHead(nn.Module):
    """8 channels: h/v offsets at distances 1,2,4,8."""

    def __init__(self, in_ch: int = 64, num_channels: int = AFFINITY_NUM_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, num_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SourceEncoder(nn.Module):
    """Encode a single source tensor for fusion."""

    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LevelOrdinalHead(nn.Module):
    """Level (3-class) + ordinal (2-threshold) heads with gated fusion."""

    def __init__(self, feat_ch: int = 64, cond_ch: int = 64, global_dim: int = 256):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, feat_ch)
        self.gate = nn.Conv2d(feat_ch * 4, 4, 1)
        self.level = nn.Conv2d(feat_ch, NUM_LEVEL_CLASSES, 1)
        self.ordinal = nn.Conv2d(feat_ch, NUM_ORDINAL_THRESHOLDS, 1)

    def forward(
        self,
        feat: torch.Tensor,
        where_enc: torch.Tensor,
        trans_enc: torch.Tensor,
        global_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = feat.shape
        g = self.global_proj(global_vec).view(B, C, 1, 1).expand(-1, -1, H, W)
        stacked = torch.cat([feat, where_enc, trans_enc, g], dim=1)
        w = F.softmax(self.gate(stacked), dim=1)
        fused = (
            w[:, 0:1] * feat
            + w[:, 1:2] * where_enc
            + w[:, 2:3] * trans_enc
            + w[:, 3:4] * g
        )
        return self.level(fused), self.ordinal(fused)


class JointDecisionHead(nn.Module):
    def __init__(self, feat_ch: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.enc_feat = SourceEncoder(feat_ch, feat_ch)
        self.enc_where = SourceEncoder(1, feat_ch)
        self.enc_trans = SourceEncoder(1, feat_ch)
        self.enc_aff = SourceEncoder(8, feat_ch)
        self.enc_level = SourceEncoder(NUM_LEVEL_CLASSES, feat_ch)
        self.enc_base = SourceEncoder(num_classes, feat_ch)
        self.enc_global = nn.Linear(256, feat_ch)
        self.gate_in = nn.Conv2d(feat_ch * 7, feat_ch, 1)
        self.delta = ZeroInitConv2d(feat_ch, num_classes, 3)
        self.correction_gate = nn.Conv2d(feat_ch, num_classes, 1)

    def forward(
        self,
        feat: torch.Tensor,
        where_logits: torch.Tensor,
        transition_logits: torch.Tensor,
        affinity_logits: torch.Tensor,
        level_logits: torch.Tensor,
        base_logits: torch.Tensor,
        global_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = feat.shape
        g = self.enc_global(global_vec).view(B, -1, 1, 1).expand(-1, -1, H, W)
        parts = torch.cat([
            self.enc_feat(feat),
            self.enc_where(where_logits),
            self.enc_trans(transition_logits),
            self.enc_aff(affinity_logits),
            self.enc_level(level_logits),
            self.enc_base(base_logits),
            g,
        ], dim=1)
        h = F.gelu(self.gate_in(parts))
        return self.delta(h), self.correction_gate(h)


class LineAwareRefiner(nn.Module):
    """4 residual blocks with dilated depthwise convolutions."""

    def __init__(self, feat_ch: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.line_enc = SourceEncoder(1, feat_ch)
        self.where_enc = SourceEncoder(1, feat_ch)
        self.trans_enc = SourceEncoder(1, feat_ch)
        self.aff_enc = SourceEncoder(8, feat_ch)
        dilations = [1, 2, 4, 1]
        blocks = []
        in_ch = num_classes + feat_ch * 5
        mid = feat_ch
        for d in dilations:
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch if not blocks else mid, mid, 3, padding=d, dilation=d, groups=1, bias=False),
                nn.GroupNorm(min(32, mid), mid),
                nn.GELU(),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.delta = ZeroInitConv2d(mid, num_classes, 3)
        self.refine_gate = nn.Conv2d(mid, num_classes, 1)

    def forward(
        self,
        joint_logits: torch.Tensor,
        feat: torch.Tensor,
        line_emb: torch.Tensor,
        where_logits: torch.Tensor,
        transition_logits: torch.Tensor,
        affinity_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        where_p = torch.sigmoid(where_logits)
        trans_p = torch.sigmoid(transition_logits)
        x = torch.cat([
            joint_logits,
            feat,
            self.line_enc(line_emb),
            self.where_enc(where_p),
            self.trans_enc(trans_p),
            self.aff_enc(affinity_logits),
        ], dim=1)
        h = x
        for blk in self.blocks:
            h = blk(h)
        return self.delta(h), self.refine_gate(h)

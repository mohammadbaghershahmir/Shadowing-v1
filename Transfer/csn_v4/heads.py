"""V4 prediction heads — categorical shade subtype only, no ordinal branch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from csn_v4.constants import NUM_SHADE_SUBTYPES


class CategoricalLevelHead(nn.Module):
    """Three-class categorical shade subtype head (200/150/100)."""

    def __init__(self, feat_ch: int = 64, cond_ch: int = 64, global_dim: int = 256):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, feat_ch)
        self.gate = nn.Conv2d(feat_ch * 4, 4, 1)
        self.level = nn.Conv2d(feat_ch, NUM_SHADE_SUBTYPES, 1)

    def forward(
        self,
        feat: torch.Tensor,
        where_enc: torch.Tensor,
        trans_enc: torch.Tensor,
        global_vec: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = feat.shape
        g = self.global_proj(global_vec).view(b, c, 1, 1).expand(-1, -1, h, w)
        stacked = torch.cat([feat, where_enc, trans_enc, g], dim=1)
        wgt = F.softmax(self.gate(stacked), dim=1)
        fused = (
            wgt[:, 0:1] * feat
            + wgt[:, 1:2] * where_enc
            + wgt[:, 2:3] * trans_enc
            + wgt[:, 3:4] * g
        )
        return self.level(fused)

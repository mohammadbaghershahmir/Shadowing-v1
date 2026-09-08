"""Local DINO view fusion with gated residual on baseline C3 lateral."""
from __future__ import annotations

import torch
import torch.nn as nn

from shadow_model.geometry_align import crop_uv_grid, sincos_posembed_from_coords


class LocalDinoFusion(nn.Module):
    """Fuse frozen local DINO tokens with baseline lateral features."""

    def __init__(self, fusion_dim: int = 256, dino_dim: int = 1024, gate_init: float = 0.05):
        super().__init__()
        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_dim, fusion_dim, 1, bias=False),
            nn.GroupNorm(min(32, fusion_dim), fusion_dim),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, fusion_dim), fusion_dim),
            nn.GELU(),
        )
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.fusion_dim = fusion_dim

    def forward(
        self,
        l3_baseline: torch.Tensor,
        local_dino_tokens: torch.Tensor,
        spatial_h: int,
        spatial_w: int,
    ) -> torch.Tensor:
        """l3_baseline [B,256,32,32], local_dino_tokens [B,N,1024]."""
        B, _, H, W = l3_baseline.shape
        dino_grid = local_dino_tokens.transpose(1, 2).reshape(B, -1, H, W)
        dino_feat = self.dino_proj(dino_grid)
        u, v = crop_uv_grid(H, W, l3_baseline.device)
        pe = sincos_posembed_from_coords(
            u.unsqueeze(0).expand(B, -1),
            v.unsqueeze(0).expand(B, -1),
            self.fusion_dim,
        ).reshape(B, H, W, self.fusion_dim).permute(0, 3, 1, 2)
        delta = self.mix(torch.cat([l3_baseline, dino_feat + pe], dim=1))
        return l3_baseline + torch.tanh(self.gate) * delta

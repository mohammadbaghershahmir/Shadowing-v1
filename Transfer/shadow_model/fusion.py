"""Gated cross-attention global fusion with absolute letterbox coordinates."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from shadow_model.geometry_align import crop_uv_grid, map_crop_to_letterbox, sincos_posembed_from_coords


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4, gate_init: float = 0.05):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.gate_ffn = nn.Parameter(torch.tensor(gate_init))

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(query)
        k = v = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, k, v, key_padding_mask=key_padding_mask)
        query = query + torch.tanh(self.gate) * attn_out
        query = query + torch.tanh(self.gate_ffn) * self.ffn(query)
        return query


class GlobalDinoFusion(nn.Module):
    """Cross-attend local queries to cached global DINO tokens."""

    def __init__(
        self,
        fusion_dim: int = 256,
        global_dim: int = 1024,
        num_blocks: int = 1,
        num_heads: int = 4,
        gate_init: float = 0.05,
        global_size: int = 512,
    ):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 1, bias=False),
            nn.GroupNorm(min(32, fusion_dim), fusion_dim),
        )
        self.global_proj = nn.Linear(global_dim, fusion_dim, bias=False)
        self.blocks = nn.ModuleList([
            GatedCrossAttentionBlock(fusion_dim, num_heads, gate_init) for _ in range(num_blocks)
        ])
        self.delta_proj = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 1, bias=False),
            nn.GroupNorm(min(32, fusion_dim), fusion_dim),
        )
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.fusion_dim = fusion_dim
        self.global_size = global_size

    def forward(
        self,
        h_local: torch.Tensor,
        global_tokens: torch.Tensor,
        crop_box: torch.Tensor,
        letterbox_meta: dict,
        global_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_fused, delta) where h_fused = h_local + gate*delta."""
        B, C, H, W = h_local.shape
        q = self.query_proj(h_local)
        q_tokens = q.flatten(2).transpose(1, 2)

        u, v = crop_uv_grid(H, W, h_local.device)
        meta_t = {
            "scale": letterbox_meta["scale"],
            "offset_x": letterbox_meta["offset_x"],
            "offset_y": letterbox_meta["offset_y"],
            "image_width": letterbox_meta["image_width"],
            "image_height": letterbox_meta["image_height"],
        }
        x_lb, y_lb = map_crop_to_letterbox(crop_box, u, v, meta_t, self.global_size)
        q_tokens = q_tokens + sincos_posembed_from_coords(x_lb, y_lb, self.fusion_dim)

        g_proj = self.global_proj(global_tokens)
        n_global = global_tokens.shape[1]
        gh = gw = int(math.sqrt(n_global))
        if gh * gw == n_global:
            gy = torch.arange(gh, device=h_local.device, dtype=torch.float32)
            gx = torch.arange(gw, device=h_local.device, dtype=torch.float32)
            if gh > 1:
                gy = gy / (gh - 1)
            if gw > 1:
                gx = gx / (gw - 1)
            vv, uu = torch.meshgrid(gy, gx, indexing="ij")
            g_pe = sincos_posembed_from_coords(
                uu.reshape(1, -1).expand(B, -1),
                vv.reshape(1, -1).expand(B, -1),
                self.fusion_dim,
            )
            g_proj = g_proj + g_pe

        key_padding_mask = None
        if global_valid_mask is not None:
            key_padding_mask = ~global_valid_mask.bool()

        out = q_tokens
        for block in self.blocks:
            out = block(out, g_proj, key_padding_mask)

        delta = self.delta_proj(out.transpose(1, 2).reshape(B, self.fusion_dim, H, W))
        return h_local + torch.tanh(self.gate) * delta, delta

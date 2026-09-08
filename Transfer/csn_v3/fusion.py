"""4-branch gated fusion with context/global cross-attention at stride 16."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from csn_v3.adapters import BranchAdapter, CropCoordEmbedding, ResidualFusionBlock, TokenAdapter


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Nq, C = query.shape
        q = self.q_proj(self.norm_q(query))
        k = self.k_proj(self.norm_kv(kv))
        v = self.v_proj(self.norm_kv(kv))
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            k = k * valid
            v = v * valid
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        out = attn.transpose(1, 2).reshape(B, Nq, C)
        return self.out_proj(out)


class DynamicSpatialChannelGate(nn.Module):
    """Factorized spatial + channel gate over 4 branches."""

    def __init__(self, dim: int = 256, num_branches: int = 4):
        super().__init__()
        self.spatial_logits = nn.Conv2d(dim * num_branches, num_branches, 1)
        self.channel_logits = nn.Conv2d(dim * num_branches, num_branches * dim, 1)

    def forward(self, branches: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stacked = torch.cat(branches, dim=1)
        B, _, H, W = stacked.shape
        dim = branches[0].shape[1]
        spatial = self.spatial_logits(stacked)
        spatial_w = F.softmax(spatial, dim=1)
        channel = self.channel_logits(stacked).view(B, 4, dim, H, W)
        channel_w = F.softmax(channel.mean(dim=(-2, -1), keepdim=True), dim=1)
        combined = spatial_w.unsqueeze(2) * channel_w
        combined = combined / combined.sum(dim=1, keepdim=True).clamp(min=1e-6)
        fused = sum(combined[:, i] * branches[i] for i in range(4))
        return fused, {"spatial": spatial_w, "channel": channel_w}


class FourBranchFusion(nn.Module):
    """ConvNeXt + local/context/global DINO fusion at stride 16."""

    def __init__(
        self,
        fusion_dim: int = 256,
        convnext_dim: int = 512,
        dino_dim: int = 1024,
        num_heads: int = 4,
        num_blocks: int = 1,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.conv_proj = BranchAdapter(convnext_dim, fusion_dim)
        self.local_dino_adapter = TokenAdapter(dino_dim, fusion_dim)
        self.context_dino_adapter = TokenAdapter(dino_dim, fusion_dim)
        self.global_dino_adapter = nn.Linear(dino_dim, fusion_dim, bias=False)
        self.coord_embed = CropCoordEmbedding(6, 128, fusion_dim)

        self.context_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(fusion_dim, num_heads) for _ in range(num_blocks)
        ])
        self.global_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(fusion_dim, num_heads) for _ in range(num_blocks)
        ])

        self.branch_adapters = nn.ModuleList([
            BranchAdapter(fusion_dim, fusion_dim) for _ in range(4)
        ])
        self.gate = DynamicSpatialChannelGate(fusion_dim, 4)
        self.residual = ResidualFusionBlock(fusion_dim)

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, D = tokens.shape
        gh, gw = h // 16, w // 16
        if N == gh * gw:
            return tokens.transpose(1, 2).reshape(B, D, gh, gw)
        return F.interpolate(
            tokens.transpose(1, 2).reshape(B, D, int(N**0.5), int(N**0.5)),
            size=(h // 16, w // 16),
            mode="nearest",
        )

    def forward(
        self,
        c3: torch.Tensor,
        dino_local: torch.Tensor,
        dino_context: torch.Tensor,
        dino_global_tokens: torch.Tensor,
        crop_coords: torch.Tensor,
        local_size: int = 512,
        global_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, _, H, W = c3.shape
        t_conv = self.branch_adapters[0](self.conv_proj(c3))
        t_local = self.branch_adapters[1](self.local_dino_adapter(dino_local))

        ctx_map = self.local_dino_adapter(dino_context)
        q = t_conv.flatten(2).transpose(1, 2)
        kv_ctx = ctx_map.flatten(2).transpose(1, 2)
        for blk in self.context_blocks:
            q = q + blk(q, kv_ctx)
        t_context = self.branch_adapters[2](q.transpose(1, 2).reshape(B, self.fusion_dim, H, W))

        kv_g = self.global_dino_adapter(dino_global_tokens)
        coord = self.coord_embed(crop_coords).unsqueeze(1)
        q_g = q + coord
        for blk in self.global_blocks:
            q_g = q_g + blk(q_g, kv_g, key_padding_mask=global_key_padding_mask)
        t_global = self.branch_adapters[3](q_g.transpose(1, 2).reshape(B, self.fusion_dim, H, W))

        f_mix, gate_maps = self.gate([t_conv, t_local, t_context, t_global])
        f16 = self.residual(f_mix, t_conv)
        gate_maps["branches"] = torch.stack([t_conv, t_local, t_context, t_global], dim=1)
        return f16, gate_maps

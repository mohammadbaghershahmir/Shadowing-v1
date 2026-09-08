"""DETR-style palette set prediction head."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def build_2d_sincos_pos_embed(grid_hw: tuple[int, int], dim: int, device: torch.device) -> torch.Tensor:
    """Return [1, H*W, dim] sin-cos positional embeddings."""
    if dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2D sin-cos, got {dim}.")
    h, w = grid_hw
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / max(1, (dim // 4) - 1)))
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    return pos.unsqueeze(0)


class PalettePredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.rgb = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.presence = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, queries: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_rgb = torch.sigmoid(self.rgb(queries))
        presence_logits = self.presence(queries).squeeze(-1)
        return {"pred_rgb": pred_rgb, "presence_logits": presence_logits}


class PaletteSetHead(nn.Module):
    """Query-based set predictor for unordered palettes."""

    def __init__(
        self,
        *,
        fused_dim: int,
        raw_color_dim: int = 15,
        cls_dim: int,
        max_num_colors: int,
        spare_queries: int = 4,
        head_dim: int = 256,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_ffn_dim: int = 1024,
        dropout: float = 0.1,
        aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.max_num_colors = int(max_num_colors)
        self.num_queries = int(max_num_colors + spare_queries)
        self.head_dim = int(head_dim)
        self.aux_loss = bool(aux_loss)

        self.patch_proj = nn.Linear(fused_dim + raw_color_dim, head_dim)
        self.cls_proj = nn.Linear(cls_dim, head_dim)
        self.query_embed = nn.Embedding(self.num_queries, head_dim)
        self.query_pos = nn.Parameter(torch.randn(1, self.num_queries, head_dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=head_dim,
                    nhead=decoder_heads,
                    dim_feedforward=decoder_ffn_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(head_dim)
        self.prediction_head = PalettePredictionHeads(head_dim)
        self.count_head = nn.Sequential(
            nn.LayerNorm(head_dim * 3),
            nn.Linear(head_dim * 3, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, self.max_num_colors + 1),
        )
        self.count_attn = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        raw_color_features: torch.Tensor | None = None,
        cls_token: torch.Tensor | None = None,
        *,
        grid_hw: tuple[int, int],
        patch_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if cls_token is None:
            if raw_color_features is None:
                raise ValueError("cls_token is required.")
            cls_token = raw_color_features
            raw_color_features = patch_tokens.new_zeros(
                patch_tokens.shape[0], patch_tokens.shape[1], self.patch_proj.in_features - patch_tokens.shape[-1]
            )
        if raw_color_features is None:
            raw_color_features = patch_tokens.new_zeros(
                patch_tokens.shape[0], patch_tokens.shape[1], self.patch_proj.in_features - patch_tokens.shape[-1]
            )
        memory = self.patch_proj(torch.cat([patch_tokens, raw_color_features], dim=-1))
        memory = memory + build_2d_sincos_pos_embed(grid_hw, self.head_dim, memory.device)
        memory_key_padding_mask = None
        if patch_valid_mask is not None:
            memory_key_padding_mask = ~patch_valid_mask.bool()

        batch_size = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + self.query_pos

        aux_outputs: list[dict[str, torch.Tensor]] = []
        hidden = queries
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(hidden, memory, memory_key_padding_mask=memory_key_padding_mask)
            if self.aux_loss and layer_idx < len(self.layers) - 1:
                aux_outputs.append(self.prediction_head(self.final_norm(hidden)))

        hidden = self.final_norm(hidden)
        out = self.prediction_head(hidden)
        cls_hidden = self.cls_proj(cls_token)
        if patch_valid_mask is None:
            pooled_mean = memory.mean(dim=1)
            pooled_attn = memory.mean(dim=1)
        else:
            weights = patch_valid_mask.float().unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            pooled_mean = (memory * weights).sum(dim=1) / denom
            attn_logits = self.count_attn(memory).squeeze(-1).masked_fill(~patch_valid_mask.bool(), float("-inf"))
            attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
            pooled_attn = (memory * attn).sum(dim=1)
        count_features = torch.cat([cls_hidden, pooled_mean, pooled_attn], dim=-1)
        out["count_logits"] = self.count_head(count_features)
        out["aux_outputs"] = aux_outputs
        return out

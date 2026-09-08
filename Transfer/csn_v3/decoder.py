"""Multiscale gated decoder D32 -> D1."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from csn_v3.adapters import BranchAdapter, ResidualFusionBlock


class PPM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scales: tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.GELU(),
            )
            for s in scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + out_ch * len(scales), out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feats = [x]
        for br in self.branches:
            f = br(x)
            feats.append(F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(feats, dim=1))


class PixelShuffleUp(nn.Module):
    """Trainable conv + PixelShuffle upsampling."""

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch * scale * scale), out_ch * scale * scale),
            nn.GELU(),
        )
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


class GatedLateralFusion(nn.Module):
    def __init__(self, dim: int, num_sources: int = 2):
        super().__init__()
        self.gate = nn.Conv2d(dim * num_sources, num_sources, 1)

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.cat(sources, dim=1)
        w = F.softmax(self.gate(stacked), dim=1)
        out = sum(w[:, i : i + 1] * sources[i] for i in range(len(sources)))
        return out


class MultiscaleGatedDecoder(nn.Module):
    """Top-down decoder producing F [B,64,512,512]."""

    def __init__(
        self,
        backbone_channels: list[int],
        stem_channels: tuple[int, int, int],
        decoder_channels: list[int],
        fusion_dim: int = 256,
    ):
        super().__init__()
        # decoder_channels: [D32, D16, D8, D4, D2, D1] channel dims
        c32, c16, c8, c4, c2, c1 = decoder_channels
        self.ppm = PPM(backbone_channels[3], c32)
        self.f16_in = BranchAdapter(fusion_dim, c16)

        self.up_32_16 = PixelShuffleUp(c32, c16, 2)
        self.lat_c2 = BranchAdapter(backbone_channels[1], c8)
        self.up_16_8 = PixelShuffleUp(c16, c8, 2)
        self.lat_c1 = BranchAdapter(backbone_channels[0], c4)
        self.lat_h2 = BranchAdapter(stem_channels[2], c4)
        self.up_8_4 = PixelShuffleUp(c8, c4, 2)
        self.lat_h1 = BranchAdapter(stem_channels[1], c2)
        self.up_4_2 = PixelShuffleUp(c4, c2, 2)
        self.lat_h0 = BranchAdapter(stem_channels[0], c1)
        self.up_2_1 = PixelShuffleUp(c2, c1, 2)

        self.gate_16 = GatedLateralFusion(c16, 2)
        self.gate_8 = GatedLateralFusion(c8, 2)
        self.gate_4 = GatedLateralFusion(c4, 3)
        self.gate_2 = GatedLateralFusion(c2, 2)
        self.gate_1 = GatedLateralFusion(c1, 2)

        self.refine_16 = ResidualFusionBlock(c16)
        self.refine_8 = ResidualFusionBlock(c8)
        self.refine_4 = ResidualFusionBlock(c4)
        self.refine_2 = ResidualFusionBlock(c2)
        self.refine_1 = ResidualFusionBlock(c1)
        self.out_channels = c1

    def forward(
        self,
        features: list[torch.Tensor],
        f16: torch.Tensor,
        stem_h0: torch.Tensor,
        stem_h1: torch.Tensor,
        stem_h2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        c1, c2, c3, c4 = features
        d32 = self.ppm(c4)
        d16 = self.refine_16(self.gate_16([self.up_32_16(d32), self.f16_in(f16)]), self.f16_in(f16))
        d8 = self.refine_8(
            self.gate_8([self.up_16_8(d16), self.lat_c2(c2)]),
            self.lat_c2(c2),
        )
        d4 = self.refine_4(
            self.gate_4([self.up_8_4(d8), self.lat_c1(c1), self.lat_h2(stem_h2)]),
            self.lat_c1(c1),
        )
        d2 = self.refine_2(
            self.gate_2([self.up_4_2(d4), self.lat_h1(stem_h1)]),
            self.lat_h1(stem_h1),
        )
        d1 = self.refine_1(
            self.gate_1([self.up_2_1(d2), self.lat_h0(stem_h0)]),
            self.lat_h0(stem_h0),
        )
        return {"D32": d32, "D16": d16, "D8": d8, "D4": d4, "D2": d2, "D1": d1, "F": d1}

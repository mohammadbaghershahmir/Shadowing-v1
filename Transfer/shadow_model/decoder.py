"""UPerNet/FPN decoder with PPM, stem fusion, and stride-1 refinement."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """Pyramid Pooling Module on C4 features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for scale in pool_scales:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.GELU(),
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_scales), out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feats = [x]
        for branch in self.branches:
            feat = branch(x)
            feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(feat)
        return self.bottleneck(torch.cat(feats, dim=1))


class UPerDecoder(nn.Module):
    """Full UPerNet decoder with PPM, FPN, stem fusion, and refinement."""

    def __init__(
        self,
        backbone_channels: list[int],   # [128, 256, 512, 1024]
        stem_channels_512: int = 32,
        stem_channels_128: int = 96,
        decoder_dim: int = 256,
        refine_channels: int = 64,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim

        self.ppm = PPM(backbone_channels[3], decoder_dim)

        self.laterals = nn.ModuleList()
        for ch in backbone_channels[:3]:
            self.laterals.append(nn.Sequential(
                nn.Conv2d(ch, decoder_dim, 1, bias=False),
                nn.GroupNorm(32, decoder_dim),
            ))

        self.smooth_convs = nn.ModuleList()
        for _ in range(3):
            self.smooth_convs.append(nn.Sequential(
                nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
                nn.GroupNorm(32, decoder_dim),
                nn.GELU(),
            ))

        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, decoder_dim),
            nn.GELU(),
        )

        self.stem_fuse_128 = nn.Sequential(
            nn.Conv2d(decoder_dim + stem_channels_128, decoder_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, decoder_dim),
            nn.GELU(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(decoder_dim + stem_channels_512, refine_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, refine_channels), refine_channels),
            nn.GELU(),
            nn.Conv2d(refine_channels, refine_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, refine_channels), refine_channels),
            nn.GELU(),
        )
        self.refine_channels = refine_channels

    def forward(
        self,
        features: list[torch.Tensor],   # C1..C4 from backbone
        stem_512: torch.Tensor,          # [B, 32, 512, 512]
        stem_128: torch.Tensor,          # [B, 96, 128, 128]
        fused_c3: torch.Tensor | None = None,  # optional cross-attn fused C3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (refined_stride1, fpn_stride4) for class head and aux head."""
        c1, c2, c3, c4 = features

        p4 = self.ppm(c4)  # [B, 256, 16, 16]

        l1 = self.laterals[0](c1)  # [B, 256, 128, 128]
        l2 = self.laterals[1](c2)  # [B, 256, 64, 64]
        if fused_c3 is not None:
            l3 = fused_c3
        else:
            l3 = self.laterals[2](c3)  # [B, 256, 32, 32]

        p3 = self.smooth_convs[2](l3 + F.interpolate(p4, size=l3.shape[2:], mode="bilinear", align_corners=False))
        p2 = self.smooth_convs[1](l2 + F.interpolate(p3, size=l2.shape[2:], mode="bilinear", align_corners=False))
        p1 = self.smooth_convs[0](l1 + F.interpolate(p2, size=l1.shape[2:], mode="bilinear", align_corners=False))

        target_size = p1.shape[2:]
        fpn_out = torch.cat([
            p1,
            F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False),
        ], dim=1)
        fpn_out = self.fpn_fuse(fpn_out)  # [B, 256, 128, 128]

        fpn_out = self.stem_fuse_128(torch.cat([fpn_out, stem_128], dim=1))

        aux_feat = fpn_out

        up = F.interpolate(fpn_out, size=stem_512.shape[2:], mode="bilinear", align_corners=False)
        refined = self.refine(torch.cat([up, stem_512], dim=1))

        return refined, aux_feat
